"""Strands-specific persisted background-task records."""

from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from typing import Any, TypeAlias, cast

from opentelemetry.trace import SpanContext, TraceFlags
from typing_extensions import NotRequired, TypedDict

from ..interrupt import _InterruptState
from ..types.session import decode_bytes_values, encode_bytes_values
from ..types.tools import ToolResult
from ._engine import validate_stored_engine_task
from ._engine_types import BackgroundTaskFailure, BackgroundTaskStatus, StoredEngineTask
from .types import BackgroundTask, BackgroundTaskError, BackgroundTaskResult

_FAILURE_TYPES = frozenset({"tool_error", "execution_error", "timeout", "recovery_error"})
_FRAMEWORK_INVOCATION_STATE_KEYS = frozenset(
    {
        "agent",
        "event_loop_cycle_id",
        "event_loop_cycle_span",
        "event_loop_cycle_trace",
        "event_loop_parent_cycle_id",
        "messages",
        "model",
        "system_prompt",
        "tool_config",
    }
)


class OriginTraceContext(TypedDict):
    """Serializable OpenTelemetry span context."""

    trace_id: int
    span_id: int
    trace_flags: int
    is_remote: NotRequired[bool]


class ToolTaskDescriptor(TypedDict):
    """Execution descriptor for one detached tool call."""

    original_tool_use_id: str
    tool_name: str
    input: NotRequired[Any]
    invocation_state: dict[str, Any]
    origin_trace_context: NotRequired[OriginTraceContext]


StoredBackgroundTask: TypeAlias = StoredEngineTask[ToolTaskDescriptor, ToolResult, dict[str, Any]]
_RECOVERY_MESSAGE = "Background task execution was interrupted while restoring persisted state"
_PERSISTED_RECORD_KEYS = frozenset(
    {
        "task_id",
        "tool_use_id",
        "tool_name",
        "status",
        "created_at",
        "updated_at",
        "cancellation_reason",
        "state",
        "result",
        "failure",
    }
)


def capture_json_value(value: Any, path: str) -> Any:
    """Defensively copy a value that can be persisted by Agent state."""
    try:
        copied = copy.deepcopy(value)
        json.dumps(encode_bytes_values(copied))
    except (TypeError, ValueError, copy.Error) as error:
        raise ValueError(f"{path} must contain only persistable JSON values") from error
    return copied


def capture_invocation_state(invocation_state: dict[str, Any]) -> dict[str, Any]:
    """Copy caller-owned invocation state without framework runtime objects."""
    user_state = {key: value for key, value in invocation_state.items() if key not in _FRAMEWORK_INVOCATION_STATE_KEYS}
    return cast(dict[str, Any], capture_json_value(user_state, "background task invocation_state"))


def serialize_span_context(span_context: SpanContext | None) -> OriginTraceContext | None:
    """Serialize a valid span context for detached trace linking."""
    if span_context is None or not span_context.is_valid:
        return None
    return {
        "trace_id": span_context.trace_id,
        "span_id": span_context.span_id,
        "trace_flags": int(span_context.trace_flags),
        "is_remote": span_context.is_remote,
    }


def deserialize_span_context(value: OriginTraceContext | None) -> SpanContext | None:
    """Reconstruct a persisted span context."""
    if value is None:
        return None
    context = SpanContext(
        trace_id=value["trace_id"],
        span_id=value["span_id"],
        is_remote=value.get("is_remote", False),
        trace_flags=TraceFlags(value["trace_flags"]),
    )
    if not context.is_valid:
        raise ValueError("task.descriptor.origin_trace_context must be a valid span context")
    return context


def encode_stored_task(record: StoredBackgroundTask) -> dict[str, Any]:
    """Project an engine record into the durable Agent-state shape."""
    validate_stored_task(record)
    descriptor = record["descriptor"]
    persisted: dict[str, Any] = {
        "task_id": record["task_id"],
        "tool_use_id": descriptor["original_tool_use_id"],
        "tool_name": descriptor["tool_name"],
        "status": record["status"],
        "created_at": record["created_at"],
        "updated_at": record["updated_at"],
    }
    if "cancellation_reason" in record:
        persisted["cancellation_reason"] = record["cancellation_reason"]
    if "state" in record:
        persisted["state"] = copy.deepcopy(record["state"])
    if "result" in record:
        persisted["result"] = copy.deepcopy(record["result"])
    if "failure" in record:
        persisted["failure"] = copy.deepcopy(record["failure"])
    return cast(dict[str, Any], encode_bytes_values(persisted))


def decode_stored_task(value: object) -> StoredBackgroundTask:
    """Decode a durable record into an inert engine record."""
    decoded = decode_bytes_values(copy.deepcopy(value))
    persisted = _require_dict(decoded, "task")
    if "descriptor" in persisted:
        validate_stored_task(persisted)
        compact = encode_stored_task(cast(StoredBackgroundTask, persisted))
        return decode_stored_task(compact)

    unknown_keys = sorted(persisted.keys() - _PERSISTED_RECORD_KEYS)
    if unknown_keys:
        raise ValueError(f"task contains unknown field(s): {', '.join(unknown_keys)}")

    task_id = _require_string_value(persisted.get("task_id"), "task.task_id")
    tool_use_id = _require_string_value(persisted.get("tool_use_id"), "task.tool_use_id")
    tool_name = _require_string_value(persisted.get("tool_name"), "task.tool_name")
    status = persisted.get("status")
    if status not in ("queued", "working", "paused", "completed", "failed", "cancelled"):
        raise ValueError(f"task.status '{status}' is invalid")
    created_at = _require_string_value(persisted.get("created_at"), "task.created_at")
    updated_at = _require_string_value(persisted.get("updated_at"), "task.updated_at")

    record = cast(
        StoredBackgroundTask,
        {
            "task_id": task_id,
            "descriptor": {
                "original_tool_use_id": tool_use_id,
                "tool_name": tool_name,
                "invocation_state": {},
            },
            "status": cast(BackgroundTaskStatus, status),
            "attempt_count": 0,
            "created_at": created_at,
            "updated_at": updated_at,
        },
    )
    if "state" in persisted:
        _validate_interrupt_state(persisted["state"])
        record["state"] = copy.deepcopy(persisted["state"])
    if "result" in persisted:
        _validate_tool_result(persisted["result"], "task.result")
        record["result"] = cast(ToolResult, copy.deepcopy(persisted["result"]))
    if "failure" in persisted:
        failure = _validate_failure(persisted["failure"])
        record["failure"] = failure
    if "cancellation_reason" in persisted:
        record["cancellation_reason"] = _require_string_value(
            persisted["cancellation_reason"],
            "task.cancellation_reason",
        )

    if status == "paused" and "state" not in record:
        raise ValueError("task.state is required while paused")
    if status == "completed" and "result" not in record:
        raise ValueError("task.result is required while completed")
    if status == "failed" and "failure" not in record:
        raise ValueError("task.failure is required while failed")
    if status == "cancelled" and "cancellation_reason" not in record:
        record["cancellation_reason"] = "Background task was cancelled"

    validate_stored_task(record)
    if status in ("queued", "working", "paused"):
        record["status"] = "failed"
        record["updated_at"] = datetime.now(timezone.utc).isoformat()
        record["failure"] = cast(
            BackgroundTaskFailure,
            {
                "type": "recovery_error",
                "message": _RECOVERY_MESSAGE,
            },
        )
        record.pop("state", None)
    return record


def validate_stored_task(value: object) -> None:
    """Validate a Strands tool-task record loaded from persistence."""
    validate_stored_engine_task(value)
    record = cast(dict[str, Any], value)
    _validate_descriptor(record["descriptor"])

    if "result" in record:
        _validate_tool_result(record["result"], "task.result")
    if "state" in record:
        _validate_interrupt_state(record["state"])
    if "failure" in record and record["failure"]["type"] not in _FAILURE_TYPES:
        raise ValueError(f"task.failure.type '{record['failure']['type']}' is invalid")


def to_background_task(record: StoredBackgroundTask) -> BackgroundTask:
    """Project an internal engine record into the public read-only shape."""
    descriptor = record["descriptor"]
    task: dict[str, Any] = {
        "task_id": record["task_id"],
        "tool_use_id": descriptor["original_tool_use_id"],
        "tool_name": descriptor["tool_name"],
        "status": record["status"],
        "created_at": record["created_at"],
        "updated_at": record["updated_at"],
    }
    if "result" in record:
        task["result"] = cast(
            BackgroundTaskResult,
            {"content": copy.deepcopy(record["result"]["content"])},
        )
    if "failure" in record:
        task["error"] = cast(
            BackgroundTaskError,
            {
                "type": record["failure"]["type"],
                "message": record["failure"]["message"],
            },
        )
    if "state" in record:
        interrupt_state = _InterruptState.from_dict(copy.deepcopy(record["state"]))
        unanswered = [
            copy.deepcopy(interrupt) for interrupt in interrupt_state.interrupts.values() if interrupt.response is None
        ]
        if unanswered:
            task["interrupts"] = unanswered
    return cast(BackgroundTask, copy.deepcopy(task))


def _validate_descriptor(value: object) -> None:
    descriptor = _require_dict(value, "task.descriptor")
    _require_string(descriptor.get("original_tool_use_id"), "task.descriptor.original_tool_use_id")
    _require_string(descriptor.get("tool_name"), "task.descriptor.tool_name")
    if "input" in descriptor:
        capture_json_value(descriptor["input"], "task.descriptor.input")
    invocation_state = _require_dict(descriptor.get("invocation_state"), "task.descriptor.invocation_state")
    capture_json_value(invocation_state, "task.descriptor.invocation_state")
    if "origin_trace_context" in descriptor:
        trace_context = _require_dict(
            descriptor["origin_trace_context"],
            "task.descriptor.origin_trace_context",
        )
        for key in ("trace_id", "span_id", "trace_flags"):
            value_for_key = trace_context.get(key)
            if isinstance(value_for_key, bool) or not isinstance(value_for_key, int):
                raise ValueError(f"task.descriptor.origin_trace_context.{key} must be an integer")
        if "is_remote" in trace_context and not isinstance(trace_context["is_remote"], bool):
            raise ValueError("task.descriptor.origin_trace_context.is_remote must be a boolean")
        deserialize_span_context(cast(OriginTraceContext, trace_context))


def _validate_tool_result(value: object, path: str) -> None:
    result = _require_dict(value, path)
    _require_string(result.get("toolUseId"), f"{path}.toolUseId")
    if result.get("status") not in ("success", "error"):
        raise ValueError(f"{path}.status must be 'success' or 'error'")
    content = result.get("content")
    if not isinstance(content, list) or not all(isinstance(block, dict) for block in content):
        raise ValueError(f"{path}.content must be an array of objects")
    capture_json_value(content, f"{path}.content")


def _validate_failure(value: object) -> BackgroundTaskFailure:
    failure = _require_dict(value, "task.failure")
    failure_type = _require_string_value(failure.get("type"), "task.failure.type")
    message = _require_string_value(failure.get("message"), "task.failure.message")
    if failure_type not in _FAILURE_TYPES:
        raise ValueError(f"task.failure.type '{failure_type}' is invalid")
    return {"type": failure_type, "message": message}


def _validate_interrupt_state(value: object) -> None:
    state = _require_dict(value, "task.state")
    try:
        restored = _InterruptState.from_dict(copy.deepcopy(state))
        restored.to_dict()
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("task.state cannot be reconstructed") from error


def _require_dict(value: object, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be an object")
    return cast(dict[str, Any], value)


def _require_string(value: object, path: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{path} must be a non-empty string")


def _require_string_value(value: object, path: str) -> str:
    _require_string(value, path)
    return cast(str, value)
