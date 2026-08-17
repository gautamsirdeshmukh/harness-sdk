"""Canonical synthetic delivery messages for completed background tasks."""

from __future__ import annotations

import copy
import json
from typing import cast

from ..agent.conversation_manager.compression.pin_message import unpin_message
from ..types.content import Message, MessageMetadata, Messages
from ..types.session import encode_bytes_values
from ..types.tools import ToolResult
from ._engine import is_engine_terminal_status
from ._record import StoredBackgroundTask

BACKGROUND_RESULT_TOOL_NAME = "strands_background_task_result"


def render_background_delivery(record: StoredBackgroundTask) -> tuple[Message, Message]:
    """Render the canonical assistant tool-use and user tool-result pair."""
    if not is_engine_terminal_status(record["status"]):
        raise RuntimeError(f"Background task '{record['task_id']}' is not terminal")

    failure = record.get("failure")
    delivery_input: dict[str, object] = {
        "task_id": record["task_id"],
        "tool_name": record["descriptor"]["tool_name"],
        "status": record["status"],
    }
    if failure is not None:
        delivery_input["error"] = {
            "type": failure["type"],
            "message": failure["message"],
        }

    result_content = copy.deepcopy(record.get("result", {}).get("content", []))
    tool_result: ToolResult = {
        "toolUseId": record["task_id"],
        "status": "success" if record["status"] == "completed" else "error",
        "content": [
            {
                "text": _render_terminal_header(
                    record,
                    has_result="result" in record,
                )
            },
            *result_content,
        ],
    }
    metadata = cast(MessageMetadata, {"custom": {"pinned": True}})
    return (
        cast(
            Message,
            {
                "role": "assistant",
                "content": [
                    {
                        "toolUse": {
                            "name": BACKGROUND_RESULT_TOOL_NAME,
                            "toolUseId": record["task_id"],
                            "input": delivery_input,
                        }
                    }
                ],
                "metadata": copy.deepcopy(metadata),
            },
        ),
        cast(
            Message,
            {
                "role": "user",
                "content": [{"toolResult": tool_result}],
                "metadata": copy.deepcopy(metadata),
            },
        ),
    )


def assert_delivery_consumed(task_id: str, expected: tuple[Message, Message], model_messages: Messages) -> None:
    """Verify middleware preserved the authoritative pair in the provider request."""
    candidates = _find_background_delivery_pairs(model_messages, task_id)
    if not candidates:
        raise RuntimeError(f"Background task delivery '{task_id}' was not present in the provider request")
    if not any(_deliveries_match(candidate, expected) for candidate in candidates):
        raise RuntimeError(f"Background task delivery '{task_id}' did not match its authoritative record")


def history_contains_background_delivery(messages: Messages, record: StoredBackgroundTask) -> bool:
    """Return whether history already contains the canonical adjacent pair."""
    expected = render_background_delivery(record)
    return any(
        _deliveries_match(candidate, expected)
        for candidate in _find_background_delivery_pairs(messages, record["task_id"])
    )


def unpin_background_deliveries(messages: Messages, task_ids: set[str]) -> None:
    """Unpin committed delivery pairs while retaining all other metadata."""
    for index in range(len(messages) - 1):
        task_id = _background_delivery_id(messages[index], messages[index + 1])
        if task_id is None or task_id not in task_ids:
            continue
        unpin_message(messages, index)
        unpin_message(messages, index + 1)


def _render_terminal_header(record: StoredBackgroundTask, *, has_result: bool) -> str:
    task_id = record["task_id"]
    tool_name = record["descriptor"]["tool_name"]
    status = record["status"]
    if status == "completed":
        return "\n".join(
            [
                "Background task completed.",
                "",
                f"Task ID: {task_id}",
                f"Tool: {tool_name}",
                "Status: completed",
                "",
                "The final result follows.",
            ]
        )
    if status == "failed":
        failure = record.get("failure")
        if failure is None:
            raise RuntimeError(f"Failed background task '{task_id}' has no failure detail")
        return "\n".join(
            [
                "Background task failed.",
                "",
                f"Task ID: {task_id}",
                f"Tool: {tool_name}",
                "Status: failed",
                f"Error type: {failure['type']}",
                f"Reason: {failure['message']}",
                "",
                "The tool error follows." if has_result else "No result is available.",
            ]
        )
    return "\n".join(
        [
            "Background task cancelled.",
            "",
            f"Task ID: {task_id}",
            f"Tool: {tool_name}",
            "Status: cancelled",
            "",
            "The task was cancelled before producing a final result.",
        ]
    )


def _find_background_delivery_pairs(messages: Messages, delivery_id: str) -> list[tuple[Message, Message]]:
    pairs: list[tuple[Message, Message]] = []
    for index in range(len(messages) - 1):
        left = messages[index]
        right = messages[index + 1]
        if _background_delivery_id(left, right) == delivery_id:
            pairs.append((left, right))
    return pairs


def _background_delivery_id(tool_use_message: Message, tool_result_message: Message) -> str | None:
    if tool_use_message["role"] != "assistant" or tool_result_message["role"] != "user":
        return None
    for content in tool_use_message["content"]:
        tool_use = content.get("toolUse")
        if not tool_use or tool_use.get("name") != BACKGROUND_RESULT_TOOL_NAME:
            continue
        tool_use_id = tool_use.get("toolUseId")
        if any(
            result_content.get("toolResult", {}).get("toolUseId") == tool_use_id
            for result_content in tool_result_message["content"]
        ):
            return str(tool_use_id)
    return None


def _deliveries_match(left: tuple[Message, Message], right: tuple[Message, Message]) -> bool:
    def project(messages: tuple[Message, Message]) -> object | None:
        tool_use_message, tool_result_message = messages
        tool_use = None
        for content in tool_use_message["content"]:
            tool_use_candidate = content.get("toolUse")
            if tool_use_candidate is not None and tool_use_candidate.get("name") == BACKGROUND_RESULT_TOOL_NAME:
                tool_use = tool_use_candidate
                break
        if tool_use is None:
            return None
        tool_result = None
        for content in tool_result_message["content"]:
            tool_result_candidate = content.get("toolResult")
            if tool_result_candidate is not None and tool_result_candidate.get("toolUseId") == tool_use.get(
                "toolUseId"
            ):
                tool_result = tool_result_candidate
                break
        if tool_result is None:
            return None
        return [encode_bytes_values(tool_use), encode_bytes_values(tool_result)]

    left_delivery = project(left)
    right_delivery = project(right)
    return (
        left_delivery is not None
        and right_delivery is not None
        and json.dumps(
            left_delivery,
            sort_keys=True,
            separators=(",", ":"),
        )
        == json.dumps(
            right_delivery,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
