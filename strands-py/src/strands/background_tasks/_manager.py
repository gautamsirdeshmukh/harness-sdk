"""Agent-owned orchestration for in-process background tool tasks."""

from __future__ import annotations

import asyncio
import concurrent.futures
import copy
import json
import math
import threading
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal, cast

from opentelemetry.trace import SpanContext

from ..agent.agent_result import AgentResult
from ..hooks import (
    AfterInvocationEvent,
    AfterModelCallEvent,
    BeforeModelCallEvent,
    HookOrder,
)
from ..interrupt import Interrupt, _InterruptState
from ..types.content import Messages
from ..types.interrupt import InterruptResponseContent
from ..types.tools import ToolResult, ToolUse
from ._delivery import (
    assert_delivery_consumed,
    history_contains_background_delivery,
    render_background_delivery,
    unpin_background_deliveries,
)
from ._engine import BackgroundTaskEngine, is_engine_terminal_status
from ._engine_types import (
    BackgroundTaskEngineEvent,
    BackgroundTaskExecutionContext,
    BackgroundTaskExecutionOutcome,
    CompletedExecutionOutcome,
    FailedExecutionOutcome,
)
from ._record import (
    StoredBackgroundTask,
    ToolTaskDescriptor,
    capture_invocation_state,
    capture_json_value,
    decode_stored_task,
    deserialize_span_context,
    encode_stored_task,
    serialize_span_context,
    to_background_task,
)
from ._runtime import get_background_task_runtime
from ._telemetry import _BackgroundTaskTelemetry
from .errors import BackgroundTaskNotFoundError, BackgroundTasksTimeoutError
from .types import BackgroundTask, BackgroundTasksConfig

if TYPE_CHECKING:
    from ..agent import Agent
    from ..hooks.events import _ContinuationIntent

_DEFAULT_MAX_CONCURRENCY = 4
_STATE_RELOAD_TIMEOUT = 30.0
_STATE_KEY = "strands.background_tasks"
_DeliveryState = Literal["pending", "ready", "delivered"]


class _InProcessTaskManager:
    """Coordinate Agent state, delivery, interrupts, and detached execution."""

    def __init__(self, agent: Agent, config: BackgroundTasksConfig) -> None:
        self._agent = agent
        self._max_concurrency = config.get("max_concurrency", _DEFAULT_MAX_CONCURRENCY)
        self._timeout = config.get("timeout", math.inf)
        self._wait_for_completion = config.get("wait_for_completion", True)
        self._runtime = get_background_task_runtime()
        self._records: dict[str, StoredBackgroundTask] = {}
        self._delivery_states: dict[str, _DeliveryState] = {}
        self._delivering: set[str] = set()
        self._delivery_changed = asyncio.Event()
        self._state_lock = threading.RLock()
        self._generation = 0
        self._reload_future: concurrent.futures.Future[None] | None = None
        self._telemetry = _BackgroundTaskTelemetry()
        self._engine = self._create_engine(self._generation)
        self._initialized = False

    def register_hooks(self) -> None:
        """Register manager lifecycle callbacks on the Agent."""
        self._agent.add_hook(self._on_before_model_call, BeforeModelCallEvent)
        self._agent.add_hook(self._on_after_model_call, AfterModelCallEvent)
        self._agent.add_hook(self._on_after_invocation, AfterInvocationEvent, order=HookOrder.SDK_FIRST)

    def initialize(self) -> None:
        """Load persisted state and start restored work."""
        self._runtime.run_sync(self._initialize)

    def _initialize(self) -> None:
        if self._initialized:
            return
        self._load_state(self._agent.state.get(_STATE_KEY))
        self._engine.initialize(tuple(self._records.values()))
        self._initialized = True
        self._persist_state()
        delivered = [
            task_id
            for task_id, state in self._delivery_states.items()
            if state == "delivered" and task_id in self._records
        ]
        if delivered:
            self._prune_delivered(delivered)
        reconciled = self._reconcile_ready_deliveries(copy.deepcopy(self._agent.messages))
        if reconciled:
            unpin_background_deliveries(self._agent.messages, set(reconciled))

    def _create_engine(
        self,
        generation: int,
    ) -> BackgroundTaskEngine[ToolTaskDescriptor, ToolResult, dict[str, Any]]:
        return BackgroundTaskEngine(
            max_concurrency=self._max_concurrency,
            timeout=self._timeout,
            execute=self._execute_tool_task,
            on_task_updated=lambda record: self._on_task_updated(generation, record),
            on_event=self._record_engine_event,
        )

    def _on_task_updated(self, generation: int, record: StoredBackgroundTask) -> None:
        if generation != self._generation:
            return
        self._records[record["task_id"]] = record
        self._delivery_states[record["task_id"]] = _delivery_state_for(
            record,
            self._delivery_states.get(record["task_id"]),
        )
        self._persist_state()

    def _load_state(self, value: object) -> None:
        self._records.clear()
        self._delivery_states.clear()
        if value is None:
            return
        if not isinstance(value, dict):
            raise ValueError(f"{_STATE_KEY} must be an object")

        for task_id, envelope_value in value.items():
            if not isinstance(task_id, str) or not isinstance(envelope_value, dict):
                raise ValueError(f"{_STATE_KEY}.{task_id} is invalid")
            if "record" not in envelope_value or "delivery_state" not in envelope_value:
                raise ValueError(f"{_STATE_KEY}.{task_id} is invalid")
            record = decode_stored_task(envelope_value["record"])
            if record["task_id"] != task_id:
                raise ValueError(f"{_STATE_KEY}.{task_id}.record.task_id must match its map key")
            delivery_state = envelope_value["delivery_state"]
            if delivery_state not in ("pending", "ready", "delivered"):
                raise ValueError(f"{_STATE_KEY}.{task_id}.delivery_state is invalid")
            self._records[task_id] = record
            self._delivery_states[task_id] = _delivery_state_for(record, cast(_DeliveryState, delivery_state))

    def _persist_state(self) -> None:
        with self._state_lock:
            if not self._records:
                self._agent.state.delete(_STATE_KEY)
                return
            value = {
                task_id: {
                    "record": encode_stored_task(record),
                    "delivery_state": self._delivery_states.get(task_id, _delivery_state_for(record)),
                }
                for task_id, record in self._records.items()
            }
            self._agent.state.set(_STATE_KEY, value)

    def app_state_loaded(self) -> None:
        """Schedule an atomic engine reload after Agent state replacement."""
        restored_value = self._agent.state.get(_STATE_KEY)
        with self._state_lock:
            self._generation += 1
            generation = self._generation
            previous = self._reload_future
            self._reload_future = self._runtime.submit(lambda: self._reload_after(previous, restored_value, generation))

    async def _reload_after(
        self,
        previous: concurrent.futures.Future[None] | None,
        restored_value: object,
        generation: int,
    ) -> None:
        if previous is not None:
            try:
                await asyncio.wrap_future(previous)
            except Exception:
                pass

        deadline = asyncio.get_running_loop().time() + _STATE_RELOAD_TIMEOUT
        await self._wait_for_deliveries(_STATE_RELOAD_TIMEOUT)
        old_engine = self._engine
        remaining = max(0.001, deadline - asyncio.get_running_loop().time())
        await old_engine.shutdown(mode="cancel", timeout=remaining)
        self._delivering.clear()
        self._load_state(restored_value)
        engine = self._create_engine(generation)
        try:
            engine.initialize(tuple(self._records.values()))
            self._persist_state()
            task_ids_to_unpin = {
                task_id
                for task_id, state in self._delivery_states.items()
                if state == "delivered" and task_id in self._records
            }
            if task_ids_to_unpin:
                self._prune_delivered(tuple(task_ids_to_unpin), engine=engine)
            reconciled = self._reconcile_ready_deliveries(
                copy.deepcopy(self._agent.messages),
                engine=engine,
            )
            task_ids_to_unpin.update(reconciled)
            if task_ids_to_unpin:
                unpin_background_deliveries(self._agent.messages, task_ids_to_unpin)
        except Exception:
            await engine.shutdown(mode="cancel", timeout=1.0)
            raise
        self._engine = engine

    async def _wait_for_reload(self) -> None:
        while True:
            with self._state_lock:
                reload_future = self._reload_future
            if reload_future is None:
                return
            await asyncio.wrap_future(reload_future)
            with self._state_lock:
                if self._reload_future is reload_future:
                    self._reload_future = None
                    return

    async def submit_tool_call(
        self,
        *,
        tool_name: str,
        original_tool_use_id: str,
        tool_input: Any,
        invocation_state: dict[str, Any],
        pass_id: str,
        origin_span_context: SpanContext | None,
    ) -> BackgroundTask:
        """Durably admit one detached tool call."""
        await self._wait_for_reload()
        descriptor: ToolTaskDescriptor = {
            "original_tool_use_id": original_tool_use_id,
            "tool_name": tool_name,
            "input": capture_json_value(tool_input, "background task input"),
            "invocation_state": capture_invocation_state(invocation_state),
        }
        serialized_context = serialize_span_context(origin_span_context)
        if serialized_context is not None:
            descriptor["origin_trace_context"] = serialized_context

        def submit() -> BackgroundTask:
            record = self._engine.submit(
                descriptor,
                idempotency_key=json.dumps([pass_id, original_tool_use_id], separators=(",", ":")),
            )
            return to_background_task(record)

        return await self._runtime.run(submit)

    async def get_task(self, task_id: str) -> BackgroundTask | None:
        """Return one visible task snapshot."""
        await self._wait_for_reload()
        return await self._runtime.run(
            lambda: to_background_task(record) if (record := self._engine.get(task_id)) is not None else None
        )

    async def list_tasks(self) -> list[BackgroundTask]:
        """Return visible task snapshots in admission order."""
        await self._wait_for_reload()
        return await self._runtime.run(lambda: [to_background_task(record) for record in self._engine.list()])

    async def cancel_task(self, task_id: str) -> BackgroundTask:
        """Request cooperative cancellation for a task."""
        await self._wait_for_reload()

        def cancel() -> BackgroundTask:
            current = self._engine.get(task_id)
            if current is None:
                raise BackgroundTaskNotFoundError(task_id)
            record = self._engine.cancel(task_id, "Cancellation requested")
            self._telemetry.record_cancellation(record["descriptor"]["tool_name"])
            return to_background_task(record)

        return await self._runtime.run(cancel)

    async def wait_for_tasks(self, timeout: float | None = None) -> None:
        """Wait for physical idleness without cancelling work on observation timeout."""
        if timeout is not None and (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise TypeError(f"wait timeout must be a positive finite number, got {timeout!r}")
        resolved_timeout = float(timeout) if timeout is not None else None
        if resolved_timeout is None:
            await self._wait_for_reload()
            await self._runtime.run_void(self._engine.wait_for_idle)
            return

        deadline = asyncio.get_running_loop().time() + resolved_timeout
        try:
            await asyncio.wait_for(self._wait_for_reload(), timeout=resolved_timeout)
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise asyncio.TimeoutError
            await asyncio.wait_for(
                self._runtime.run_void(self._engine.wait_for_idle),
                timeout=remaining,
            )
        except (TimeoutError, asyncio.TimeoutError) as error:
            raise BackgroundTasksTimeoutError(resolved_timeout) from error

    async def shutdown(self, *, timeout: float = 5.0) -> None:
        """Cancel active work and wait for the configured executor to settle."""
        await self._wait_for_reload()
        await self._runtime.run_void(lambda: self._engine.shutdown(mode="cancel", timeout=timeout))

    async def _on_before_model_call(self, event: BeforeModelCallEvent) -> None:
        await self._wait_for_reload()
        await self._route_interrupt_responses(event.agent._interrupt_state)
        interrupts = await self._paused_interrupts()
        if interrupts:
            self._activate_foreground_interrupts(interrupts)
            event._interrupt_with(list(interrupts))
            return
        await self._deliver_ready(event)

    async def _on_after_model_call(self, event: AfterModelCallEvent) -> None:
        await self._wait_for_reload()
        if event.exception is not None or event.stop_response is None:
            return
        interrupts = await self._paused_interrupts()
        if interrupts:
            self._activate_foreground_interrupts(interrupts)
            event._interrupt_with(list(interrupts))

    async def _on_after_invocation(self, event: AfterInvocationEvent) -> None:
        await self._wait_for_reload()
        result = event.result
        if not self._wait_for_completion or result is None or result.stop_reason in ("cancelled", "interrupt"):
            return

        cannot_continue = result.stop_reason in {
            "checkpoint",
            "limit_turns",
            "limit_output_tokens",
            "limit_total_tokens",
        }
        await self._wait_for_task_result(wait_for_all=cannot_continue)
        if self._agent._cancel_signal.is_set():
            return
        interrupts = await self._paused_interrupts()
        if interrupts:
            self._activate_foreground_interrupts(interrupts)
            object.__setattr__(
                event,
                "result",
                AgentResult(
                    stop_reason="interrupt",
                    message=result.message,
                    metrics=result.metrics,
                    state=result.state,
                    interrupts=interrupts,
                    structured_output=result.structured_output,
                    checkpoint=result.checkpoint,
                ),
            )
            return
        if cannot_continue:
            return
        await self._deliver_ready(event)

    async def _deliver_ready(self, event: BeforeModelCallEvent | AfterInvocationEvent) -> None:
        messages = copy.deepcopy(self._agent.messages)
        already_delivered = await self._runtime.run(lambda: self._reconcile_ready_deliveries(messages))
        if already_delivered:
            unpin_background_deliveries(self._agent.messages, set(already_delivered))

        task_ids: list[str] = []
        continuation_registered = False
        try:
            records = await self._runtime.run(self._reserve_ready_deliveries)
            if not records:
                return
            task_ids = [record["task_id"] for record in records]
            deliveries = [render_background_delivery(record) for record in records]

            async def on_selected(model_messages: Messages) -> None:
                for record, delivery in zip(records, deliveries, strict=True):
                    assert_delivery_consumed(record["task_id"], delivery, model_messages)

            async def on_committed() -> None:
                try:
                    await self._runtime.run(lambda: self._prune_delivered(task_ids))
                    unpin_background_deliveries(self._agent.messages, set(task_ids))
                finally:
                    await self._runtime.run(lambda: self._finish_delivery(task_ids))

            async def on_rejected(_reason: object) -> None:
                await self._runtime.run(lambda: self._finish_delivery(task_ids))

            intent = cast(
                "_ContinuationIntent",
                {
                    "phase": "deferred_result",
                    "input": [message for delivery in deliveries for message in delivery],
                    "on_selected": on_selected,
                    "on_committed": on_committed,
                    "on_rejected": on_rejected,
                },
            )
            event._continue_with(intent)
            continuation_registered = True
        finally:
            if task_ids and not continuation_registered:
                await self._runtime.run(lambda: self._finish_delivery(task_ids))

    def _reserve_ready_deliveries(self) -> list[StoredBackgroundTask]:
        records = [
            record
            for record in self._engine.list()
            if is_engine_terminal_status(record["status"])
            and self._delivery_states.get(record["task_id"]) == "ready"
            and record["task_id"] not in self._delivering
        ]
        self._delivering.update(record["task_id"] for record in records)
        return records

    def _finish_delivery(self, task_ids: Sequence[str]) -> None:
        self._delivering.difference_update(task_ids)
        self._delivery_changed.set()

    def _reconcile_ready_deliveries(
        self,
        messages: Messages,
        *,
        engine: BackgroundTaskEngine[ToolTaskDescriptor, ToolResult, dict[str, Any]] | None = None,
    ) -> list[str]:
        resolved_engine = engine or self._engine
        delivered = [
            record["task_id"]
            for record in resolved_engine.list()
            if is_engine_terminal_status(record["status"])
            and self._delivery_states.get(record["task_id"]) == "ready"
            and record["task_id"] not in self._delivering
            and history_contains_background_delivery(messages, record)
        ]
        if delivered:
            self._prune_delivered(delivered, engine=resolved_engine)
        return delivered

    def _prune_delivered(
        self,
        task_ids: Sequence[str],
        *,
        engine: BackgroundTaskEngine[ToolTaskDescriptor, ToolResult, dict[str, Any]] | None = None,
    ) -> None:
        resolved_engine = engine or self._engine
        for task_id in task_ids:
            record = self._records.get(task_id)
            if record is None:
                raise BackgroundTaskNotFoundError(task_id)
            state = self._delivery_states.get(task_id)
            if state not in ("ready", "delivered") or not is_engine_terminal_status(record["status"]):
                raise RuntimeError(f"Background task '{task_id}' does not have a ready result")
            resolved_engine.remove(task_id)
            self._records.pop(task_id, None)
            self._delivery_states.pop(task_id, None)
        self._persist_state()

    async def _wait_for_deliveries(self, timeout: float) -> None:
        deadline = asyncio.get_running_loop().time() + timeout
        while self._delivering:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise TimeoutError(f"Background Tasks state reload timed out after {timeout}s")
            self._delivery_changed.clear()
            if not self._delivering:
                return
            try:
                await asyncio.wait_for(self._delivery_changed.wait(), timeout=remaining)
            except asyncio.TimeoutError as error:
                raise TimeoutError(f"Background Tasks state reload timed out after {timeout}s") from error

    async def _wait_for_task_result(self, *, wait_for_all: bool) -> None:
        async def wait() -> None:
            while not self._agent._cancel_signal.is_set():
                tasks = self._engine.list()
                if any(task["status"] == "paused" for task in tasks):
                    return
                if not wait_for_all and any(
                    is_engine_terminal_status(task["status"])
                    and self._delivery_states.get(task["task_id"]) == "ready"
                    and task["task_id"] not in self._delivering
                    for task in tasks
                ):
                    return
                if not any(task["status"] in ("queued", "working") for task in tasks):
                    return
                await asyncio.sleep(0.01)

        await self._runtime.run_void(wait)

    async def _paused_interrupts(self) -> list[Interrupt]:
        def collect() -> list[Interrupt]:
            interrupts: list[Interrupt] = []
            for record in self._engine.list():
                if record["status"] != "paused":
                    continue
                interrupts.extend(to_background_task(record).get("interrupts", []))
            return interrupts

        return await self._runtime.run(collect)

    async def _route_interrupt_responses(self, foreground_state: _InterruptState) -> None:
        responses = foreground_state.context.get("responses", [])
        if not responses:
            return

        def route() -> set[str]:
            paused = [record for record in self._engine.list() if record["status"] == "paused"]
            task_by_interrupt_id: dict[str, str] = {}
            for record in paused:
                for interrupt in to_background_task(record).get("interrupts", []):
                    owner = task_by_interrupt_id.get(interrupt.id)
                    if owner is not None and owner != record["task_id"]:
                        raise RuntimeError(f"Background interrupt '{interrupt.id}' is ambiguous across paused tasks")
                    task_by_interrupt_id[interrupt.id] = record["task_id"]

            responses_by_task: dict[str, list[InterruptResponseContent]] = {}
            for content in responses:
                interrupt_id = content["interruptResponse"]["interruptId"]
                task_id = task_by_interrupt_id.get(interrupt_id)
                if task_id is not None:
                    responses_by_task.setdefault(task_id, []).append(content)

            for task_id, task_responses in responses_by_task.items():
                self._resume_task(task_id, task_responses)
            return set(task_by_interrupt_id)

        background_interrupt_ids = await self._runtime.run(route)
        foreground_interrupt_ids = set(foreground_state.interrupts)
        if foreground_interrupt_ids and foreground_interrupt_ids <= background_interrupt_ids:
            foreground_state.deactivate()

    def _resume_task(self, task_id: str, responses: Sequence[InterruptResponseContent]) -> None:
        current = self._engine.get(task_id)
        if current is None:
            raise BackgroundTaskNotFoundError(task_id)
        if current["status"] != "paused":
            if "state" in current and _responses_already_applied(current["state"], responses):
                return
            raise RuntimeError(
                f"Background task '{task_id}' cannot transition: status is '{current['status']}', not 'paused'"
            )

        def update(state_value: dict[str, Any]) -> dict[str, object]:
            interrupt_state = _InterruptState.from_dict(state_value)
            known_ids = set(interrupt_state.interrupts)
            for content in responses:
                interrupt_id = content["interruptResponse"]["interruptId"]
                if interrupt_id not in known_ids:
                    raise RuntimeError(
                        f"Background task '{task_id}' cannot transition: unknown interrupt '{interrupt_id}'"
                    )
            interrupt_state.resume(list(responses))
            ready = all(interrupt.response is not None for interrupt in interrupt_state.interrupts.values())
            return {"state": interrupt_state.to_dict(), "ready": ready}

        self._engine.resume(task_id, update)  # type: ignore[arg-type]

    def _activate_foreground_interrupts(self, interrupts: Sequence[Interrupt]) -> None:
        for interrupt in interrupts:
            self._agent._interrupt_state.interrupts.setdefault(interrupt.id, copy.deepcopy(interrupt))
        self._agent._interrupt_state.context["background_tasks"] = True
        self._agent._interrupt_state.activate()

    async def _execute_tool_task(
        self,
        context: BackgroundTaskExecutionContext[ToolTaskDescriptor, dict[str, Any]],
    ) -> BackgroundTaskExecutionOutcome[ToolResult, dict[str, Any]]:
        descriptor = context.descriptor
        tool = self._agent.tool_registry.dynamic_tools.get(descriptor["tool_name"])
        if tool is None:
            tool = self._agent.tool_registry.registry.get(descriptor["tool_name"])
        if tool is None:
            outcome: BackgroundTaskExecutionOutcome[ToolResult, dict[str, Any]] = {
                "status": "failed",
                "failure": {
                    "type": "recovery_error",
                    "message": (
                        f"Tool '{descriptor['tool_name']}' is not registered on Agent '{self._agent.agent_id}'"
                    ),
                },
            }
            if context.state is not None:
                outcome["state"] = context.state
            return outcome

        tool_use: ToolUse = {
            "name": descriptor["tool_name"],
            "toolUseId": descriptor["original_tool_use_id"],
            "input": copy.deepcopy(descriptor.get("input")),
        }
        try:

            async def execute() -> dict[str, Any]:
                return await self._agent._execute_detached_tool(
                    tool_use=tool_use,
                    invocation_state=copy.deepcopy(descriptor["invocation_state"]),
                    cancel_signal=context.cancel_signal,
                    interrupt_state=context.state,
                    task_id=context.task_id,
                    origin_span_context=deserialize_span_context(descriptor.get("origin_trace_context")),
                )

            detached_outcome: dict[str, Any] = await execute()
        except Exception as error:
            failed: BackgroundTaskExecutionOutcome[ToolResult, dict[str, Any]] = {
                "status": "failed",
                "failure": {
                    "type": "execution_error",
                    "message": str(error) or type(error).__name__,
                },
            }
            if context.state is not None:
                failed["state"] = context.state
            return failed

        if "interrupt_state" in detached_outcome:
            return {
                "status": "paused",
                "state": detached_outcome["interrupt_state"],
            }

        result = cast(ToolResult, detached_outcome["result"])
        if result["status"] == "error":
            message = next(
                (content["text"] for content in result["content"] if "text" in content and content["text"]),
                "Tool returned an error without a message",
            )
            failed_result = cast(
                FailedExecutionOutcome[ToolResult, dict[str, Any]],
                {
                    "status": "failed",
                    "failure": {"type": "tool_error", "message": message},
                    "result": copy.deepcopy(result),
                },
            )
            if context.state is not None:
                failed_result["state"] = context.state
            return failed_result

        completed = cast(
            CompletedExecutionOutcome[ToolResult, dict[str, Any]],
            {
                "status": "completed",
                "result": copy.deepcopy(result),
            },
        )
        if context.state is not None:
            completed["state"] = context.state
        return completed

    def _record_engine_event(
        self,
        event: BackgroundTaskEngineEvent[ToolTaskDescriptor, ToolResult, dict[str, Any]],
    ) -> None:
        tool_name = event["task"]["descriptor"]["tool_name"]
        if event["type"] == "admitted":
            self._telemetry.record_admission(tool_name)
            return
        if event["type"] == "execution_started":
            self._telemetry.record_execution_started(
                tool_name=tool_name,
                attempt=event["task"]["attempt_count"],
                resumed=event["resumed"],
                queue_duration=event["queue_duration"],
            )
            return
        if event["type"] == "execution_finished":
            task = event["task"]
            outcome = (
                "execution_error"
                if task["status"] == "failed" and task.get("failure", {}).get("type") == "execution_error"
                else task["status"]
            )
            self._telemetry.record_execution_finished(
                tool_name=tool_name,
                outcome=outcome,
                duration=event["duration"],
            )
            if "failure" in task:
                self._telemetry.record_failure(tool_name, task["failure"]["type"])
            if task["status"] in ("completed", "failed"):
                self._telemetry.record_terminal(tool_name, task["status"])
            return
        self._telemetry.record_terminal(tool_name, "cancelled")


def _delivery_state_for(
    record: StoredBackgroundTask,
    current: _DeliveryState | None = None,
) -> _DeliveryState:
    if not is_engine_terminal_status(record["status"]):
        return "pending"
    return "delivered" if current == "delivered" else "ready"


def _responses_already_applied(
    state_value: dict[str, Any],
    responses: Sequence[InterruptResponseContent],
) -> bool:
    interrupt_state = _InterruptState.from_dict(state_value)
    missing = object()
    return all(
        (
            interrupt_state.interrupts.get(content["interruptResponse"]["interruptId"], missing) is not missing
            and interrupt_state.interrupts[content["interruptResponse"]["interruptId"]].response
            == content["interruptResponse"]["response"]
        )
        for content in responses
    )
