import asyncio
import copy
import threading
from collections.abc import AsyncGenerator, AsyncIterator
from dataclasses import replace
from typing import Any, cast

import pytest

from strands import Agent, tool
from strands._middleware.stages import InvokeModelContext, InvokeModelStage
from strands.agent.conversation_manager import NullConversationManager
from strands.background_tasks._delivery import history_contains_background_delivery, render_background_delivery
from strands.background_tasks._record import StoredBackgroundTask, encode_stored_task
from strands.hooks import BeforeModelCallEvent
from strands.interrupt import Interrupt, _InterruptState
from strands.types.content import Message, Messages
from strands.types.exceptions import ContextWindowOverflowException, EventLoopException
from strands.types.streaming import StreamEvent
from strands.types.tools import ToolSpec
from strands.vended_plugins.context_injector import ContextInjector
from tests.fixtures.mocked_model_provider import MockedModelProvider

_STATE_KEY = "strands.background_tasks"
_DELIVERY_TOOL_NAME = "strands_background_task_result"


def _assistant_text(text: str) -> Message:
    return {"role": "assistant", "content": [{"text": text}]}


def _assistant_tool_use() -> Message:
    return {
        "role": "assistant",
        "content": [
            {
                "toolUse": {
                    "name": "work",
                    "toolUseId": "work-use",
                    "input": {},
                }
            }
        ],
    }


def _contains_delivery(messages: Messages) -> bool:
    return any(
        content.get("toolUse", {}).get("name") == _DELIVERY_TOOL_NAME
        for message in messages
        for content in message["content"]
    )


def _delivery_count(messages: Messages) -> int:
    return sum(
        content.get("toolUse", {}).get("name") == _DELIVERY_TOOL_NAME
        for message in messages
        for content in message["content"]
    )


def _delivery_ids(messages: Messages) -> list[str]:
    return [
        str(content["toolUse"]["toolUseId"])
        for message in messages
        for content in message["content"]
        if content.get("toolUse", {}).get("name") == _DELIVERY_TOOL_NAME
    ]


def _terminal_record() -> StoredBackgroundTask:
    return cast(
        StoredBackgroundTask,
        {
            "task_id": "task-1",
            "idempotency_key": '["pass-1","original-tool-use"]',
            "descriptor": {
                "original_tool_use_id": "original-tool-use",
                "tool_name": "work",
                "input": {"value": "stored"},
                "invocation_state": {},
            },
            "status": "completed",
            "attempt_count": 1,
            "created_at": "2026-08-10T12:00:00",
            "updated_at": "2026-08-10T12:00:00",
            "result": {
                "toolUseId": "original-tool-use",
                "status": "success",
                "content": [{"text": "stored result"}],
            },
        },
    )


def _background_task_state(
    record: StoredBackgroundTask,
    delivery_state: str = "ready",
) -> dict[str, Any]:
    return {
        _STATE_KEY: {
            record["task_id"]: {
                "record": encode_stored_task(record),
                "delivery_state": delivery_state,
            }
        }
    }


def _delivery_states(agent: Agent) -> list[str]:
    stored = agent.state.get(_STATE_KEY)
    if not isinstance(stored, dict):
        return []
    return [envelope["delivery_state"] for envelope in stored.values()]


def _create_ready_agent(
    model: MockedModelProvider,
    *,
    conversation_manager: NullConversationManager | None = None,
) -> Agent:
    return Agent(
        model=model,
        state=_background_task_state(_terminal_record()),
        background_tasks=True,
        conversation_manager=conversation_manager,
        callback_handler=None,
        retry_strategy=None,
    )


async def _drain(stream: AsyncIterator[Any]) -> None:
    async for _event in stream:
        pass


class _RecordingModel(MockedModelProvider):
    def __init__(self, responses: list[Message]) -> None:
        super().__init__(responses)
        self.requests: list[Messages] = []

    async def stream(
        self,
        messages: Messages,
        tool_specs: list[ToolSpec] | None = None,
        system_prompt: str | None = None,
        tool_choice: Any | None = None,
        *,
        system_prompt_content: Any = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamEvent, None]:
        self.requests.append(copy.deepcopy(messages))
        async for event in super().stream(
            messages,
            tool_specs,
            system_prompt,
            tool_choice,
            system_prompt_content=system_prompt_content,
            **kwargs,
        ):
            yield event


class _FailFirstDeliveryModel(_RecordingModel):
    def __init__(self, responses: list[Message]) -> None:
        super().__init__(responses)
        self._fail_delivery = True

    async def stream(
        self,
        messages: Messages,
        tool_specs: list[ToolSpec] | None = None,
        system_prompt: str | None = None,
        tool_choice: Any | None = None,
        *,
        system_prompt_content: Any = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamEvent, None]:
        if self._fail_delivery and _contains_delivery(messages):
            self.requests.append(copy.deepcopy(messages))
            self._fail_delivery = False
            raise RuntimeError("injected provider failure")
        async for event in super().stream(
            messages,
            tool_specs,
            system_prompt,
            tool_choice,
            system_prompt_content=system_prompt_content,
            **kwargs,
        ):
            yield event


class _PauseFirstRequestModel(_RecordingModel):
    def __init__(
        self,
        responses: list[Message],
        requested: asyncio.Event,
        release: asyncio.Event,
    ) -> None:
        super().__init__(responses)
        self._requested = requested
        self._release = release
        self._pause = True

    async def stream(
        self,
        messages: Messages,
        tool_specs: list[ToolSpec] | None = None,
        system_prompt: str | None = None,
        tool_choice: Any | None = None,
        *,
        system_prompt_content: Any = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamEvent, None]:
        if self._pause:
            self.requests.append(copy.deepcopy(messages))
            self._pause = False
            self._requested.set()
            await self._release.wait()
            async for event in MockedModelProvider.stream(
                self,
                messages,
                tool_specs,
                system_prompt,
                tool_choice,
                system_prompt_content=system_prompt_content,
                **kwargs,
            ):
                yield event
            return
        async for event in super().stream(
            messages,
            tool_specs,
            system_prompt,
            tool_choice,
            system_prompt_content=system_prompt_content,
            **kwargs,
        ):
            yield event


class _OverflowFirstDeliveryModel(_RecordingModel):
    def __init__(self, responses: list[Message]) -> None:
        super().__init__(responses)
        self._overflow_delivery = True

    async def stream(
        self,
        messages: Messages,
        tool_specs: list[ToolSpec] | None = None,
        system_prompt: str | None = None,
        tool_choice: Any | None = None,
        *,
        system_prompt_content: Any = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamEvent, None]:
        if self._overflow_delivery and _contains_delivery(messages):
            self.requests.append(copy.deepcopy(messages))
            self._overflow_delivery = False
            raise ContextWindowOverflowException("injected context overflow")
        async for event in super().stream(
            messages,
            tool_specs,
            system_prompt,
            tool_choice,
            system_prompt_content=system_prompt_content,
            **kwargs,
        ):
            yield event


def test_history_contains_delivery_ignores_unrelated_blocks_but_requires_canonical_delivery() -> None:
    record = _terminal_record()
    persisted = list(copy.deepcopy(render_background_delivery(record)))

    assert history_contains_background_delivery(persisted, record)

    without_metadata = copy.deepcopy(persisted)
    for message in without_metadata:
        message.pop("metadata", None)
    assert history_contains_background_delivery(without_metadata, record)
    assert not history_contains_background_delivery(persisted[:1], record)

    # Context injection may append text without changing the authoritative delivery blocks
    # (https://github.com/strands-agents/stan/issues/16).
    injected_content = copy.deepcopy(persisted)
    injected_content[0]["content"].append({"text": "assistant context"})
    injected_content[1]["content"].append({"text": "user context"})
    assert history_contains_background_delivery(injected_content, record)

    altered_tool_use = copy.deepcopy(persisted)
    altered_tool_use[0]["content"][0] = {
        "toolUse": {
            "name": _DELIVERY_TOOL_NAME,
            "toolUseId": record["task_id"],
            "input": {"altered": True},
        }
    }
    assert not history_contains_background_delivery(altered_tool_use, record)

    altered_tool_result = copy.deepcopy(persisted)
    altered_tool_result[1]["content"][0] = {
        "toolResult": {
            "toolUseId": record["task_id"],
            "status": "success",
            "content": [{"text": "altered result"}],
        }
    }
    assert not history_contains_background_delivery(altered_tool_result, record)


@pytest.mark.asyncio
async def test_ready_delivery_accepts_every_turn_context_injection() -> None:
    model = _RecordingModel([_assistant_text("Delivered.")])
    agent = _create_ready_agent(model)
    render_calls = 0

    def render_content(_context: Any) -> str:
        nonlocal render_calls
        render_calls += 1
        return "INJECTED"

    ContextInjector(render_content, trigger="everyTurn").init_agent(agent)

    try:
        await agent.invoke_async("Deliver result.")

        assert render_calls > 0
        assert agent.state.get(_STATE_KEY) is None
        exp_delivery_count = 1
        assert _delivery_count(agent.messages) == exp_delivery_count
        tru_injected_delivery = any(
            any("toolResult" in content for content in message["content"])
            and any(content.get("text", "").endswith("INJECTED") for content in message["content"])
            for message in model.requests[0]
        )
        assert tru_injected_delivery
    finally:
        await asyncio.to_thread(agent.cleanup)


@pytest.mark.asyncio
async def test_ready_delivery_retries_after_provider_failure_without_partial_history() -> None:
    started = threading.Event()
    release = threading.Event()

    @tool(name="work")
    async def work() -> str:
        """Complete after the initial model turn."""
        started.set()
        if not await asyncio.to_thread(release.wait, 5):
            raise TimeoutError("test did not release background work")
        return "complete"

    model = _FailFirstDeliveryModel(
        [
            _assistant_tool_use(),
            _assistant_text("Task admitted."),
            _assistant_text("Delivery recovered."),
        ]
    )
    agent = Agent(
        model=model,
        tools=[work],
        background_tasks={"always": [work]},
        callback_handler=None,
    )
    invocation = asyncio.create_task(agent.invoke_async("Start work."))

    try:
        tru_started = await asyncio.to_thread(started.wait, 1)
        exp_started = True
        assert tru_started == exp_started
        while model.index < 2:
            await asyncio.sleep(0)
        release.set()

        with pytest.raises(RuntimeError, match="injected provider failure"):
            await invocation

        stored = agent.state.get(_STATE_KEY)
        assert isinstance(stored, dict)
        tru_delivery_states = [envelope["delivery_state"] for envelope in stored.values()]
        exp_delivery_states = ["ready"]
        assert tru_delivery_states == exp_delivery_states
        exp_partial_delivery_count = 0
        assert _delivery_count(agent.messages) == exp_partial_delivery_count

        tru_result = await agent.invoke_async("Retry delivery.")
        exp_final_text = "Delivery recovered."
        assert tru_result.message["content"][0].get("text") == exp_final_text
        assert agent.state.get(_STATE_KEY) is None
        exp_delivery_count = 1
        assert _delivery_count(agent.messages) == exp_delivery_count
    finally:
        release.set()
        if not invocation.done():
            await asyncio.wait_for(asyncio.gather(invocation, return_exceptions=True), timeout=2)
        await asyncio.to_thread(agent.cleanup)


@pytest.mark.asyncio
async def test_ready_delivery_retries_when_stream_closes_before_provider_request() -> None:
    model = _RecordingModel([_assistant_text("Delivery recovered.")])
    agent = _create_ready_agent(model)
    hook_entered = asyncio.Event()
    release_hook = asyncio.Event()
    pause_hook = True

    async def pause_before_model(_event: BeforeModelCallEvent) -> None:
        nonlocal pause_hook
        if not pause_hook:
            return
        pause_hook = False
        hook_entered.set()
        await release_hook.wait()

    agent.add_hook(pause_before_model, BeforeModelCallEvent)
    stream = agent.stream_async("Abandon delivery.")
    consumer = asyncio.create_task(_drain(stream))

    try:
        await asyncio.wait_for(hook_entered.wait(), timeout=1)
        consumer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await consumer
        await stream.aclose()

        tru_delivery_states = _delivery_states(agent)
        exp_delivery_states = ["ready"]
        assert tru_delivery_states == exp_delivery_states
        exp_request_count = 0
        assert len(model.requests) == exp_request_count
        exp_delivery_count = 0
        assert _delivery_count(agent.messages) == exp_delivery_count

        await agent.invoke_async("Retry delivery.")

        assert agent.state.get(_STATE_KEY) is None
        exp_delivery_count = 1
        assert _delivery_count(agent.messages) == exp_delivery_count
    finally:
        release_hook.set()
        if not consumer.done():
            consumer.cancel()
            await asyncio.gather(consumer, return_exceptions=True)
        await asyncio.to_thread(agent.cleanup)


@pytest.mark.asyncio
async def test_ready_delivery_retries_when_stream_closes_after_provider_request() -> None:
    requested = asyncio.Event()
    release_model = asyncio.Event()
    model = _PauseFirstRequestModel(
        [_assistant_text("Delivery recovered.")],
        requested,
        release_model,
    )
    agent = _create_ready_agent(model)
    stream = agent.stream_async("Abandon delivery.")
    consumer = asyncio.create_task(_drain(stream))

    try:
        await asyncio.wait_for(requested.wait(), timeout=1)
        consumer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await consumer
        await stream.aclose()

        tru_delivery_states = _delivery_states(agent)
        exp_delivery_states = ["ready"]
        assert tru_delivery_states == exp_delivery_states
        exp_request_delivery_ids = ["task-1"]
        assert _delivery_ids(model.requests[0]) == exp_request_delivery_ids
        exp_delivery_count = 0
        assert _delivery_count(agent.messages) == exp_delivery_count

        await agent.invoke_async("Retry delivery.")

        assert agent.state.get(_STATE_KEY) is None
        exp_delivery_count = 1
        assert _delivery_count(agent.messages) == exp_delivery_count
    finally:
        release_model.set()
        if not consumer.done():
            consumer.cancel()
            await asyncio.gather(consumer, return_exceptions=True)
        await asyncio.to_thread(agent.cleanup)


@pytest.mark.asyncio
async def test_ready_delivery_retries_after_before_model_hook_failure() -> None:
    model = _RecordingModel([_assistant_text("Delivery recovered.")])
    agent = _create_ready_agent(model)
    fail_hook = True

    def reject_delivery(_event: BeforeModelCallEvent) -> None:
        nonlocal fail_hook
        if not fail_hook:
            return
        fail_hook = False
        raise RuntimeError("injected hook failure")

    agent.add_hook(reject_delivery, BeforeModelCallEvent)

    try:
        with pytest.raises(RuntimeError, match="injected hook failure"):
            await agent.invoke_async("Reject delivery.")

        tru_delivery_states = _delivery_states(agent)
        exp_delivery_states = ["ready"]
        assert tru_delivery_states == exp_delivery_states
        exp_request_count = 0
        assert len(model.requests) == exp_request_count
        exp_delivery_count = 0
        assert _delivery_count(agent.messages) == exp_delivery_count

        await agent.invoke_async("Retry delivery.")

        assert agent.state.get(_STATE_KEY) is None
        exp_delivery_count = 1
        assert _delivery_count(agent.messages) == exp_delivery_count
    finally:
        await asyncio.to_thread(agent.cleanup)


@pytest.mark.asyncio
async def test_ready_delivery_retries_when_input_middleware_removes_delivery_result() -> None:
    model = _RecordingModel(
        [
            _assistant_text("Incomplete delivery."),
            _assistant_text("Delivery recovered."),
        ]
    )
    agent = _create_ready_agent(model)
    remove_result = True

    def remove_delivery_result(context: InvokeModelContext) -> InvokeModelContext:
        nonlocal remove_result
        if not remove_result:
            return context
        for index, message in enumerate(context.messages):
            if _contains_delivery([message]):
                remove_result = False
                return replace(
                    context,
                    messages=[
                        candidate
                        for candidate_index, candidate in enumerate(context.messages)
                        if candidate_index != index + 1
                    ],
                )
        return context

    agent._middleware_registry.add_middleware(InvokeModelStage.Input, remove_delivery_result)

    try:
        with pytest.raises(
            EventLoopException,
            match="Background task delivery 'task-1' was not present in the provider request",
        ):
            await agent.invoke_async("Deliver result.")

        tru_delivery_states = _delivery_states(agent)
        exp_delivery_states = ["ready"]
        assert tru_delivery_states == exp_delivery_states
        exp_delivery_count = 0
        assert _delivery_count(agent.messages) == exp_delivery_count

        await agent.invoke_async("Retry delivery.")

        assert agent.state.get(_STATE_KEY) is None
        exp_delivery_count = 1
        assert _delivery_count(agent.messages) == exp_delivery_count
    finally:
        await asyncio.to_thread(agent.cleanup)


@pytest.mark.asyncio
async def test_ready_delivery_retries_when_wrapping_middleware_rejects_response() -> None:
    model = _RecordingModel(
        [
            _assistant_text("Rejected delivery."),
            _assistant_text("Delivery recovered."),
        ]
    )
    agent = _create_ready_agent(model)
    reject_response = True

    async def reject_delivery_response(context: InvokeModelContext, next_fn: Any) -> AsyncGenerator[Any, None]:
        nonlocal reject_response
        async for event in next_fn(context):
            yield event
        if reject_response and _contains_delivery(context.messages):
            reject_response = False
            raise RuntimeError("injected wrapping middleware failure")

    agent._middleware_registry.add_middleware(InvokeModelStage, reject_delivery_response)

    try:
        with pytest.raises(RuntimeError, match="injected wrapping middleware failure"):
            await agent.invoke_async("Deliver result.")

        tru_delivery_states = _delivery_states(agent)
        exp_delivery_states = ["ready"]
        assert tru_delivery_states == exp_delivery_states
        exp_delivery_count = 0
        assert _delivery_count(agent.messages) == exp_delivery_count

        await agent.invoke_async("Retry delivery.")

        assert agent.state.get(_STATE_KEY) is None
        exp_delivery_count = 1
        assert _delivery_count(agent.messages) == exp_delivery_count
    finally:
        await asyncio.to_thread(agent.cleanup)


@pytest.mark.asyncio
async def test_ready_delivery_retries_when_model_streaming_is_cancelled() -> None:
    model = _RecordingModel([_assistant_text("Delivery recovered.")])
    agent = _create_ready_agent(model)
    cancel_model_call = True

    def cancel_delivery(event: BeforeModelCallEvent) -> None:
        nonlocal cancel_model_call
        if not cancel_model_call:
            return
        cancel_model_call = False
        event.agent.cancel()

    agent.add_hook(cancel_delivery, BeforeModelCallEvent)

    try:
        tru_cancelled = await agent.invoke_async("Cancel delivery.")
        exp_stop_reason = "cancelled"
        assert tru_cancelled.stop_reason == exp_stop_reason
        tru_delivery_states = _delivery_states(agent)
        exp_delivery_states = ["ready"]
        assert tru_delivery_states == exp_delivery_states
        exp_delivery_count = 0
        assert _delivery_count(agent.messages) == exp_delivery_count

        await agent.invoke_async("Retry delivery.")

        assert agent.state.get(_STATE_KEY) is None
        exp_delivery_count = 1
        assert _delivery_count(agent.messages) == exp_delivery_count
    finally:
        await asyncio.to_thread(agent.cleanup)


@pytest.mark.asyncio
async def test_ready_delivery_retries_after_context_overflow() -> None:
    model = _OverflowFirstDeliveryModel([_assistant_text("Delivery recovered.")])
    agent = _create_ready_agent(
        model,
        conversation_manager=NullConversationManager(),
    )

    try:
        with pytest.raises(ContextWindowOverflowException, match="injected context overflow"):
            await agent.invoke_async("Overflow delivery.")

        tru_delivery_states = _delivery_states(agent)
        exp_delivery_states = ["ready"]
        assert tru_delivery_states == exp_delivery_states
        exp_delivery_count = 0
        assert _delivery_count(agent.messages) == exp_delivery_count

        await agent.invoke_async("Retry delivery.")

        assert agent.state.get(_STATE_KEY) is None
        exp_delivery_count = 1
        assert _delivery_count(agent.messages) == exp_delivery_count
    finally:
        await asyncio.to_thread(agent.cleanup)


@pytest.mark.asyncio
async def test_snapshot_loaded_after_initialization_delivers_restored_result() -> None:
    record = _terminal_record()
    source = Agent(
        model=MockedModelProvider([]),
        state=_background_task_state(record),
        callback_handler=None,
    )
    snapshot = source.take_snapshot(include=["state"])
    model = _RecordingModel([_assistant_text("Delivery restored.")])
    restored = Agent(
        model=model,
        background_tasks=True,
        callback_handler=None,
        retry_strategy=None,
    )

    try:
        restored.load_snapshot(snapshot)
        restored.state.set("written_after_load", "keep-me")

        await restored.invoke_async("Continue.")

        exp_request_count = 1
        assert len(model.requests) == exp_request_count
        exp_delivery_ids = ["task-1"]
        assert _delivery_ids(model.requests[0]) == exp_delivery_ids
        assert restored.state.get(_STATE_KEY) is None
        exp_marker = "keep-me"
        assert restored.state.get("written_after_load") == exp_marker
    finally:
        await asyncio.to_thread(restored.cleanup)
        await asyncio.to_thread(source.cleanup)


@pytest.mark.parametrize("status", ["queued", "working", "paused"])
@pytest.mark.asyncio
async def test_manager_restoration_marks_live_work_failed_without_reexecution(status: str) -> None:
    executions: list[str] = []

    @tool(name="work")
    def work() -> str:
        """Record unexpected replay."""
        executions.append("executed")
        return "unexpected"

    record = _terminal_record()
    record["status"] = cast(Any, status)
    record.pop("result")
    if status == "working":
        record["attempt_id"] = "stale-attempt"
    if status == "paused":
        record["state"] = _InterruptState(
            interrupts={
                "approval": Interrupt(
                    id="approval",
                    name="approve",
                    reason="Approve restored work?",
                )
            },
            activated=True,
        ).to_dict()
    agent = Agent(
        model=MockedModelProvider([]),
        tools=[work],
        state=_background_task_state(record, "pending"),
        background_tasks=True,
        callback_handler=None,
        retry_strategy=None,
    )

    try:
        assert agent.background_tasks is not None
        tru_restored = await agent.background_tasks.get_async(record["task_id"])
        assert tru_restored is not None
        exp_status = "failed"
        assert tru_restored["status"] == exp_status
        exp_error = {
            "type": "recovery_error",
            "message": "Background task execution was interrupted while restoring persisted state",
        }
        assert tru_restored["error"] == exp_error
        assert executions == []
        tru_delivery_states = _delivery_states(agent)
        exp_delivery_states = ["ready"]
        assert tru_delivery_states == exp_delivery_states
    finally:
        await asyncio.to_thread(agent.cleanup)
