import asyncio
import copy
import json
import threading
from collections.abc import AsyncGenerator
from typing import Any

import pytest
from pydantic import BaseModel

from strands import Agent, tool
from strands._middleware.stages import AgentStreamStage
from strands.agent.conversation_manager import SlidingWindowConversationManager
from strands.types.content import Message, Messages, SystemContentBlock
from strands.types.streaming import StreamEvent
from strands.types.tools import ToolSpec
from strands.vended_plugins.context_offloader import ContextOffloader, InMemoryStorage
from strands.vended_plugins.goal import GoalLoop, ValidationOutcome
from tests.fixtures.mocked_model_provider import MockedModelProvider

_DELIVERY_TOOL_NAME = "strands_background_task_result"


class _StructuredResult(BaseModel):
    value: int


class _RecordingModel(MockedModelProvider):
    def __init__(self, agent_responses: list[Message]) -> None:
        super().__init__(agent_responses)
        self.requests: list[Messages] = []
        self.config: dict[str, Any] = {}

    def get_config(self) -> dict[str, Any]:
        return self.config

    def update_config(self, **model_config: Any) -> None:
        self.config.update(model_config)

    async def stream(
        self,
        messages: Messages,
        tool_specs: list[ToolSpec] | None = None,
        system_prompt: str | None = None,
        tool_choice: Any | None = None,
        *,
        system_prompt_content: list[SystemContentBlock] | None = None,
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


def _assistant_text(text: str) -> Message:
    return {"role": "assistant", "content": [{"text": text}]}


def _assistant_tool_uses(*tool_uses: tuple[str, str, dict[str, Any]]) -> Message:
    return {
        "role": "assistant",
        "content": [
            {
                "toolUse": {
                    "name": tool_name,
                    "toolUseId": tool_use_id,
                    "input": tool_input,
                }
            }
            for tool_name, tool_use_id, tool_input in tool_uses
        ],
    }


def _delivery_index(messages: Messages) -> int:
    return next(
        (
            index
            for index, message in enumerate(messages)
            if any(content.get("toolUse", {}).get("name") == _DELIVERY_TOOL_NAME for content in message["content"])
        ),
        -1,
    )


@pytest.mark.asyncio
async def test_background_delivery_precedes_goal_loop_retry() -> None:
    started = threading.Event()
    release = threading.Event()
    validations = 0

    @tool(name="work")
    async def work() -> str:
        """Complete after the first draft."""
        started.set()
        if not await asyncio.to_thread(release.wait, 5):
            raise TimeoutError("test did not release background work")
        return "background complete"

    def validate(_result: object, _attempt: int) -> bool | ValidationOutcome:
        nonlocal validations
        validations += 1
        if validations == 2:
            return True
        return ValidationOutcome(passed=False, feedback="use the completed background result")

    goal = GoalLoop(goal=validate, max_attempts=2)
    model = MockedModelProvider(
        [
            _assistant_tool_uses(("work", "work-use", {})),
            _assistant_text("draft"),
            _assistant_text("background-aware draft"),
        ]
    )
    agent = Agent(
        model=model,
        tools=[work],
        plugins=[goal],
        background_tasks={"always": [work]},
        callback_handler=None,
    )
    invocation = asyncio.create_task(agent.invoke_async("Start."))

    try:
        tru_started = await asyncio.to_thread(started.wait, 1)
        exp_started = True
        assert tru_started == exp_started
        while model.index < 2:
            await asyncio.sleep(0)
        release.set()

        tru_result = await invocation
        exp_final_text = "background-aware draft"
        assert tru_result.message["content"][0].get("text") == exp_final_text
        exp_validation_count = 2
        assert validations == exp_validation_count

        delivery_index = _delivery_index(agent.messages)
        retry_index = next(
            index
            for index, message in enumerate(agent.messages)
            if any("use the completed background result" in content.get("text", "") for content in message["content"])
        )
        assert delivery_index < retry_index
    finally:
        release.set()
        if not invocation.done():
            await asyncio.wait_for(asyncio.gather(invocation, return_exceptions=True), timeout=2)
        await asyncio.to_thread(agent.cleanup)


@pytest.mark.asyncio
async def test_background_delivery_survives_agent_stream_interrupt() -> None:
    started = threading.Event()
    release = threading.Event()

    @tool(name="work")
    async def work() -> str:
        """Complete after the initial agent pass."""
        started.set()
        if not await asyncio.to_thread(release.wait, 5):
            raise TimeoutError("test did not release background work")
        return "background complete"

    model = _RecordingModel(
        [
            _assistant_tool_uses(("work", "work-use", {})),
            _assistant_text("admitted"),
            _assistant_text("delivered"),
        ]
    )
    agent = Agent(
        model=model,
        tools=[work],
        background_tasks={"always": [work]},
        callback_handler=None,
    )
    gate_calls = 0

    async def gate_delivery(context, next_fn):
        nonlocal gate_calls
        if gate_calls == 0 and _delivery_index(context.messages) >= 0:
            gate_calls += 1
            context.interrupt("approve_background_delivery", reason="Deliver the completed task?")
        async for event in next_fn(context):
            yield event

    agent._middleware_registry.add_middleware(AgentStreamStage, gate_delivery)
    invocation = asyncio.create_task(agent.invoke_async("Start."))

    async def wait_for_initial_pass() -> None:
        while model.index < 2:
            await asyncio.sleep(0)

    try:
        assert await asyncio.to_thread(started.wait, 1)
        await asyncio.wait_for(wait_for_initial_pass(), timeout=1)
        release.set()

        interrupted = await invocation
        assert interrupted.stop_reason == "interrupt"
        assert interrupted.interrupts is not None
        assert [interrupt.name for interrupt in interrupted.interrupts] == ["approve_background_delivery"]

        resumed = await agent.invoke_async(
            [
                {
                    "interruptResponse": {
                        "interruptId": interrupted.interrupts[0].id,
                        "response": "continue",
                    }
                }
            ]
        )

        assert resumed.stop_reason == "end_turn"
        assert resumed.message["content"][0].get("text") == "delivered"
        assert gate_calls == 1
        assert any(_delivery_index(request) >= 0 for request in model.requests)

        background_tasks = agent.background_tasks
        assert background_tasks is not None
        assert await background_tasks.list_async() == []
    finally:
        release.set()
        if not invocation.done():
            await asyncio.wait_for(asyncio.gather(invocation, return_exceptions=True), timeout=2)
        await asyncio.to_thread(agent.cleanup)


@pytest.mark.asyncio
async def test_background_delivery_retains_structured_output() -> None:
    @tool(name="work")
    async def work(value: str) -> str:
        """Perform background work."""
        return f"background complete: {value}"

    model = _RecordingModel(
        [
            _assistant_tool_uses(
                ("work", "work-use", {"value": "x"}),
                ("_StructuredResult", "structured-1", {"value": 1}),
            ),
            _assistant_tool_uses(("_StructuredResult", "structured-2", {"value": 2})),
        ]
    )
    agent = Agent(
        model=model,
        tools=[work],
        background_tasks={"always": [work]},
        structured_output_model=_StructuredResult,
        callback_handler=None,
    )

    try:
        tru_result = await agent.invoke_async("Start.")
        exp_structured_output = _StructuredResult(value=2)
        assert tru_result.structured_output == exp_structured_output

        background_tasks = agent.background_tasks
        assert background_tasks is not None
        tru_tasks = await background_tasks.list_async()
        exp_tasks: list[object] = []
        assert tru_tasks == exp_tasks
        assert any(_delivery_index(request) >= 0 for request in model.requests)
    finally:
        await asyncio.to_thread(agent.cleanup)


@pytest.mark.asyncio
async def test_background_delivery_uses_context_offloader_result() -> None:
    large_result = "x" * 2_000

    @tool(name="work")
    async def work(value: str) -> str:
        """Perform background work."""
        return large_result

    offloader = ContextOffloader(
        storage=InMemoryStorage(),
        max_result_tokens=10,
        preview_tokens=2,
        include_retrieval_tool=False,
    )
    model = _RecordingModel(
        [
            _assistant_tool_uses(("work", "work-use", {"value": "x"})),
            _assistant_text("admitted"),
            _assistant_text("delivered"),
        ]
    )
    agent = Agent(
        model=model,
        tools=[work],
        plugins=[offloader],
        background_tasks={"always": [work]},
        callback_handler=None,
    )

    try:
        await agent.invoke_async("Start.")

        delivery_index = _delivery_index(agent.messages)
        delivered_result = agent.messages[delivery_index + 1]["content"][0]["toolResult"]
        delivered_text = [content["text"] for content in delivered_result["content"] if "text" in content]
        assert any("[Offloaded:" in text for text in delivered_text)
        assert large_result not in json.dumps(delivered_result)
    finally:
        await asyncio.to_thread(agent.cleanup)


@pytest.mark.asyncio
async def test_background_delivery_pair_survives_proactive_compaction() -> None:
    @tool(name="work")
    async def work(value: str) -> str:
        """Perform background work."""
        return f"background complete: {value}"

    model = _RecordingModel(
        [
            _assistant_tool_uses(("work", "work-use", {"value": "x"})),
            _assistant_text("admitted"),
            _assistant_text("delivery consumed"),
        ]
    )
    model.update_config(context_window_limit=500)
    conversation_manager = SlidingWindowConversationManager(
        window_size=0,
        proactive_compression={"compression_threshold": 0.1},
    )
    agent = Agent(
        model=model,
        tools=[work],
        background_tasks={"always": [work]},
        conversation_manager=conversation_manager,
        callback_handler=None,
    )

    try:
        await agent.invoke_async("Start.")

        delivery_request = next(request for request in model.requests if _delivery_index(request) >= 0)
        delivery_index = _delivery_index(delivery_request)
        delivery_messages = delivery_request[delivery_index : delivery_index + 2]

        assert conversation_manager.removed_message_count > 0
        assert delivery_messages[0]["content"][0]["toolUse"]["name"] == _DELIVERY_TOOL_NAME
        assert "toolResult" in delivery_messages[1]["content"][0]
        tru_tool_result_id = delivery_messages[1]["content"][0]["toolResult"]["toolUseId"]
        exp_tool_result_id = delivery_messages[0]["content"][0]["toolUse"]["toolUseId"]
        assert tru_tool_result_id == exp_tool_result_id
    finally:
        await asyncio.to_thread(agent.cleanup)
