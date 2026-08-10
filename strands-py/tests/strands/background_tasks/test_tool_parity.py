import asyncio
from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from mcp.types import Tool as MCPTool

from strands import Agent, tool
from strands.tools.executors import SequentialToolExecutor
from strands.tools.mcp import MCPAgentTool, MCPClient
from strands.types._events import TypedEvent
from strands.types.content import Message, Messages
from strands.types.tools import ToolResult, ToolUse
from strands.vended_tools.sleep import make_sleep
from tests.fixtures.mocked_model_provider import MockedModelProvider

_DELIVERY_TOOL_NAME = "strands_background_task_result"


class _RecordingExecutor(SequentialToolExecutor):
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    async def _execute(
        self,
        agent: Agent,
        tool_uses: list[ToolUse],
        tool_results: list[ToolResult],
        cycle_trace: Any,
        cycle_span: Any,
        invocation_state: dict[str, Any],
        structured_output_context: Any = None,
    ) -> AsyncGenerator[TypedEvent, None]:
        self.batch_sizes.append(len(tool_uses))
        async for event in super()._execute(
            agent,
            tool_uses,
            tool_results,
            cycle_trace,
            cycle_span,
            invocation_state,
            structured_output_context,
        ):
            yield event


def _assistant_text(text: str) -> Message:
    return {"role": "assistant", "content": [{"text": text}]}


def _assistant_tool_use(tool_name: str, tool_use_id: str, tool_input: dict[str, Any]) -> Message:
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
        ],
    }


def _background_deliveries(messages: Messages) -> list[tuple[ToolUse, ToolResult]]:
    tool_results = {
        content["toolResult"]["toolUseId"]: content["toolResult"]
        for message in messages
        for content in message["content"]
        if "toolResult" in content
    }
    return [
        (content["toolUse"], tool_results[content["toolUse"]["toolUseId"]])
        for message in messages
        for content in message["content"]
        if "toolUse" in content
        and content["toolUse"]["name"] == _DELIVERY_TOOL_NAME
        and content["toolUse"]["toolUseId"] in tool_results
    ]


@pytest.mark.asyncio
async def test_configured_executor_handles_original_batch_and_detached_singleton() -> None:
    executions: list[str] = []

    @tool(name="background")
    async def background() -> str:
        """Run background work."""
        executions.append("background")
        return "background"

    @tool(name="foreground")
    async def foreground() -> str:
        """Run foreground work."""
        executions.append("foreground")
        return "foreground"

    executor = _RecordingExecutor()
    model = MockedModelProvider(
        [
            {
                "role": "assistant",
                "content": [
                    {
                        "toolUse": {
                            "name": "background",
                            "toolUseId": "background-use",
                            "input": {},
                        }
                    },
                    {
                        "toolUse": {
                            "name": "foreground",
                            "toolUseId": "foreground-use",
                            "input": {},
                        }
                    },
                ],
            },
            _assistant_text("Task admitted."),
            _assistant_text("Result delivered."),
        ]
    )
    agent = Agent(
        model=model,
        tools=[background, foreground],
        tool_executor=executor,
        background_tasks={"always": [background], "never": [foreground]},
        callback_handler=None,
    )

    try:
        await agent.invoke_async("Run both tools.")

        tru_batch_sizes = executor.batch_sizes
        exp_batch_sizes = [2, 1]
        assert tru_batch_sizes == exp_batch_sizes
        tru_executions = sorted(executions)
        exp_executions = ["background", "foreground"]
        assert tru_executions == exp_executions
    finally:
        await asyncio.to_thread(agent.cleanup)


@pytest.mark.asyncio
async def test_background_mcp_call_strips_selector_and_uses_task_cancel_signal() -> None:
    started = asyncio.Event()
    stopped = asyncio.Event()
    captured_arguments: dict[str, Any] = {}
    captured_signal: Any = None

    async def call_tool_async(**kwargs: Any) -> ToolResult:
        nonlocal captured_arguments, captured_signal
        captured_arguments = kwargs["arguments"]
        captured_signal = kwargs["cancel_signal"]
        started.set()
        await asyncio.to_thread(captured_signal.wait, 5)
        stopped.set()
        return {
            "toolUseId": kwargs["tool_use_id"],
            "status": "success",
            "content": [{"text": "remote stopped"}],
        }

    client = MagicMock(spec=MCPClient)
    client.call_tool_async = AsyncMock(side_effect=call_tool_async)
    remote = MCPAgentTool(
        MCPTool(
            name="remote",
            description="Remote work",
            inputSchema={
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
            },
        ),
        client,
    )
    model = MockedModelProvider(
        [
            _assistant_tool_use("remote", "remote-use", {"value": "x", "_background": True}),
            _assistant_text("Task admitted."),
        ]
    )
    agent = Agent(
        model=model,
        tools=[remote],
        background_tasks={
            "agentic": [remote],
            "wait_for_completion": False,
        },
        callback_handler=None,
    )
    background_tasks = agent.background_tasks
    assert background_tasks is not None

    try:
        await agent.invoke_async("Run remote work.")
        await asyncio.wait_for(started.wait(), timeout=1)
        tasks = await background_tasks.list_async()
        exp_task_count = 1
        assert len(tasks) == exp_task_count

        tru_cancelled = await background_tasks.cancel_async(tasks[0]["task_id"])
        exp_status = "cancelled"
        assert tru_cancelled["status"] == exp_status
        await asyncio.wait_for(stopped.wait(), timeout=1)
        await background_tasks.wait_async(timeout=2)

        exp_arguments = {"value": "x"}
        assert captured_arguments == exp_arguments
        assert captured_signal is not agent._cancel_signal
        assert captured_signal.is_set()
    finally:
        await asyncio.to_thread(agent.cleanup)


@pytest.mark.asyncio
async def test_background_delivery_preserves_multimodal_tool_content() -> None:
    @tool(name="media")
    async def media() -> dict[str, Any]:
        """Return image content."""
        return {
            "status": "success",
            "content": [
                {
                    "image": {
                        "format": "png",
                        "source": {"bytes": b"\x01\x02\x03"},
                    }
                }
            ],
        }

    model = MockedModelProvider(
        [
            _assistant_tool_use("media", "media-use", {}),
            _assistant_text("Task admitted."),
            _assistant_text("Result delivered."),
        ]
    )
    agent = Agent(
        model=model,
        tools=[media],
        background_tasks={"always": [media]},
        callback_handler=None,
    )

    try:
        await agent.invoke_async("Create media.")
        deliveries = _background_deliveries(agent.messages)
        exp_delivery_count = 1
        assert len(deliveries) == exp_delivery_count
        delivery_tool_use, delivery_tool_result = deliveries[0]
        exp_status = "completed"
        assert delivery_tool_use["input"]["status"] == exp_status
        exp_image = {
            "image": {
                "format": "png",
                "source": {"bytes": b"\x01\x02\x03"},
            }
        }
        assert exp_image in delivery_tool_result["content"]
    finally:
        await asyncio.to_thread(agent.cleanup)


@pytest.mark.asyncio
async def test_backgrounds_agent_as_tool_through_normal_delivery() -> None:
    child = Agent(
        name="child",
        description="Child agent",
        model=MockedModelProvider([_assistant_text("child complete")]),
        callback_handler=None,
    )
    child_tool = child.as_tool()
    parent = Agent(
        model=MockedModelProvider(
            [
                _assistant_tool_use("child", "child-use", {"input": "run child"}),
                _assistant_text("Task admitted."),
                _assistant_text("Result delivered."),
            ]
        ),
        tools=[child_tool],
        background_tasks={"always": [child_tool]},
        callback_handler=None,
    )

    try:
        await parent.invoke_async("Run child.")
        deliveries = _background_deliveries(parent.messages)
        exp_delivery_count = 1
        assert len(deliveries) == exp_delivery_count
        delivery_tool_use, delivery_tool_result = deliveries[0]
        exp_tool_name = "child"
        assert delivery_tool_use["input"]["tool_name"] == exp_tool_name
        assert any(content.get("text", "").strip() == "child complete" for content in delivery_tool_result["content"])
    finally:
        await asyncio.to_thread(parent.cleanup)
        await asyncio.to_thread(child.cleanup)


@pytest.mark.asyncio
async def test_background_sleep_releases_execution_capacity_on_task_cancellation() -> None:
    sleep_tool = make_sleep(max_duration=10, name="background_sleep")
    model = MockedModelProvider(
        [
            _assistant_tool_use("background_sleep", "sleep-use", {"duration": 5}),
            _assistant_text("Task admitted."),
        ]
    )
    agent = Agent(
        model=model,
        tools=[sleep_tool],
        background_tasks={
            "always": [sleep_tool],
            "wait_for_completion": False,
        },
        callback_handler=None,
    )
    background_tasks = agent.background_tasks
    assert background_tasks is not None

    try:
        await agent.invoke_async("Sleep.")
        tasks = await background_tasks.list_async()
        exp_task_count = 1
        assert len(tasks) == exp_task_count

        await background_tasks.cancel_async(tasks[0]["task_id"])
        await background_tasks.wait_async(timeout=1)

        tru_task = await background_tasks.get_async(tasks[0]["task_id"])
        assert tru_task is not None
        exp_status = "cancelled"
        assert tru_task["status"] == exp_status
    finally:
        await asyncio.to_thread(agent.cleanup)


@pytest.mark.asyncio
async def test_background_agent_as_tool_forwards_task_cancellation_to_child() -> None:
    child_sleep = make_sleep(max_duration=10, name="child_sleep")
    child = Agent(
        name="child",
        description="Child agent",
        model=MockedModelProvider(
            [
                _assistant_tool_use("child_sleep", "child-sleep-use", {"duration": 5}),
                _assistant_text("child complete"),
            ]
        ),
        tools=[child_sleep],
        callback_handler=None,
    )
    child_tool = child.as_tool()
    parent = Agent(
        model=MockedModelProvider(
            [
                _assistant_tool_use("child", "child-use", {"input": "sleep"}),
                _assistant_text("Task admitted."),
            ]
        ),
        tools=[child_tool],
        background_tasks={
            "always": [child_tool],
            "wait_for_completion": False,
        },
        callback_handler=None,
    )
    background_tasks = parent.background_tasks
    assert background_tasks is not None

    try:
        await parent.invoke_async("Run child.")
        tasks = await background_tasks.list_async()
        exp_task_count = 1
        assert len(tasks) == exp_task_count

        await background_tasks.cancel_async(tasks[0]["task_id"])
        await background_tasks.wait_async(timeout=1)

        tru_task = await background_tasks.get_async(tasks[0]["task_id"])
        assert tru_task is not None
        exp_status = "cancelled"
        assert tru_task["status"] == exp_status
    finally:
        await asyncio.to_thread(parent.cleanup)
        await asyncio.to_thread(child.cleanup)
