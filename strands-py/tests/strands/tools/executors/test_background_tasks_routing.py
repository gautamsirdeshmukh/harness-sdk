"""Focused tests for Background Tasks tool policy integration."""

import threading
from unittest.mock import AsyncMock, Mock

import pytest

from strands.agent.agent import Agent
from strands.background_tasks._background_tasks import _BackgroundTasks
from strands.hooks.events import AfterToolCallEvent, BeforeToolCallEvent
from strands.tools.executors._executor import (
    ToolExecutor,
    _get_model_tool_specs,
    _tool_execution_context,
    _ToolExecutionContext,
)
from strands.types._events import ToolResultEvent
from strands.types.tools import ToolUse


@pytest.fixture
def executor() -> ToolExecutor:
    class RoutingToolExecutor(ToolExecutor):
        def _execute(self, *args, **kwargs):
            raise NotImplementedError

    return RoutingToolExecutor()


def _execution_context(
    agent,
    *,
    route_background: bool,
    background_task_id: str | None = None,
) -> _ToolExecutionContext:
    return _ToolExecutionContext(
        cancel_signal=threading.Event(),
        interrupt_state=agent._interrupt_state,
        route_background=route_background,
        pass_id="pass-1",
        background_task_id=background_task_id,
    )


def test_get_model_tool_specs_applies_background_policy(agent, weather_tool) -> None:
    original_spec = weather_tool.tool_spec
    agent._background_tasks = _BackgroundTasks({"agentic": [weather_tool]})

    tool_specs = _get_model_tool_specs(agent)

    weather_spec = next(spec for spec in tool_specs if spec["name"] == "weather_tool")
    assert weather_spec["inputSchema"]["json"]["properties"]["_background"]["type"] == "boolean"
    assert "_background" not in original_spec["inputSchema"]["json"]["properties"]


@pytest.mark.asyncio
async def test_agentic_route_strips_foreground_selector(
    executor,
    agent,
    weather_tool,
    tool_results,
    invocation_state,
    hook_events,
    alist,
) -> None:
    agent._background_tasks = _BackgroundTasks({"agentic": [weather_tool]})
    tool_use: ToolUse = {
        "name": "weather_tool",
        "toolUseId": "tool-1",
        "input": {"_background": False},
    }

    with _tool_execution_context(_execution_context(agent, route_background=True)):
        events = await alist(executor._stream(agent, tool_use, tool_results, invocation_state))

    assert events == [ToolResultEvent({"toolUseId": "tool-1", "status": "success", "content": [{"text": "sunny"}]})]
    before_event = next(event for event in hook_events if isinstance(event, BeforeToolCallEvent))
    assert before_event.tool_use["input"] == {}
    assert tool_use["input"] == {"_background": False}
    model_specs = [entry["toolSpec"] for entry in invocation_state["tool_config"]["tools"]]
    weather_spec = next(spec for spec in model_specs if spec["name"] == "weather_tool")
    assert weather_spec["inputSchema"]["json"]["properties"]["_background"]["type"] == "boolean"


@pytest.mark.asyncio
async def test_agentic_route_short_circuits_admitted_call(
    executor,
    agent,
    weather_tool,
    tool_results,
    invocation_state,
    hook_events,
    alist,
) -> None:
    manager = AsyncMock()
    manager.submit_tool_call.return_value = {"task_id": "task-1", "tool_name": "weather_tool"}
    background_tasks = _BackgroundTasks({"agentic": [weather_tool]})
    background_tasks._agent = agent
    background_tasks._manager = manager
    agent._background_tasks = background_tasks
    tool_use: ToolUse = {
        "name": "weather_tool",
        "toolUseId": "tool-1",
        "input": {"_background": True},
    }

    with _tool_execution_context(_execution_context(agent, route_background=True)):
        events = await alist(executor._stream(agent, tool_use, tool_results, invocation_state))

    assert len(events) == 1
    assert isinstance(events[0], ToolResultEvent)
    assert events[0].tool_result["toolUseId"] == "tool-1"
    assert events[0].tool_result["status"] == "success"
    assert "Background task dispatched." in events[0].tool_result["content"][0]["text"]
    manager.submit_tool_call.assert_awaited_once_with(
        tool_name="weather_tool",
        original_tool_use_id="tool-1",
        tool_input={},
        invocation_state=invocation_state,
        pass_id="pass-1",
        origin_span_context=None,
    )
    assert hook_events == []
    assert tool_results == [events[0].tool_result]


@pytest.mark.asyncio
async def test_direct_tool_execution_bypasses_background_routing(
    executor,
    agent,
    weather_tool,
    tool_results,
    invocation_state,
    alist,
) -> None:
    manager = AsyncMock()
    background_tasks = _BackgroundTasks({"always": [weather_tool]})
    background_tasks._agent = agent
    background_tasks._manager = manager
    agent._background_tasks = background_tasks
    tool_use: ToolUse = {"name": "weather_tool", "toolUseId": "tool-1", "input": {}}

    with _tool_execution_context(_execution_context(agent, route_background=False)):
        events = await alist(executor._stream(agent, tool_use, tool_results, invocation_state))

    assert events[-1] == ToolResultEvent({"toolUseId": "tool-1", "status": "success", "content": [{"text": "sunny"}]})
    manager.submit_tool_call.assert_not_awaited()


def test_direct_agent_tool_call_bypasses_always_policy(weather_tool) -> None:
    direct_agent = Agent(model=Mock(), tools=[weather_tool])
    direct_agent._background_tasks = _BackgroundTasks({"always": [weather_tool]})

    try:
        result = direct_agent.tool.weather_tool()
    finally:
        direct_agent._background_tasks = None
        direct_agent.cleanup()

    assert result["toolUseId"].startswith("tooluse_weather_tool_")
    assert result["status"] == "success"
    assert result["content"] == [{"text": "sunny"}]


@pytest.mark.asyncio
async def test_detached_execution_rejects_forbidden_hook_replacement(
    executor,
    agent,
    weather_tool,
    temperature_tool,
    tool_results,
    invocation_state,
    alist,
) -> None:
    agent._background_tasks = _BackgroundTasks(
        {
            "always": [weather_tool],
            "never": [temperature_tool],
        }
    )

    def replace_tool(event: BeforeToolCallEvent) -> None:
        event.selected_tool = temperature_tool

    agent.hooks.add_callback(BeforeToolCallEvent, replace_tool)
    tool_use: ToolUse = {"name": "weather_tool", "toolUseId": "tool-1", "input": {}}

    with _tool_execution_context(_execution_context(agent, route_background=False, background_task_id="task-1")):
        events = await alist(executor._stream(agent, tool_use, tool_results, invocation_state))

    result = events[-1]
    assert isinstance(result, ToolResultEvent)
    assert result.tool_result == {
        "toolUseId": "tool-1",
        "status": "error",
        "content": [{"text": "Error: Tool 'temperature_tool' is forbidden by background task policy"}],
    }


@pytest.mark.asyncio
async def test_detached_execution_rejects_hook_tool_use_id_mutation(
    executor,
    agent,
    weather_tool,
    tool_results,
    invocation_state,
    alist,
) -> None:
    agent._background_tasks = _BackgroundTasks({"always": [weather_tool]})

    def change_tool_use_id(event: BeforeToolCallEvent) -> None:
        event.tool_use = {**event.tool_use, "toolUseId": "changed-tool-id"}

    agent.hooks.add_callback(BeforeToolCallEvent, change_tool_use_id)
    tool_use: ToolUse = {"name": "weather_tool", "toolUseId": "tool-1", "input": {}}

    with _tool_execution_context(_execution_context(agent, route_background=False, background_task_id="task-1")):
        events = await alist(executor._stream(agent, tool_use, tool_results, invocation_state))

    result = events[-1]
    assert isinstance(result, ToolResultEvent)
    assert result.tool_result == {
        "toolUseId": "tool-1",
        "status": "error",
        "content": [{"text": "Error: Background task hooks cannot change the original tool-use ID"}],
    }


@pytest.mark.asyncio
async def test_detached_execution_retains_original_tool_when_hook_clears_selection(
    executor,
    agent,
    weather_tool,
    tool_results,
    invocation_state,
    alist,
) -> None:
    agent._background_tasks = _BackgroundTasks({"always": [weather_tool]})

    def clear_selected_tool(event: BeforeToolCallEvent) -> None:
        event.selected_tool = None

    agent.hooks.add_callback(BeforeToolCallEvent, clear_selected_tool)
    tool_use: ToolUse = {"name": "weather_tool", "toolUseId": "tool-1", "input": {}}

    with _tool_execution_context(_execution_context(agent, route_background=False, background_task_id="task-1")):
        events = await alist(executor._stream(agent, tool_use, tool_results, invocation_state))

    result = events[-1]
    assert isinstance(result, ToolResultEvent)
    assert result.tool_result == {
        "toolUseId": "tool-1",
        "status": "success",
        "content": [{"text": "sunny"}],
    }


@pytest.mark.asyncio
async def test_result_hook_cannot_change_model_tool_use_id(
    executor,
    agent,
    weather_tool,
    tool_results,
    invocation_state,
    alist,
) -> None:
    agent._background_tasks = _BackgroundTasks({"never": [weather_tool]})

    def change_result_id(event: AfterToolCallEvent) -> None:
        event.result = {**event.result, "toolUseId": "changed-tool-id"}

    agent.hooks.add_callback(AfterToolCallEvent, change_result_id)
    tool_use: ToolUse = {"name": "weather_tool", "toolUseId": "tool-1", "input": {}}

    with _tool_execution_context(_execution_context(agent, route_background=True)):
        events = await alist(executor._stream(agent, tool_use, tool_results, invocation_state))

    result = events[-1]
    assert isinstance(result, ToolResultEvent)
    assert result.tool_result == {
        "toolUseId": "tool-1",
        "status": "success",
        "content": [{"text": "sunny"}],
    }
