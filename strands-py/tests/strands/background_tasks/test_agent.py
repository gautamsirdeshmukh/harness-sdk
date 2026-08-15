import asyncio
import json
import threading
import time
from collections.abc import AsyncGenerator
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from mcp.types import Tool as MCPTool
from opentelemetry import trace as trace_api
from opentelemetry.trace import NonRecordingSpan, SpanContext

from strands import Agent, ToolContext, tool
from strands.agent.agent_result import AgentResult
from strands.background_tasks import BackgroundTaskNotFoundError, BackgroundTasksTimeoutError
from strands.tools.mcp import MCPAgentTool, MCPClient
from strands.types._events import ToolResultEvent, TypedEvent
from strands.types.content import Message, Messages
from strands.types.tools import AgentTool, ToolResult, ToolUse
from tests.fixtures.mocked_model_provider import MockedModelProvider

_BACKGROUND_RESULT_TOOL_NAME = "strands_background_task_result"


def _assistant_text(text: str) -> Message:
    return {"role": "assistant", "content": [{"text": text}]}


def _assistant_tool_use(
    tool_use_id: str,
    *,
    select_background: bool,
    tool_name: str = "work",
) -> Message:
    tool_input = {"_background": True} if select_background else {}
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


def _assistant_tool_uses(*tool_uses: tuple[str, str]) -> Message:
    return {
        "role": "assistant",
        "content": [
            {
                "toolUse": {
                    "name": tool_name,
                    "toolUseId": tool_use_id,
                    "input": {},
                }
            }
            for tool_name, tool_use_id in tool_uses
        ],
    }


def _gated_work(
    started: threading.Event,
    release: threading.Event,
    executions: list[str],
) -> AgentTool:
    @tool(name="work")
    async def work() -> str:
        """Wait for the test gate, then complete."""
        executions.append("started")
        started.set()
        if not await asyncio.to_thread(release.wait, 5):
            raise TimeoutError("test did not release background work")
        executions.append("completed")
        return "complete"

    return work


def _background_deliveries(messages: Messages) -> list[tuple[ToolUse, ToolResult]]:
    tool_results: dict[str, ToolResult] = {}
    for message in messages:
        for content in message["content"]:
            if "toolResult" in content:
                tool_result = content["toolResult"]
                tool_results[tool_result["toolUseId"]] = tool_result

    deliveries: list[tuple[ToolUse, ToolResult]] = []
    for message in messages:
        for content in message["content"]:
            if "toolUse" not in content:
                continue
            tool_use = content["toolUse"]
            if tool_use["name"] != _BACKGROUND_RESULT_TOOL_NAME:
                continue
            tool_result = tool_results.get(tool_use["toolUseId"])
            if tool_result is not None:
                deliveries.append((tool_use, tool_result))
    return deliveries


async def _wait_for_model_calls(model: MockedModelProvider, count: int) -> None:
    async def wait() -> None:
        while model.index < count:
            await asyncio.sleep(0)

    await asyncio.wait_for(wait(), timeout=1)


async def _wait_for_task_status(agent: Agent, tool_name: str, status: str) -> None:
    background_tasks = agent.background_tasks
    assert background_tasks is not None

    async def wait() -> None:
        while not any(
            task["tool_name"] == tool_name and task["status"] == status for task in await background_tasks.list_async()
        ):
            await asyncio.sleep(0)

    await asyncio.wait_for(wait(), timeout=2)


@pytest.mark.asyncio
async def test_detached_tool_executes_under_background_task_span() -> None:
    agent = Agent(model=MockedModelProvider([]), callback_handler=None)
    task_span = NonRecordingSpan(SpanContext(trace_id=1, span_id=2, is_remote=False))
    agent.tracer = MagicMock()
    agent.tracer.start_background_task_span.return_value = task_span
    observed_spans: list[tuple[object, object]] = []

    async def execute(
        _agent: Agent,
        _tool_uses: list[ToolUse],
        _tool_results: list[ToolResult],
        _cycle_trace: Any,
        cycle_span: object,
        _invocation_state: dict[str, Any],
        _structured_output_context: object | None = None,
    ) -> AsyncGenerator[TypedEvent, None]:
        observed_spans.append((cycle_span, trace_api.get_current_span()))
        yield ToolResultEvent(
            {
                "toolUseId": "tool-use",
                "status": "success",
                "content": [{"text": "done"}],
            }
        )

    cast(Any, agent.tool_executor)._execute = execute

    try:
        result = await agent._execute_detached_tool(
            tool_use={"name": "work", "toolUseId": "tool-use", "input": {}},
            invocation_state={},
            cancel_signal=threading.Event(),
            interrupt_state=None,
            task_id="task-1",
            attempt=2,
            attempt_id="attempt-1",
            execution_id="execution-1",
            origin_span_context=None,
        )
    finally:
        agent.cleanup()

    assert result["result"]["status"] == "success"
    assert observed_spans == [(task_span, task_span)]
    agent.tracer.end_background_task_span.assert_called_once_with(task_span, outcome="completed")


@pytest.mark.asyncio
async def test_invoke_async_applies_background_policy_to_model_specs() -> None:
    @tool(name="work")
    def work(value: str) -> str:
        """Return the supplied value."""
        return value

    @tool(name="foreground")
    def foreground(value: str) -> str:
        """Return the supplied value in the foreground."""
        return value

    model = MockedModelProvider([_assistant_text("Done.")])
    agent = Agent(
        model=model,
        tools=[work, foreground],
        background_tasks={"agentic": [work], "never": [foreground]},
        callback_handler=None,
    )

    try:
        await agent.invoke_async("Start work.")

        assert model.last_tool_specs is not None
        work_spec = next(spec for spec in model.last_tool_specs if spec["name"] == "work")
        foreground_spec = next(spec for spec in model.last_tool_specs if spec["name"] == "foreground")
        management_spec = next(
            spec for spec in model.last_tool_specs if spec["name"] == "strands_manage_background_task"
        )
        assert "_background" in work_spec["inputSchema"]["json"]["properties"]
        assert "_background" not in foreground_spec["inputSchema"]["json"]["properties"]
        assert "_background" not in management_spec["inputSchema"]["json"]["properties"]
    finally:
        await asyncio.to_thread(agent.cleanup)


@pytest.mark.asyncio
async def test_persisted_state_excludes_tool_input_and_invocation_state() -> None:
    started = threading.Event()
    release = threading.Event()

    @tool(name="work")
    async def work(secret_input: str) -> str:
        """Wait for release without returning the input."""
        started.set()
        if not await asyncio.to_thread(release.wait, 5):
            raise TimeoutError("test did not release background work")
        return "complete"

    model = MockedModelProvider(
        [
            {
                "role": "assistant",
                "content": [
                    {
                        "toolUse": {
                            "name": "work",
                            "toolUseId": "secret-use",
                            "input": {"secret_input": "tool-secret"},
                        }
                    }
                ],
            },
            _assistant_text("Task admitted."),
        ]
    )
    agent = Agent(
        model=model,
        tools=[work],
        background_tasks={"always": [work], "wait_for_completion": False},
        callback_handler=None,
    )

    try:
        await agent.invoke_async("Start work.", invocation_state={"api_token": "invocation-secret"})
        assert await asyncio.to_thread(started.wait, 1)

        persisted = agent.state.get("strands.background_tasks")
        assert isinstance(persisted, dict)
        serialized = json.dumps(persisted)
        assert "tool-secret" not in serialized
        assert "invocation-secret" not in serialized
        stored_record = next(iter(persisted.values()))["record"]
        assert "descriptor" not in stored_record
        assert "attempt_count" not in stored_record
        assert stored_record["tool_name"] == "work"
    finally:
        release.set()
        await asyncio.to_thread(agent.cleanup)


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"wait_for_completin": False}, "Unknown Background Tasks configuration key"),
        ({"wait_for_completion": "false"}, "wait_for_completion must be a boolean"),
        ({"max_concurrency": 0}, "max_concurrency must be a positive integer"),
        ({"max_concurrency": True}, "max_concurrency must be a positive integer"),
        ({"timeout": 0}, "timeout must be a positive number"),
        ({"timeout": float("nan")}, "timeout must be a positive number"),
        ({"agentic": "*"}, "agentic must be a list"),
        ({"always": ["work"]}, "always entries must be AgentTool instances"),
    ],
)
def test_init_rejects_invalid_background_task_config(config: object, message: str) -> None:
    with pytest.raises(TypeError, match=message):
        Agent(
            model=MockedModelProvider([]),
            background_tasks=config,  # type: ignore[arg-type]
            callback_handler=None,
        )


def test_public_controls_raise_public_exceptions() -> None:
    started = threading.Event()
    release = threading.Event()
    executions: list[str] = []
    work = _gated_work(started, release, executions)
    model = MockedModelProvider(
        [
            _assistant_tool_use("work-timeout", select_background=False),
            _assistant_text("Task admitted."),
        ]
    )
    agent = Agent(
        model=model,
        tools=[work],
        background_tasks={"always": [work], "wait_for_completion": False},
        callback_handler=None,
    )
    background_tasks = agent.background_tasks
    assert background_tasks is not None

    try:
        agent("Start work.")
        assert started.wait(1)

        with pytest.raises(BackgroundTasksTimeoutError) as timeout_info:
            background_tasks.wait(timeout=0.01)
        assert timeout_info.value.timeout == 0.01

        with pytest.raises(BackgroundTaskNotFoundError, match="was not found"):
            background_tasks.cancel("missing")
    finally:
        release.set()
        background_tasks.wait(timeout=2)
        agent.cleanup()


@pytest.mark.asyncio
async def test_invoke_async_default_wait_delivers_agentic_result_in_same_invocation() -> None:
    started = threading.Event()
    release = threading.Event()
    executions: list[str] = []
    work = _gated_work(started, release, executions)
    model = MockedModelProvider(
        [
            _assistant_tool_use("work-1", select_background=True),
            _assistant_text("Task admitted."),
            _assistant_text("Result received."),
        ]
    )
    agent = Agent(
        model=model,
        tools=[work],
        background_tasks=True,
        callback_handler=None,
    )
    background_tasks = agent.background_tasks
    assert background_tasks is not None
    invocation = asyncio.create_task(agent.invoke_async("Start work."))

    try:
        tru_started = await asyncio.to_thread(started.wait, 1)
        exp_started = True
        assert tru_started == exp_started

        await _wait_for_model_calls(model, 2)
        tru_invocation_done = invocation.done()
        exp_invocation_done = False
        assert tru_invocation_done == exp_invocation_done

        release.set()
        tru_result = await invocation

        tru_final_text = tru_result.message["content"][0].get("text")
        exp_final_text = "Result received."
        assert tru_final_text == exp_final_text

        tru_model_calls = model.index
        exp_model_calls = 3
        assert tru_model_calls == exp_model_calls

        tru_executions = executions
        exp_executions = ["started", "completed"]
        assert tru_executions == exp_executions

        tru_deliveries = _background_deliveries(agent.messages)
        exp_delivery_count = 1
        assert len(tru_deliveries) == exp_delivery_count
        delivery_tool_use, delivery_tool_result = tru_deliveries[0]

        tru_delivery_status = delivery_tool_use["input"]["status"]
        exp_delivery_status = "completed"
        assert tru_delivery_status == exp_delivery_status

        tru_result_content = delivery_tool_result["content"]
        exp_result_block = {"text": "complete"}
        assert exp_result_block in tru_result_content

        tru_retained_tasks = await background_tasks.list_async()
        exp_retained_tasks: list[object] = []
        assert tru_retained_tasks == exp_retained_tasks
    finally:
        release.set()
        if not invocation.done():
            await asyncio.wait_for(asyncio.gather(invocation, return_exceptions=True), timeout=2)
        await asyncio.to_thread(agent.cleanup)


@pytest.mark.asyncio
async def test_cleanup_cancels_work_without_blocking_caller_loop() -> None:
    started = threading.Event()

    @tool(name="work", context=True)
    async def work(tool_context: ToolContext) -> str:
        """Wait until cleanup requests cancellation."""
        started.set()
        while not tool_context.cancel_signal.is_set():
            await asyncio.sleep(0.01)
        return "cancelled"

    model = MockedModelProvider(
        [
            _assistant_tool_use("work-cleanup", select_background=True),
            _assistant_text("Task admitted."),
            _assistant_text("Result received."),
        ]
    )
    agent = Agent(model=model, tools=[work], background_tasks=True, callback_handler=None)
    invocation = asyncio.create_task(agent.invoke_async("Start work."))

    try:
        assert await asyncio.to_thread(started.wait, 1)
        started_at = time.perf_counter()
        agent.cleanup()
        elapsed = time.perf_counter() - started_at

        assert elapsed < 1
        await invocation
    finally:
        if not invocation.done():
            await asyncio.wait_for(asyncio.gather(invocation, return_exceptions=True), timeout=2)


def test_interrupted_asyncio_run_keeps_background_sibling_alive() -> None:
    background_started = threading.Event()
    release_background = threading.Event()

    @tool(name="survivor")
    async def survivor() -> str:
        """Complete after the invocation event loop closes."""
        background_started.set()
        if not await asyncio.to_thread(release_background.wait, 5):
            raise TimeoutError("test did not release background work")
        return "survived"

    @tool(name="interrupting", context=True)
    async def interrupting(tool_context: ToolContext) -> str:
        """Interrupt after the background sibling starts."""
        if not await asyncio.to_thread(background_started.wait, 5):
            raise TimeoutError("background work did not start")
        tool_context.interrupt("foreground_interrupt", reason="Pause the invocation")
        return "unreachable"

    model = MockedModelProvider(
        [
            _assistant_tool_uses(("survivor", "survivor-use"), ("interrupting", "interrupt-use")),
        ]
    )
    agent = Agent(
        model=model,
        tools=[survivor, interrupting],
        background_tasks={"always": [survivor], "never": [interrupting]},
        callback_handler=None,
    )
    background_tasks = agent.background_tasks
    assert background_tasks is not None

    try:
        interrupted = asyncio.run(agent.invoke_async("Run both tools."))

        assert interrupted.stop_reason == "interrupt"
        assert interrupted.interrupts is not None
        assert interrupted.interrupts[0].name == "foreground_interrupt"

        release_background.set()
        background_tasks.wait(timeout=2)

        tasks = background_tasks.list()
        assert len(tasks) == 1
        assert tasks[0]["status"] == "completed"
        assert tasks[0]["result"]["content"] == [{"text": "survived"}]
    finally:
        release_background.set()
        agent.cleanup()


def test_invoke_async_non_waiting_task_survives_caller_loop_shutdown() -> None:
    started = threading.Event()
    release = threading.Event()

    @tool(name="survivor")
    async def survivor() -> str:
        """Complete after the invoking event loop has closed."""
        started.set()
        if not await asyncio.to_thread(release.wait, 5):
            raise TimeoutError("test did not release background work")
        return "survived"

    model = MockedModelProvider(
        [
            _assistant_tool_use("survivor-use", select_background=False, tool_name="survivor"),
            _assistant_text("Task admitted."),
        ]
    )
    agent = Agent(
        model=model,
        tools=[survivor],
        background_tasks={"always": [survivor], "wait_for_completion": False},
        callback_handler=None,
    )
    background_tasks = agent.background_tasks
    assert background_tasks is not None

    try:
        asyncio.run(agent.invoke_async("Run surviving work."))
        tru_started = started.wait(1)
        exp_started = True
        assert tru_started == exp_started

        release.set()
        background_tasks.wait(timeout=2)

        tru_tasks = background_tasks.list()
        exp_task_count = 1
        assert len(tru_tasks) == exp_task_count
        exp_status = "completed"
        assert tru_tasks[0]["status"] == exp_status
        exp_content = [{"text": "survived"}]
        assert tru_tasks[0]["result"]["content"] == exp_content
    finally:
        release.set()
        agent.cleanup()


@pytest.mark.asyncio
async def test_invoke_async_surfaces_and_resumes_background_interrupt_without_stopping_sibling() -> None:
    independent_started = threading.Event()
    release_independent = threading.Event()
    allow_interrupt = threading.Event()
    approval_resumed = threading.Event()
    release_approval = threading.Event()

    @tool(name="approval", context=True)
    async def approval(tool_context: ToolContext) -> str:
        """Wait for approval before completing."""
        if not await asyncio.to_thread(allow_interrupt.wait, 5):
            raise TimeoutError("test did not allow background interrupt")
        response = tool_context.interrupt("approve_background_work", reason="Approve background work?")
        approval_resumed.set()
        if not await asyncio.to_thread(release_approval.wait, 5):
            raise TimeoutError("test did not release resumed approval")
        return f"approved: {response}"

    @tool(name="independent")
    async def independent() -> str:
        """Complete independent work after the test gate opens."""
        independent_started.set()
        if not await asyncio.to_thread(release_independent.wait, 5):
            raise TimeoutError("test did not release independent work")
        return "independent complete"

    model = MockedModelProvider(
        [
            _assistant_tool_uses(("approval", "approval-use"), ("independent", "independent-use")),
            _assistant_text("Background work started."),
            _assistant_text("Independent work received."),
            _assistant_text("All background work received."),
        ]
    )
    agent = Agent(
        model=model,
        tools=[approval, independent],
        background_tasks={"always": [approval, independent]},
        callback_handler=None,
    )
    background_tasks = agent.background_tasks
    assert background_tasks is not None
    invocation = asyncio.create_task(agent.invoke_async("Run both tasks."))
    resumed_invocation: asyncio.Task[AgentResult] | None = None

    try:
        tru_independent_started = await asyncio.to_thread(independent_started.wait, 1)
        exp_independent_started = True
        assert tru_independent_started == exp_independent_started
        await _wait_for_model_calls(model, 2)
        allow_interrupt.set()

        tru_interrupted = await invocation
        exp_stop_reason = "interrupt"
        assert tru_interrupted.stop_reason == exp_stop_reason
        assert tru_interrupted.interrupts is not None
        exp_interrupt_count = 1
        assert len(tru_interrupted.interrupts) == exp_interrupt_count
        interrupt = tru_interrupted.interrupts[0]
        exp_interrupt_name = "approve_background_work"
        assert interrupt.name == exp_interrupt_name
        exp_interrupt_reason = "Approve background work?"
        assert interrupt.reason == exp_interrupt_reason

        tru_tasks_while_interrupted = {
            task["tool_name"]: task["status"] for task in await background_tasks.list_async()
        }
        exp_tasks_while_interrupted = {"approval": "paused", "independent": "working"}
        assert tru_tasks_while_interrupted == exp_tasks_while_interrupted

        release_independent.set()
        await _wait_for_task_status(agent, "independent", "completed")

        resumed_invocation = asyncio.create_task(
            agent.invoke_async([{"interruptResponse": {"interruptId": interrupt.id, "response": "yes"}}])
        )
        tru_approval_resumed = await asyncio.to_thread(approval_resumed.wait, 1)
        exp_approval_resumed = True
        assert tru_approval_resumed == exp_approval_resumed
        await _wait_for_model_calls(model, 3)
        release_approval.set()
        tru_completed = await resumed_invocation
        exp_completed_stop_reason = "end_turn"
        assert tru_completed.stop_reason == exp_completed_stop_reason
        exp_final_text = "All background work received."
        assert tru_completed.message["content"][0].get("text") == exp_final_text

        tru_deliveries = _background_deliveries(agent.messages)
        tru_delivery_tools = [delivery[0]["input"]["tool_name"] for delivery in tru_deliveries]
        exp_delivery_tools = ["independent", "approval"]
        assert tru_delivery_tools == exp_delivery_tools
        tru_delivery_text = [
            content["text"]
            for _, result in tru_deliveries
            for content in result["content"]
            if "text" in content and content["text"] in {"independent complete", "approved: yes"}
        ]
        exp_delivery_text = ["independent complete", "approved: yes"]
        assert tru_delivery_text == exp_delivery_text

        tru_retained_tasks = await background_tasks.list_async()
        exp_retained_tasks: list[object] = []
        assert tru_retained_tasks == exp_retained_tasks
    finally:
        allow_interrupt.set()
        release_approval.set()
        release_independent.set()
        if not invocation.done():
            await asyncio.wait_for(asyncio.gather(invocation, return_exceptions=True), timeout=2)
        if resumed_invocation is not None and not resumed_invocation.done():
            await asyncio.wait_for(asyncio.gather(resumed_invocation, return_exceptions=True), timeout=2)
        await asyncio.to_thread(agent.cleanup)


def test_call_non_waiting_task_survives_sync_loop_and_delivers_once_next_invocation() -> None:
    started = threading.Event()
    release = threading.Event()
    executions: list[str] = []
    work = _gated_work(started, release, executions)
    model = MockedModelProvider(
        [
            _assistant_tool_use("work-2", select_background=False),
            _assistant_text("Task admitted."),
            _assistant_text("Result received."),
        ]
    )
    agent = Agent(
        model=model,
        tools=[work],
        background_tasks={
            "always": [work],
            "wait_for_completion": False,
        },
        callback_handler=None,
    )
    background_tasks = agent.background_tasks
    assert background_tasks is not None

    try:
        tru_initial_result = agent("Start work.")
        exp_initial_text = "Task admitted."
        assert tru_initial_result.message["content"][0].get("text") == exp_initial_text

        tru_started = started.wait(1)
        exp_started = True
        assert tru_started == exp_started

        tru_running_tasks = background_tasks.list()
        exp_running_task_count = 1
        assert len(tru_running_tasks) == exp_running_task_count
        exp_running_status = "working"
        assert tru_running_tasks[0]["status"] == exp_running_status
        task_id = tru_running_tasks[0]["task_id"]

        tru_running_executions = executions
        exp_running_executions = ["started"]
        assert tru_running_executions == exp_running_executions

        tru_initial_deliveries = _background_deliveries(agent.messages)
        exp_initial_deliveries: list[object] = []
        assert tru_initial_deliveries == exp_initial_deliveries

        release.set()
        background_tasks.wait(timeout=2)

        tru_completed_tasks = background_tasks.list()
        exp_completed_task_count = 1
        assert len(tru_completed_tasks) == exp_completed_task_count
        exp_completed_status = "completed"
        assert tru_completed_tasks[0]["status"] == exp_completed_status
        exp_result_block = {"text": "complete"}
        assert exp_result_block in tru_completed_tasks[0]["result"]["content"]

        tru_result = agent("Continue.")
        exp_final_text = "Result received."
        assert tru_result.message["content"][0].get("text") == exp_final_text

        tru_deliveries = _background_deliveries(agent.messages)
        exp_delivery_count = 1
        assert len(tru_deliveries) == exp_delivery_count
        delivery_tool_use, delivery_tool_result = tru_deliveries[0]
        assert delivery_tool_use["toolUseId"] == task_id
        assert delivery_tool_result["toolUseId"] == task_id
        assert exp_result_block in delivery_tool_result["content"]

        tru_model_calls = model.index
        exp_model_calls = 3
        assert tru_model_calls == exp_model_calls

        tru_retained_tasks = background_tasks.list()
        exp_retained_tasks: list[object] = []
        assert tru_retained_tasks == exp_retained_tasks
    finally:
        release.set()
        agent.cleanup()


def test_init_preserves_exact_mcp_tool_selector_identity() -> None:
    mcp_client = MCPClient(MagicMock())
    mcp_tool = MCPAgentTool(
        MCPTool(
            name="remote_work",
            description="Remote work",
            inputSchema={"type": "object", "properties": {}},
        ),
        mcp_client,
    )
    agent = Agent(
        model=MockedModelProvider([]),
        tools=[mcp_tool],
        background_tasks={"never": [mcp_tool]},
        callback_handler=None,
    )

    agent.cleanup()
