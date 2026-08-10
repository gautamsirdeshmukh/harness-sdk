"""Abstract base class for tool executors.

Tool executors are responsible for determining how tools are executed (e.g., concurrently, sequentially, with custom
thread pools, etc.).
"""

import abc
import contextvars
import logging
import threading
import time
import uuid
from collections.abc import AsyncGenerator, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from opentelemetry import trace as trace_api
from opentelemetry.trace import SpanContext

from ..._middleware.stages import ExecuteToolContext, ExecuteToolStage
from ...experimental.hooks.events import BidiAfterToolCallEvent, BidiBeforeToolCallEvent
from ...hooks import AfterToolCallEvent, BeforeToolCallEvent
from ...interrupt import InterruptException, _InterruptState
from ...telemetry.metrics import EventLoopMetrics, Trace
from ...telemetry.tracer import get_tracer, serialize
from ...types._events import ToolCancelEvent, ToolInterruptEvent, ToolResultEvent, ToolStreamEvent, TypedEvent
from ...types.content import Message
from ...types.interrupt import Interrupt
from ...types.tools import ToolChoice, ToolChoiceAuto, ToolConfig, ToolResult, ToolSpec, ToolUse
from ..structured_output._structured_output_context import StructuredOutputContext

if TYPE_CHECKING:  # pragma: no cover
    from ...agent import Agent
    from ...experimental.bidi import BidiAgent

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _ToolExecutionContext:
    """Execution-scoped state that must not be read from the mutable Agent."""

    cancel_signal: threading.Event
    interrupt_state: _InterruptState
    route_background: bool = True
    event_loop_metrics: EventLoopMetrics | None = None
    pass_id: str | None = None
    origin_span_context: SpanContext | None = None
    background_task_id: str | None = None

    def __post_init__(self) -> None:
        if self.route_background and self.pass_id is None:
            object.__setattr__(self, "pass_id", str(uuid.uuid4()))


_CURRENT_TOOL_EXECUTION_CONTEXT: contextvars.ContextVar[_ToolExecutionContext | None] = contextvars.ContextVar(
    "strands_tool_execution_context",
    default=None,
)


def _get_current_tool_execution_context() -> _ToolExecutionContext | None:
    """Return the execution context for the current tool coroutine."""
    return _CURRENT_TOOL_EXECUTION_CONTEXT.get()


@contextmanager
def _tool_execution_context(context: _ToolExecutionContext) -> Generator[None, None, None]:
    """Scope tool execution state to the current async context."""
    token = _CURRENT_TOOL_EXECUTION_CONTEXT.set(context)
    try:
        yield
    finally:
        _CURRENT_TOOL_EXECUTION_CONTEXT.reset(token)


def _get_background_tasks(agent: Any) -> Any | None:
    """Return the Agent's Background Tasks integration without triggering mock attributes."""
    return vars(agent).get("_background_tasks")


def _get_model_tool_specs(agent: Any) -> list[ToolSpec]:
    """Assemble current model-facing tool specs with Background Tasks policy applied."""
    tool_specs = cast(list[ToolSpec], agent.tool_registry.get_all_tool_specs())
    background_tasks = _get_background_tasks(agent)
    if background_tasks is None:
        return tool_specs

    registry = agent.tool_registry
    tools = list(registry.registry.values())
    tools.extend(tool for name, tool in registry.dynamic_tools.items() if name not in registry.registry)
    return cast(list[ToolSpec], background_tasks.transform_tool_specs(tools, tool_specs))


def _normalize_tool_result_id(result: ToolResult, tool_use_id: str) -> ToolResult:
    """Keep results correlated to the model-issued tool-use ID."""
    if "toolUseId" not in result or result["toolUseId"] == tool_use_id:
        return result
    return cast(ToolResult, {**result, "toolUseId": tool_use_id})


class ToolExecutor(abc.ABC):
    """Abstract base class for tool executors."""

    @staticmethod
    def _is_agent(agent: "Agent | BidiAgent") -> bool:
        """Check if the agent is an Agent instance, otherwise we assume BidiAgent.

        Note, we use a runtime import to avoid a circular dependency error.
        """
        from ...agent import Agent

        return isinstance(agent, Agent)

    @staticmethod
    async def _invoke_before_tool_call_hook(
        agent: "Agent | BidiAgent",
        tool_func: Any,
        tool_use: ToolUse,
        invocation_state: dict[str, Any],
        interrupt_state: _InterruptState | None = None,
    ) -> tuple[BeforeToolCallEvent | BidiBeforeToolCallEvent, list[Interrupt]]:
        """Invoke the appropriate before tool call hook based on agent type."""
        kwargs = {
            "selected_tool": tool_func,
            "tool_use": tool_use,
            "invocation_state": invocation_state,
        }
        event = (
            BeforeToolCallEvent(agent=cast("Agent", agent), **kwargs)
            if ToolExecutor._is_agent(agent)
            else BidiBeforeToolCallEvent(agent=cast("BidiAgent", agent), **kwargs)
        )
        if interrupt_state is not None:
            object.__setattr__(event, "_interrupt_state_override", interrupt_state)

        return await agent.hooks.invoke_callbacks_async(event)

    @staticmethod
    async def _invoke_after_tool_call_hook(
        agent: "Agent | BidiAgent",
        selected_tool: Any,
        tool_use: ToolUse,
        invocation_state: dict[str, Any],
        result: ToolResult,
        exception: Exception | None = None,
        cancel_message: str | None = None,
        duration: float | None = None,
    ) -> tuple[AfterToolCallEvent | BidiAfterToolCallEvent, list[Interrupt]]:
        """Invoke the appropriate after tool call hook based on agent type."""
        kwargs = {
            "selected_tool": selected_tool,
            "tool_use": tool_use,
            "invocation_state": invocation_state,
            "result": result,
            "exception": exception,
            "cancel_message": cancel_message,
            "duration": duration,
        }
        event = (
            AfterToolCallEvent(agent=cast("Agent", agent), **kwargs)
            if ToolExecutor._is_agent(agent)
            else BidiAfterToolCallEvent(agent=cast("BidiAgent", agent), **kwargs)
        )

        return await agent.hooks.invoke_callbacks_async(event)

    @staticmethod
    def _should_retry(
        agent: "Agent | BidiAgent",
        after_event: AfterToolCallEvent | BidiAfterToolCallEvent,
        cancel_signal: threading.Event | None = None,
    ) -> bool:
        """Return whether a hook-requested retry should run.

        Cancellation is terminal: retrying a locally cancelled tool can spin indefinitely
        while delaying the caller's requested agent stop.
        """
        if not getattr(after_event, "retry", False):
            return False
        if cast(dict[str, Any], after_event.result).get("cancelled") is True:
            return False
        resolved_cancel_signal = cancel_signal
        if resolved_cancel_signal is None and ToolExecutor._is_agent(agent):
            resolved_cancel_signal = cast("Agent", agent)._cancel_signal
        return resolved_cancel_signal is None or not resolved_cancel_signal.is_set()

    @staticmethod
    async def _stream(
        agent: "Agent | BidiAgent",
        tool_use: ToolUse,
        tool_results: list[ToolResult],
        invocation_state: dict[str, Any],
        structured_output_context: StructuredOutputContext | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[TypedEvent, None]:
        """Stream tool events.

        This method adds additional logic to the stream invocation including:

        - Tool lookup and validation
        - Before/after hook execution
        - Tracing and metrics collection
        - Error handling and recovery
        - Interrupt handling for human-in-the-loop workflows

        Args:
            agent: The agent (Agent or BidiAgent) for which the tool is being executed.
            tool_use: Metadata and inputs for the tool to be executed.
            tool_results: List of tool results from each tool execution.
            invocation_state: Context for the tool invocation.
            structured_output_context: Context for structured output management.
            **kwargs: Additional keyword arguments for future extensibility.

        Yields:
            Tool events with the last being the tool result.
        """
        logger.debug("tool_use=<%s> | streaming", tool_use)
        tool_name = tool_use["name"]
        original_tool_use_id = tool_use["toolUseId"]
        structured_output_context = structured_output_context or StructuredOutputContext()
        execution_context = _get_current_tool_execution_context()
        background_tasks = _get_background_tasks(agent)
        interrupt_state = (
            execution_context.interrupt_state
            if execution_context is not None
            else getattr(agent, "_interrupt_state", None)
        )
        cancel_signal = (
            execution_context.cancel_signal if execution_context is not None else getattr(agent, "_cancel_signal", None)
        )
        if interrupt_state is None:
            raise RuntimeError("tool execution requires interrupt state")

        tool_info = agent.tool_registry.dynamic_tools.get(tool_name)
        tool_func = tool_info if tool_info is not None else agent.tool_registry.registry.get(tool_name)
        tool_spec = tool_func.tool_spec if tool_func is not None else None

        if background_tasks is not None and execution_context is not None and execution_context.route_background:
            if execution_context.pass_id is None:
                raise RuntimeError("background task routing requires a pass ID")
            route_kind, routed = await background_tasks.route_tool_call(
                agent=agent,
                tool_use=tool_use,
                tool_instance=tool_func,
                invocation_state=invocation_state,
                pass_id=execution_context.pass_id,
                origin_span_context=execution_context.origin_span_context,
            )
            if route_kind == "result":
                result = _normalize_tool_result_id(cast(ToolResult, routed), original_tool_use_id)
                tool_results.append(result)
                yield ToolResultEvent(result)
                return
            tool_use = cast(ToolUse, {**tool_use, "input": routed})

        current_span = trace_api.get_current_span()
        if current_span and tool_spec is not None:
            current_span.set_attribute("gen_ai.tool.description", tool_spec["description"])
            input_schema = tool_spec["inputSchema"]
            if "json" in input_schema:
                current_span.set_attribute("gen_ai.tool.json_schema", serialize(input_schema["json"]))

        invocation_state.update(
            {
                "agent": agent,
                "model": agent.model,
                "messages": agent.messages,
                "system_prompt": agent.system_prompt,
                "tool_config": ToolConfig(  # for backwards compatibility
                    tools=[{"toolSpec": tool_spec} for tool_spec in _get_model_tool_specs(agent)],
                    toolChoice=cast(ToolChoice, {"auto": ToolChoiceAuto()}),
                ),
            }
        )

        # Retry loop for tool execution - hooks can set after_event.retry = True to retry
        while True:
            selected_tool = tool_func
            before_event, interrupts = await ToolExecutor._invoke_before_tool_call_hook(
                agent, tool_func, tool_use, invocation_state, interrupt_state
            )

            if interrupts:
                yield ToolInterruptEvent(tool_use, interrupts)
                return

            try:
                tool_start_time = time.monotonic()
                selected_tool = before_event.selected_tool
                hook_tool_use = before_event.tool_use
                if selected_tool is None:
                    if hook_tool_use["name"] == tool_name:
                        selected_tool = tool_func
                    else:
                        selected_tool = agent.tool_registry.dynamic_tools.get(
                            hook_tool_use["name"]
                        ) or agent.tool_registry.registry.get(hook_tool_use["name"])
                if (
                    background_tasks is not None
                    and execution_context is not None
                    and execution_context.background_task_id is not None
                ):
                    background_tasks.validate_tool_replacement(
                        original_tool_use_id=original_tool_use_id,
                        effective_tool=selected_tool,
                        tool_use=hook_tool_use,
                    )
                tool_use = cast(ToolUse, {**hook_tool_use, "toolUseId": original_tool_use_id})
                invocation_state = before_event.invocation_state

                if before_event.cancel_tool:
                    cancel_message = (
                        before_event.cancel_tool
                        if isinstance(before_event.cancel_tool, str)
                        else "tool cancelled by user"
                    )
                    yield ToolCancelEvent(tool_use, cancel_message)

                    cancel_result: ToolResult = {
                        "toolUseId": original_tool_use_id,
                        "status": "error",
                        "content": [{"text": cancel_message}],
                    }
                    after_event, _ = await ToolExecutor._invoke_after_tool_call_hook(
                        agent,
                        selected_tool,
                        tool_use,
                        invocation_state,
                        cancel_result,
                        cancel_message=cancel_message,
                    )
                    final_result = _normalize_tool_result_id(after_event.result, original_tool_use_id)
                    yield ToolResultEvent(final_result)
                    tool_results.append(final_result)
                    return

                if not selected_tool:
                    # Unknown tool: log here, but do NOT short-circuit. The middleware chain
                    # still runs with ctx.tool = None (matching TS), so middleware can observe
                    # or mock the call; the terminal produces the unknown-tool error result.
                    if tool_func == selected_tool:
                        logger.error(
                            "tool_name=<%s>, available_tools=<%s> | tool not found in registry",
                            tool_name,
                            list(agent.tool_registry.registry.keys()),
                        )
                    else:
                        logger.debug(
                            "tool_name=<%s>, tool_use_id=<%s> | a hook resulted in a non-existing tool call",
                            tool_name,
                            str(tool_use.get("toolUseId")),
                        )
                if structured_output_context.is_enabled:
                    kwargs["structured_output_context"] = structured_output_context

                # Run tool execution through the ExecuteToolStage middleware chain. The
                # terminal streams the tool and yields a plain ToolResultEvent as the last
                # (result) event; middleware can transform inputs/result, short-circuit with
                # a cached result, or gate execution behind an interrupt. A shallow copy of
                # tool_use guards its top-level keys (e.g. name, toolUseId) from accidental
                # in-place edits; its `input` can hold arbitrary, non-copyable objects (e.g.
                # the agent injected on direct tool calls) so it is shared by reference.
                # Middleware wanting an isolated tool_use should pass one via replace().
                middleware_context = ExecuteToolContext(
                    agent=agent,
                    tool=selected_tool,
                    tool_use=dict(tool_use),  # type: ignore[arg-type]
                    invocation_state=invocation_state,
                    _interrupt_state=interrupt_state,
                )

                result_event: ToolResultEvent | None = None
                async for event in agent._middleware_registry.invoke(
                    ExecuteToolStage,
                    middleware_context,
                    _make_execute_tool_terminal(kwargs),
                ):
                    # Tool-originated interrupt: a ToolInterruptEvent yielded from tool.stream()
                    # (including sub-agent interrupts propagated via _AgentAsTool). Distinct from
                    # the middleware-initiated InterruptException handled below — this one rides
                    # the event stream rather than unwinding it. Register its interrupts so
                    # _interrupt_state.resume() can locate them by id, surface the event, and
                    # short-circuit here: a halted tool has no result, so the after-hook and the
                    # result handling below are intentionally skipped.
                    if isinstance(event, ToolInterruptEvent):
                        for interrupt in event.interrupts:
                            interrupt_state.interrupts.setdefault(interrupt.id, interrupt)
                        yield event
                        return

                    # Capture the result but keep draining: middleware may yield trailing
                    # events after it, and the last ToolResultEvent wins (matching the model
                    # stage). It is re-emitted only after AfterToolCallEvent runs, since hooks
                    # may rewrite it. All non-result events flow through as they arrive.
                    if isinstance(event, ToolResultEvent):
                        result_event = event
                    else:
                        yield event

                if result_event is None:
                    raise RuntimeError(
                        "ExecuteToolStage middleware chain did not yield a ToolResultEvent. "
                        "Ensure middleware forwards events from next()."
                    )

                result = result_event.tool_result
                exception = result_event.exception

                tool_duration = time.monotonic() - tool_start_time
                after_event, _ = await ToolExecutor._invoke_after_tool_call_hook(
                    agent,
                    selected_tool,
                    tool_use,
                    invocation_state,
                    result,
                    exception=exception,
                    duration=tool_duration,
                )

                if ToolExecutor._should_retry(agent, after_event, cancel_signal):
                    logger.debug("tool_name=<%s> | retry requested, retrying tool call", tool_name)
                    continue

                final_result = _normalize_tool_result_id(after_event.result, original_tool_use_id)
                yield ToolResultEvent(final_result, exception=after_event.exception)
                tool_results.append(final_result)
                return

            except InterruptException as interrupt_exception:
                # Middleware-initiated interrupt (context.interrupt() with no response yet).
                # interrupt() is read-only, so this handler is the single place the interrupt
                # is registered before surfacing a ToolInterruptEvent to halt the agent,
                # matching how hook/tool interrupts are reported.
                interrupt_state.interrupts.setdefault(interrupt_exception.interrupt.id, interrupt_exception.interrupt)
                yield ToolInterruptEvent(tool_use, [interrupt_exception.interrupt])
                return

            except Exception as e:
                logger.exception("tool_name=<%s> | failed to process tool", tool_name)
                tool_duration = time.monotonic() - tool_start_time
                error_result: ToolResult = {
                    "toolUseId": str(tool_use.get("toolUseId")),
                    "status": "error",
                    "content": [{"text": f"Error: {str(e)}"}],
                }

                after_event, _ = await ToolExecutor._invoke_after_tool_call_hook(
                    agent, selected_tool, tool_use, invocation_state, error_result, exception=e, duration=tool_duration
                )
                if ToolExecutor._should_retry(agent, after_event, cancel_signal):
                    logger.debug("tool_name=<%s> | retry requested after exception, retrying tool call", tool_name)
                    continue
                final_result = _normalize_tool_result_id(after_event.result, original_tool_use_id)
                yield ToolResultEvent(final_result, exception=after_event.exception)
                tool_results.append(final_result)
                return

    @staticmethod
    async def _stream_with_trace(
        agent: "Agent",
        tool_use: ToolUse,
        tool_results: list[ToolResult],
        cycle_trace: Trace,
        cycle_span: Any,
        invocation_state: dict[str, Any],
        structured_output_context: StructuredOutputContext | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[TypedEvent, None]:
        """Execute tool with tracing and metrics collection.

        Args:
            agent: The agent for which the tool is being executed.
            tool_use: Metadata and inputs for the tool to be executed.
            tool_results: List of tool results from each tool execution.
            cycle_trace: Trace object for the current event loop cycle.
            cycle_span: Span object for tracing the cycle.
            invocation_state: Context for the tool invocation.
            structured_output_context: Context for structured output management.
            **kwargs: Additional keyword arguments for future extensibility.

        Yields:
            Tool events with the last being the tool result.
        """
        tool_name = tool_use["name"]
        structured_output_context = structured_output_context or StructuredOutputContext()
        execution_context = _get_current_tool_execution_context()

        tracer = get_tracer()

        tool_call_span = tracer.start_tool_call_span(
            tool_use, cycle_span, custom_trace_attributes=agent.trace_attributes
        )
        tool_trace = Trace(f"Tool: {tool_name}", parent_id=cycle_trace.id, raw_name=tool_name)
        tool_start_time = time.time()

        with trace_api.use_span(tool_call_span):
            last_event: TypedEvent | None = None
            async for event in ToolExecutor._stream(
                agent, tool_use, tool_results, invocation_state, structured_output_context, **kwargs
            ):
                last_event = event
                yield event

            if last_event is None:
                raise RuntimeError("tool execution did not yield a terminal event")

            if isinstance(last_event, ToolInterruptEvent):
                tool_duration = time.time() - tool_start_time
                if ToolExecutor._is_agent(agent):
                    metrics = (
                        execution_context.event_loop_metrics
                        if execution_context is not None and execution_context.event_loop_metrics is not None
                        else agent.event_loop_metrics
                    )
                    metrics.add_tool_usage(tool_use, tool_duration, tool_trace, False)
                cycle_trace.add_child(tool_trace)
                tracer.end_tool_call_span(tool_call_span, tool_result=None)
                return

            result_event = cast(ToolResultEvent, last_event)
            result = result_event.tool_result

            tool_success = result.get("status") == "success"
            tool_duration = time.time() - tool_start_time
            message = Message(role="user", content=[{"toolResult": result}])
            if ToolExecutor._is_agent(agent):
                metrics = (
                    execution_context.event_loop_metrics
                    if execution_context is not None and execution_context.event_loop_metrics is not None
                    else agent.event_loop_metrics
                )
                metrics.add_tool_usage(tool_use, tool_duration, tool_trace, tool_success, message)
            cycle_trace.add_child(tool_trace)

            tracer.end_tool_call_span(tool_call_span, result, error=result_event.exception)

    @abc.abstractmethod
    # pragma: no cover
    def _execute(
        self,
        agent: "Agent",
        tool_uses: list[ToolUse],
        tool_results: list[ToolResult],
        cycle_trace: Trace,
        cycle_span: Any,
        invocation_state: dict[str, Any],
        structured_output_context: "StructuredOutputContext | None" = None,
    ) -> AsyncGenerator[TypedEvent, None]:
        """Execute the given tools according to this executor's strategy.

        Args:
            agent: The agent for which tools are being executed.
            tool_uses: Metadata and inputs for the tools to be executed.
            tool_results: List of tool results from each tool execution.
            cycle_trace: Trace object for the current event loop cycle.
            cycle_span: Span object for tracing the cycle.
            invocation_state: Context for the tool invocation.
            structured_output_context: Context for structured output management.

        Yields:
            Events from the tool execution stream.
        """
        pass


def _make_execute_tool_terminal(
    extra_kwargs: dict[str, Any],
) -> "Any":
    """Build the terminal for the ExecuteToolStage middleware chain.

    The terminal streams the resolved tool and yields a plain ``ToolResultEvent`` as its
    last (result) event, matching the SDK-wide "last event is the result" convention.
    Intermediate ``ToolStreamEvent``s flow through unchanged. A tool-originated
    ``ToolInterruptEvent`` flows through as a normal event; the Output-phase adapter and
    the executor both recognize it as a control-flow signal rather than a result.

    A raw exception from ``tool.stream()`` is converted to an error ``ToolResultEvent`` here,
    inside the terminal, so ExecuteToolStage middleware always observes a result rather than a
    thrown exception (matching the TypeScript SDK). ``InterruptException`` is re-raised so a
    tool-raised interrupt still halts the agent instead of becoming an error result.

    All events are derived from ``ctx.tool_use`` (the possibly Input-transformed value the tool
    actually ran with), so the streamed, wrapped, and error events agree on identity fields
    (e.g. ``toolUseId``) even when an Input handler rewrote them.

    Args:
        extra_kwargs: Extra keyword arguments forwarded to ``tool.stream()``.

    Returns:
        An async generator function suitable as a middleware terminal.
    """

    async def terminal(ctx: ExecuteToolContext) -> AsyncGenerator[TypedEvent, None]:
        tool_use = ctx.tool_use

        # Unknown tool (not in the registry): the chain still ran so middleware could observe
        # or mock it, but with no tool to invoke the terminal yields the error result. The
        # message/exception mirror the pre-middleware unknown-tool contract.
        if ctx.tool is None:
            tool_name = tool_use["name"]
            yield ToolResultEvent(
                {
                    "toolUseId": str(tool_use.get("toolUseId")),
                    "status": "error",
                    "content": [{"text": f"Unknown tool: {tool_name}"}],
                },
                exception=Exception(f"Unknown tool: {tool_name}"),
            )
            return

        # Mirrors ToolExecutor._stream's original dispatch: built-in AgentTools yield
        # TypedEvents directly (ending in a ToolResultEvent); other tools yield raw values
        # we wrap in ToolStreamEvent, and their last raw value is the result.
        yielded_any = False
        last_raw_event: Any = None
        try:
            async for event in ctx.tool.stream(tool_use, ctx.invocation_state, **extra_kwargs):
                if isinstance(event, ToolInterruptEvent):
                    yield event
                    return

                if isinstance(event, ToolResultEvent):
                    # Re-emit so the exception decorated tools attach rides along as the result.
                    yield ToolResultEvent(event.tool_result, exception=event.exception)
                    return

                if isinstance(event, ToolStreamEvent):
                    yield event
                else:
                    yield ToolStreamEvent(tool_use, event)
                yielded_any = True
                last_raw_event = event
        except InterruptException:
            # A tool-raised interrupt must halt the agent — let it unwind rather than
            # becoming an error result (matches TS re-throwing InterruptError).
            raise
        except Exception as error:
            # Convert a raw tool failure to an error result inside the terminal so middleware
            # sees a result, not an exception. The executor's after-hook still receives the
            # exception via the ToolResultEvent below.
            logger.exception("tool_name=<%s> | tool execution failed", tool_use["name"])
            yield ToolResultEvent(
                {
                    "toolUseId": str(tool_use.get("toolUseId")),
                    "status": "error",
                    "content": [{"text": f"Error: {error}"}],
                },
                exception=error,
            )
            return

        # Non-SDK tool: no ToolResultEvent was emitted, so the last raw value is the result.
        # A tool that streamed nothing at all has no result — surface an error result rather
        # than a null one so the agent can continue (matches the pre-middleware degradation).
        if not yielded_any:
            yield ToolResultEvent(
                {
                    "toolUseId": str(tool_use.get("toolUseId")),
                    "status": "error",
                    "content": [{"text": f"Tool '{tool_use['name']}' did not return a result"}],
                }
            )
            return
        yield ToolResultEvent(cast(ToolResult, last_raw_event))

    return terminal
