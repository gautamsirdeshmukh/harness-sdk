"""Background Tasks policy, routing, management tool, and Agent facade."""

from __future__ import annotations

import copy
import logging
import math
from typing import TYPE_CHECKING, Any, Literal, cast

from opentelemetry.trace import SpanContext

from .._async import run_async
from ..hooks import AgentInitializedEvent, HookOrder
from ..plugins import Plugin
from ..tools import tool
from ..types.tools import AgentTool, ToolContext, ToolResult, ToolSpec, ToolUse
from ._delivery import BACKGROUND_RESULT_TOOL_NAME
from ._manager import _InProcessTaskManager
from ._schema import add_background_selection, strip_background_selection
from .control import BackgroundTasks
from .errors import BackgroundTaskNotFoundError
from .types import BackgroundTasksConfig

if TYPE_CHECKING:
    from ..agent import Agent

logger = logging.getLogger(__name__)

MANAGE_BACKGROUND_TASK_TOOL_NAME = "strands_manage_background_task"
_BackgroundMode = Literal["never", "agentic", "always"]
_CONFIG_KEYS = frozenset(BackgroundTasksConfig.__annotations__)


class _BackgroundTasks(Plugin):
    """Internal plugin connecting Background Tasks policy to an Agent."""

    def __init__(self, config: BackgroundTasksConfig | None = None) -> None:
        raw_config = dict(config or {})
        _validate_config(raw_config)
        config_copy = cast(BackgroundTasksConfig, raw_config)
        policy = _resolve_policy(config_copy)
        if "agentic" in config_copy:
            config_copy["agentic"] = list(config_copy["agentic"])
        if "always" in config_copy:
            config_copy["always"] = list(config_copy["always"])
        if "never" in config_copy:
            config_copy["never"] = list(config_copy["never"])
        self._config = config_copy
        self._policy = policy
        self._warned_wildcard_tools: set[str] = set()
        self._agent: Agent | None = None
        self._manager: _InProcessTaskManager | None = None
        super().__init__()

    @property
    def name(self) -> str:
        """Return the plugin registry identifier."""
        return "strands:background-tasks"

    def init_agent(self, agent: Agent) -> None:
        """Attach one manager and register lifecycle validation."""
        self._agent = agent
        manager = _InProcessTaskManager(agent, self._config)
        self._manager = manager
        manager.register_hooks()
        agent.add_hook(self._on_initialized, AgentInitializedEvent, order=HookOrder.SDK_LAST)

    def _on_initialized(self, event: AgentInitializedEvent) -> None:
        self._validate_current_tools(event.agent)
        self._require_manager(event.agent).initialize()

    def app_state_loaded(self) -> None:
        """Reload manager state after the Agent replaces its snapshot state."""
        self._require_manager(self._agent).app_state_loaded()

    def shutdown(self) -> None:
        """Cancel active work and stop this Agent's manager."""
        manager = self._require_manager(self._agent)
        run_async(manager.shutdown)

    @tool(
        name=MANAGE_BACKGROUND_TASK_TOOL_NAME,
        description=(
            "Inspect or cancel a background task. Completed results are delivered automatically; "
            "do not poll with this tool."
        ),
        context=True,
    )
    async def _manage_background_task(
        self,
        mode: Literal["get", "cancel"],
        task_id: str,
        tool_context: ToolContext,
    ) -> Any:
        """Inspect or cancel a background task.

        Args:
            mode: Whether to inspect or cancel the task.
            task_id: Task ID returned by the dispatch acknowledgement.
            tool_context: Framework-provided execution context.
        """
        manager = self._require_manager(tool_context.agent)
        try:
            if mode == "get":
                task = await manager.get_task(task_id)
                if task is None:
                    raise BackgroundTaskNotFoundError(task_id)
                return copy.deepcopy(task)
            task = await manager.cancel_task(task_id)
            return {"task_id": task["task_id"], "status": task["status"]}
        except BackgroundTaskNotFoundError:
            raise
        except Exception as error:
            logger.warning("error=<%s> | background task management failed", error)
            raise RuntimeError("Background task management failed") from error

    @property
    def control(self) -> BackgroundTasks:
        """Return the Agent-facing programmatic control facade."""
        return BackgroundTasks(self._require_manager(self._agent))

    def _require_manager(self, agent: Agent | None) -> _InProcessTaskManager:
        if agent is None or agent is not self._agent or self._manager is None:
            raise RuntimeError("Background Tasks is not initialized for this Agent")
        return self._manager

    def _validate_reserved_tool_names(self, tools: list[AgentTool]) -> None:
        management_tool = next(
            tool_instance for tool_instance in self.tools if tool_instance.tool_name == MANAGE_BACKGROUND_TASK_TOOL_NAME
        )
        for tool_instance in tools:
            if tool_instance.tool_name == MANAGE_BACKGROUND_TASK_TOOL_NAME and tool_instance is not management_tool:
                raise TypeError(
                    f"Tool name '{MANAGE_BACKGROUND_TASK_TOOL_NAME}' is reserved for Background Tasks management"
                )
            if tool_instance.tool_name == BACKGROUND_RESULT_TOOL_NAME:
                raise TypeError(f"Tool name '{BACKGROUND_RESULT_TOOL_NAME}' is reserved for Background Tasks delivery")

    def transform_tool_specs(self, tools: list[AgentTool], tool_specs: list[ToolSpec]) -> list[ToolSpec]:
        """Copy and transform model-facing specs for current eligible tools."""
        self._validate_reserved_tool_names(tools)
        tools_by_name = {tool_instance.tool_name: tool_instance for tool_instance in tools}
        transformed_specs: list[ToolSpec] = []
        for tool_spec in tool_specs:
            tool_name = tool_spec["name"]
            tool_instance = tools_by_name.get(tool_name)
            policy = self._resolve_policy(tool_name)
            if tool_instance is None or policy is None:
                transformed_specs.append(tool_spec)
                continue
            mode, exact = policy
            if _is_framework_tool(tool_instance):
                if exact and mode != "never":
                    raise TypeError(f"Tool '{tool_name}' is framework-owned and cannot run in the background")
                transformed_specs.append(tool_spec)
                continue
            if mode != "agentic":
                transformed_specs.append(tool_spec)
                continue

            transformed = add_background_selection(tool_spec)
            if transformed["compatible"]:
                transformed_specs.append(transformed["tool_spec"])
            elif exact:
                raise TypeError(f"Tool '{tool_name}' cannot use agentic background selection: {transformed['reason']}")
            else:
                self._warn_wildcard_skip(tool_name, transformed["reason"])
                transformed_specs.append(tool_spec)
        return transformed_specs

    async def route_tool_call(
        self,
        *,
        agent: Agent,
        tool_use: ToolUse,
        tool_instance: AgentTool | None,
        invocation_state: dict[str, Any],
        pass_id: str,
        origin_span_context: SpanContext | None,
    ) -> tuple[Literal["execute"], Any] | tuple[Literal["result"], ToolResult]:
        """Choose foreground execution or durable background admission."""
        if tool_use["name"] == BACKGROUND_RESULT_TOOL_NAME:
            return (
                "result",
                _routing_error(
                    tool_use,
                    "This tool is reserved for Strands. Do not call it. "
                    "Background task results are delivered automatically.",
                ),
            )

        policy = self._resolve_policy(tool_use["name"])
        if policy is None or policy[0] == "never":
            return ("execute", tool_use["input"])
        mode, exact = policy

        if tool_instance is not None and _is_framework_tool(tool_instance):
            if exact:
                return (
                    "result",
                    _routing_error(
                        tool_use,
                        f"Tool '{tool_use['name']}' is framework-owned and cannot run in the background",
                    ),
                )
            return ("execute", tool_use["input"])

        routed_input = tool_use["input"]
        selected = mode == "always"
        if mode == "agentic":
            compatibility = add_background_selection(tool_instance.tool_spec) if tool_instance is not None else None
            if compatibility is not None and not compatibility["compatible"]:
                if exact:
                    return (
                        "result",
                        _routing_error(
                            tool_use,
                            f"Tool '{tool_use['name']}' cannot run in the background: {compatibility['reason']}",
                        ),
                    )
                self._warn_wildcard_skip(tool_use["name"], compatibility["reason"])
                return ("execute", tool_use["input"])
            if compatibility is None or not compatibility["compatible"]:
                return ("execute", tool_use["input"])
            try:
                stripped = strip_background_selection(tool_use["input"])
            except TypeError as error:
                return ("result", _routing_error(tool_use, str(error)))
            routed_input = stripped["input"]
            selected = stripped.get("selected") is True

        if not selected:
            return ("execute", routed_input)
        if tool_instance is None:
            return (
                "result",
                _routing_error(
                    tool_use,
                    f"Tool '{tool_use['name']}' is not registered and cannot be admitted",
                ),
            )

        try:
            task = await self._require_manager(agent).submit_tool_call(
                tool_name=tool_instance.tool_name,
                original_tool_use_id=tool_use["toolUseId"],
                tool_input=routed_input,
                invocation_state=invocation_state,
                pass_id=pass_id,
                origin_span_context=origin_span_context,
            )
        except Exception as error:
            logger.warning("error=<%s> | background task admission failed", error)
            return ("result", _routing_error(tool_use, "Background task admission failed"))

        result: ToolResult = {
            "toolUseId": tool_use["toolUseId"],
            "status": "success",
            "content": [{"text": _render_dispatch_acknowledgement(task["task_id"], task["tool_name"])}],
        }
        return ("result", result)

    def validate_tool_replacement(
        self,
        *,
        original_tool_use_id: str,
        effective_tool: AgentTool | None,
        tool_use: ToolUse,
    ) -> None:
        """Enforce background identity, policy, and framework ownership after hooks."""
        if tool_use["name"] == BACKGROUND_RESULT_TOOL_NAME or (
            effective_tool is not None and effective_tool.tool_name == BACKGROUND_RESULT_TOOL_NAME
        ):
            raise RuntimeError("The Background Tasks delivery tool name is reserved")
        if tool_use["toolUseId"] != original_tool_use_id:
            raise RuntimeError("Background task hooks cannot change the original tool-use ID")
        if effective_tool is None:
            raise RuntimeError(f"Background task tool '{tool_use['name']}' is not registered")
        if _is_framework_tool(effective_tool):
            raise RuntimeError(f"Framework-owned tool '{effective_tool.tool_name}' cannot run in the background")
        policy = self._resolve_policy(effective_tool.tool_name)
        if policy is None or policy[0] == "never":
            raise RuntimeError(f"Tool '{effective_tool.tool_name}' is forbidden by background task policy")
        if policy[0] == "agentic":
            compatibility = add_background_selection(effective_tool.tool_spec)
            if not compatibility["compatible"]:
                raise RuntimeError(
                    f"Tool '{effective_tool.tool_name}' cannot run in the background: {compatibility['reason']}"
                )

    def _validate_current_tools(self, agent: Agent) -> None:
        tools = _all_tools(agent)
        self._validate_reserved_tool_names(tools)
        for tool_instance in tools:
            policy = self._resolve_policy(tool_instance.tool_name)
            if policy is None or not policy[1]:
                continue
            mode = policy[0]
            if _is_framework_tool(tool_instance) and mode != "never":
                raise TypeError(f"Tool '{tool_instance.tool_name}' is framework-owned and cannot run in the background")
            if mode == "agentic":
                compatibility = add_background_selection(tool_instance.tool_spec)
                if not compatibility["compatible"]:
                    raise TypeError(
                        f"Tool '{tool_instance.tool_name}' cannot use agentic background selection: "
                        f"{compatibility['reason']}"
                    )

    def _resolve_policy(self, tool_name: str) -> tuple[_BackgroundMode, bool] | None:
        if tool_name in self._policy:
            return self._policy[tool_name], True
        if "*" in self._policy:
            return self._policy["*"], False
        return None

    def _warn_wildcard_skip(self, tool_name: str, reason: str) -> None:
        if tool_name in self._warned_wildcard_tools:
            return
        self._warned_wildcard_tools.add(tool_name)
        logger.warning(
            "tool_name=<%s>, reason=<%s> | wildcard background policy skipped incompatible tool",
            tool_name,
            reason,
        )


def _resolve_policy(config: BackgroundTasksConfig) -> dict[str, _BackgroundMode]:
    policy: dict[str, _BackgroundMode] = {}
    configured_selectors: dict[str, AgentTool] = {}
    explicit_wildcard = "*" in config.get("always", []) or "*" in config.get("never", [])
    assignments: list[tuple[_BackgroundMode, object]] = [
        ("agentic", config.get("agentic", [] if explicit_wildcard else ["*"])),
        ("always", config.get("always", [])),
        ("never", config.get("never", [])),
    ]
    for mode, selectors_value in assignments:
        if not isinstance(selectors_value, list):
            raise TypeError(f"Background Tasks {mode} must be a list")
        for selector in selectors_value:
            tool_name = _selector_name(selector, mode)
            if selector != "*":
                tool_instance = cast(AgentTool, selector)
                existing_tool = configured_selectors.get(tool_name)
                if existing_tool is not None and existing_tool is not tool_instance:
                    raise TypeError(f"Background Tasks policy contains multiple Tool instances named '{tool_name}'")
                configured_selectors[tool_name] = tool_instance
            existing_mode = policy.get(tool_name)
            if existing_mode is not None and existing_mode != mode:
                raise TypeError(f"Tool '{tool_name}' cannot be configured as both '{existing_mode}' and '{mode}'")
            policy[tool_name] = mode
    return policy


def _validate_config(config: dict[str, object]) -> None:
    unknown_keys = sorted(config.keys() - _CONFIG_KEYS)
    if unknown_keys:
        raise TypeError(f"Unknown Background Tasks configuration key(s): {', '.join(unknown_keys)}")

    wait_for_completion = config.get("wait_for_completion")
    if "wait_for_completion" in config and not isinstance(wait_for_completion, bool):
        raise TypeError("Background Tasks wait_for_completion must be a boolean")

    max_concurrency = config.get("max_concurrency")
    if "max_concurrency" in config and (
        isinstance(max_concurrency, bool) or not isinstance(max_concurrency, int) or max_concurrency <= 0
    ):
        raise TypeError("Background Tasks max_concurrency must be a positive integer")

    timeout = config.get("timeout")
    if "timeout" in config and (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or math.isnan(float(timeout))
        or float(timeout) <= 0
    ):
        raise TypeError("Background Tasks timeout must be a positive number")


def _selector_name(selector: object, mode: _BackgroundMode) -> str:
    if selector == "*":
        return "*"
    if not isinstance(selector, AgentTool):
        raise TypeError(f"Background Tasks {mode} entries must be AgentTool instances or '*'")
    if not selector.tool_name:
        raise TypeError(f"Background Tasks {mode} tool name must be a non-empty string")
    if selector.tool_name == BACKGROUND_RESULT_TOOL_NAME:
        raise TypeError(f"Tool name '{BACKGROUND_RESULT_TOOL_NAME}' is reserved for Background Tasks delivery")
    return selector.tool_name


def _is_framework_tool(tool_instance: AgentTool) -> bool:
    return tool_instance.tool_name == MANAGE_BACKGROUND_TASK_TOOL_NAME or tool_instance.tool_type == "structured_output"


def _all_tools(agent: Agent) -> list[AgentTool]:
    tools = list(agent.tool_registry.registry.values())
    tools.extend(
        tool
        for tool_name, tool in agent.tool_registry.dynamic_tools.items()
        if tool_name not in agent.tool_registry.registry
    )
    return tools


def _routing_error(tool_use: ToolUse, message: str) -> ToolResult:
    return {
        "toolUseId": tool_use["toolUseId"],
        "status": "error",
        "content": [{"text": message}],
    }


def _render_dispatch_acknowledgement(task_id: str, tool_name: str) -> str:
    return "\n".join(
        [
            "Background task dispatched.",
            "",
            f"Task ID: {task_id}",
            f"Tool: {tool_name}",
            "Status: queued",
            "",
            "The task is running in the background. Continue without waiting or polling.",
            "The final result will be delivered automatically when the task completes.",
        ]
    )
