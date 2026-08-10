"""Tests for public background-task data contracts."""

from typing import Literal, get_type_hints

from typing_extensions import NotRequired, ReadOnly

import strands.background_tasks as background_tasks
import strands.background_tasks.types as background_task_types
from strands.background_tasks import (
    BackgroundTask,
    BackgroundTaskError,
    BackgroundTaskNotFoundError,
    BackgroundTaskResult,
    BackgroundTasks,
    BackgroundTasksConfig,
    BackgroundTasksTimeoutError,
)
from strands.types.tools import AgentTool, ToolContext


def test_package_exports_public_contract_types() -> None:
    tru_package_exports = background_tasks.__all__
    tru_types_exports = background_task_types.__all__
    exp_package_exports = [
        "BackgroundTask",
        "BackgroundTaskNotFoundError",
        "BackgroundTaskError",
        "BackgroundTaskResult",
        "BackgroundTasks",
        "BackgroundTasksConfig",
        "BackgroundTasksTimeoutError",
    ]
    exp_types_exports = [
        "BackgroundTask",
        "BackgroundTaskError",
        "BackgroundTaskResult",
        "BackgroundTasksConfig",
    ]

    assert tru_package_exports == exp_package_exports
    assert tru_types_exports == exp_types_exports
    assert background_tasks.BackgroundTasks is BackgroundTasks


def test_public_exception_contracts() -> None:
    tru_not_found = BackgroundTaskNotFoundError("task-1")
    tru_timeout = BackgroundTasksTimeoutError(1.5)

    assert isinstance(tru_not_found, LookupError)
    assert str(tru_not_found) == "Background task 'task-1' was not found"
    assert isinstance(tru_timeout, TimeoutError)
    assert tru_timeout.timeout == 1.5
    assert str(tru_timeout) == "Background Tasks wait timed out after 1.5s"


def test_background_tasks_config_shape() -> None:
    tru_required_keys = BackgroundTasksConfig.__required_keys__
    tru_optional_keys = BackgroundTasksConfig.__optional_keys__
    exp_required_keys: frozenset[str] = frozenset()
    exp_optional_keys = frozenset(
        {
            "wait_for_completion",
            "agentic",
            "always",
            "never",
            "max_concurrency",
            "timeout",
        }
    )

    assert tru_required_keys == exp_required_keys
    assert tru_optional_keys == exp_optional_keys

    tru_hints = get_type_hints(BackgroundTasksConfig, include_extras=True)
    exp_selector_type = list[AgentTool | Literal["*"]]
    assert tru_hints["agentic"] == exp_selector_type
    assert tru_hints["always"] == exp_selector_type
    assert tru_hints["never"] == exp_selector_type
    assert tru_hints["timeout"] is float


def test_tool_context_equality_ignores_cancel_signal() -> None:
    agent = object()
    tool_use = {"toolUseId": "tool-use", "name": "work", "input": {}}

    tru_left = ToolContext(tool_use=tool_use, agent=agent, invocation_state={})
    tru_right = ToolContext(tool_use=tool_use, agent=agent, invocation_state={})

    assert tru_left == tru_right
    assert tru_left.cancel_signal is not tru_right.cancel_signal


def test_background_task_shape_is_read_only() -> None:
    tru_required_keys = BackgroundTask.__required_keys__
    tru_optional_keys = BackgroundTask.__optional_keys__
    tru_readonly_keys = BackgroundTask.__readonly_keys__
    exp_required_keys = frozenset(
        {
            "task_id",
            "tool_use_id",
            "tool_name",
            "status",
            "created_at",
            "updated_at",
        }
    )
    exp_optional_keys = frozenset({"result", "error", "interrupts"})
    exp_readonly_keys = exp_required_keys | exp_optional_keys

    assert tru_required_keys == exp_required_keys
    assert tru_optional_keys == exp_optional_keys
    assert tru_readonly_keys == exp_readonly_keys

    tru_hints = get_type_hints(BackgroundTask, include_extras=True)
    exp_status_type = ReadOnly[Literal["queued", "working", "paused", "completed", "failed", "cancelled"]]
    assert tru_hints["status"] == exp_status_type
    assert tru_hints["result"] == NotRequired[ReadOnly[BackgroundTaskResult]]
    assert tru_hints["error"] == NotRequired[ReadOnly[BackgroundTaskError]]


def test_background_task_nested_shapes_are_read_only() -> None:
    tru_result_required_keys = BackgroundTaskResult.__required_keys__
    tru_result_readonly_keys = BackgroundTaskResult.__readonly_keys__
    tru_error_required_keys = BackgroundTaskError.__required_keys__
    tru_error_readonly_keys = BackgroundTaskError.__readonly_keys__

    assert tru_result_required_keys == frozenset({"content"})
    assert tru_result_readonly_keys == frozenset({"content"})
    assert tru_error_required_keys == frozenset({"type", "message"})
    assert tru_error_readonly_keys == frozenset({"type", "message"})

    tru_error_hints = get_type_hints(BackgroundTaskError, include_extras=True)
    exp_error_type = ReadOnly[Literal["tool_error", "execution_error", "timeout", "recovery_error"]]
    assert tru_error_hints["type"] == exp_error_type
