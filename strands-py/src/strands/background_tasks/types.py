"""Public data contracts for background tool execution."""

from typing import Literal

from typing_extensions import NotRequired, ReadOnly, TypedDict

from ..interrupt import Interrupt
from ..types.tools import AgentTool, ToolResultContent

__all__ = [
    "BackgroundTask",
    "BackgroundTaskError",
    "BackgroundTaskResult",
    "BackgroundTasksConfig",
]


class BackgroundTasksConfig(TypedDict, total=False):
    """Configure background tool execution for an Agent.

    Attributes:
        wait_for_completion: Wait for all background tasks before an invocation returns. Defaults to ``True``.
        agentic: Tools whose execution mode is selected by the model. Defaults to ``["*"]``.
        always: Tools that always execute in the background.
        never: Tools that never execute in the background.
        max_concurrency: Maximum number of physically executing background tasks. Defaults to ``4``.
        timeout: Per-execution timeout in seconds. Defaults to infinity.
    """

    wait_for_completion: bool
    agentic: list[AgentTool | Literal["*"]]
    always: list[AgentTool | Literal["*"]]
    never: list[AgentTool | Literal["*"]]
    max_concurrency: int
    timeout: float


class BackgroundTaskResult(TypedDict):
    """Serialized result produced by a background task.

    Attributes:
        content: Tool-result content blocks.
    """

    content: ReadOnly[list[ToolResultContent]]


class BackgroundTaskError(TypedDict):
    """Failure details for a background task.

    Attributes:
        type: Failure category.
        message: Failure message.
    """

    type: ReadOnly[Literal["tool_error", "execution_error", "timeout", "recovery_error"]]
    message: ReadOnly[str]


class BackgroundTask(TypedDict):
    """Read-only snapshot of a background task.

    Attributes:
        task_id: Stable identifier for task inspection and cancellation.
        tool_use_id: Tool-use identifier from the original model request.
        tool_name: Registered name of the executing tool.
        status: Current task lifecycle status.
        created_at: ISO timestamp recorded when the task was admitted.
        updated_at: ISO timestamp recorded at the latest task state change.
        result: Serialized tool result when execution produced one.
        error: Task failure details.
        interrupts: Unanswered interrupts when the task is paused.
    """

    task_id: ReadOnly[str]
    tool_use_id: ReadOnly[str]
    tool_name: ReadOnly[str]
    status: ReadOnly[Literal["queued", "working", "paused", "completed", "failed", "cancelled"]]
    created_at: ReadOnly[str]
    updated_at: ReadOnly[str]
    result: NotRequired[ReadOnly[BackgroundTaskResult]]
    error: NotRequired[ReadOnly[BackgroundTaskError]]
    interrupts: NotRequired[ReadOnly[list[Interrupt]]]
