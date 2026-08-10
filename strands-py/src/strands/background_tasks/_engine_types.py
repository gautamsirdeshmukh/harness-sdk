"""Internal types for bounded background task execution."""

from __future__ import annotations

import threading
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Generic, Literal, Protocol, TypeAlias, TypeVar

from typing_extensions import NotRequired, TypedDict

DescriptorT = TypeVar("DescriptorT")
ResultT = TypeVar("ResultT")
StateT = TypeVar("StateT")

BackgroundTaskStatus: TypeAlias = Literal["queued", "working", "paused", "completed", "failed", "cancelled"]
BackgroundTaskShutdownMode: TypeAlias = Literal["drain", "cancel"]


class BackgroundTaskFailure(TypedDict):
    """A classified background task failure."""

    type: str
    message: str


class StoredEngineTask(TypedDict, Generic[DescriptorT, ResultT, StateT]):
    """Durable state for one engine task."""

    task_id: str
    descriptor: DescriptorT
    status: BackgroundTaskStatus
    attempt_count: int
    created_at: str
    updated_at: str
    idempotency_key: NotRequired[str]
    attempt_id: NotRequired[str]
    cancellation_reason: NotRequired[str]
    state: NotRequired[StateT]
    result: NotRequired[ResultT]
    failure: NotRequired[BackgroundTaskFailure]


@dataclass(frozen=True)
class BackgroundTaskExecutionContext(Generic[DescriptorT, StateT]):
    """Runtime context for one physical execution."""

    task_id: str
    descriptor: DescriptorT
    state: StateT | None
    attempt: int
    attempt_id: str
    execution_id: str
    cancel_signal: threading.Event


class CompletedExecutionOutcome(TypedDict, Generic[ResultT, StateT]):
    """A successful execution outcome."""

    status: Literal["completed"]
    result: ResultT
    state: NotRequired[StateT]


class PausedExecutionOutcome(TypedDict, Generic[StateT]):
    """An execution outcome waiting for external state."""

    status: Literal["paused"]
    state: StateT


class FailedExecutionOutcome(TypedDict, Generic[ResultT, StateT]):
    """An unsuccessful execution outcome."""

    status: Literal["failed"]
    failure: BackgroundTaskFailure
    result: NotRequired[ResultT]
    state: NotRequired[StateT]


BackgroundTaskExecutionOutcome: TypeAlias = (
    CompletedExecutionOutcome[ResultT, StateT]
    | PausedExecutionOutcome[StateT]
    | FailedExecutionOutcome[ResultT, StateT]
)


class BackgroundTaskResume(TypedDict, Generic[StateT]):
    """State returned while attempting to resume a paused task."""

    state: StateT
    ready: bool


class BackgroundTaskAdmittedEvent(TypedDict, Generic[DescriptorT, ResultT, StateT]):
    """Event emitted after a task is durably admitted."""

    type: Literal["admitted"]
    task: StoredEngineTask[DescriptorT, ResultT, StateT]


class BackgroundTaskExecutionStartedEvent(TypedDict, Generic[DescriptorT, ResultT, StateT]):
    """Event emitted when physical execution starts."""

    type: Literal["execution_started"]
    task: StoredEngineTask[DescriptorT, ResultT, StateT]
    resumed: bool
    queue_duration: float


class BackgroundTaskExecutionFinishedEvent(TypedDict, Generic[DescriptorT, ResultT, StateT]):
    """Event emitted when physical execution finishes."""

    type: Literal["execution_finished"]
    task: StoredEngineTask[DescriptorT, ResultT, StateT]
    duration: float


class BackgroundTaskCancelledEvent(TypedDict, Generic[DescriptorT, ResultT, StateT]):
    """Event emitted when a task is logically cancelled."""

    type: Literal["cancelled"]
    task: StoredEngineTask[DescriptorT, ResultT, StateT]


BackgroundTaskEngineEvent: TypeAlias = (
    BackgroundTaskAdmittedEvent[DescriptorT, ResultT, StateT]
    | BackgroundTaskExecutionStartedEvent[DescriptorT, ResultT, StateT]
    | BackgroundTaskExecutionFinishedEvent[DescriptorT, ResultT, StateT]
    | BackgroundTaskCancelledEvent[DescriptorT, ResultT, StateT]
)


class BackgroundTaskExecutor(Protocol[DescriptorT, ResultT, StateT]):
    """Async execution callback used by the engine."""

    def __call__(
        self,
        context: BackgroundTaskExecutionContext[DescriptorT, StateT],
    ) -> Awaitable[BackgroundTaskExecutionOutcome[ResultT, StateT]]:
        """Execute one physical task attempt."""
        ...


BackgroundTaskUpdatedCallback: TypeAlias = Callable[[StoredEngineTask[DescriptorT, ResultT, StateT]], None]
BackgroundTaskEventCallback: TypeAlias = Callable[
    [BackgroundTaskEngineEvent[DescriptorT, ResultT, StateT]],
    None,
]
BackgroundTaskResumeCallback: TypeAlias = Callable[[StateT], BackgroundTaskResume[StateT]]
