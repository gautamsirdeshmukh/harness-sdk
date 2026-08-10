"""Loop-owned bounded execution for persistent background tasks."""

from __future__ import annotations

import asyncio
import copy
import math
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from typing import Any, Generic, TypeVar, cast
from uuid import uuid4

from ._engine_types import (
    BackgroundTaskAdmittedEvent,
    BackgroundTaskCancelledEvent,
    BackgroundTaskEngineEvent,
    BackgroundTaskEventCallback,
    BackgroundTaskExecutionContext,
    BackgroundTaskExecutionFinishedEvent,
    BackgroundTaskExecutionOutcome,
    BackgroundTaskExecutionStartedEvent,
    BackgroundTaskExecutor,
    BackgroundTaskResumeCallback,
    BackgroundTaskShutdownMode,
    BackgroundTaskStatus,
    BackgroundTaskUpdatedCallback,
    StoredEngineTask,
)
from .errors import BackgroundTaskNotFoundError

DescriptorT = TypeVar("DescriptorT")
ResultT = TypeVar("ResultT")
StateT = TypeVar("StateT")

_TERMINAL_STATUSES = frozenset[BackgroundTaskStatus]({"completed", "failed", "cancelled"})
_VALID_STATUSES = frozenset[BackgroundTaskStatus]({"queued", "working", "paused", "completed", "failed", "cancelled"})
_RECOVERY_MESSAGE = "Background task execution was interrupted while restoring persisted state"
_MAX_SAFE_INTEGER = (1 << 53) - 1


@dataclass
class _ActiveExecution:
    """Physical execution state retained until the callback exits."""

    cancel_signal: threading.Event
    timeout_handle: asyncio.TimerHandle | None = None


def is_engine_terminal_status(status: BackgroundTaskStatus) -> bool:
    """Return whether execution has permanently stopped."""
    return status in _TERMINAL_STATUSES


def validate_stored_engine_task(value: object) -> None:
    """Validate one persisted engine task.

    Args:
        value: Candidate persisted record.

    Raises:
        ValueError: If the record is malformed.
    """
    if not isinstance(value, dict):
        raise ValueError("Stored background task must be an object")

    _require_string(value.get("task_id"), "task.task_id")
    if "descriptor" not in value:
        raise ValueError("task.descriptor is required")

    status = value.get("status")
    if not isinstance(status, str) or status not in _VALID_STATUSES:
        raise ValueError(f"task.status '{status}' is invalid")

    attempt_count = value.get("attempt_count")
    if (
        isinstance(attempt_count, bool)
        or not isinstance(attempt_count, int)
        or attempt_count < 0
        or attempt_count > _MAX_SAFE_INTEGER
    ):
        raise ValueError("task.attempt_count must be a non-negative integer")

    _require_timestamp(value.get("created_at"), "task.created_at")
    _require_timestamp(value.get("updated_at"), "task.updated_at")

    for key in ("idempotency_key", "attempt_id", "cancellation_reason"):
        if key in value:
            _require_string(value[key], f"task.{key}")

    if "failure" in value:
        _validate_failure(value["failure"])

    if status == "paused" and "state" not in value:
        raise ValueError("task.state is required while paused")
    if status == "completed" and "result" not in value:
        raise ValueError("task.result is required while completed")
    if status == "failed" and "failure" not in value:
        raise ValueError("task.failure is required while failed")
    if status == "cancelled" and "cancellation_reason" not in value:
        raise ValueError("task.cancellation_reason is required while cancelled")
    if status != "cancelled" and "cancellation_reason" in value:
        raise ValueError("task.cancellation_reason is only valid while cancelled")


class BackgroundTaskEngine(Generic[DescriptorT, ResultT, StateT]):
    """Execute generic persistent tasks with bounded physical concurrency."""

    def __init__(
        self,
        *,
        max_concurrency: int,
        execute: BackgroundTaskExecutor[DescriptorT, ResultT, StateT],
        timeout: float = math.inf,
        on_task_updated: BackgroundTaskUpdatedCallback[DescriptorT, ResultT, StateT] | None = None,
        on_event: BackgroundTaskEventCallback[DescriptorT, ResultT, StateT] | None = None,
    ) -> None:
        """Initialize an engine without starting work.

        Args:
            max_concurrency: Maximum number of physically executing callbacks.
            execute: Async callback that executes one task attempt.
            timeout: Active execution timeout in seconds, or infinity to disable it.
            on_task_updated: Synchronous persistence callback for each record update.
            on_event: Best-effort synchronous lifecycle observer.

        Raises:
            TypeError: If a concurrency or timeout value is invalid.
        """
        _require_positive_integer(max_concurrency, "max_concurrency")
        _require_positive_timeout(timeout, "timeout", allow_infinity=True)

        self._max_concurrency = max_concurrency
        self._timeout = float(timeout)
        self._execute_callback = execute
        self._on_task_updated = on_task_updated
        self._on_event = on_event

        self._tasks: dict[str, StoredEngineTask[DescriptorT, ResultT, StateT]] = {}
        self._queue: dict[str, None] = {}
        self._active_executions: dict[str, _ActiveExecution] = {}
        self._pending_removal: set[str] = set()
        self._waiters: dict[str, set[asyncio.Future[StoredEngineTask[DescriptorT, ResultT, StateT]]]] = {}
        self._idle_waiters: set[asyncio.Future[None]] = set()

        self._loop: asyncio.AbstractEventLoop | None = None
        self._initialized = False
        self._accepting = True
        self._stopping = False
        self._shutdown_task: asyncio.Task[None] | None = None

    def initialize(
        self,
        restored: Sequence[StoredEngineTask[DescriptorT, ResultT, StateT]] = (),
    ) -> None:
        """Load persisted records and restart queued work.

        Args:
            restored: Previously persisted engine records.

        Raises:
            RuntimeError: If called outside the owning event loop.
            ValueError: If any restored record is malformed.
        """
        if self._initialized:
            return

        loop_was_unbound = self._loop is None
        self._bind_loop()
        self._tasks.clear()
        self._pending_removal.clear()
        try:
            for restored_task in restored:
                task = _snapshot(restored_task)
                validate_stored_engine_task(task)
                self._tasks[task["task_id"]] = task

            self._initialized = True
            for task in tuple(self._tasks.values()):
                if task["status"] != "working":
                    continue

                def recover(record: StoredEngineTask[DescriptorT, ResultT, StateT]) -> bool:
                    record.pop("attempt_id", None)
                    record["status"] = "failed"
                    record["failure"] = {
                        "type": "recovery_error",
                        "message": _RECOVERY_MESSAGE,
                    }
                    return True

                self._update_task(task["task_id"], recover)

            for task in tuple(self._tasks.values()):
                if task["status"] == "queued":
                    self._enqueue(task["task_id"])
            self._pump()
        except BaseException:
            self._initialized = False
            self._tasks.clear()
            self._queue.clear()
            self._pending_removal.clear()
            if loop_was_unbound:
                self._loop = None
            raise

    def submit(
        self,
        descriptor: DescriptorT,
        idempotency_key: str | None = None,
    ) -> StoredEngineTask[DescriptorT, ResultT, StateT]:
        """Admit work or return the visible task with the same idempotency key.

        Args:
            descriptor: Integration-owned execution descriptor.
            idempotency_key: Optional key used to deduplicate admission.

        Returns:
            A defensive copy of the admitted or existing task.

        Raises:
            RuntimeError: If the engine is uninitialized or admission is closed.
            ValueError: If ``idempotency_key`` is empty.
        """
        self._assert_initialized()
        if not self._accepting:
            raise RuntimeError("Background task admission is closed")
        if idempotency_key is not None:
            _require_string(idempotency_key, "idempotency_key")
            for task in self._tasks.values():
                if task["task_id"] not in self._pending_removal and task.get("idempotency_key") == idempotency_key:
                    return _snapshot(task)

        now = _utc_now_iso()
        task_id = str(uuid4())
        stored = cast(
            StoredEngineTask[DescriptorT, ResultT, StateT],
            {
                "task_id": task_id,
                "descriptor": copy.deepcopy(descriptor),
                "status": "queued",
                "attempt_count": 0,
                "created_at": now,
                "updated_at": now,
            },
        )
        if idempotency_key is not None:
            stored["idempotency_key"] = idempotency_key

        validate_stored_engine_task(stored)
        self._persist_update(stored)
        self._tasks[task_id] = stored
        event: BackgroundTaskAdmittedEvent[DescriptorT, ResultT, StateT] = {
            "type": "admitted",
            "task": _snapshot(stored),
        }
        self._emit(event)
        self._enqueue(task_id)
        return _snapshot(stored)

    def get(self, task_id: str) -> StoredEngineTask[DescriptorT, ResultT, StateT] | None:
        """Return a defensive copy of a visible task."""
        self._assert_initialized()
        if task_id in self._pending_removal:
            return None
        task = self._tasks.get(task_id)
        return _snapshot(task) if task is not None else None

    def list(self) -> list[StoredEngineTask[DescriptorT, ResultT, StateT]]:
        """Return defensive copies of all visible tasks in admission order."""
        self._assert_initialized()
        return [_snapshot(task) for task in self._tasks.values() if task["task_id"] not in self._pending_removal]

    def remove(self, task_id: str) -> None:
        """Remove a terminal task, hiding it while physical execution settles.

        Args:
            task_id: Task to remove.

        Raises:
            BackgroundTaskNotFoundError: If the task is not visible.
            RuntimeError: If the task is not terminal.
        """
        self._assert_initialized()
        task = self._require_visible_task(task_id)
        if not is_engine_terminal_status(task["status"]):
            raise RuntimeError(f"Background task '{task_id}' cannot be removed before reaching a terminal status")
        if task_id in self._active_executions:
            self._pending_removal.add(task_id)
            return
        self._tasks.pop(task_id)

    def cancel(
        self,
        task_id: str,
        reason: str,
    ) -> StoredEngineTask[DescriptorT, ResultT, StateT]:
        """Logically cancel a task and signal any active execution.

        Args:
            task_id: Task to cancel.
            reason: Persisted cancellation reason.

        Returns:
            The resulting task record.

        Raises:
            BackgroundTaskNotFoundError: If the task is not visible.
            ValueError: If ``reason`` is empty.
        """
        self._assert_initialized()
        _require_string(reason, "reason")
        self._require_visible_task(task_id)

        def cancel_record(record: StoredEngineTask[DescriptorT, ResultT, StateT]) -> bool:
            if is_engine_terminal_status(record["status"]):
                return False
            record["cancellation_reason"] = reason
            record["status"] = "cancelled"
            record.pop("attempt_id", None)
            record.pop("failure", None)
            record.pop("result", None)
            return True

        updated = self._update_task(task_id, cancel_record)
        task = updated if updated is not None else self._require_task(task_id)
        if updated is not None:
            self._queue.pop(task_id, None)
            active_execution = self._active_executions.get(task_id)
            if active_execution is not None:
                if active_execution.timeout_handle is not None:
                    active_execution.timeout_handle.cancel()
                    active_execution.timeout_handle = None
                active_execution.cancel_signal.set()
            event: BackgroundTaskCancelledEvent[DescriptorT, ResultT, StateT] = {
                "type": "cancelled",
                "task": _snapshot(task),
            }
            self._emit(event)

        self._signal_idle()
        self._pump()
        return task

    async def wait(self, task_id: str) -> StoredEngineTask[DescriptorT, ResultT, StateT]:
        """Wait until a task pauses or reaches a terminal status.

        Cancelling this coroutine stops only the observation.

        Args:
            task_id: Task to observe.

        Returns:
            The settled task record.

        Raises:
            BackgroundTaskNotFoundError: If the task is not visible.
        """
        self._assert_initialized()
        current = self._require_visible_task(task_id)
        if _is_settled(current):
            return current

        loop = self._require_loop()
        waiter: asyncio.Future[StoredEngineTask[DescriptorT, ResultT, StateT]] = loop.create_future()
        waiters = self._waiters.setdefault(task_id, set())
        waiters.add(waiter)
        try:
            return await waiter
        finally:
            waiters.discard(waiter)
            if not waiters:
                self._waiters.pop(task_id, None)

    async def wait_for_idle(self) -> None:
        """Wait until no task is queued or physically executing.

        Cancelling this coroutine stops only the observation.
        """
        self._assert_initialized()
        await self._wait_for_idle()

    def resume(
        self,
        task_id: str,
        update: BackgroundTaskResumeCallback[StateT],
    ) -> StoredEngineTask[DescriptorT, ResultT, StateT]:
        """Update paused state and optionally queue the same logical attempt.

        Args:
            task_id: Paused task to update.
            update: Callback returning replacement state and whether execution is ready.

        Returns:
            The updated task record.

        Raises:
            BackgroundTaskNotFoundError: If the task is not visible.
            RuntimeError: If execution is closed or the task is not paused.
            ValueError: If the resume callback returns an invalid value.
        """
        self._assert_initialized()
        self._require_visible_task(task_id)

        def resume_record(record: StoredEngineTask[DescriptorT, ResultT, StateT]) -> bool:
            if not self._accepting:
                raise RuntimeError("Background task execution is closed")
            if record["status"] != "paused":
                raise RuntimeError(
                    f"Background task '{task_id}' cannot transition: status is '{record['status']}', not 'paused'"
                )
            if "state" not in record:
                raise RuntimeError(f"Background task '{task_id}' cannot transition: paused state is missing")

            resumed = update(copy.deepcopy(record["state"]))
            _validate_resume(resumed)
            record["state"] = copy.deepcopy(resumed["state"])
            if resumed["ready"]:
                record["status"] = "queued"
            return True

        task = self._update_task(task_id, resume_record)
        if task is None:
            raise BackgroundTaskNotFoundError(task_id)
        if task["status"] == "queued":
            self._enqueue(task_id)
        return task

    async def shutdown(self, *, mode: BackgroundTaskShutdownMode, timeout: float) -> None:
        """Stop admission and wait for physical execution to settle.

        Args:
            mode: ``"drain"`` to finish queued work or ``"cancel"`` to cancel it.
            timeout: Maximum wait in seconds.

        Raises:
            TimeoutError: If physical execution does not settle before ``timeout``.
            TypeError: If ``mode`` or ``timeout`` is invalid.
        """
        self._bind_loop()
        if mode not in ("drain", "cancel"):
            raise TypeError(f"shutdown mode must be 'drain' or 'cancel', got {mode!r}")
        _require_positive_timeout(timeout, "shutdown timeout", allow_infinity=False)
        self._accepting = False

        if self._shutdown_task is None:
            loop = self._require_loop()
            self._shutdown_task = loop.create_task(self._shutdown_engine(mode=mode, timeout=float(timeout)))
        shutdown_task = self._shutdown_task
        try:
            await asyncio.shield(shutdown_task)
        except Exception:
            if self._shutdown_task is shutdown_task:
                self._shutdown_task = None
            raise

    def _enqueue(self, task_id: str) -> None:
        if self._stopping or task_id in self._queue or task_id in self._active_executions:
            return
        self._queue[task_id] = None
        self._pump()

    def _pump(self) -> None:
        if self._stopping or not self._initialized:
            return
        loop = self._require_loop()
        while len(self._active_executions) < self._max_concurrency and self._queue:
            task_id = next(iter(self._queue))
            self._queue.pop(task_id)
            active_execution = _ActiveExecution(cancel_signal=threading.Event())
            self._active_executions[task_id] = active_execution
            execution_task = loop.create_task(self._run_execution(task_id, active_execution))
            execution_task.add_done_callback(partial(self._execution_done, task_id, active_execution))

    async def _run_execution(self, task_id: str, active_execution: _ActiveExecution) -> None:
        before = self._require_task(task_id)
        if before["status"] != "queued":
            return

        resumed = "attempt_id" in before
        attempt = before["attempt_count"] + (0 if resumed else 1)
        attempt_id = before.get("attempt_id", str(uuid4()))
        execution_id = str(uuid4())
        started_at = self._require_loop().time()

        def start(record: StoredEngineTask[DescriptorT, ResultT, StateT]) -> bool:
            if record["status"] != "queued" or "cancellation_reason" in record:
                return False
            record["status"] = "working"
            if not resumed:
                record["attempt_count"] = attempt
            record["attempt_id"] = attempt_id
            record.pop("failure", None)
            return True

        working = self._update_task(task_id, start)
        if working is None:
            return

        started_event: BackgroundTaskExecutionStartedEvent[DescriptorT, ResultT, StateT] = {
            "type": "execution_started",
            "task": _snapshot(working),
            "resumed": resumed,
            "queue_duration": max(0.0, (_utc_now() - _parse_timestamp(before["updated_at"])).total_seconds() * 1000),
        }
        self._emit(started_event)

        if math.isfinite(self._timeout):
            active_execution.timeout_handle = self._require_loop().call_later(
                self._timeout,
                self._handle_timeout,
                task_id,
                attempt_id,
                active_execution,
            )

        try:
            outcome: BackgroundTaskExecutionOutcome[ResultT, StateT]
            try:
                outcome = await self._execute_callback(
                    BackgroundTaskExecutionContext(
                        task_id=task_id,
                        descriptor=copy.deepcopy(working["descriptor"]),
                        state=copy.deepcopy(working.get("state")),
                        attempt=attempt,
                        attempt_id=attempt_id,
                        execution_id=execution_id,
                        cancel_signal=active_execution.cancel_signal,
                    )
                )
                _validate_execution_outcome(outcome)
                outcome = copy.deepcopy(outcome)
            except asyncio.CancelledError as error:
                outcome = {
                    "status": "failed",
                    "failure": {
                        "type": "execution_error",
                        "message": _exception_message(error),
                    },
                }
            except Exception as error:
                outcome = {
                    "status": "failed",
                    "failure": {
                        "type": "execution_error",
                        "message": _exception_message(error),
                    },
                }
            self._finish_outcome(task_id, attempt_id, outcome)
        finally:
            latest = self._require_task(task_id)
            finished_event: BackgroundTaskExecutionFinishedEvent[DescriptorT, ResultT, StateT] = {
                "type": "execution_finished",
                "task": latest,
                "duration": max(0.0, (self._require_loop().time() - started_at) * 1000),
            }
            self._emit(finished_event)

    def _execution_done(
        self,
        task_id: str,
        active_execution: _ActiveExecution,
        execution_task: asyncio.Task[None],
    ) -> None:
        error: BaseException | None = None
        try:
            execution_task.result()
        except BaseException as execution_error:
            error = execution_error

        if active_execution.timeout_handle is not None:
            active_execution.timeout_handle.cancel()
            active_execution.timeout_handle = None
        if self._active_executions.get(task_id) is active_execution:
            self._active_executions.pop(task_id)

        if task_id in self._pending_removal:
            self._pending_removal.remove(task_id)
            self._tasks.pop(task_id, None)
        else:
            current = self._tasks.get(task_id)
            if current is not None and current["status"] == "queued":
                self._enqueue(task_id)

        self._signal_idle()
        self._pump()
        if error is not None:
            self._reject_waiters(task_id, error)

    def _finish_outcome(
        self,
        task_id: str,
        attempt_id: str,
        outcome: BackgroundTaskExecutionOutcome[ResultT, StateT],
    ) -> None:
        if outcome["status"] == "paused":

            def pause(record: StoredEngineTask[DescriptorT, ResultT, StateT]) -> bool:
                if record["status"] != "working" or record.get("attempt_id") != attempt_id:
                    return False
                record["status"] = "paused"
                record["state"] = copy.deepcopy(outcome["state"])
                return True

            self._update_task(task_id, pause)
            return

        def finish(record: StoredEngineTask[DescriptorT, ResultT, StateT]) -> bool:
            if record["status"] != "working" or record.get("attempt_id") != attempt_id:
                return False
            if outcome["status"] == "failed":
                if "state" in outcome:
                    record["state"] = copy.deepcopy(outcome["state"])
                record["status"] = "failed"
                record.pop("attempt_id", None)
                record["failure"] = copy.deepcopy(outcome["failure"])
                if "result" in outcome:
                    record["result"] = copy.deepcopy(outcome["result"])
                else:
                    record.pop("result", None)
                return True

            record["status"] = "completed"
            record.pop("attempt_id", None)
            record["result"] = copy.deepcopy(outcome["result"])
            if "state" in outcome:
                record["state"] = copy.deepcopy(outcome["state"])
            else:
                record.pop("state", None)
            record.pop("failure", None)
            return True

        self._update_task(task_id, finish)

    def _handle_timeout(
        self,
        task_id: str,
        attempt_id: str,
        active_execution: _ActiveExecution,
    ) -> None:
        active_execution.timeout_handle = None
        try:
            self._timeout_task(task_id, attempt_id, active_execution)
        except BaseException as error:
            self._reject_waiters(task_id, error)

    def _timeout_task(
        self,
        task_id: str,
        attempt_id: str,
        active_execution: _ActiveExecution,
    ) -> None:
        reason = f"Timed out after {_format_seconds(self._timeout)}s"

        def timeout(record: StoredEngineTask[DescriptorT, ResultT, StateT]) -> bool:
            if record["status"] != "working" or record.get("attempt_id") != attempt_id:
                return False
            record["status"] = "failed"
            record.pop("attempt_id", None)
            record["failure"] = {
                "type": "timeout",
                "message": reason,
            }
            return True

        task = self._update_task(task_id, timeout)
        if task is not None:
            active_execution.cancel_signal.set()

    async def _shutdown_engine(self, *, mode: BackgroundTaskShutdownMode, timeout: float) -> None:
        self._accepting = False
        if not self._initialized:
            return

        if mode == "cancel":
            self._stopping = True
            for task in self.list():
                if not is_engine_terminal_status(task["status"]):
                    self.cancel(task["task_id"], "Coordinator shutdown")

        try:
            await asyncio.wait_for(self._wait_for_idle(), timeout=timeout)
        except asyncio.TimeoutError as error:
            raise TimeoutError(
                f"Background Task Engine shutdown timed out after {_format_seconds(timeout)}s"
            ) from error

    def _update_task(
        self,
        task_id: str,
        update: Callable[[StoredEngineTask[DescriptorT, ResultT, StateT]], bool],
    ) -> StoredEngineTask[DescriptorT, ResultT, StateT] | None:
        current = self._tasks.get(task_id)
        if current is None:
            raise BackgroundTaskNotFoundError(task_id)
        next_task = _snapshot(current)
        if not update(next_task):
            return None
        next_task["updated_at"] = _utc_now_iso()
        validate_stored_engine_task(next_task)
        self._persist_update(next_task)
        self._tasks[task_id] = next_task
        result = _snapshot(next_task)
        self._notify(result)
        return result

    def _persist_update(self, task: StoredEngineTask[DescriptorT, ResultT, StateT]) -> None:
        if self._on_task_updated is not None:
            self._on_task_updated(_snapshot(task))

    def _require_task(self, task_id: str) -> StoredEngineTask[DescriptorT, ResultT, StateT]:
        task = self._tasks.get(task_id)
        if task is None:
            raise BackgroundTaskNotFoundError(task_id)
        return _snapshot(task)

    def _require_visible_task(self, task_id: str) -> StoredEngineTask[DescriptorT, ResultT, StateT]:
        if task_id in self._pending_removal:
            raise BackgroundTaskNotFoundError(task_id)
        return self._require_task(task_id)

    def _notify(self, task: StoredEngineTask[DescriptorT, ResultT, StateT]) -> None:
        if not _is_settled(task):
            return
        waiters = self._waiters.pop(task["task_id"], set())
        for waiter in waiters:
            if not waiter.done():
                waiter.set_result(_snapshot(task))

    def _reject_waiters(self, task_id: str, error: BaseException) -> None:
        waiters = self._waiters.pop(task_id, set())
        for waiter in waiters:
            if not waiter.done():
                waiter.set_exception(error)

    def _emit(self, event: BackgroundTaskEngineEvent[DescriptorT, ResultT, StateT]) -> None:
        if self._on_event is None:
            return
        try:
            self._on_event(copy.deepcopy(event))
        except Exception:
            pass

    def _signal_idle(self) -> None:
        waiters = tuple(self._idle_waiters)
        self._idle_waiters.clear()
        for waiter in waiters:
            if not waiter.done():
                waiter.set_result(None)

    async def _wait_for_idle(self) -> None:
        loop = self._require_loop()
        while self._queue or self._active_executions:
            waiter: asyncio.Future[None] = loop.create_future()
            self._idle_waiters.add(waiter)
            try:
                await waiter
            finally:
                self._idle_waiters.discard(waiter)

    def _assert_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError("Background Task Engine is not initialized")
        self._bind_loop()

    def _bind_loop(self) -> asyncio.AbstractEventLoop:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError as error:
            raise RuntimeError("Background Task Engine operations require a running event loop") from error
        if self._loop is None:
            self._loop = loop
        elif self._loop is not loop:
            raise RuntimeError("Background Task Engine operations must execute on the owning event loop")
        return loop

    def _require_loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is None:
            raise RuntimeError("Background Task Engine does not have an owning event loop")
        return self._loop


def _snapshot(
    task: StoredEngineTask[DescriptorT, ResultT, StateT],
) -> StoredEngineTask[DescriptorT, ResultT, StateT]:
    return copy.deepcopy(task)


def _is_settled(task: StoredEngineTask[Any, Any, Any]) -> bool:
    return task["status"] == "paused" or is_engine_terminal_status(task["status"])


def _validate_failure(value: object) -> None:
    if not isinstance(value, dict):
        raise ValueError("task.failure must be an object")
    _require_string(value.get("type"), "task.failure.type")
    _require_string(value.get("message"), "task.failure.message")


def _validate_execution_outcome(value: object) -> None:
    if not isinstance(value, dict):
        raise ValueError("Background task execution outcome must be an object")
    status = value.get("status")
    if status == "completed":
        if "result" not in value:
            raise ValueError("execution outcome result is required while completed")
        return
    if status == "paused":
        if "state" not in value:
            raise ValueError("execution outcome state is required while paused")
        return
    if status == "failed":
        if "failure" not in value:
            raise ValueError("execution outcome failure is required while failed")
        _validate_failure(value["failure"])
        return
    raise ValueError(f"execution outcome status '{status}' is invalid")


def _validate_resume(value: object) -> None:
    if not isinstance(value, dict) or "state" not in value or "ready" not in value:
        raise ValueError("Background task resume update must contain state and ready")
    if not isinstance(value["ready"], bool):
        raise ValueError("Background task resume ready must be a boolean")


def _require_string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{path} must be a non-empty string")
    return value


def _require_timestamp(value: object, path: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{path} must be a non-empty string")
    try:
        _parse_timestamp(value)
    except ValueError as error:
        raise ValueError(f"{path} must be an ISO-8601 timestamp") from error


def _parse_timestamp(value: str) -> datetime:
    normalized = f"{value[:-1]}+00:00" if value.endswith(("Z", "z")) else value
    timestamp = datetime.fromisoformat(normalized)
    return timestamp.replace(tzinfo=timezone.utc) if timestamp.tzinfo is None else timestamp


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_now_iso() -> str:
    return _utc_now().isoformat().replace("+00:00", "Z")


def _require_positive_integer(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0 or value > _MAX_SAFE_INTEGER:
        raise TypeError(f"{name} must be a positive integer, got {value!r}")


def _require_positive_timeout(value: object, name: str, *, allow_infinity: bool) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a positive number, got {value!r}")
    numeric = float(value)
    if math.isnan(numeric) or numeric <= 0 or (math.isinf(numeric) and not allow_infinity):
        raise TypeError(f"{name} must be a positive finite number, got {value!r}")


def _exception_message(error: BaseException) -> str:
    return str(error) or type(error).__name__


def _format_seconds(value: float) -> str:
    return f"{value:g}"
