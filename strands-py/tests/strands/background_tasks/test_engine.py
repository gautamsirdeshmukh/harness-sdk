from __future__ import annotations

import asyncio
import copy
import math
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from datetime import datetime
from typing import TypeAlias, cast
from unittest.mock import ANY
from uuid import UUID

import pytest
import pytest_asyncio
from typing_extensions import TypedDict

from strands.background_tasks._engine import (
    BackgroundTaskEngine,
    validate_stored_engine_task,
)
from strands.background_tasks._engine_types import (
    BackgroundTaskEngineEvent,
    BackgroundTaskExecutionContext,
    BackgroundTaskExecutionOutcome,
    BackgroundTaskStatus,
    StoredEngineTask,
)
from strands.background_tasks.errors import BackgroundTaskNotFoundError


class _TestDescriptor(TypedDict):
    value: str


class _TestResult(TypedDict):
    value: str


class _TestState(TypedDict):
    phase: str


TestContext: TypeAlias = BackgroundTaskExecutionContext[_TestDescriptor, _TestState]
TestOutcome: TypeAlias = BackgroundTaskExecutionOutcome[_TestResult, _TestState]
TestTask: TypeAlias = StoredEngineTask[_TestDescriptor, _TestResult, _TestState]
TestEvent: TypeAlias = BackgroundTaskEngineEvent[_TestDescriptor, _TestResult, _TestState]
TestExecute: TypeAlias = Callable[[TestContext], Awaitable[TestOutcome]]
TestEngine: TypeAlias = BackgroundTaskEngine[_TestDescriptor, _TestResult, _TestState]


class _EngineFactory:
    def __init__(self) -> None:
        self.engines: list[TestEngine] = []

    def __call__(
        self,
        execute: TestExecute,
        *,
        max_concurrency: int = 2,
        timeout: float = 1,
        on_task_updated: Callable[[TestTask], None] | None = None,
        on_event: Callable[[TestEvent], None] | None = None,
    ) -> TestEngine:
        engine = BackgroundTaskEngine[_TestDescriptor, _TestResult, _TestState](
            max_concurrency=max_concurrency,
            timeout=timeout,
            execute=execute,
            on_task_updated=on_task_updated,
            on_event=on_event,
        )
        self.engines.append(engine)
        return engine


@pytest_asyncio.fixture
async def engine_factory() -> AsyncIterator[_EngineFactory]:
    factory = _EngineFactory()
    try:
        yield factory
    finally:
        await asyncio.gather(
            *(engine.shutdown(mode="cancel", timeout=1) for engine in factory.engines),
            return_exceptions=True,
        )


def _initialize(engine: TestEngine, restored: Sequence[TestTask] = ()) -> TestEngine:
    engine.initialize(restored)
    return engine


async def _wait_for_status(engine: TestEngine, task_id: str, status: BackgroundTaskStatus) -> TestTask:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + 1
    while loop.time() < deadline:
        task = engine.get(task_id)
        if task is not None and task["status"] == status:
            return task
        await asyncio.sleep(0)
    raise RuntimeError(f"Task '{task_id}' did not reach '{status}'")


async def _abortable(context: TestContext) -> TestOutcome:
    while not context.cancel_signal.is_set():
        await asyncio.sleep(0)
    raise RuntimeError("execution cancelled")


def _stored_task(
    task_id: str,
    *,
    status: BackgroundTaskStatus = "queued",
    attempt_count: int = 0,
) -> TestTask:
    timestamp = "2026-08-10T12:00:00Z"
    return cast(
        TestTask,
        {
            "task_id": task_id,
            "descriptor": {"value": task_id},
            "status": status,
            "attempt_count": attempt_count,
            "created_at": timestamp,
            "updated_at": timestamp,
        },
    )


@pytest.mark.asyncio
async def test_executes_reports_updates_lists_and_deduplicates_work(engine_factory: _EngineFactory) -> None:
    statuses: list[BackgroundTaskStatus] = []
    events: list[str] = []

    async def execute(context: TestContext) -> TestOutcome:
        return {
            "status": "completed",
            "result": {"value": context.descriptor["value"].upper()},
        }

    def on_task_updated(task: TestTask) -> None:
        statuses.append(task["status"])
        task["descriptor"]["value"] = "update mutation"
        if "result" in task:
            task["result"]["value"] = "update mutation"

    def on_event(event: TestEvent) -> None:
        events.append(event["type"])
        event["task"]["descriptor"]["value"] = "event mutation"
        raise RuntimeError("observer failure")

    engine = _initialize(
        engine_factory(
            execute,
            on_task_updated=on_task_updated,
            on_event=on_event,
        )
    )
    descriptor: _TestDescriptor = {"value": "hello"}
    admitted = engine.submit(descriptor, idempotency_key="work-1")
    duplicate = engine.submit({"value": "ignored"}, idempotency_key="work-1")
    descriptor["value"] = "caller mutation"
    admitted["descriptor"]["value"] = "return mutation"

    tru_duplicate_id = duplicate["task_id"]
    exp_duplicate_id = admitted["task_id"]
    assert tru_duplicate_id == exp_duplicate_id

    tru_result = await engine.wait(admitted["task_id"])
    exp_result: TestTask = {
        "task_id": admitted["task_id"],
        "idempotency_key": "work-1",
        "descriptor": {"value": "hello"},
        "status": "completed",
        "attempt_count": 1,
        "result": {"value": "HELLO"},
        "created_at": ANY,
        "updated_at": ANY,
    }
    assert tru_result == exp_result

    tru_tasks = engine.list()
    exp_tasks = [exp_result]
    assert tru_tasks == exp_tasks

    tru_statuses = statuses
    exp_statuses = ["queued", "working", "completed"]
    assert tru_statuses == exp_statuses

    tru_events = events
    exp_events = ["admitted", "execution_started", "execution_finished"]
    assert tru_events == exp_events

    assert UUID(admitted["task_id"]).version == 4
    for timestamp_key in ("created_at", "updated_at"):
        parsed = datetime.fromisoformat(tru_result[timestamp_key].replace("Z", "+00:00"))
        assert parsed.utcoffset() is not None

    external = engine.get(admitted["task_id"])
    assert external is not None
    external["descriptor"]["value"] = "changed"
    external["result"]["value"] = "changed"
    tru_stored = engine.get(admitted["task_id"])
    assert tru_stored is not None
    exp_stored_result = {"value": "HELLO"}
    assert tru_stored["result"] == exp_stored_result
    exp_stored_descriptor = {"value": "hello"}
    assert tru_stored["descriptor"] == exp_stored_descriptor


@pytest.mark.asyncio
async def test_bounds_physical_execution_concurrency(engine_factory: _EngineFactory) -> None:
    loop = asyncio.get_running_loop()
    releases: list[asyncio.Future[TestOutcome]] = [loop.create_future() for _ in range(3)]
    active = 0
    maximum = 0

    async def execute(context: TestContext) -> TestOutcome:
        nonlocal active, maximum
        active += 1
        maximum = max(maximum, active)
        try:
            return await releases[int(context.descriptor["value"])]
        finally:
            active -= 1

    engine = _initialize(engine_factory(execute, max_concurrency=2))
    tasks = [engine.submit({"value": str(index)}) for index in range(3)]
    await _wait_for_status(engine, tasks[0]["task_id"], "working")
    await _wait_for_status(engine, tasks[1]["task_id"], "working")

    tru_queued_status = engine.get(tasks[2]["task_id"])
    assert tru_queued_status is not None
    exp_queued_status = "queued"
    assert tru_queued_status["status"] == exp_queued_status

    releases[0].set_result({"status": "completed", "result": {"value": "0"}})
    await _wait_for_status(engine, tasks[2]["task_id"], "working")
    releases[1].set_result({"status": "completed", "result": {"value": "1"}})
    releases[2].set_result({"status": "completed", "result": {"value": "2"}})
    await asyncio.gather(*(engine.wait(task["task_id"]) for task in tasks))

    tru_maximum = maximum
    exp_maximum = 2
    assert tru_maximum == exp_maximum


@pytest.mark.asyncio
async def test_wait_for_idle_observation_can_stop_without_cancelling_work(engine_factory: _EngineFactory) -> None:
    finish: asyncio.Future[TestOutcome] = asyncio.get_running_loop().create_future()

    async def execute(_context: TestContext) -> TestOutcome:
        return await finish

    engine = _initialize(engine_factory(execute))
    admitted = engine.submit({"value": "work"})
    await _wait_for_status(engine, admitted["task_id"], "working")
    waiting = asyncio.create_task(engine.wait_for_idle())
    await asyncio.sleep(0)

    waiting.cancel()

    with pytest.raises(asyncio.CancelledError):
        await waiting
    tru_status = engine.get(admitted["task_id"])
    assert tru_status is not None
    exp_status = "working"
    assert tru_status["status"] == exp_status

    finish.set_result({"status": "completed", "result": {"value": "done"}})
    await engine.wait_for_idle()


@pytest.mark.asyncio
async def test_pauses_persists_state_and_resumes_same_logical_attempt(engine_factory: _EngineFactory) -> None:
    contexts: list[TestContext] = []

    async def execute(context: TestContext) -> TestOutcome:
        contexts.append(context)
        if context.state is None:
            return {"status": "paused", "state": {"phase": "waiting"}}
        return {"status": "completed", "result": {"value": context.state["phase"]}}

    engine = _initialize(engine_factory(execute))
    admitted = engine.submit({"value": "work"})
    tru_paused = await engine.wait(admitted["task_id"])
    exp_paused_status = "paused"
    assert tru_paused["status"] == exp_paused_status
    exp_paused_state = {"phase": "waiting"}
    assert tru_paused["state"] == exp_paused_state

    not_ready = engine.resume(
        admitted["task_id"],
        lambda state: {"state": {"phase": f"{state['phase']}-updated"}, "ready": False},
    )
    tru_not_ready_status = not_ready["status"]
    exp_not_ready_status = "paused"
    assert tru_not_ready_status == exp_not_ready_status
    await asyncio.sleep(0)
    assert len(contexts) == 1

    engine.resume(admitted["task_id"], lambda state: {"state": state, "ready": True})
    tru_completed = await engine.wait(admitted["task_id"])
    exp_completed_status = "completed"
    assert tru_completed["status"] == exp_completed_status
    exp_completed_result = {"value": "waiting-updated"}
    assert tru_completed["result"] == exp_completed_result
    exp_attempt_count = 1
    assert tru_completed["attempt_count"] == exp_attempt_count

    tru_attempt_ids = [context.attempt_id for context in contexts]
    assert tru_attempt_ids[0] == tru_attempt_ids[1]
    tru_execution_ids = [context.execution_id for context in contexts]
    assert tru_execution_ids[0] != tru_execution_ids[1]
    assert contexts[0].cancel_signal is not contexts[1].cancel_signal


@pytest.mark.asyncio
async def test_cancels_running_work_signals_execution_and_wakes_waiters(engine_factory: _EngineFactory) -> None:
    contexts: list[TestContext] = []
    event_types: list[str] = []

    async def execute(context: TestContext) -> TestOutcome:
        contexts.append(context)
        return await _abortable(context)

    engine = _initialize(
        engine_factory(
            execute,
            on_event=lambda event: event_types.append(event["type"]),
        )
    )
    admitted = engine.submit({"value": "work"})
    await _wait_for_status(engine, admitted["task_id"], "working")
    waiting = asyncio.create_task(engine.wait(admitted["task_id"]))
    await asyncio.sleep(0)

    cancelled = engine.cancel(admitted["task_id"], "Stop work")

    tru_waited = await waiting
    exp_waited = cancelled
    assert tru_waited == exp_waited
    exp_status = "cancelled"
    assert tru_waited["status"] == exp_status
    exp_reason = "Stop work"
    assert tru_waited["cancellation_reason"] == exp_reason
    assert contexts[0].cancel_signal.is_set()

    duplicate_cancel = engine.cancel(admitted["task_id"], "Different reason")
    assert duplicate_cancel == cancelled
    await engine.wait_for_idle()
    exp_cancel_events = 1
    assert event_types.count("cancelled") == exp_cancel_events


@pytest.mark.asyncio
async def test_removes_delivered_cancellation_after_active_execution_settles(
    engine_factory: _EngineFactory,
) -> None:
    finish: asyncio.Future[TestOutcome] = asyncio.get_running_loop().create_future()
    events: list[str] = []

    async def execute(_context: TestContext) -> TestOutcome:
        return await finish

    engine = _initialize(
        engine_factory(
            execute,
            on_event=lambda event: events.append(event["type"]),
        )
    )
    admitted = engine.submit({"value": "work"}, idempotency_key="delivered-cancellation")
    await _wait_for_status(engine, admitted["task_id"], "working")

    engine.cancel(admitted["task_id"], "Stop work")
    engine.remove(admitted["task_id"])

    assert engine.get(admitted["task_id"]) is None
    tru_tasks = engine.list()
    exp_tasks: list[TestTask] = []
    assert tru_tasks == exp_tasks
    with pytest.raises(BackgroundTaskNotFoundError, match="was not found"):
        engine.cancel(admitted["task_id"], "Again")

    finish.set_result({"status": "completed", "result": {"value": "late"}})
    await engine.shutdown(mode="drain", timeout=1)
    assert "execution_finished" in events
    assert engine.get(admitted["task_id"]) is None


@pytest.mark.asyncio
async def test_times_out_work_and_records_classified_execution_failures(engine_factory: _EngineFactory) -> None:
    timeout_contexts: list[TestContext] = []

    async def timeout_execute(context: TestContext) -> TestOutcome:
        timeout_contexts.append(context)
        return await _abortable(context)

    timeout_engine = _initialize(engine_factory(timeout_execute, timeout=0.01))
    timed = timeout_engine.submit({"value": "timeout"})
    tru_timed = await timeout_engine.wait(timed["task_id"])
    exp_timed_failure = {"type": "timeout", "message": "Timed out after 0.01s"}
    assert tru_timed["status"] == "failed"
    assert tru_timed["failure"] == exp_timed_failure
    assert timeout_contexts[0].cancel_signal.is_set()

    async def raise_execution_error(_context: TestContext) -> TestOutcome:
        raise TypeError("Execution exploded")

    execution_error_engine = _initialize(engine_factory(raise_execution_error))
    execution_error = execution_error_engine.submit({"value": "throw"})
    tru_execution_error = await execution_error_engine.wait(execution_error["task_id"])
    exp_execution_failure = {"type": "execution_error", "message": "Execution exploded"}
    assert tru_execution_error["status"] == "failed"
    assert tru_execution_error["failure"] == exp_execution_failure

    async def return_tool_error(_context: TestContext) -> TestOutcome:
        return {
            "status": "failed",
            "failure": {"type": "tool_error", "message": "Tool failed"},
            "result": {"value": "tool detail"},
        }

    failure_engine = _initialize(engine_factory(return_tool_error))
    failed = failure_engine.submit({"value": "work"})
    tru_failed = await failure_engine.wait(failed["task_id"])
    exp_tool_failure = {"type": "tool_error", "message": "Tool failed"}
    assert tru_failed["failure"] == exp_tool_failure
    exp_failure_result = {"value": "tool detail"}
    assert tru_failed["result"] == exp_failure_result


@pytest.mark.asyncio
async def test_timeout_retains_capacity_until_noncooperative_work_exits(engine_factory: _EngineFactory) -> None:
    release_hung: asyncio.Future[TestOutcome] = asyncio.get_running_loop().create_future()
    active_executions = 0
    maximum_executions = 0
    hung_contexts: list[TestContext] = []

    async def execute(context: TestContext) -> TestOutcome:
        nonlocal active_executions, maximum_executions
        active_executions += 1
        maximum_executions = max(maximum_executions, active_executions)
        try:
            if context.descriptor["value"] == "hang":
                hung_contexts.append(context)
                return await release_hung
            return {"status": "completed", "result": {"value": "next"}}
        finally:
            active_executions -= 1

    engine = _initialize(
        engine_factory(
            execute,
            max_concurrency=1,
            timeout=0.01,
        )
    )
    hung = engine.submit({"value": "hang"})
    next_task = engine.submit({"value": "next"})
    try:
        tru_hung = await engine.wait(hung["task_id"])
        assert tru_hung["status"] == "failed"
        assert tru_hung["failure"]["type"] == "timeout"
        assert hung_contexts[0].cancel_signal.is_set()

        tru_next_before_release = engine.get(next_task["task_id"])
        assert tru_next_before_release is not None
        exp_next_before_release = "queued"
        assert tru_next_before_release["status"] == exp_next_before_release
        exp_maximum_before_release = 1
        assert maximum_executions == exp_maximum_before_release
    finally:
        release_hung.set_result({"status": "completed", "result": {"value": "late"}})

    tru_next = await engine.wait(next_task["task_id"])
    exp_next_status = "completed"
    assert tru_next["status"] == exp_next_status
    exp_next_result = {"value": "next"}
    assert tru_next["result"] == exp_next_result
    tru_hung_after_late_result = engine.get(hung["task_id"])
    assert tru_hung_after_late_result is not None
    assert tru_hung_after_late_result["status"] == "failed"
    assert tru_hung_after_late_result["failure"]["type"] == "timeout"


@pytest.mark.asyncio
async def test_infinity_disables_execution_timeout(engine_factory: _EngineFactory) -> None:
    finish: asyncio.Future[TestOutcome] = asyncio.get_running_loop().create_future()
    contexts: list[TestContext] = []

    async def execute(context: TestContext) -> TestOutcome:
        contexts.append(context)
        return await finish

    engine = _initialize(engine_factory(execute, timeout=math.inf))
    task = engine.submit({"value": "work"})
    await _wait_for_status(engine, task["task_id"], "working")
    await asyncio.sleep(0.02)

    tru_status = engine.get(task["task_id"])
    assert tru_status is not None
    exp_status = "working"
    assert tru_status["status"] == exp_status
    assert not contexts[0].cancel_signal.is_set()

    finish.set_result({"status": "completed", "result": {"value": "done"}})
    tru_completed = await engine.wait(task["task_id"])
    exp_result = {"value": "done"}
    assert tru_completed["result"] == exp_result


@pytest.mark.asyncio
async def test_recovers_terminal_outcomes_without_executing_again(engine_factory: _EngineFactory) -> None:
    records: dict[str, TestTask] = {}
    executions = 0

    async def first_execute(_context: TestContext) -> TestOutcome:
        nonlocal executions
        executions += 1
        return {"status": "completed", "result": {"value": "done"}}

    first = _initialize(
        engine_factory(
            first_execute,
            on_task_updated=lambda task: records.__setitem__(task["task_id"], task),
        )
    )
    admitted = first.submit({"value": "work"})
    await first.wait(admitted["task_id"])
    await first.shutdown(mode="drain", timeout=1)

    async def second_execute(_context: TestContext) -> TestOutcome:
        nonlocal executions
        executions += 1
        return {"status": "completed", "result": {"value": "duplicate"}}

    second = _initialize(engine_factory(second_execute), list(records.values()))
    tru_restored = second.get(admitted["task_id"])
    assert tru_restored is not None
    exp_restored_status = "completed"
    assert tru_restored["status"] == exp_restored_status
    exp_restored_result = {"value": "done"}
    assert tru_restored["result"] == exp_restored_result
    exp_executions = 1
    assert executions == exp_executions


@pytest.mark.asyncio
async def test_recovers_working_restarts_queued_and_resumes_paused_attempt(
    engine_factory: _EngineFactory,
) -> None:
    contexts: list[TestContext] = []

    async def execute(context: TestContext) -> TestOutcome:
        contexts.append(context)
        return {"status": "completed", "result": {"value": context.descriptor["value"]}}

    working = _stored_task("working", status="working", attempt_count=1)
    working["attempt_id"] = "working-attempt"
    queued = _stored_task("queued")
    paused = _stored_task("paused", status="paused", attempt_count=1)
    paused["attempt_id"] = "paused-attempt"
    paused["state"] = {"phase": "waiting"}

    engine = _initialize(engine_factory(execute), [working, queued, paused])
    queued["descriptor"]["value"] = "caller mutation"
    paused["state"]["phase"] = "caller mutation"

    tru_working = engine.get("working")
    assert tru_working is not None
    exp_working_status = "failed"
    assert tru_working["status"] == exp_working_status
    exp_recovery_failure = {
        "type": "recovery_error",
        "message": "Background task execution was interrupted while restoring persisted state",
    }
    assert tru_working["failure"] == exp_recovery_failure
    assert "attempt_id" not in tru_working

    tru_queued = await engine.wait("queued")
    exp_queued_status = "completed"
    assert tru_queued["status"] == exp_queued_status
    exp_queued_result = {"value": "queued"}
    assert tru_queued["result"] == exp_queued_result
    exp_queued_attempt_count = 1
    assert tru_queued["attempt_count"] == exp_queued_attempt_count

    tru_paused = engine.get("paused")
    assert tru_paused is not None
    exp_paused_status = "paused"
    assert tru_paused["status"] == exp_paused_status
    exp_paused_state = {"phase": "waiting"}
    assert tru_paused["state"] == exp_paused_state
    engine.resume("paused", lambda state: {"state": state, "ready": True})
    tru_resumed = await engine.wait("paused")
    exp_resumed_status = "completed"
    assert tru_resumed["status"] == exp_resumed_status
    exp_resumed_attempt_count = 1
    assert tru_resumed["attempt_count"] == exp_resumed_attempt_count

    queued_context = next(context for context in contexts if context.task_id == "queued")
    resumed_context = next(context for context in contexts if context.task_id == "paused")
    exp_queued_context_attempt = 1
    assert queued_context.attempt == exp_queued_context_attempt
    exp_resumed_context_attempt = 1
    assert resumed_context.attempt == exp_resumed_context_attempt
    exp_resumed_attempt_id = "paused-attempt"
    assert resumed_context.attempt_id == exp_resumed_attempt_id


@pytest.mark.asyncio
async def test_drains_or_cancels_cleanly_during_shutdown(engine_factory: _EngineFactory) -> None:
    finish: asyncio.Future[TestOutcome] = asyncio.get_running_loop().create_future()

    async def drain_execute(_context: TestContext) -> TestOutcome:
        return await finish

    draining = _initialize(engine_factory(drain_execute))
    task = draining.submit({"value": "drain"})
    shutdown = asyncio.create_task(draining.shutdown(mode="drain", timeout=1))
    await asyncio.sleep(0)
    with pytest.raises(RuntimeError, match="admission is closed"):
        draining.submit({"value": "rejected"})
    finish.set_result({"status": "completed", "result": {"value": "done"}})
    await shutdown
    tru_drained = draining.get(task["task_id"])
    assert tru_drained is not None
    exp_drained_status = "completed"
    assert tru_drained["status"] == exp_drained_status

    cancelling = _initialize(engine_factory(_abortable))
    cancelled = cancelling.submit({"value": "cancel"})
    await _wait_for_status(cancelling, cancelled["task_id"], "working")
    await cancelling.shutdown(mode="cancel", timeout=1)
    tru_cancelled = cancelling.get(cancelled["task_id"])
    assert tru_cancelled is not None
    exp_cancelled_status = "cancelled"
    assert tru_cancelled["status"] == exp_cancelled_status


@pytest.mark.asyncio
async def test_shutdown_timeout_can_be_retried_after_noncooperative_work_exits(
    engine_factory: _EngineFactory,
) -> None:
    finish: asyncio.Future[TestOutcome] = asyncio.get_running_loop().create_future()

    async def execute(_context: TestContext) -> TestOutcome:
        return await finish

    engine = _initialize(engine_factory(execute))
    task = engine.submit({"value": "work"})
    await _wait_for_status(engine, task["task_id"], "working")

    with pytest.raises(TimeoutError, match="shutdown timed out"):
        await engine.shutdown(mode="cancel", timeout=0.01)
    tru_cancelled = engine.get(task["task_id"])
    assert tru_cancelled is not None
    exp_cancelled_status = "cancelled"
    assert tru_cancelled["status"] == exp_cancelled_status

    finish.set_result({"status": "completed", "result": {"value": "late"}})
    await engine.shutdown(mode="cancel", timeout=1)
    tru_after_late_result = engine.get(task["task_id"])
    assert tru_after_late_result is not None
    assert tru_after_late_result["status"] == exp_cancelled_status


@pytest.mark.asyncio
async def test_keeps_paused_work_stopped_after_drain_shutdown(engine_factory: _EngineFactory) -> None:
    executions = 0

    async def execute(context: TestContext) -> TestOutcome:
        nonlocal executions
        executions += 1
        if context.state is None:
            return {"status": "paused", "state": {"phase": "waiting"}}
        return {"status": "completed", "result": {"value": "resumed"}}

    engine = _initialize(engine_factory(execute))
    admitted = engine.submit({"value": "work"})
    await engine.wait(admitted["task_id"])
    await engine.shutdown(mode="drain", timeout=1)

    with pytest.raises(RuntimeError, match="execution is closed"):
        engine.resume(admitted["task_id"], lambda state: {"state": state, "ready": True})
    tru_paused = engine.get(admitted["task_id"])
    assert tru_paused is not None
    exp_paused_status = "paused"
    assert tru_paused["status"] == exp_paused_status
    exp_executions = 1
    assert executions == exp_executions


@pytest.mark.asyncio
async def test_remove_rejects_nonterminal_and_missing_tasks(engine_factory: _EngineFactory) -> None:
    finish: asyncio.Future[TestOutcome] = asyncio.get_running_loop().create_future()

    async def execute(_context: TestContext) -> TestOutcome:
        return await finish

    engine = _initialize(engine_factory(execute, max_concurrency=1))
    working = engine.submit({"value": "working"})
    queued = engine.submit({"value": "queued"})
    await _wait_for_status(engine, working["task_id"], "working")

    with pytest.raises(RuntimeError, match="cannot be removed"):
        engine.remove(working["task_id"])
    with pytest.raises(RuntimeError, match="cannot be removed"):
        engine.remove(queued["task_id"])
    with pytest.raises(BackgroundTaskNotFoundError, match="was not found"):
        engine.remove("missing")

    finish.set_result({"status": "completed", "result": {"value": "done"}})
    engine.cancel(queued["task_id"], "cleanup")


@pytest.mark.parametrize(
    ("patch", "removed_key", "message"),
    [
        ({"task_id": ""}, None, "task.task_id must be a non-empty string"),
        ({}, "descriptor", "task.descriptor is required"),
        ({"status": "unknown"}, None, "task.status 'unknown' is invalid"),
        ({"status": ["queued"]}, None, "task.status"),
        ({"attempt_count": True}, None, "task.attempt_count must be a non-negative integer"),
        ({"attempt_count": -1}, None, "task.attempt_count must be a non-negative integer"),
        ({"attempt_count": 1 << 53}, None, "task.attempt_count must be a non-negative integer"),
        ({"updated_at": "not-a-date"}, None, "ISO-8601"),
        ({"idempotency_key": ""}, None, "task.idempotency_key must be a non-empty string"),
        ({"failure": "bad"}, None, "task.failure must be an object"),
        ({"failure": {"type": "", "message": "bad"}}, None, "task.failure.type must be a non-empty string"),
        ({"status": "paused"}, "state", "task.state is required while paused"),
        ({"status": "completed"}, "result", "task.result is required while completed"),
        ({"status": "failed"}, "failure", "task.failure is required while failed"),
        ({"status": "cancelled"}, "cancellation_reason", "task.cancellation_reason is required while cancelled"),
        (
            {"cancellation_reason": "cancelled"},
            None,
            "task.cancellation_reason is only valid while cancelled",
        ),
    ],
)
def test_validate_stored_engine_task_rejects_malformed_records(
    patch: dict[str, object],
    removed_key: str | None,
    message: str,
) -> None:
    record = cast(dict[str, object], copy.deepcopy(_stored_task("task")))
    record.update(patch)
    if removed_key is not None:
        record.pop(removed_key, None)

    with pytest.raises(ValueError, match=message):
        validate_stored_engine_task(record)


def test_validate_stored_engine_task_rejects_non_objects_and_accepts_none_results() -> None:
    with pytest.raises(ValueError, match="must be an object"):
        validate_stored_engine_task([])

    record = cast(dict[str, object], _stored_task("completed", status="completed"))
    record["result"] = None
    validate_stored_engine_task(record)


@pytest.mark.asyncio
async def test_initialize_rolls_back_after_invalid_restored_record(engine_factory: _EngineFactory) -> None:
    async def execute(_context: TestContext) -> TestOutcome:
        return {"status": "completed", "result": {"value": "done"}}

    engine = engine_factory(execute)
    invalid = _stored_task("invalid")
    invalid["created_at"] = "not-a-date"

    with pytest.raises(ValueError, match="ISO-8601"):
        engine.initialize([invalid])
    with pytest.raises(RuntimeError, match="not initialized"):
        engine.list()

    engine.initialize()
    tru_tasks = engine.list()
    exp_tasks: list[TestTask] = []
    assert tru_tasks == exp_tasks


def test_validate_stored_engine_task_accepts_timestamp_without_timezone() -> None:
    record = _stored_task("naive-timestamp")
    record["created_at"] = "2026-08-10T12:00:00"
    record["updated_at"] = "2026-08-10T12:00:00"

    validate_stored_engine_task(record)


@pytest.mark.asyncio
async def test_initialize_executes_queued_task_with_timestamp_without_timezone(
    engine_factory: _EngineFactory,
) -> None:
    async def execute(_context: TestContext) -> TestOutcome:
        return {"status": "completed", "result": {"value": "done"}}

    record = _stored_task("naive-timestamp")
    record["created_at"] = "2026-08-10T12:00:00"
    record["updated_at"] = "2026-08-10T12:00:00"
    engine = _initialize(engine_factory(execute), [record])

    tru_task = await engine.wait(record["task_id"])
    assert tru_task["status"] == "completed"


@pytest.mark.asyncio
async def test_invalid_execution_outcome_becomes_execution_error(engine_factory: _EngineFactory) -> None:
    async def execute(_context: TestContext) -> TestOutcome:
        return cast(TestOutcome, {"status": "completed"})

    engine = _initialize(engine_factory(execute))
    admitted = engine.submit({"value": "work"})
    tru_failed = await engine.wait(admitted["task_id"])
    exp_status = "failed"
    assert tru_failed["status"] == exp_status
    exp_failure_type = "execution_error"
    assert tru_failed["failure"]["type"] == exp_failure_type
    assert "result is required" in tru_failed["failure"]["message"]
