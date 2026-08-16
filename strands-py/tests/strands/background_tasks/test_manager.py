import asyncio
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from strands import Agent
from strands.background_tasks import BackgroundTasksTimeoutError
from strands.background_tasks._manager import _InProcessTaskManager
from strands.background_tasks._record import StoredBackgroundTask
from strands.background_tasks._runtime import get_background_task_runtime
from strands.hooks import AfterModelCallEvent
from strands.interrupt import Interrupt, _InterruptState
from strands.types.interrupt import InterruptResponseContent
from tests.fixtures.mocked_model_provider import MockedModelProvider


def _response(interrupt_id: str, response: Any) -> InterruptResponseContent:
    return {"interruptResponse": {"interruptId": interrupt_id, "response": response}}


def _resumed_record(response: Any) -> StoredBackgroundTask:
    state = _InterruptState(
        interrupts={
            "approval": Interrupt(
                id="approval",
                name="approve",
                response=response,
            )
        },
        activated=True,
    )
    return cast(
        StoredBackgroundTask,
        {
            "task_id": "task",
            "descriptor": {
                "original_tool_use_id": "tool-use",
                "tool_name": "work",
                "invocation_state": {},
            },
            "status": "working",
            "attempt_count": 1,
            "created_at": "2026-08-10T12:00:00",
            "updated_at": "2026-08-10T12:00:00",
            "state": state.to_dict(),
        },
    )


@pytest.mark.asyncio
async def test_runtime_run_sync_allows_sync_operation_on_runtime_thread() -> None:
    runtime = get_background_task_runtime()

    async def run_on_runtime() -> None:
        tru_result = runtime.run_sync(lambda: "ready")
        exp_result = "ready"
        assert tru_result == exp_result

        async def async_operation() -> str:
            return "not allowed"

        with pytest.raises(RuntimeError, match="Cannot synchronously wait on an awaitable"):
            runtime.run_sync(async_operation)

    await runtime.run(run_on_runtime)


def test__resume_task_accepts_only_already_applied_response_for_nonpaused_task() -> None:
    manager = cast(_InProcessTaskManager, object.__new__(_InProcessTaskManager))
    manager._engine = MagicMock()
    manager._engine.get.return_value = _resumed_record("yes")

    manager._resume_task("task", [_response("approval", "yes")])
    manager._engine.resume.assert_not_called()

    with pytest.raises(RuntimeError, match="status is 'working', not 'paused'"):
        manager._resume_task("task", [_response("approval", "no")])

    with pytest.raises(RuntimeError, match="status is 'working', not 'paused'"):
        manager._resume_task("task", [_response("unknown", None)])


@pytest.mark.asyncio
async def test__on_after_model_call_waits_for_state_reload() -> None:
    manager = cast(_InProcessTaskManager, object.__new__(_InProcessTaskManager))
    manager._wait_for_reload = AsyncMock()
    manager._paused_interrupts = AsyncMock(return_value=[])
    event = MagicMock(spec=AfterModelCallEvent)
    event.exception = None
    event.stop_response = {"stopReason": "end_turn"}

    await manager._on_after_model_call(event)

    manager._wait_for_reload.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_wait_for_tasks_timeout_includes_state_reload() -> None:
    async def wait_for_reload() -> None:
        await asyncio.sleep(1)

    manager = cast(_InProcessTaskManager, object.__new__(_InProcessTaskManager))
    manager._wait_for_reload = AsyncMock(side_effect=wait_for_reload)
    manager._runtime = MagicMock()

    with pytest.raises(BackgroundTasksTimeoutError) as timeout_info:
        await manager.wait_for_tasks(timeout=0.01)

    assert timeout_info.value.timeout == 0.01
    manager._runtime.run_void.assert_not_called()


@pytest.mark.asyncio
async def test_wait_for_tasks_maps_state_reload_timeout() -> None:
    manager = cast(_InProcessTaskManager, object.__new__(_InProcessTaskManager))
    manager._wait_for_reload = AsyncMock(side_effect=TimeoutError("reload timed out"))
    manager._runtime = MagicMock()

    with pytest.raises(BackgroundTasksTimeoutError) as timeout_info:
        await manager.wait_for_tasks(timeout=1)

    assert timeout_info.value.timeout == 1
    manager._runtime.run_void.assert_not_called()


def test__load_state_accepts_extra_envelope_fields_and_recovers_live_work() -> None:
    agent = Agent(
        model=MockedModelProvider([]),
        background_tasks=True,
        callback_handler=None,
    )
    assert agent._background_tasks is not None
    manager = agent._background_tasks._manager
    assert manager is not None
    record = _resumed_record(None)
    record["status"] = "queued"
    record.pop("state")
    record["descriptor"]["input"] = {"secret": "tool-secret"}
    record["descriptor"]["invocation_state"] = {"api_token": "invocation-secret"}

    try:
        manager._load_state(
            {
                "task": {
                    "record": record,
                    "delivery_state": "pending",
                    "future_field": True,
                }
            }
        )
        manager._persist_state()

        tru_record = manager._records["task"]
        assert tru_record["status"] == "failed"
        assert tru_record["failure"]["type"] == "recovery_error"
        assert tru_record["descriptor"] == {
            "original_tool_use_id": "tool-use",
            "tool_name": "work",
            "invocation_state": {},
        }
        persisted = agent.state.get("strands.background_tasks")
        assert isinstance(persisted, dict)
        persisted_record = persisted["task"]["record"]
        assert "descriptor" not in persisted_record
        assert "tool-secret" not in str(persisted_record)
        assert "invocation-secret" not in str(persisted_record)
    finally:
        agent.cleanup()


@pytest.mark.asyncio
async def test_cancel_task_records_each_successful_request() -> None:
    completed = _resumed_record(None)
    completed["status"] = "completed"
    completed["result"] = {
        "toolUseId": "tool-use",
        "status": "success",
        "content": [{"text": "done"}],
    }
    completed.pop("state")

    manager = cast(_InProcessTaskManager, object.__new__(_InProcessTaskManager))
    manager._wait_for_reload = AsyncMock()
    manager._runtime = MagicMock()
    manager._runtime.run = AsyncMock(side_effect=lambda operation: operation())
    manager._engine = MagicMock()
    manager._engine.get.return_value = completed
    manager._engine.cancel.return_value = completed
    manager._telemetry = MagicMock()

    await manager.cancel_task("task")
    await manager.cancel_task("task")

    exp_calls = 2
    assert manager._telemetry.record_cancellation.call_count == exp_calls
