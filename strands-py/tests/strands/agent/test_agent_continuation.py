import pytest

from strands import Agent
from strands.hooks import AfterInvocationEvent, MessageAddedEvent
from strands.types.content import Message
from strands.types.exceptions import EventLoopException
from tests.fixtures.mocked_model_provider import MockedModelProvider


def _assistant_text(text: str) -> Message:
    return {"role": "assistant", "content": [{"text": text}]}


@pytest.mark.asyncio
async def test_stream_async_orders_continuations_and_emits_one_final_result(alist):
    agent = Agent(
        model=MockedModelProvider([_assistant_text("initial"), _assistant_text("final")]),
        callback_handler=None,
    )
    lifecycle: list[str] = []
    armed = False
    deferred_assistant: Message = {
        "role": "assistant",
        "content": [
            {
                "toolUse": {
                    "name": "deferred_result",
                    "toolUseId": "delivery-1",
                    "input": {"taskId": "task-1"},
                }
            }
        ],
    }
    deferred_result: Message = {
        "role": "user",
        "content": [
            {
                "toolResult": {
                    "toolUseId": "delivery-1",
                    "status": "success",
                    "content": [{"text": "background result"}],
                }
            }
        ],
    }

    def register_continuations(event: AfterInvocationEvent) -> None:
        nonlocal armed
        if armed:
            return
        armed = True
        event._continue_with(
            {
                "phase": "guidance",
                "input": "review the completed work",
                "on_accepted": lambda: lifecycle.append("accepted:guidance"),
                "on_committed": lambda: lifecycle.append("committed:guidance"),
            }
        )
        event._continue_with(
            {
                "phase": "deferred_result",
                "input": [deferred_assistant, deferred_result],
                "on_accepted": lambda: lifecycle.append("accepted:deferred"),
                "on_committed": lambda: lifecycle.append("committed:deferred"),
            }
        )

    def record_deferred_message(event: MessageAddedEvent) -> None:
        if event.message is deferred_assistant:
            lifecycle.append("message:deferred")

    def record_continuation_invocation(_event: AfterInvocationEvent) -> None:
        if any(message is deferred_assistant for message in agent.messages):
            lifecycle.append("after:continuation")

    agent.add_hook(register_continuations, AfterInvocationEvent)
    agent.add_hook(record_deferred_message, MessageAddedEvent)
    agent.add_hook(record_continuation_invocation, AfterInvocationEvent)

    tru_events = await alist(agent.stream_async("start"))
    tru_result_events = [event["result"] for event in tru_events if "result" in event]
    exp_result_event_count = 1
    assert len(tru_result_events) == exp_result_event_count

    tru_final_message = {
        "role": tru_result_events[0].message["role"],
        "content": tru_result_events[0].message["content"],
    }
    exp_final_message = {"role": "assistant", "content": [{"text": "final"}]}
    assert tru_final_message == exp_final_message

    tru_lifecycle = lifecycle
    exp_lifecycle = [
        "accepted:deferred",
        "accepted:guidance",
        "message:deferred",
        "after:continuation",
        "committed:deferred",
        "committed:guidance",
    ]
    assert tru_lifecycle == exp_lifecycle

    tru_messages = [{"role": message["role"], "content": message["content"]} for message in agent.messages]
    exp_messages = [
        {"role": "user", "content": [{"text": "start"}]},
        {"role": "assistant", "content": [{"text": "initial"}]},
        {
            "role": "assistant",
            "content": [
                {
                    "toolUse": {
                        "name": "deferred_result",
                        "toolUseId": "delivery-1",
                        "input": {"taskId": "task-1"},
                    }
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "toolResult": {
                        "toolUseId": "delivery-1",
                        "status": "success",
                        "content": [{"text": "background result"}],
                    }
                }
            ],
        },
        {"role": "user", "content": [{"text": "review the completed work"}]},
        {"role": "assistant", "content": [{"text": "final"}]},
    ]
    assert tru_messages == exp_messages


@pytest.mark.asyncio
async def test_invoke_async_makes_complete_continuation_batch_visible_before_message_hooks():
    agent = Agent(
        model=MockedModelProvider([_assistant_text("initial"), _assistant_text("unreachable")]),
        callback_handler=None,
    )
    committed: list[str] = []
    rejected: list[str] = []
    armed = False
    first: Message = {"role": "user", "content": [{"text": "first"}]}
    second: Message = {"role": "user", "content": [{"text": "second"}]}

    def register_continuation(event: AfterInvocationEvent) -> None:
        nonlocal armed
        if armed:
            return
        armed = True
        event._continue_with(
            {
                "phase": "deferred_result",
                "input": [first, second],
                "on_committed": lambda: committed.append("committed"),
                "on_rejected": lambda _reason: rejected.append("rejected"),
            }
        )

    def fail_on_first_message(event: MessageAddedEvent) -> None:
        if event.message is first:
            tru_batch = agent.messages[-3:-1]
            exp_batch = [first, second]
            assert tru_batch == exp_batch
            raise RuntimeError("message hook failed")

    agent.add_hook(register_continuation, AfterInvocationEvent)
    agent.add_hook(fail_on_first_message, MessageAddedEvent)

    with pytest.raises(EventLoopException, match="message hook failed"):
        await agent.invoke_async("start")

    tru_committed = committed
    exp_committed: list[str] = []
    assert tru_committed == exp_committed

    tru_rejected = rejected
    exp_rejected = ["rejected"]
    assert tru_rejected == exp_rejected

    tru_batch = agent.messages[-3:-1]
    exp_batch = [first, second]
    assert tru_batch == exp_batch


@pytest.mark.asyncio
async def test_invoke_async_rejects_continuation_when_after_invocation_hook_fails():
    agent = Agent(
        model=MockedModelProvider([_assistant_text("initial"), _assistant_text("final")]),
        callback_handler=None,
    )
    deferred: Message = {"role": "user", "content": [{"text": "deferred"}]}
    lifecycle: list[str] = []
    armed = False

    def register_continuation(event: AfterInvocationEvent) -> None:
        nonlocal armed
        if armed:
            return
        armed = True
        event._continue_with(
            {
                "phase": "deferred_result",
                "input": [deferred],
                "on_committed": lambda: lifecycle.append("committed"),
                "on_rejected": lambda _reason: lifecycle.append("rejected"),
            }
        )

    def fail_snapshot_write(_event: AfterInvocationEvent) -> None:
        if any(message is deferred for message in agent.messages):
            raise RuntimeError("snapshot write failed")

    agent.add_hook(register_continuation, AfterInvocationEvent)
    agent.add_hook(fail_snapshot_write, AfterInvocationEvent)

    with pytest.raises(RuntimeError, match="snapshot write failed"):
        await agent.invoke_async("start")

    tru_lifecycle = lifecycle
    exp_lifecycle = ["rejected"]
    assert tru_lifecycle == exp_lifecycle
