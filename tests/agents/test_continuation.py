from unittest.mock import Mock

import pytest

from aisuite import Agent, Client, RunState, Runner
from tests.agents.helpers import chat_response


def test_run_result_to_state_round_trips_through_dict():
    client = Client()
    client.chat.completions.create = Mock(return_value=chat_response("first"))
    agent = Agent(name="assistant", model="openai:gpt-4o")

    result = Runner.run_sync(
        agent,
        "Hello",
        client=client,
        run_name="chat",
        group_id="group_1",
        tags=["tag"],
        metadata={"task_type": "chat"},
    )

    state = RunState.from_dict(result.to_state().to_dict())

    assert state.agent_name == "assistant"
    assert state.run_name == "chat"
    assert state.group_id == "group_1"
    assert state.tags == ["tag"]
    assert state.metadata == {"task_type": "chat"}
    assert state.messages == result.messages
    assert len(state.steps) == len(result.steps)


def test_run_sync_accepts_state_and_resumes_messages():
    client = Client()
    client.chat.completions.create = Mock(return_value=chat_response("second"))
    agent = Agent(name="assistant", model="openai:gpt-4o")
    state = RunState(
        agent_name="assistant",
        messages=[
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "first"},
        ],
        group_id="group_1",
        metadata={"task_type": "chat"},
    )
    state.add_user_message("Follow up")

    result = Runner.run_sync(agent, state, client=client)

    assert client.chat.completions.create.call_args.kwargs["messages"] == [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "first"},
        {"role": "user", "content": "Follow up"},
    ]
    assert result.final_output == "second"
    assert result.group_id == "group_1"
    assert result.metadata == {"task_type": "chat"}


def test_continue_sync_reuses_result_context_and_appends_input():
    client = Client()
    client.chat.completions.create = Mock(
        side_effect=[chat_response("first"), chat_response("second")]
    )
    agent = Agent(
        name="assistant",
        model="openai:gpt-4o",
        tags=["agent"],
        metadata={"team": "growth"},
    )

    first = Runner.run_sync(
        agent,
        "Hello",
        client=client,
        run_name="chat",
        group_id="group_1",
        tags=["run"],
        metadata={"request_id": "req_1"},
    )
    second = Runner.continue_sync(first, "Follow up")

    assert client.chat.completions.create.call_args_list[1].kwargs["messages"] == [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "first"},
        {"role": "user", "content": "Follow up"},
    ]
    assert second.final_output == "second"
    assert second.run_name == "chat"
    assert second.group_id == "group_1"
    assert second.tags == ["agent", "run"]
    assert second.metadata == {"team": "growth", "request_id": "req_1"}
    assert second.trace_id != first.trace_id


def test_state_serialization_rejects_non_json_metadata():
    state = RunState(
        agent_name="assistant",
        messages=[],
        metadata={"bad": object()},
    )

    with pytest.raises(TypeError, match="JSON serializable"):
        state.to_dict()
