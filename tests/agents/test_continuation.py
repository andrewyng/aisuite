from unittest.mock import Mock

import pytest

import aisuite as ai
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


def test_run_sync_with_state_store_persists_first_turn():
    client = Client()
    client.chat.completions.create = Mock(return_value=chat_response("first"))
    agent = Agent(name="assistant", model="openai:gpt-4o")
    store = ai.InMemoryStateStore()

    result = Runner.run_sync(
        agent,
        "Hello",
        client=client,
        state_store=store,
        thread_id="thread_1",
        run_name="chat",
    )

    stored = store.load_state("thread_1")
    assert stored.revision == 1
    assert stored.state.messages == result.messages
    assert stored.state.run_name == "chat"


def test_run_sync_with_existing_thread_raises():
    client = Client()
    client.chat.completions.create = Mock(return_value=chat_response("first"))
    agent = Agent(name="assistant", model="openai:gpt-4o")
    store = ai.InMemoryStateStore()
    Runner.run_sync(
        agent, "Hello", client=client, state_store=store, thread_id="thread_1"
    )

    with pytest.raises(ai.ThreadAlreadyExistsError):
        Runner.run_sync(
            agent,
            "Start over",
            client=client,
            state_store=store,
            thread_id="thread_1",
        )


def test_continue_sync_with_agent_loads_and_saves_persisted_state():
    client = Client()
    client.chat.completions.create = Mock(return_value=chat_response("second"))
    agent = Agent(name="assistant", model="openai:gpt-4o")
    store = ai.InMemoryStateStore()
    store.save_state(
        "thread_1",
        RunState(
            agent_name="assistant",
            messages=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "first"},
            ],
            run_name="chat",
            metadata={"request_id": "req_1"},
        ),
    )

    result = Runner.continue_sync(
        agent,
        "Follow up",
        client=client,
        state_store=store,
        thread_id="thread_1",
    )

    assert client.chat.completions.create.call_args.kwargs["messages"] == [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "first"},
        {"role": "user", "content": "Follow up"},
    ]
    stored = store.load_state("thread_1")
    assert stored.revision == 2
    assert stored.state.messages == result.messages
    assert stored.state.messages[-1] == {"role": "assistant", "content": "second"}


def test_continue_sync_with_missing_persisted_state_raises():
    agent = Agent(name="assistant", model="openai:gpt-4o")

    with pytest.raises(ai.StateNotFoundError):
        Runner.continue_sync(
            agent,
            "Follow up",
            state_store=ai.InMemoryStateStore(),
            thread_id="missing",
        )


def test_continue_sync_result_can_persist_after_in_memory_continuation():
    client = Client()
    client.chat.completions.create = Mock(
        side_effect=[chat_response("first"), chat_response("second")]
    )
    agent = Agent(name="assistant", model="openai:gpt-4o")
    store = ai.InMemoryStateStore()

    first = Runner.run_sync(agent, "Hello", client=client)
    second = Runner.continue_sync(
        first,
        "Follow up",
        state_store=store,
        thread_id="thread_1",
    )

    stored = store.load_state("thread_1")
    assert stored.revision == 1
    assert stored.state.messages == second.messages


def test_persisted_state_requires_store_and_thread_id_together():
    agent = Agent(name="assistant", model="openai:gpt-4o")

    with pytest.raises(ValueError, match="state_store and thread_id"):
        Runner.run_sync(agent, "Hello", state_store=ai.InMemoryStateStore())

    with pytest.raises(ValueError, match="Persisted continuation requires"):
        Runner.continue_sync(agent, "Follow up", state_store=ai.InMemoryStateStore())


def test_state_serialization_rejects_non_json_metadata():
    state = RunState(
        agent_name="assistant",
        messages=[],
        metadata={"bad": object()},
    )

    with pytest.raises(TypeError, match="JSON serializable"):
        state.to_dict()
