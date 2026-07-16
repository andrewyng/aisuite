from unittest.mock import Mock

from aisuite import Agent, Client, Runner
from tests.agents.helpers import chat_response


def test_run_sync_builds_user_message_and_returns_result():
    client = Client()
    client.chat.completions.create = Mock(return_value=chat_response("hello"))
    agent = Agent(name="assistant", model="openai:gpt-4o")

    result = Runner.run_sync(agent, "Say hi", client=client)

    client.chat.completions.create.assert_called_once_with(
        model="openai:gpt-4o",
        messages=[{"role": "user", "content": "Say hi"}],
    )
    assert result.final_output == "hello"
    assert result.status == "completed"
    assert result.agent is agent
    assert result.last_agent is agent
    assert result.messages == [
        {"role": "user", "content": "Say hi"},
        {"role": "assistant", "content": "hello"},
    ]
    assert result.new_items == [{"role": "assistant", "content": "hello"}]
    assert result.raw_responses


def test_run_sync_prepends_instructions():
    client = Client()
    client.chat.completions.create = Mock(return_value=chat_response("ok"))
    agent = Agent(
        name="assistant",
        model="openai:gpt-4o",
        instructions="Answer briefly.",
    )

    Runner.run_sync(agent, "Hi", client=client)

    assert client.chat.completions.create.call_args.kwargs["messages"] == [
        {"role": "system", "content": "Answer briefly."},
        {"role": "user", "content": "Hi"},
    ]


def test_run_sync_preserves_existing_system_message():
    client = Client()
    client.chat.completions.create = Mock(return_value=chat_response("ok"))
    agent = Agent(
        name="assistant",
        model="openai:gpt-4o",
        instructions="Do not duplicate.",
    )
    messages = [
        {"role": "system", "content": "Existing."},
        {"role": "user", "content": "Hi"},
    ]

    Runner.run_sync(agent, messages, client=client)

    assert client.chat.completions.create.call_args.kwargs["messages"] == messages


def test_run_sync_passes_model_settings_and_runtime_overrides():
    client = Client()
    client.chat.completions.create = Mock(return_value=chat_response("ok"))
    agent = Agent(
        name="assistant",
        model="openai:gpt-4o",
        model_settings={"temperature": 0.2, "max_tokens": 100},
    )

    Runner.run_sync(agent, "Hi", client=client, temperature=0.7)

    assert client.chat.completions.create.call_args.kwargs["temperature"] == 0.7
    assert client.chat.completions.create.call_args.kwargs["max_tokens"] == 100


def test_run_sync_enables_tool_loop_when_agent_has_tools():
    client = Client()
    client.chat.completions.create = Mock(return_value=chat_response("ok"))

    def lookup(city: str) -> str:
        """Lookup a city."""
        return city

    agent = Agent(name="assistant", model="openai:gpt-4o", tools=[lookup])

    Runner.run_sync(agent, "Hi", client=client, max_turns=3)

    assert client.chat.completions.create.call_args.kwargs["tools"] == [lookup]
    assert client.chat.completions.create.call_args.kwargs["max_turns"] == 3


def test_run_sync_merges_tags_metadata_and_observability_fields():
    client = Client()
    client.chat.completions.create = Mock(return_value=chat_response("ok"))
    agent = Agent(
        name="assistant",
        model="openai:gpt-4o",
        tags=["agent", "shared"],
        metadata={"team": "growth", "env": "dev"},
    )

    result = Runner.run_sync(
        agent,
        "Hi",
        client=client,
        run_name="support_reply",
        group_id="conversation_1",
        tags=["run", "shared"],
        metadata={"request_id": "req_1", "env": "prod"},
    )

    assert result.run_name == "support_reply"
    assert result.group_id == "conversation_1"
    assert result.tags == ["agent", "shared", "run"]
    assert result.metadata == {
        "team": "growth",
        "env": "prod",
        "request_id": "req_1",
    }
    assert result.trace_id.startswith("trace_")
    assert [step.type for step in result.steps] == ["agent", "model_response"]
    assert result.steps[0].name == "assistant"
