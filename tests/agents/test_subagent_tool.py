from unittest.mock import Mock

import aisuite as ai
from aisuite.framework.message import ChatCompletionMessageToolCall, Function, Message
from tests.agents.helpers import chat_response


def tool_call(name, arguments, call_id="call_1"):
    return ChatCompletionMessageToolCall(
        id=call_id,
        type="function",
        function=Function(name=name, arguments=arguments),
    )


def test_agent_tool_runs_subagent_and_returns_output():
    client = ai.Client()
    provider = Mock()
    first_response = chat_response(None)
    first_response.choices[0].message = Message(
        role="assistant",
        tool_calls=[tool_call("research_topic", '{"input": "Paris"}')],
    )
    provider.chat_completions_create.side_effect = [
        first_response,
        chat_response("Paris notes"),
        chat_response("Final answer"),
    ]
    client.providers["openai"] = provider
    researcher = ai.Agent(name="researcher", model="openai:gpt-4o-mini")
    writer = ai.Agent(
        name="writer",
        model="openai:gpt-4o",
        tools=[ai.agent_tool(researcher, name="research_topic")],
    )

    result = ai.Runner.run_sync(writer, "Write about Paris", client=client)

    assert result.final_output == "Final answer"
    assert provider.chat_completions_create.call_args_list[1].args[0] == "gpt-4o-mini"
    tool_message = provider.chat_completions_create.call_args_list[2].args[1][-1]
    assert tool_message["role"] == "tool"
    assert tool_message["content"] == '"Paris notes"'


def test_agent_tool_inherits_trace_context_for_subagent_run():
    client = ai.Client()
    provider = Mock()
    first_response = chat_response(None)
    first_response.choices[0].message = Message(
        role="assistant",
        tool_calls=[tool_call("research_topic", '{"input": "Paris"}')],
    )
    provider.chat_completions_create.side_effect = [
        first_response,
        chat_response("Paris notes"),
        chat_response("Final answer"),
    ]
    client.providers["openai"] = provider
    sink = ai.tracing.InMemoryTraceSink()
    researcher = ai.Agent(name="researcher", model="openai:gpt-4o-mini")
    writer = ai.Agent(
        name="writer",
        model="openai:gpt-4o",
        tools=[ai.agent_tool(researcher, name="research_topic")],
    )

    parent = ai.Runner.run_sync(
        writer,
        "Write about Paris",
        client=client,
        group_id="group",
        tags=["workflow"],
        metadata={"request_id": "req_1"},
        trace_sinks=[sink],
    )

    completed_runs = [
        event.data["run"]
        for event in sink.events
        if event.event_type == "run.completed"
    ]
    child = next(run for run in completed_runs if run["agent_name"] == "researcher")
    assert child["parent_run_id"] == parent.trace_id
    assert child["group_id"] == "group"
    assert child["tags"] == ["workflow"]
    assert child["metadata"] == {"request_id": "req_1"}
