from io import StringIO
from unittest.mock import Mock

from aisuite import Agent, Client, Runner
from tests.agents.helpers import chat_response


def test_trace_to_dict_includes_observability_fields():
    client = Client()
    client.chat.completions.create = Mock(return_value=chat_response("ok"))
    agent = Agent(name="assistant", model="openai:gpt-4o", tags=["agent"])

    result = Runner.run_sync(
        agent,
        "Hello",
        client=client,
        run_name="test_run",
        group_id="group_1",
        tags=["run"],
        metadata={"request_id": "req_1"},
    )

    trace = result.trace_to_dict()

    assert trace["trace_id"] == result.trace_id
    assert trace["group_id"] == "group_1"
    assert trace["run_name"] == "test_run"
    assert trace["agent_name"] == "assistant"
    assert trace["status"] == "completed"
    assert trace["tags"] == ["agent", "run"]
    assert trace["metadata"] == {"request_id": "req_1"}
    assert trace["final_output"] == "ok"
    assert trace["message_count"] == len(result.messages)
    assert trace["step_count"] == len(result.steps)
    assert trace["messages"] == result.messages
    assert trace["new_items"] == result.new_items
    assert [step["type"] for step in trace["steps"]] == ["agent", "model_response"]


def test_print_trace_outputs_human_readable_trace():
    client = Client()
    client.chat.completions.create = Mock(return_value=chat_response("ok"))
    agent = Agent(name="assistant", model="openai:gpt-4o")
    result = Runner.run_sync(
        agent,
        "Hello",
        client=client,
        run_name="test_run",
        group_id="group_1",
        metadata={"request_id": "req_1"},
    )
    output = StringIO()

    result.print_trace(file=output)

    text = output.getvalue()
    assert f"Trace {result.trace_id}" in text
    assert "run=test_run" in text
    assert "agent=assistant" in text
    assert "status=completed" in text
    assert "Group: group_1" in text
    assert "Metadata: request_id=req_1" in text
    assert "Final output: ok" in text
    assert "- agent: assistant" in text
    assert "- model_response: model_response" in text
