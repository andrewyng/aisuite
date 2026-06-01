import json
from unittest.mock import Mock

import aisuite as ai
from tests.agents.helpers import chat_response


def test_runner_emits_trace_events_to_memory_sink():
    client = ai.Client()
    client.chat.completions.create = Mock(return_value=chat_response("ok"))
    sink = ai.tracing.InMemoryTraceSink()
    agent = ai.Agent(name="assistant", model="openai:gpt-4o")

    result = ai.Runner.run_sync(
        agent,
        "Hello",
        client=client,
        run_name="run",
        group_id="group",
        trace_sinks=[sink],
    )

    event_types = [event.event_type for event in sink.events]
    assert event_types == [
        "run.started",
        "model.started",
        "model.completed",
        "run.completed",
    ]
    assert sink.events[0].trace_id == result.trace_id
    assert sink.events[0].group_id == "group"
    assert sink.events[-1].data["run"]["final_output"] == "ok"


def test_local_trace_sink_writes_event_jsonl(tmp_path):
    sink = ai.tracing.LocalTraceSink(tmp_path / "events.jsonl")
    event = ai.tracing.TraceEvent(
        event_type="run.started",
        trace_id="trace_1",
        agent_name="assistant",
        group_id="group",
    )

    sink.emit(event)

    line = (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()[0]
    payload = json.loads(line)
    assert payload["record_type"] == "trace_event"
    assert payload["schema_version"] == ai.tracing.TRACE_SCHEMA_VERSION
    assert payload["event_type"] == "run.started"
    assert payload["trace_id"] == "trace_1"


def test_jsonl_trace_store_lists_runs_and_events(tmp_path):
    store = ai.tracing.JsonlTraceStore(tmp_path / "events.jsonl")
    started = ai.tracing.TraceEvent(
        event_type="run.started",
        trace_id="trace_1",
        agent_name="assistant",
        group_id="group",
        run_name="run",
    )
    completed = ai.tracing.TraceEvent(
        event_type="run.completed",
        trace_id="trace_1",
        agent_name="assistant",
        group_id="group",
        run_name="run",
        data={
            "run": {
                "trace_id": "trace_1",
                "agent_name": "assistant",
                "run_name": "run",
                "group_id": "group",
                "status": "completed",
                "messages": [],
                "steps": [],
                "tags": [],
                "metadata": {},
                "final_output": "ok",
            }
        },
    )

    store.write_event(started)
    store.write_event(completed)

    run = store.get_run("trace_1")
    assert run["status"] == "completed"
    assert run["final_output"] == "ok"
    assert [event["event_type"] for event in store.list_events("trace_1")] == [
        "run.started",
        "run.completed",
    ]


def test_global_trace_configuration_is_used_by_runner():
    client = ai.Client()
    client.chat.completions.create = Mock(return_value=chat_response("ok"))
    sink = ai.tracing.InMemoryTraceSink()
    ai.tracing.configure(sink)
    try:
        ai.Runner.run_sync(
            ai.Agent(name="assistant", model="openai:gpt-4o"),
            "Hello",
            client=client,
        )
    finally:
        ai.tracing.configure()

    assert [event.event_type for event in sink.events][-1] == "run.completed"
