import json
from types import SimpleNamespace
from urllib.error import URLError
from urllib.request import Request
from unittest.mock import Mock

import aisuite as ai
from aisuite.tracing.normalize import normalize_usage
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
        "model.send",
        "model.response",
        "run.completed",
    ]
    assert sink.events[0].trace_id == result.trace_id
    assert sink.events[0].group_id == "group"
    assert sink.events[1].data["input"]["items"][-1]["text_preview"] == "Hello"
    assert sink.events[2].data["response"]["text_preview"] == "ok"
    assert sink.events[-1].data["run"]["final_output"] == "ok"


def test_trace_store_sink_writes_events_to_store():
    store = ai.tracing.InMemoryTraceStore()
    sink = ai.tracing.TraceStoreSink(store)
    event = ai.tracing.TraceEvent(
        event_type="run.started",
        trace_id="trace_1",
        agent_name="assistant",
        run_name="run",
    )

    sink.emit(event)

    records = store.list_records()
    assert records[0]["event_type"] == "run.started"
    assert store.list_events("trace_1")[0]["event_id"] == event.event_id


def test_runner_can_emit_to_trace_store_sink_and_viewer_reads_same_store():
    client = ai.Client()
    client.chat.completions.create = Mock(return_value=chat_response("ok"))
    store = ai.tracing.InMemoryTraceStore()
    sink = ai.tracing.TraceStoreSink(store)

    result = ai.Runner.run_sync(
        ai.Agent(name="assistant", model="openai:gpt-4o"),
        "Hello",
        client=client,
        run_name="store_run",
        trace_sinks=[sink],
    )

    run = store.get_run(result.trace_id)
    assert run["run_name"] == "store_run"
    assert run["final_output"] == "ok"
    assert [event["event_type"] for event in store.list_events(result.trace_id)] == [
        "run.started",
        "model.send",
        "model.response",
        "run.completed",
    ]


def test_normalize_usage_preserves_aisuite_and_provider_fields():
    usage = normalize_usage(
        SimpleNamespace(
            prompt_tokens=1200,
            completion_tokens=34,
            total_tokens=None,
            prompt_tokens_details={"cached_tokens": 100},
        )
    )

    assert usage["input_tokens"] == 1200
    assert usage["output_tokens"] == 34
    assert usage["total_tokens"] == 1234
    assert usage["prompt_tokens"] == 1200
    assert usage["completion_tokens"] == 34
    assert usage["provider_raw"]["prompt_tokens_details"] == {"cached_tokens": 100}


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


def test_http_trace_sink_posts_event(monkeypatch):
    calls = []

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    def fake_urlopen(request: Request, timeout):
        calls.append((request, timeout))
        return Response()

    monkeypatch.setattr("aisuite.tracing.sinks.urlopen", fake_urlopen)
    sink = ai.tracing.HttpTraceSink("http://127.0.0.1:8780/api/events", timeout=3)
    event = ai.tracing.TraceEvent(
        event_type="run.started",
        trace_id="trace_1",
        agent_name="assistant",
    )

    sink.emit(event)

    request, timeout = calls[0]
    assert request.full_url == "http://127.0.0.1:8780/api/events"
    assert timeout == 3
    payload = json.loads(request.data.decode("utf-8"))
    assert payload["event_type"] == "run.started"
    assert payload["trace_id"] == "trace_1"


def test_http_trace_sink_can_fail_silently(monkeypatch):
    def fake_urlopen(_request, timeout=None):
        raise URLError("offline")

    monkeypatch.setattr("aisuite.tracing.sinks.urlopen", fake_urlopen)
    sink = ai.tracing.HttpTraceSink("http://127.0.0.1:8780/api/events")

    sink.emit(
        ai.tracing.TraceEvent(
            event_type="run.started",
            trace_id="trace_1",
            agent_name="assistant",
        )
    )


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

    store.append_event(started)
    store.append_events([completed])

    records = store.list_records()
    assert [record["event_type"] for record in records] == [
        "run.started",
        "run.completed",
    ]

    run = store.get_run("trace_1")
    assert run["status"] == "completed"
    assert run["final_output"] == "ok"
    assert [event["event_type"] for event in store.list_events("trace_1")] == [
        "run.started",
        "run.completed",
    ]


def test_trace_stores_import_jsonl_and_ignore_invalid_lines(tmp_path):
    valid_event = ai.tracing.TraceEvent(
        event_type="run.started",
        trace_id="trace_1",
        agent_name="assistant",
    ).to_dict()
    content = "\n".join([json.dumps(valid_event), "not json", ""])

    jsonl_store = ai.tracing.JsonlTraceStore(tmp_path / "events.jsonl")
    memory_store = ai.tracing.InMemoryTraceStore()

    assert jsonl_store.import_jsonl(content) == 1
    assert memory_store.import_jsonl(content) == 1
    assert jsonl_store.list_events("trace_1")[0]["event_type"] == "run.started"
    assert memory_store.list_events("trace_1")[0]["event_type"] == "run.started"


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
