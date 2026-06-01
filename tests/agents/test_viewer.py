import json
from urllib.request import urlopen
from unittest.mock import Mock

import pytest

import aisuite as ai
from aisuite.tracing.viewer import VIEWER_HTML
from tests.agents.helpers import chat_response


def test_write_trace_jsonl_and_read_trace_file(tmp_path):
    client = ai.Client()
    client.chat.completions.create = Mock(return_value=chat_response("ok"))
    agent = ai.Agent(name="assistant", model="openai:gpt-4o")
    result = ai.Runner.run_sync(agent, "Hello", client=client, run_name="run")
    trace_file = tmp_path / "runs.jsonl"

    result.write_trace_jsonl(trace_file)

    lines = trace_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["trace_id"] == result.trace_id
    trace = ai.tracing.read_trace_file(trace_file)[0]
    assert trace["run_name"] == "run"
    assert trace["final_output"] == "ok"
    assert trace["messages"] == result.messages
    assert trace["message_count"] == len(result.messages)
    assert trace["step_count"] == len(result.steps)


def test_start_viewer_serves_runs_api(tmp_path):
    client = ai.Client()
    client.chat.completions.create = Mock(return_value=chat_response("ok"))
    agent = ai.Agent(name="assistant", model="openai:gpt-4o")
    result = ai.Runner.run_sync(agent, "Hello", client=client, run_name="run")
    trace_file = tmp_path / "runs.jsonl"
    result.write_trace_jsonl(trace_file)

    try:
        viewer = ai.tracing.start_viewer(trace_file, port=0)
    except PermissionError:
        pytest.skip("Local socket binding is not permitted in this environment")
    try:
        with urlopen(f"{viewer.url}/api/runs", timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
    finally:
        viewer.stop()

    assert payload["runs"][0]["trace_id"] == result.trace_id
    assert payload["runs"][0]["run_name"] == "run"
    assert payload["runs"][0]["final_output"] == "ok"


def test_viewer_html_renders_run_transcript_sections():
    assert "Final Output" in VIEWER_HTML
    assert "Transcript" in VIEWER_HTML
    assert "message_count" in VIEWER_HTML
    assert "Events" in VIEWER_HTML
    assert "Details" in VIEWER_HTML


def test_read_trace_file_reconstructs_runs_from_events(tmp_path):
    trace_file = tmp_path / "events.jsonl"
    sink = ai.tracing.LocalTraceSink(trace_file)
    client = ai.Client()
    client.chat.completions.create = Mock(return_value=chat_response("ok"))
    agent = ai.Agent(name="assistant", model="openai:gpt-4o")

    result = ai.Runner.run_sync(
        agent,
        "Hello",
        client=client,
        run_name="run",
        group_id="group",
        trace_sinks=[sink],
    )

    runs = ai.tracing.read_trace_file(trace_file)

    assert len(runs) == 1
    assert runs[0]["trace_id"] == result.trace_id
    assert runs[0]["run_name"] == "run"
    assert runs[0]["group_id"] == "group"
    assert runs[0]["final_output"] == "ok"
    assert [event["event_type"] for event in runs[0]["events"]] == [
        "run.started",
        "model.started",
        "model.completed",
        "run.completed",
    ]
    json.dumps({"runs": runs})
