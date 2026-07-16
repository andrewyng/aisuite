from pathlib import Path
from unittest.mock import Mock
import json
from urllib.request import Request, urlopen

import pytest

import aisuite as ai
from aisuite.tracing.viewer import (
    VIEWER_HTML,
    ViewerTraceState,
    prepare_viewer_run_summaries,
    prepare_viewer_runs,
)
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
    assert "messages" not in payload["runs"][0]
    assert "steps" not in payload["runs"][0]
    assert "events" not in payload["runs"][0]


def test_start_viewer_can_use_explicit_trace_store(tmp_path):
    store = ai.tracing.InMemoryTraceStore()
    store.append_record(
        {
            "trace_id": "trace_store",
            "run_name": "store run",
            "agent_name": "assistant",
            "status": "completed",
            "messages": [],
            "steps": [],
            "final_output": "from store",
        }
    )

    try:
        viewer = ai.tracing.start_viewer(None, port=0, trace_store=store)
    except PermissionError:
        pytest.skip("Local socket binding is not permitted in this environment")
    try:
        with urlopen(f"{viewer.url}/api/runs", timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
    finally:
        viewer.stop()

    assert payload["runs"][0]["trace_id"] == "trace_store"
    assert payload["runs"][0]["final_output"] == "from store"


def test_start_viewer_serves_artifact_api(tmp_path):
    artifact_store = ai.FileArtifactStore(tmp_path / "artifacts")
    ref = artifact_store.put(
        "large output",
        media_type="text/plain; charset=utf-8",
        metadata={"kind": "stdout"},
    )

    try:
        viewer = ai.tracing.start_viewer(
            tmp_path / "runs.jsonl",
            port=0,
            artifact_store=artifact_store,
        )
    except PermissionError:
        pytest.skip("Local socket binding is not permitted in this environment")
    try:
        with urlopen(
            f"{viewer.url}/api/artifacts/{ref.artifact_id}", timeout=5
        ) as response:
            payload = json.loads(response.read().decode("utf-8"))
    finally:
        viewer.stop()

    assert payload["artifact"]["artifact_id"] == ref.artifact_id
    assert payload["artifact"]["metadata"]["kind"] == "stdout"
    assert payload["text"] == "large output"


def test_prepare_viewer_run_summaries_excludes_heavy_fields():
    runs = prepare_viewer_run_summaries(
        [
            {
                "trace_id": "trace_1",
                "run_name": "run",
                "agent_name": "assistant",
                "status": "completed",
                "messages": [{"role": "user", "content": "x" * 20_000}],
                "steps": [{"id": "step_1", "data": {"big": "y" * 20_000}}],
                "events": [
                    {
                        "event_type": "run.started",
                        "timestamp": "2026-05-16T00:00:00+00:00",
                    }
                ],
                "final_output": "ok",
            }
        ]
    )

    summary = runs[0]
    assert summary["trace_id"] == "trace_1"
    assert summary["final_output"] == "ok"
    assert summary["display"]["title"] == "run"
    assert "messages" not in summary
    assert "steps" not in summary
    assert "events" not in summary


def test_viewer_state_get_run_uses_trace_store_get_run_contract():
    class SpyStore(ai.tracing.InMemoryTraceStore):
        def __init__(self):
            super().__init__()
            self.get_run_calls = []

        def get_run(self, trace_id):
            self.get_run_calls.append(trace_id)
            return super().get_run(trace_id)

    store = SpyStore()
    event = ai.tracing.TraceEvent(
        event_type="run.started",
        trace_id="trace_store",
        agent_name="assistant",
        run_name="store-backed",
    )
    store.append_event(event)
    state = ViewerTraceState(store=store)

    run = state.get_run("trace_store")

    assert store.get_run_calls == ["trace_store"]
    assert run["trace_id"] == "trace_store"
    assert run["run_name"] == "store-backed"


def test_viewer_server_exposes_trace_store_sink_for_embedded_runs(tmp_path):
    store = ai.tracing.InMemoryTraceStore()
    try:
        viewer = ai.tracing.start_viewer(None, port=0, trace_store=store)
    except PermissionError:
        pytest.skip("Local socket binding is not permitted in this environment")
    try:
        viewer.trace_sink.emit(
            ai.tracing.TraceEvent(
                event_type="run.started",
                trace_id="trace_embedded",
                agent_name="assistant",
                run_name="embedded",
            )
        )
        with urlopen(f"{viewer.url}/api/runs/trace_embedded", timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
    finally:
        viewer.stop()

    assert payload["run"]["trace_id"] == "trace_embedded"
    assert payload["run"]["run_name"] == "embedded"


def test_viewer_detail_sanitizes_large_strings_and_counts_artifacts():
    artifact_ref = {
        "type": "artifact_ref",
        "artifact_ref": {
            "artifact_id": "art_1",
            "size_bytes": 12000,
            "media_type": "text/plain",
        },
        "preview": "preview",
    }
    store = ai.tracing.InMemoryTraceStore()
    store.append_record(
        {
            "trace_id": "trace_1",
            "run_name": "run",
            "agent_name": "assistant",
            "status": "completed",
            "messages": [{"role": "tool", "content": "x" * 9000}],
            "steps": [
                {
                    "id": "step_1",
                    "type": "tool_call",
                    "name": "write_file",
                    "data": {"arguments": {"content": artifact_ref}},
                }
            ],
            "events": [
                {
                    "event_type": "tool.allowed",
                    "timestamp": "2026-05-16T00:00:00+00:00",
                    "data": {
                        "tool_name": "write_file",
                        "tool_call_id": "call_1",
                        "arguments": {"content": artifact_ref},
                    },
                }
            ],
            "final_output": "ok",
        }
    )
    state = ViewerTraceState(store=store)

    summary = state.list_runs()[0]
    detail = state.get_run("trace_1")

    assert "messages" not in summary
    assert summary["display"]["artifact_count"] == 1
    assert detail["messages"][0]["content"]["type"] == "text_preview"
    assert detail["display"]["timeline"][0]["artifact_count"] == 1
    rendered = json.dumps(detail)
    assert "x" * 9000 not in rendered
    assert "art_1" in rendered


def test_prepare_viewer_runs_links_subagent_activity_by_tool_name():
    runs = prepare_viewer_runs(
        [
            {
                "trace_id": "parent",
                "run_name": "main",
                "agent_name": "assistant",
                "status": "completed",
                "messages": [],
                "steps": [],
                "events": [
                    {
                        "event_type": "tool.completed",
                        "timestamp": "2026-05-16T00:00:02+00:00",
                        "data": {
                            "tool_name": "review_changes",
                            "tool_call_id": "call_review",
                            "status": "success",
                            "result_preview": "ok",
                        },
                    }
                ],
            },
            {
                "trace_id": "child",
                "parent_run_id": "parent",
                "run_name": "review_changes",
                "agent_name": "reviewer",
                "status": "completed",
                "messages": [],
                "steps": [],
                "events": [],
                "final_output": "No issues.",
            },
        ]
    )

    parent = next(run for run in runs if run["trace_id"] == "parent")
    child = next(run for run in runs if run["trace_id"] == "child")

    assert parent["display"]["child_count"] == 1
    assert child["display"]["relationship"] == "child"
    assert child["display"]["parent_title"] == "main"
    assert parent["display"]["activities"][0]["type"] == "subagent_call"
    assert parent["display"]["activities"][0]["schema_version"] == 1
    assert parent["display"]["activities"][0]["event_types"] == ["tool.completed"]
    assert parent["display"]["activities"][0]["child_run"]["trace_id"] == "child"


def test_viewer_imports_jsonl_and_accepts_live_events(tmp_path):
    trace_file = tmp_path / "viewer.jsonl"
    try:
        viewer = ai.tracing.start_viewer(trace_file, port=0)
    except PermissionError:
        pytest.skip("Local socket binding is not permitted in this environment")
    try:
        started = ai.tracing.TraceEvent(
            event_type="run.started",
            trace_id="trace_live",
            agent_name="assistant",
            run_name="live",
        ).to_dict()
        completed = ai.tracing.TraceEvent(
            event_type="run.completed",
            trace_id="trace_live",
            agent_name="assistant",
            run_name="live",
            data={
                "run": {
                    "trace_id": "trace_live",
                    "agent_name": "assistant",
                    "run_name": "live",
                    "status": "completed",
                    "messages": [],
                    "steps": [],
                    "tags": [],
                    "metadata": {},
                    "final_output": "ok",
                }
            },
        ).to_dict()

        request = Request(
            f"{viewer.url}/api/import-jsonl",
            data=(json.dumps(started) + "\n").encode("utf-8"),
            method="POST",
        )
        with urlopen(request, timeout=5) as response:
            imported = json.loads(response.read().decode("utf-8"))

        request = Request(
            f"{viewer.url}/api/events",
            data=json.dumps(completed).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=5) as response:
            posted = json.loads(response.read().decode("utf-8"))

        with urlopen(f"{viewer.url}/api/runs/trace_live", timeout=5) as response:
            run_payload = json.loads(response.read().decode("utf-8"))
        with urlopen(f"{viewer.url}/api/events/trace_live", timeout=5) as response:
            events_payload = json.loads(response.read().decode("utf-8"))
    finally:
        viewer.stop()

    assert imported["imported"] == 1
    assert posted == {"ok": True}
    assert run_payload["run"]["final_output"] == "ok"
    assert [event["event_type"] for event in events_payload["events"]] == [
        "run.started",
        "run.completed",
    ]
    assert len(trace_file.read_text(encoding="utf-8").splitlines()) == 2


def test_viewer_html_renders_run_transcript_sections():
    assert "Final Output" in VIEWER_HTML
    assert "Transcript" in VIEWER_HTML
    assert "message_count" in VIEWER_HTML
    assert "Events" in VIEWER_HTML
    assert "Details" in VIEWER_HTML
    assert "eventSummary" in VIEWER_HTML
    assert "result_preview" in VIEWER_HTML
    assert "latestToolActivity" in VIEWER_HTML
    assert "Search runs, tools, traces" in VIEWER_HTML
    assert "Timeline" in VIEWER_HTML
    assert "renderTimelineItem" in VIEWER_HTML


def test_viewer_ui_source_exposes_operational_scan_controls():
    source = (
        Path(__file__).resolve().parents[2] / "viewer-ui" / "src" / "App.jsx"
    ).read_text(encoding="utf-8")

    assert "Copy focused link" in source
    assert "TIMELINE_FILTERS" in source
    for label in ["All", "Model", "Tools", "Approvals", "Errors", "Subagents"]:
        assert f'label: "{label}"' in source
    assert "ops-summary-item" in source
    assert "?embed=1" not in source  # URL is built through URLSearchParams
    assert 'url.searchParams.set("embed", "1")' in source


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
        "model.send",
        "model.response",
        "run.completed",
    ]
    json.dumps({"runs": runs})


def test_prepare_viewer_runs_adds_display_model():
    runs = prepare_viewer_runs(
        [
            {
                "trace_id": "trace_1",
                "run_name": "run",
                "agent_name": "assistant",
                "status": "completed",
                "group_id": "group",
                "messages": [],
                "steps": [],
                "events": [
                    {
                        "event_type": "run.started",
                        "timestamp": "2026-05-16T00:00:00+00:00",
                        "data": {"input": "hello", "model": "openai:gpt-4o-mini"},
                    },
                    {
                        "event_type": "model.send",
                        "timestamp": "2026-05-16T00:00:00.500000+00:00",
                        "data": {
                            "model": "openai:gpt-4o-mini",
                            "input": {
                                "message_count": 1,
                                "modalities": ["text"],
                                "items": [
                                    {
                                        "type": "user_message",
                                        "role": "user",
                                        "modalities": ["text"],
                                        "text_preview": "hello",
                                        "text_length": 5,
                                        "truncated": False,
                                    }
                                ],
                            },
                        },
                    },
                    {
                        "event_type": "model.response",
                        "timestamp": "2026-05-16T00:00:00.750000+00:00",
                        "data": {
                            "model": "openai:gpt-4o-mini",
                            "usage": {
                                "input_tokens": 1200,
                                "output_tokens": 25,
                                "total_tokens": 1225,
                            },
                            "response": {
                                "kind": "tool_calls",
                                "modalities": ["tool_call"],
                                "tool_call_count": 1,
                                "tool_calls": [
                                    {
                                        "id": "call_1",
                                        "name": "read_file_lines",
                                        "arguments": {
                                            "path": "tests/toolkits/test_shell.py"
                                        },
                                    }
                                ],
                                "finish_reason": "tool_calls",
                            },
                        },
                    },
                    {
                        "event_type": "tool.allowed",
                        "timestamp": "2026-05-16T00:00:01+00:00",
                        "data": {
                            "tool_name": "read_file_lines",
                            "arguments": {"path": "tests/toolkits/test_shell.py"},
                            "allowed": True,
                            "reason": "low risk",
                        },
                    },
                    {
                        "event_type": "tool.completed",
                        "timestamp": "2026-05-16T00:00:02+00:00",
                        "data": {
                            "tool_name": "read_file_lines",
                            "status": "success",
                            "result_preview": (
                                '{"path": "tests/toolkits/test_shell.py", '
                                '"content": "from unittest.mock import Mock"}'
                            ),
                        },
                    },
                    {
                        "event_type": "run.completed",
                        "timestamp": "2026-05-16T00:00:03+00:00",
                        "data": {},
                    },
                ],
            }
        ]
    )

    display = runs[0]["display"]
    assert display["title"] == "run"
    assert display["duration"] == "3.0 s"
    assert display["model"] == "openai:gpt-4o-mini"
    assert display["usage"] == {
        "input_tokens": 1200,
        "output_tokens": 25,
        "total_tokens": 1225,
        "model_call_count": 1,
    }
    assert display["tools"] == ["read_file_lines"]
    assert display["tool_count"] == 2
    assert display["approval_count"] == 1
    assert (
        display["timeline"][2]["summary"]
        == "tool_calls · read_file_lines (call_1) · 1.2k tok · 1.2k in · 25 out"
    )
    assert display["timeline"][2]["usage"]["total_tokens"] == 1225
    assert display["timeline"][4]["result"]["path"] == "tests/toolkits/test_shell.py"
    assert [activity["type"] for activity in display["activities"]] == [
        "event",
        "model_call",
        "tool_call",
        "event",
    ]
    model_activity = display["activities"][1]
    assert model_activity["schema_version"] == 1
    assert model_activity["event_count"] == 2
    assert model_activity["event_types"] == ["model.send", "model.response"]
    assert model_activity["duration_ms"] == 250
    assert model_activity["usage"]["total_tokens"] == 1225
    assert model_activity["response"]["tool_calls"][0]["name"] == "read_file_lines"
    tool_activity = display["activities"][2]
    assert tool_activity["schema_version"] == 1
    assert tool_activity["event_count"] == 2
    assert tool_activity["event_types"] == [
        "tool.allowed",
        "tool.completed",
    ]
    assert tool_activity["tool_name"] == "read_file_lines"
    assert tool_activity["status"] == "completed"
    assert tool_activity["approval"] == {
        "required": False,
        "allowed": True,
        "reason": "low risk",
    }
    assert tool_activity["duration_ms"] == 1000
    assert tool_activity["result"]["path"] == "tests/toolkits/test_shell.py"
    assert display["latency"]["model_call_count"] == 1
    assert display["latency"]["tool_call_count"] == 1
    assert display["latency"]["slowest_model_ms"] == 250
    assert display["latency"]["slowest_model"]["model"] == "openai:gpt-4o-mini"
    assert display["latency"]["slowest_tool_ms"] == 1000
    assert display["latency"]["slowest_tool"]["tool_name"] == "read_file_lines"


def test_prepare_viewer_runs_builds_tool_activity_for_failed_approved_tool():
    runs = prepare_viewer_runs(
        [
            {
                "trace_id": "trace_1",
                "run_name": "run",
                "agent_name": "assistant",
                "status": "failed",
                "messages": [],
                "steps": [],
                "events": [
                    {
                        "event_type": "tool.allowed",
                        "timestamp": "2026-05-16T00:00:00+00:00",
                        "data": {
                            "tool_name": "apply_patch",
                            "tool_call_id": "call_patch",
                            "arguments": {"patch": "*** Begin Patch\n*** End Patch"},
                            "allowed": True,
                            "reason": "approved by user",
                            "tool_metadata": {
                                "requires_approval": True,
                            },
                        },
                    },
                    {
                        "event_type": "tool.started",
                        "timestamp": "2026-05-16T00:00:01+00:00",
                        "data": {
                            "tool_name": "apply_patch",
                            "tool_call_id": "call_patch",
                            "arguments": {"patch": "*** Begin Patch\n*** End Patch"},
                        },
                    },
                    {
                        "event_type": "tool.failed",
                        "timestamp": "2026-05-16T00:00:03.500000+00:00",
                        "data": {
                            "tool_name": "apply_patch",
                            "tool_call_id": "call_patch",
                            "status": "failed",
                            "error": "Patch contains no file operations.",
                        },
                    },
                ],
            }
        ]
    )

    activity = runs[0]["display"]["activities"][0]
    assert activity["type"] == "tool_call"
    assert activity["tool_call_id"] == "call_patch"
    assert activity["status"] == "failed"
    assert activity["duration_ms"] == 2500
    assert activity["approval"] == {
        "required": True,
        "allowed": True,
        "reason": "approved by user",
    }
    assert activity["error"] == "Patch contains no file operations."


def test_prepare_viewer_runs_summarizes_multiple_model_tool_calls():
    runs = prepare_viewer_runs(
        [
            {
                "trace_id": "trace_1",
                "run_name": "run",
                "agent_name": "assistant",
                "status": "running",
                "messages": [],
                "steps": [],
                "events": [
                    {
                        "event_type": "model.response",
                        "timestamp": "2026-05-16T00:00:00+00:00",
                        "data": {
                            "model": "openai:gpt-5.5",
                            "response": {
                                "kind": "mixed",
                                "text_preview": "I will inspect and test this.",
                                "tool_call_count": 2,
                                "tool_calls": [
                                    {"id": "call_list", "name": "list_files"},
                                    {"id": "call_test", "name": "run_shell"},
                                ],
                            },
                        },
                    }
                ],
            }
        ]
    )

    assert runs[0]["display"]["timeline"][0]["summary"] == (
        "I will inspect and test this. · "
        "list_files (call_list), run_shell (call_test)"
    )


def test_prepare_viewer_runs_summarizes_coding_tool_events():
    runs = prepare_viewer_runs(
        [
            {
                "trace_id": "trace_1",
                "run_name": "run",
                "agent_name": "assistant",
                "status": "completed",
                "messages": [],
                "steps": [],
                "events": [
                    {
                        "event_type": "tool.allowed",
                        "timestamp": "2026-05-16T00:00:00+00:00",
                        "data": {
                            "tool_name": "write_file",
                            "arguments": {
                                "path": "src/App.jsx",
                                "content": "one\ntwo\nthree\n",
                                "overwrite": True,
                            },
                            "allowed": True,
                            "reason": "approved",
                        },
                    },
                    {
                        "event_type": "tool.completed",
                        "timestamp": "2026-05-16T00:00:01+00:00",
                        "data": {
                            "tool_name": "run_shell",
                            "status": "success",
                            "result_preview": (
                                '{"command": "npm run build", "exit_code": 0, '
                                '"timed_out": false, "stdout": "built"}'
                            ),
                        },
                    },
                    {
                        "event_type": "tool.completed",
                        "timestamp": "2026-05-16T00:00:02+00:00",
                        "data": {
                            "tool_name": "apply_unified_diff",
                            "status": "success",
                            "result_preview": (
                                '{"changed_files": ["src/App.jsx"], '
                                '"added_files": [], "deleted_files": [], '
                                '"file_count": 1, "hunk_count": 2}'
                            ),
                        },
                    },
                ],
            }
        ]
    )

    timeline = runs[0]["display"]["timeline"]
    assert timeline[0]["summary"] == (
        "write_file(write src/App.jsx · 14 chars · 3 lines) · allowed · approved"
    )
    assert timeline[1]["summary"] == "run_shell · success · npm run build · exit 0"
    assert timeline[2]["summary"] == (
        "apply_unified_diff · success · 1 files · 2 hunks · src/App.jsx"
    )


def test_prepare_viewer_runs_marks_parent_and_child_runs():
    runs = prepare_viewer_runs(
        [
            {
                "trace_id": "parent",
                "run_name": "writer",
                "agent_name": "writer",
                "status": "completed",
                "messages": [],
                "steps": [],
                "events": [],
            },
            {
                "trace_id": "child",
                "parent_run_id": "parent",
                "run_name": "researcher",
                "agent_name": "researcher",
                "status": "completed",
                "messages": [],
                "steps": [],
                "events": [],
            },
        ]
    )

    parent = next(run for run in runs if run["trace_id"] == "parent")
    child = next(run for run in runs if run["trace_id"] == "child")
    assert parent["display"]["relationship"] == "root"
    assert parent["display"]["child_count"] == 1
    assert child["display"]["relationship"] == "child"
    assert child["display"]["parent_title"] == "writer"


def test_prepare_viewer_runs_links_subagent_child_to_parent_tool():
    runs = prepare_viewer_runs(
        [
            {
                "trace_id": "parent",
                "run_name": "writer",
                "agent_name": "writer",
                "status": "completed",
                "messages": [],
                "steps": [],
                "events": [
                    {
                        "event_type": "tool.completed",
                        "timestamp": "2026-05-16T00:00:03+00:00",
                        "data": {
                            "tool_name": "review_changes",
                            "status": "success",
                            "result_preview": '"No issues found"',
                        },
                    }
                ],
            },
            {
                "trace_id": "child",
                "parent_run_id": "parent",
                "run_name": "review_changes",
                "agent_name": "reviewer",
                "status": "completed",
                "final_output": "No issues found",
                "messages": [],
                "steps": [],
                "events": [
                    {
                        "event_type": "run.started",
                        "timestamp": "2026-05-16T00:00:01+00:00",
                        "data": {},
                    },
                    {
                        "event_type": "run.completed",
                        "timestamp": "2026-05-16T00:00:02+00:00",
                        "data": {},
                    },
                ],
            },
        ]
    )

    parent = next(run for run in runs if run["trace_id"] == "parent")
    child_run = parent["display"]["timeline"][0]["child_run"]
    assert child_run["trace_id"] == "child"
    assert child_run["agent_name"] == "reviewer"
    assert child_run["duration"] == "1.0 s"
    assert child_run["final_output_preview"] == "No issues found"
