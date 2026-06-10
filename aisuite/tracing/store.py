from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Optional, Protocol


class TraceStore(Protocol):
    """Storage contract used by the local viewer and trace sinks.

    Implementations may store raw trace_event records, legacy run snapshots,
    or both. Read methods return reconstructed run dictionaries and raw event
    records so the viewer API does not depend on the physical storage backend.
    """

    def append_event(self, event: Any) -> None: ...

    def append_events(self, events: list[Any]) -> None: ...

    def append_record(self, record: dict[str, Any]) -> None: ...

    def append_records(self, records: list[dict[str, Any]]) -> None: ...

    def import_jsonl(self, content: str) -> int: ...

    def list_records(self) -> list[dict[str, Any]]: ...

    def list_runs(self) -> list[dict[str, Any]]: ...

    def get_run(self, trace_id: str) -> Optional[dict[str, Any]]: ...

    def list_events(self, trace_id: str) -> list[dict[str, Any]]: ...


class JsonlTraceStore:
    """TraceStore implementation backed by a local JSONL file."""

    def __init__(self, path: str | Path = ".aisuite/events.jsonl"):
        self.path = Path(path)

    def append_event(self, event: Any) -> None:
        self.append_record(event.to_dict())

    def append_events(self, events: list[Any]) -> None:
        self.append_records([event.to_dict() for event in events])

    def append_record(self, record: dict[str, Any]) -> None:
        self.append_records([record])

    def append_records(self, records: list[dict[str, Any]]) -> None:
        if not records:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")

    def write_event(self, event: Any) -> None:
        self.append_event(event)

    def write_record(self, record: dict[str, Any]) -> None:
        self.append_record(record)

    def import_jsonl(self, content: str) -> int:
        records = parse_jsonl_records(content)
        self.append_records(records)
        return len(records)

    def list_records(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []

        records = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return records

    def list_runs(self) -> list[dict[str, Any]]:
        return reconstruct_runs(self.list_records())

    def get_run(self, trace_id: str) -> Optional[dict[str, Any]]:
        for run in self.list_runs():
            if run.get("trace_id") == trace_id:
                return run
        return None

    def list_events(self, trace_id: str) -> list[dict[str, Any]]:
        return [
            record
            for record in self.list_records()
            if record.get("record_type") == "trace_event"
            and record.get("trace_id") == trace_id
        ]


class InMemoryTraceStore:
    """TraceStore implementation backed by an in-memory record list."""

    def __init__(self, records: Optional[list[dict[str, Any]]] = None):
        self.records = list(records or [])

    def append_event(self, event: Any) -> None:
        self.append_record(event.to_dict())

    def append_events(self, events: list[Any]) -> None:
        self.append_records([event.to_dict() for event in events])

    def append_record(self, record: dict[str, Any]) -> None:
        self.append_records([record])

    def append_records(self, records: list[dict[str, Any]]) -> None:
        self.records.extend(copy.deepcopy(records))

    def import_jsonl(self, content: str) -> int:
        records = parse_jsonl_records(content)
        self.append_records(records)
        return len(records)

    def list_records(self) -> list[dict[str, Any]]:
        return copy.deepcopy(self.records)

    def list_runs(self) -> list[dict[str, Any]]:
        return reconstruct_runs(self.list_records())

    def get_run(self, trace_id: str) -> Optional[dict[str, Any]]:
        for run in self.list_runs():
            if run.get("trace_id") == trace_id:
                return run
        return None

    def list_events(self, trace_id: str) -> list[dict[str, Any]]:
        return [
            record
            for record in self.list_records()
            if record.get("record_type") == "trace_event"
            and record.get("trace_id") == trace_id
        ]


def parse_jsonl_records(content: str) -> list[dict[str, Any]]:
    records = []
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        records.append(record)
    return records


def reconstruct_runs(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    snapshots = []
    events_by_trace_id: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        if record.get("record_type") == "trace_event":
            trace_id = record.get("trace_id")
            if trace_id:
                events_by_trace_id.setdefault(trace_id, []).append(
                    copy.deepcopy(record)
                )
            continue
        snapshots.append(copy.deepcopy(record))

    runs_by_trace_id = {
        snapshot.get("trace_id"): dict(snapshot)
        for snapshot in snapshots
        if snapshot.get("trace_id")
    }
    for trace_id, events in events_by_trace_id.items():
        run = runs_by_trace_id.setdefault(trace_id, _empty_run(trace_id))
        run["events"] = events
        for event in events:
            _merge_event_fields(run, event)
            if event.get("event_type") == "run.completed":
                completed = event.get("data", {}).get("run")
                if completed:
                    run_snapshot = dict(completed)
                    run_snapshot["events"] = events
                    runs_by_trace_id[trace_id] = run_snapshot
                    run = run_snapshot
                else:
                    run["status"] = "completed"
            elif event.get("event_type") == "run.failed":
                run["status"] = "failed"
            elif event.get("event_type") == "run.started":
                run["status"] = run.get("status") or "running"

        run.setdefault("event_count", len(events))

    for run in runs_by_trace_id.values():
        events = events_by_trace_id.get(run.get("trace_id"), [])
        run.setdefault("events", events)
        run.setdefault("event_count", len(events))
        run.setdefault("message_count", len(run.get("messages", [])))
        run.setdefault("step_count", len(run.get("steps", [])))
        run.setdefault("status", "running")

    return sorted(
        runs_by_trace_id.values(),
        key=lambda run: _run_sort_key(run),
        reverse=True,
    )


def _empty_run(trace_id: str) -> dict[str, Any]:
    return {
        "trace_id": trace_id,
        "status": "running",
        "steps": [],
        "messages": [],
        "new_items": [],
        "tags": [],
        "metadata": {},
    }


def _merge_event_fields(run: dict[str, Any], event: dict[str, Any]) -> None:
    run["group_id"] = run.get("group_id") or event.get("group_id")
    run["run_name"] = run.get("run_name") or event.get("run_name")
    run["agent_name"] = run.get("agent_name") or event.get("agent_name")
    run["parent_run_id"] = run.get("parent_run_id") or event.get("parent_run_id")
    run["tags"] = run.get("tags") or event.get("tags", [])
    run["metadata"] = run.get("metadata") or event.get("metadata", {})


def _run_sort_key(run: dict[str, Any]) -> str:
    events = run.get("events", [])
    if events:
        return events[-1].get("timestamp", "")
    return run.get("created_at") or run.get("started_at") or ""
