from __future__ import annotations

import copy
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional, Protocol

from .store import JsonlTraceStore


TRACE_SCHEMA_VERSION = "2026-05-15"


TraceEventType = Literal[
    "run.started",
    "run.completed",
    "run.failed",
    "model.started",
    "model.completed",
    "tool.allowed",
    "tool.denied",
    "tool.completed",
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


@dataclass
class TraceEvent:
    event_type: TraceEventType
    trace_id: str
    agent_name: str
    event_id: str = field(default_factory=lambda: _new_id("event"))
    timestamp: str = field(default_factory=_now)
    span_id: Optional[str] = None
    parent_span_id: Optional[str] = None
    parent_run_id: Optional[str] = None
    group_id: Optional[str] = None
    run_name: Optional[str] = None
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_type": "trace_event",
            "schema_version": TRACE_SCHEMA_VERSION,
            "event_id": self.event_id,
            "event_type": self.event_type,
            "timestamp": self.timestamp,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "parent_run_id": self.parent_run_id,
            "group_id": self.group_id,
            "run_name": self.run_name,
            "agent_name": self.agent_name,
            "tags": copy.deepcopy(self.tags),
            "metadata": copy.deepcopy(self.metadata),
            "data": copy.deepcopy(self.data),
        }


class TraceSink(Protocol):
    def emit(self, event: TraceEvent) -> None:
        ...


class LocalTraceSink:
    def __init__(self, path: str | Path = ".aisuite/events.jsonl"):
        self.path = Path(path)
        self.store = JsonlTraceStore(self.path)

    def emit(self, event: TraceEvent) -> None:
        self.store.write_event(event)


class InMemoryTraceSink:
    def __init__(self):
        self.events: list[TraceEvent] = []

    def emit(self, event: TraceEvent) -> None:
        self.events.append(event)


_configured_sinks: list[TraceSink] = []


def configure(*sinks: TraceSink | None) -> None:
    global _configured_sinks
    _configured_sinks = [sink for sink in sinks if sink is not None]


def get_configured_sinks() -> list[TraceSink]:
    return list(_configured_sinks)


def emit_event(sinks: list[TraceSink], event: TraceEvent) -> None:
    for sink in sinks:
        sink.emit(event)
