"""Automation data model — a scheduled task is its own persistent entity (see
docs/AUTOMATION-SCHEDULING.md). Each fire is a fresh Run of the task's instructions, recorded
in the task's own thread + working folder.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

_DOW = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def _now() -> float:
    return time.time()


def _human_time(hour: int, minute: int) -> str:
    ampm = "AM" if hour < 12 else "PM"
    h12 = hour % 12 or 12
    return f"{h12}:{minute:02d} {ampm}"


@dataclass
class Schedule:
    kind: str  # "cron" | "once"
    cron: Optional[str] = None
    fire_at: Optional[str] = None  # ISO datetime for one-time
    timezone: str = (
        "local"  # 'local' = the machine's clock (a local-first tool default)
    )

    def human(self) -> str:
        """Best-effort human label ('Every day at ~7:10 PM'); falls back to the raw cron."""
        if self.kind == "once":
            return f"Once at {self.fire_at}"
        parts = (self.cron or "").split()
        if len(parts) != 5:
            return self.cron or "?"
        minute, hour, dom, month, dow = parts
        try:
            t = _human_time(int(hour), int(minute))
        except ValueError:
            return self.cron  # non-trivial cron (ranges/steps) — show as-is
        if dom == "*" and dow == "*":
            return f"Every day at ~{t}"
        if dom == "*" and dow.isdigit():
            return f"Every {_DOW[int(dow) % 7]} at ~{t}"
        if dom.isdigit() and dow == "*":
            return f"Monthly on day {dom} at ~{t}"
        return self.cron

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "cron": self.cron,
            "fire_at": self.fire_at,
            "timezone": self.timezone,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Schedule":
        return cls(
            kind=d.get("kind", "cron"),
            cron=d.get("cron"),
            fire_at=d.get("fire_at"),
            timezone=d.get("timezone", "local"),
        )


@dataclass
class ScheduledTask:
    title: str
    instructions: str
    schedule: Schedule
    workspace: str
    origin_surface: str = "cowork"  # where it was launched from (a reference)
    origin_session_id: str = ""
    agent: str = "cowork"
    id: str = field(default_factory=lambda: "task-" + uuid.uuid4().hex[:10])
    task_session_id: str = ""  # the task's OWN thread (set to f"__task__{id}")
    model: Optional[str] = None
    notify_on_completion: bool = True
    notify_target: Optional[str] = None  # extra messaging target ("telegram:123")
    always_allowed_tools: list[str] = field(default_factory=list)
    always_allowed_commands: list[str] = field(default_factory=list)
    enabled: bool = True
    created_at: float = field(default_factory=_now)
    updated_at: float = field(default_factory=_now)
    next_run: Optional[float] = None  # epoch seconds; computed by the store
    last_run: Optional[float] = None
    last_status: Optional[str] = None
    run_count: int = 0
    max_runs: Optional[int] = None

    def __post_init__(self) -> None:
        if not self.task_session_id:
            self.task_session_id = f"__task__{self.id}"

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["schedule"] = self.schedule.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ScheduledTask":
        d = dict(d)
        d["schedule"] = Schedule.from_dict(d.get("schedule") or {})
        return cls(**d)

    def public(self) -> dict[str, Any]:
        """Status shape for the API/UI (no instructions truncation; never any secret)."""
        return {
            "id": self.id,
            "title": self.title,
            "instructions": self.instructions,
            "schedule": self.schedule.human(),
            "schedule_raw": self.schedule.to_dict(),
            "workspace": self.workspace,
            "agent": self.agent,
            "enabled": self.enabled,
            "next_run": self.next_run,
            "last_run": self.last_run,
            "last_status": self.last_status,
            "run_count": self.run_count,
            "notify_on_completion": self.notify_on_completion,
            "always_allowed": sorted(set(self.always_allowed_tools)),
        }


@dataclass
class TaskRun:
    task_id: str
    run_id: str = field(default_factory=lambda: "run-" + uuid.uuid4().hex[:10])
    started_at: float = field(default_factory=_now)
    finished_at: Optional[float] = None
    status: str = "running"  # running | ok | error | skipped
    result_text: Optional[str] = None
    artifacts: list[str] = field(default_factory=list)
    error: Optional[str] = None
    trigger: str = "schedule"  # schedule | manual | catchup
    session_id: str = ""  # the run's own conversation thread — persisted + continuable

    def __post_init__(self) -> None:
        if not self.session_id:
            self.session_id = f"__run__{self.run_id}"

    def to_dict(self) -> dict:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, d: dict) -> "TaskRun":
        return cls(**d)
