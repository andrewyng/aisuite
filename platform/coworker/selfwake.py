"""Self-wake — tools that let a long-running agent suspend and be re-invoked on a trigger.

Converts an always-on agent into suspend/resume (event-driven, ~zero idle cost): the session
sleeps and the runtime re-invokes it when a wake is due. Two triggers here: a **timer**
(`sleep_for` / `sleep_until`) and **on-completion** (`wake_on` a backgrounded job). This module
owns the wake records + the due/complete logic; the scheduler tick consumes ``due()`` /
``complete_job()`` and resumes the session (shares the automation scheduler — see
``PERMISSIONS-AND-INBOX.md``).
"""

from __future__ import annotations

import json
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

KIND_TIMER = "timer"
KIND_COMPLETION = "completion"

STATE_PENDING = "pending"
STATE_DUE = "due"
STATE_FIRED = "fired"


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class Wake:
    id: str
    session_id: str
    kind: str
    state: str = STATE_PENDING
    fire_at: Optional[str] = None  # ISO, for timer wakes
    job_id: Optional[str] = None  # for completion wakes
    note: str = ""
    created_at: str = field(default_factory=lambda: _now().isoformat())


class WakeStore:
    def __init__(self, path: Optional[str | Path] = None) -> None:
        self.path = Path(path) if path else None
        self._lock = threading.Lock()
        self._wakes: dict[str, Wake] = {}
        if self.path and self.path.is_file():
            for raw in json.loads(self.path.read_text(encoding="utf-8")).get("wakes", []):
                w = Wake(**raw)
                self._wakes[w.id] = w

    def _save(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({"wakes": [asdict(w) for w in self._wakes.values()]}, indent=2),
            encoding="utf-8",
        )

    def add_timer(self, session_id: str, fire_at: datetime, *, note: str = "") -> Wake:
        w = Wake(uuid.uuid4().hex, session_id, KIND_TIMER, fire_at=fire_at.isoformat(), note=note)
        with self._lock:
            self._wakes[w.id] = w
            self._save()
        return w

    def add_completion(self, session_id: str, job_id: str, *, note: str = "") -> Wake:
        w = Wake(uuid.uuid4().hex, session_id, KIND_COMPLETION, job_id=job_id, note=note)
        with self._lock:
            self._wakes[w.id] = w
            self._save()
        return w

    def due(self, now: Optional[datetime] = None) -> list[Wake]:
        """Timer wakes whose fire time has passed (and completion wakes already marked due)."""
        now = now or _now()
        out = []
        for w in self._wakes.values():
            if w.state != STATE_PENDING and w.state != STATE_DUE:
                continue
            if w.kind == KIND_TIMER and w.fire_at and datetime.fromisoformat(w.fire_at) <= now:
                out.append(w)
            elif w.kind == KIND_COMPLETION and w.state == STATE_DUE:
                out.append(w)
        return out

    def complete_job(self, job_id: str) -> list[Wake]:
        """Mark completion wakes for ``job_id`` as due (the job exited). Returns them."""
        fired = []
        with self._lock:
            for w in self._wakes.values():
                if w.kind == KIND_COMPLETION and w.job_id == job_id and w.state == STATE_PENDING:
                    w.state = STATE_DUE
                    fired.append(w)
            if fired:
                self._save()
        return fired

    def mark_fired(self, wake_id: str) -> None:
        with self._lock:
            w = self._wakes.get(wake_id)
            if w is not None:
                w.state = STATE_FIRED
                self._save()

    def pending(self, session_id: Optional[str] = None) -> list[Wake]:
        return [
            w
            for w in self._wakes.values()
            if w.state != STATE_FIRED and (session_id is None or w.session_id == session_id)
        ]


def selfwake_tools(store: WakeStore, session_id: str) -> list:
    """Tools an agent calls to schedule its own resumption."""

    def sleep_for(seconds: int, note: str = "") -> dict:
        """Suspend and wake this session after `seconds`. Use for polling/waiting without
        burning context while idle."""
        w = store.add_timer(session_id, _now() + timedelta(seconds=int(seconds)), note=note)
        return {"ok": True, "wake_id": w.id, "fire_at": w.fire_at}

    def sleep_until(when_iso: str, note: str = "") -> dict:
        """Suspend and wake this session at an ISO-8601 timestamp."""
        when = datetime.fromisoformat(when_iso)
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        w = store.add_timer(session_id, when, note=note)
        return {"ok": True, "wake_id": w.id, "fire_at": w.fire_at}

    def wake_on(job_id: str, note: str = "") -> dict:
        """Suspend and wake this session when a backgrounded job (`job_id`) completes."""
        w = store.add_completion(session_id, job_id, note=note)
        return {"ok": True, "wake_id": w.id, "job_id": job_id}

    return [sleep_for, sleep_until, wake_on]
