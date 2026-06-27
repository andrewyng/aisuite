"""The Inbox — the canonical, cross-session human-attention queue.

While a user works in one session (or is away with a session running Unattended), the Inbox
holds what other agents need from them: an **approval**, a **question**, or a **notification**.
It is the store of record; messaging connectors / mobile (Phase 3) are transports of the same
items.

Item state machine (the anti-race contract): each item is ``pending → resolved``, resolved
**once**, idempotent + first-responder-wins — so answering from any surface (in-app, Slack, the
composer after resuming) is safe. ``inbox_approver`` turns a permission request into an item and
suspends the agent until that item is resolved.
"""

from __future__ import annotations

import asyncio
import json
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

KIND_APPROVAL = "approval"
KIND_QUESTION = "question"
KIND_NOTIFICATION = "notification"

STATE_PENDING = "pending"
STATE_RESOLVED = "resolved"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class InboxItem:
    id: str
    session_id: str
    kind: str
    title: str
    body: str = ""
    state: str = STATE_PENDING
    resolution: Optional[str] = None  # approval: "allow"/"deny"/"always"; question: answer text
    inbox: str = "default"  # named inbox / delivery binding (Phase 3 routing)
    created_at: str = field(default_factory=_now)
    resolved_at: Optional[str] = None


class InboxStore:
    def __init__(self, path: Optional[str | Path] = None) -> None:
        self.path = Path(path) if path else None
        self._lock = threading.Lock()
        self._items: dict[str, InboxItem] = {}
        self._waiters: dict[str, asyncio.Event] = {}
        self._load()

    # -- persistence ------------------------------------------------------------
    def _load(self) -> None:
        if self.path and self.path.is_file():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            for raw in data.get("items", []):
                item = InboxItem(**raw)
                self._items[item.id] = item

    def _save(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({"items": [asdict(i) for i in self._items.values()]}, indent=2),
            encoding="utf-8",
        )

    # -- adding -----------------------------------------------------------------
    def add(
        self, session_id: str, kind: str, title: str, *, body: str = "", inbox: str = "default"
    ) -> InboxItem:
        item = InboxItem(
            id=uuid.uuid4().hex, session_id=session_id, kind=kind, title=title,
            body=body, inbox=inbox,
        )
        with self._lock:
            self._items[item.id] = item
            self._save()
        return item

    def add_approval(self, session_id, title, *, body="", inbox="default") -> InboxItem:
        return self.add(session_id, KIND_APPROVAL, title, body=body, inbox=inbox)

    def add_question(self, session_id, title, *, body="", inbox="default") -> InboxItem:
        return self.add(session_id, KIND_QUESTION, title, body=body, inbox=inbox)

    def add_notification(self, session_id, title, *, body="", inbox="default") -> InboxItem:
        return self.add(session_id, KIND_NOTIFICATION, title, body=body, inbox=inbox)

    # -- queries ----------------------------------------------------------------
    def get(self, item_id: str) -> Optional[InboxItem]:
        return self._items.get(item_id)

    def list(
        self, *, session_id: Optional[str] = None, state: Optional[str] = None,
        inbox: Optional[str] = None,
    ) -> list[InboxItem]:
        out = list(self._items.values())
        if session_id is not None:
            out = [i for i in out if i.session_id == session_id]
        if state is not None:
            out = [i for i in out if i.state == state]
        if inbox is not None:
            out = [i for i in out if i.inbox == inbox]
        return sorted(out, key=lambda i: i.created_at)

    def pending(self, session_id: Optional[str] = None) -> list[InboxItem]:
        return self.list(session_id=session_id, state=STATE_PENDING)

    # -- the state machine ------------------------------------------------------
    def resolve(self, item_id: str, resolution: str) -> bool:
        """Resolve an item exactly once. First responder wins; later attempts are no-ops
        (return False). Fires any awaiting agent (the suspended inbox_approver)."""
        with self._lock:
            item = self._items.get(item_id)
            if item is None or item.state == STATE_RESOLVED:
                return False
            item.state = STATE_RESOLVED
            item.resolution = resolution
            item.resolved_at = _now()
            self._save()
        waiter = self._waiters.get(item_id)
        if waiter is not None:
            waiter.set()
        return True

    async def wait(self, item_id: str) -> str:
        """Await an item's resolution; returns the resolution string. Used by the approver to
        suspend the agent until a human answers (from any surface)."""
        item = self._items.get(item_id)
        if item is not None and item.state == STATE_RESOLVED:
            return item.resolution or ""
        ev = self._waiters.setdefault(item_id, asyncio.Event())
        await ev.wait()
        resolved = self._items.get(item_id)
        return (resolved.resolution if resolved else "") or ""

    # -- resume reconciliation --------------------------------------------------
    def reconcile_on_resume(self, session_id: str) -> dict:
        """When a user resumes attended control, surface this session's still-pending items
        inline (one place to answer from now on) plus a recap of what was answered while away.
        Single source of truth: every item already has one authoritative resolution."""
        pending = self.pending(session_id)
        recap = [i for i in self.list(session_id=session_id, state=STATE_RESOLVED)]
        return {
            "pending": [asdict(i) for i in pending],
            "recap": [asdict(i) for i in recap],
        }


# -- approver routing -----------------------------------------------------------
def inbox_approver(store: InboxStore, session_id: str, *, inbox: str = "default"):
    """An Approver that routes a permission request to the Inbox and suspends until resolved.
    Maps the resolution to an ApprovalOutcome (allow → ONCE, always → ALWAYS_TOOL, else DENY)."""
    from .engine import ApprovalOutcome, PermissionRequest

    async def approve(request: "PermissionRequest") -> "ApprovalOutcome":
        item = store.add_approval(
            session_id,
            title=f"Run `{request.tool_name}`?",
            body=request.reason or "",
            inbox=inbox,
        )
        resolution = await store.wait(item.id)
        if resolution == "always":
            return ApprovalOutcome.ALWAYS_TOOL
        if resolution == "allow":
            return ApprovalOutcome.ONCE
        return ApprovalOutcome.DENY

    return approve
