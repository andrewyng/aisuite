"""Session record — the metadata + messages for one conversation.

Storage lives in `coworker.conversations.ConversationStore`: a SQLite index keyed by
project, with each conversation's messages in an append-only `.jsonl` file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class SessionRecord:
    session_id: str
    workspace: str
    model: str
    mode: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    title: Optional[str] = None
    agent: str = "code"
    message_count: int = 0
    updated_at: Optional[str] = None
    # Folders added to the session beyond its primary scratch dir, each {path, writable, label}.
    # The primary scratch is re-provisioned at engine build, so only these extras are persisted.
    extra_roots: list[dict[str, Any]] = field(default_factory=list)
    pinned: bool = False
    archived: bool = False
