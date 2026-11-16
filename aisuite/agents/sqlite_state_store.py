from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from threading import Lock
from typing import Any

from .state_store import StoredRunState, _assert_revision, _next_stored_state
from .types import RunState


class SQLiteStateStore:
    """Store run snapshots in a SQLite database shared by local workers."""

    def __init__(self, path: str | Path):
        self._lock = Lock()
        self._connection = sqlite3.connect(path, check_same_thread=False)
        self._connection.execute(
            "CREATE TABLE IF NOT EXISTS aisuite_run_states "
            "(thread_id TEXT PRIMARY KEY, snapshot TEXT NOT NULL)"
        )

    def _load_state(self, thread_id: str) -> StoredRunState | None:
        row = self._connection.execute(
            "SELECT snapshot FROM aisuite_run_states WHERE thread_id = ?",
            (thread_id,),
        ).fetchone()
        return StoredRunState.from_dict(json.loads(row[0])) if row else None

    def load_state(self, thread_id: str) -> StoredRunState | None:
        with self._lock:
            return self._load_state(thread_id)

    def save_state(
        self,
        thread_id: str,
        state: RunState,
        *,
        revision: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> StoredRunState:
        with self._lock, self._connection:
            # Reserve the write before reading the revision, including for new threads.
            self._connection.execute("BEGIN IMMEDIATE")
            current = self._load_state(thread_id)
            _assert_revision(thread_id, current.revision if current else None, revision)
            stored = _next_stored_state(
                thread_id, state, current=current, metadata=metadata
            )
            self._connection.execute(
                "INSERT INTO aisuite_run_states (thread_id, snapshot) VALUES (?, ?) "
                "ON CONFLICT(thread_id) DO UPDATE SET snapshot = excluded.snapshot",
                (thread_id, json.dumps(stored.to_dict())),
            )
        return stored

    def delete_state(self, thread_id: str) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                "DELETE FROM aisuite_run_states WHERE thread_id = ?", (thread_id,)
            )

    def close(self) -> None:
        with self._lock:
            self._connection.close()
