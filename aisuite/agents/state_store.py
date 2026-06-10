from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Protocol
from urllib.parse import quote

from .types import RunState, ensure_json_serializable
from .utils import now, new_id


class StateConflictError(RuntimeError):
    """Raised when a state write loses an optimistic concurrency check."""


@dataclass(kw_only=True)
class StoredRunState:
    thread_id: str
    state: RunState
    revision: int
    created_at: str
    updated_at: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return ensure_json_serializable(
            {
                "schema_version": 1,
                "thread_id": self.thread_id,
                "state": self.state.to_dict(),
                "revision": self.revision,
                "created_at": self.created_at,
                "updated_at": self.updated_at,
                "metadata": copy.deepcopy(self.metadata),
            }
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StoredRunState":
        return cls(
            thread_id=data["thread_id"],
            state=RunState.from_dict(data["state"]),
            revision=data["revision"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            metadata=copy.deepcopy(data.get("metadata", {})),
        )


class StateStore(Protocol):
    def save_state(
        self,
        thread_id: str,
        state: RunState,
        *,
        revision: int | None = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> StoredRunState: ...

    def load_state(self, thread_id: str) -> Optional[StoredRunState]: ...

    def delete_state(self, thread_id: str) -> None: ...


class InMemoryStateStore:
    def __init__(self):
        self._states: dict[str, StoredRunState] = {}

    def save_state(
        self,
        thread_id: str,
        state: RunState,
        *,
        revision: int | None = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> StoredRunState:
        current = self._states.get(thread_id)
        _assert_revision(thread_id, current.revision if current else None, revision)
        stored = _next_stored_state(
            thread_id,
            state,
            current=current,
            metadata=metadata,
        )
        self._states[thread_id] = StoredRunState.from_dict(stored.to_dict())
        return StoredRunState.from_dict(stored.to_dict())

    def load_state(self, thread_id: str) -> Optional[StoredRunState]:
        stored = self._states.get(thread_id)
        if stored is None:
            return None
        return StoredRunState.from_dict(stored.to_dict())

    def delete_state(self, thread_id: str) -> None:
        self._states.pop(thread_id, None)


class FileStateStore:
    def __init__(self, root: str | Path = ".aisuite/state"):
        self.root = Path(root)

    def save_state(
        self,
        thread_id: str,
        state: RunState,
        *,
        revision: int | None = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> StoredRunState:
        current = self.load_state(thread_id)
        _assert_revision(thread_id, current.revision if current else None, revision)
        stored = _next_stored_state(
            thread_id,
            state,
            current=current,
            metadata=metadata,
        )
        self._write_stored_state(stored)
        return stored

    def load_state(self, thread_id: str) -> Optional[StoredRunState]:
        path = self._path_for(thread_id)
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as handle:
            return StoredRunState.from_dict(json.load(handle))

    def delete_state(self, thread_id: str) -> None:
        path = self._path_for(thread_id)
        try:
            path.unlink()
        except FileNotFoundError:
            pass

    def _write_stored_state(self, stored: StoredRunState) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self._path_for(stored.thread_id)
        tmp_path = path.with_name(f".{path.name}.{new_id('tmp')}.tmp")
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(stored.to_dict(), handle, sort_keys=True)
            handle.write("\n")
        os.replace(tmp_path, path)

    def _path_for(self, thread_id: str) -> Path:
        return self.root / f"{quote(thread_id, safe='')}.json"


def _next_stored_state(
    thread_id: str,
    state: RunState,
    *,
    current: Optional[StoredRunState],
    metadata: Optional[dict[str, Any]],
) -> StoredRunState:
    timestamp = now()
    return StoredRunState(
        thread_id=thread_id,
        state=RunState.from_dict(state.to_dict()),
        revision=(current.revision + 1) if current else 1,
        created_at=current.created_at if current else timestamp,
        updated_at=timestamp,
        metadata=copy.deepcopy(
            metadata if metadata is not None else (current.metadata if current else {})
        ),
    )


def _assert_revision(
    thread_id: str,
    current_revision: Optional[int],
    expected_revision: Optional[int],
) -> None:
    if expected_revision is None:
        return
    if current_revision != expected_revision:
        raise StateConflictError(
            f"State revision conflict for {thread_id!r}: "
            f"expected {expected_revision}, found {current_revision}."
        )
