from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import Any, Optional

from .state_store import StateConflictError, StoredRunState, _assert_revision
from .types import RunState, ensure_json_serializable
from .utils import new_id

SCHEMA_STATEMENTS = [
    """
    create table if not exists agent_thread_heads (
      thread_id text primary key,
      model_context_message_ids text[] not null default '{}',
      full_history_message_ids text[] not null default '{}',
      step_ids text[] not null default '{}',
      compacted_from_message_ids text[] not null default '{}',
      state jsonb not null default '{}',
      revision bigint not null,
      metadata jsonb not null default '{}',
      created_at timestamptz not null default now(),
      updated_at timestamptz not null default now()
    )
    """,
    """
    create table if not exists agent_messages (
      message_id text primary key,
      thread_id text not null,
      role text,
      message jsonb not null,
      artifact_refs jsonb not null default '[]',
      created_at timestamptz not null default now()
    )
    """,
    """
    create index if not exists agent_messages_thread_id_idx
      on agent_messages (thread_id)
    """,
    """
    create table if not exists agent_compactions (
      compaction_id text primary key,
      thread_id text not null,
      source_message_ids text[] not null,
      summary_message_id text not null,
      summary_text text not null,
      reason text,
      model text,
      input_token_count bigint,
      output_token_count bigint,
      created_at timestamptz not null default now(),
      metadata jsonb not null default '{}'
    )
    """,
    """
    create index if not exists agent_compactions_thread_id_idx
      on agent_compactions (thread_id)
    """,
]


@dataclass(kw_only=True)
class CompactionRecord:
    compaction_id: str
    thread_id: str
    source_message_ids: list[str]
    summary_message_id: str
    summary_text: str
    reason: Optional[str] = None
    model: Optional[str] = None
    input_token_count: Optional[int] = None
    output_token_count: Optional[int] = None
    created_at: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return ensure_json_serializable(
            {
                "compaction_id": self.compaction_id,
                "thread_id": self.thread_id,
                "source_message_ids": copy.deepcopy(self.source_message_ids),
                "summary_message_id": self.summary_message_id,
                "summary_text": self.summary_text,
                "reason": self.reason,
                "model": self.model,
                "input_token_count": self.input_token_count,
                "output_token_count": self.output_token_count,
                "created_at": self.created_at,
                "metadata": copy.deepcopy(self.metadata),
            }
        )


class PostgresStateStore:
    def __init__(self, connection: Any, *, create_schema: bool = False):
        self.connection = connection
        if create_schema:
            self.create_schema()

    @classmethod
    def from_dsn(cls, dsn: str, *, create_schema: bool = False) -> "PostgresStateStore":
        try:
            import psycopg
        except ImportError as exc:
            raise ImportError(
                "PostgresStateStore.from_dsn() requires psycopg. "
                "Install psycopg or pass an existing connection."
            ) from exc
        return cls(psycopg.connect(dsn), create_schema=create_schema)

    def create_schema(self) -> None:
        cursor = self.connection.cursor()
        try:
            for statement in SCHEMA_STATEMENTS:
                cursor.execute(statement)
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise
        finally:
            cursor.close()

    def save_state(
        self,
        thread_id: str,
        state: RunState,
        *,
        revision: int | None = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> StoredRunState:
        cursor = self.connection.cursor()
        try:
            head = self._load_head_for_update(cursor, thread_id)
            current_revision = head[0] if head else None
            _assert_revision(thread_id, current_revision, revision)

            messages = copy.deepcopy(state.messages)
            state_payload = _state_payload_without_messages(state)
            stored_metadata = (
                metadata
                if metadata is not None
                else (_decode_json(head[5]) if head else {})
            )

            if head:
                old_context_ids = list(head[1] or [])
                old_full_history_ids = list(head[2] or [])
                old_compacted_from_ids = list(head[3] or [])
                old_context_messages = self._load_messages(
                    cursor, thread_id, old_context_ids
                )
                shared_count = _shared_message_prefix_length(
                    old_context_messages, messages
                )
                suffix_messages = messages[shared_count:]
                suffix_ids = self._insert_messages(cursor, thread_id, suffix_messages)
                message_ids = [*old_context_ids[:shared_count], *suffix_ids]
                full_history_ids = [*old_full_history_ids, *suffix_ids]
                cursor.execute(
                    """
                    update agent_thread_heads
                    set model_context_message_ids = %s,
                        full_history_message_ids = %s,
                        step_ids = %s,
                        compacted_from_message_ids = %s,
                        state = %s::jsonb,
                        revision = revision + 1,
                        metadata = %s::jsonb,
                        updated_at = now()
                    where thread_id = %s and revision = %s
                    returning revision, created_at::text, updated_at::text, metadata::text
                    """,
                    (
                        message_ids,
                        full_history_ids,
                        _step_ids(state),
                        old_compacted_from_ids,
                        json.dumps(state_payload),
                        json.dumps(stored_metadata),
                        thread_id,
                        current_revision,
                    ),
                )
            else:
                message_ids = self._insert_messages(cursor, thread_id, messages)
                cursor.execute(
                    """
                    insert into agent_thread_heads (
                      thread_id,
                      model_context_message_ids,
                      full_history_message_ids,
                      step_ids,
                      compacted_from_message_ids,
                      state,
                      revision,
                      metadata
                    ) values (%s, %s, %s, %s, %s, %s::jsonb, 1, %s::jsonb)
                    returning revision, created_at::text, updated_at::text, metadata::text
                    """,
                    (
                        thread_id,
                        message_ids,
                        message_ids,
                        _step_ids(state),
                        [],
                        json.dumps(state_payload),
                        json.dumps(stored_metadata),
                    ),
                )
            row = cursor.fetchone()
            self.connection.commit()
            return _stored_from_row(thread_id, state_payload, messages, row)
        except Exception:
            self.connection.rollback()
            raise
        finally:
            cursor.close()

    def load_state(self, thread_id: str) -> Optional[StoredRunState]:
        cursor = self.connection.cursor()
        try:
            cursor.execute(
                """
                select revision,
                       created_at::text,
                       updated_at::text,
                       state::text,
                       metadata::text,
                       model_context_message_ids
                from agent_thread_heads
                where thread_id = %s
                """,
                (thread_id,),
            )
            row = cursor.fetchone()
            if not row:
                return None
            messages = self._load_messages(cursor, thread_id, list(row[5] or []))
            return StoredRunState(
                thread_id=thread_id,
                state=_state_from_payload(row[3], messages),
                revision=row[0],
                created_at=str(row[1]),
                updated_at=str(row[2]),
                metadata=_decode_json(row[4]),
            )
        finally:
            cursor.close()

    def delete_state(self, thread_id: str) -> None:
        cursor = self.connection.cursor()
        try:
            cursor.execute(
                "delete from agent_compactions where thread_id = %s", (thread_id,)
            )
            cursor.execute(
                "delete from agent_messages where thread_id = %s", (thread_id,)
            )
            cursor.execute(
                "delete from agent_thread_heads where thread_id = %s", (thread_id,)
            )
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise
        finally:
            cursor.close()

    def compact_state(
        self,
        thread_id: str,
        source_message_ids: list[str],
        summary_message: dict[str, Any],
        *,
        revision: int | None = None,
        reason: Optional[str] = None,
        model: Optional[str] = None,
        input_token_count: Optional[int] = None,
        output_token_count: Optional[int] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> StoredRunState:
        if not source_message_ids:
            raise ValueError("source_message_ids must not be empty")

        cursor = self.connection.cursor()
        try:
            head = self._load_head_for_update(cursor, thread_id)
            if not head:
                raise KeyError(f"No state stored for thread_id {thread_id!r}")
            current_revision = head[0]
            _assert_revision(thread_id, current_revision, revision)

            model_context_ids = list(head[1] or [])
            full_history_ids = list(head[2] or [])
            compacted_from_ids = list(head[3] or [])
            state_payload = _decode_json(head[4])
            stored_metadata = _decode_json(head[5])

            summary_message_id = new_id("msg")
            self._insert_message(cursor, thread_id, summary_message_id, summary_message)
            summary_text = _message_text(summary_message)
            compaction_id = new_id("cmp")
            cursor.execute(
                """
                insert into agent_compactions (
                  compaction_id,
                  thread_id,
                  source_message_ids,
                  summary_message_id,
                  summary_text,
                  reason,
                  model,
                  input_token_count,
                  output_token_count,
                  metadata
                ) values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                """,
                (
                    compaction_id,
                    thread_id,
                    source_message_ids,
                    summary_message_id,
                    summary_text,
                    reason,
                    model,
                    input_token_count,
                    output_token_count,
                    json.dumps(metadata or {}),
                ),
            )

            next_context_ids = _replace_ordered_subsequence(
                model_context_ids,
                source_message_ids,
                [summary_message_id],
            )
            next_compacted_from_ids = [*compacted_from_ids, *source_message_ids]
            messages = self._load_messages(cursor, thread_id, next_context_ids)
            state_payload["messages"] = []
            cursor.execute(
                """
                update agent_thread_heads
                set model_context_message_ids = %s,
                    compacted_from_message_ids = %s,
                    state = %s::jsonb,
                    revision = revision + 1,
                    updated_at = now()
                where thread_id = %s and revision = %s
                returning revision, created_at::text, updated_at::text, metadata::text
                """,
                (
                    next_context_ids,
                    next_compacted_from_ids,
                    json.dumps(state_payload),
                    thread_id,
                    current_revision,
                ),
            )
            row = cursor.fetchone()
            self.connection.commit()
            return _stored_from_row(thread_id, state_payload, messages, row)
        except Exception:
            self.connection.rollback()
            raise
        finally:
            cursor.close()

    def list_compactions(self, thread_id: str) -> list[CompactionRecord]:
        cursor = self.connection.cursor()
        try:
            cursor.execute(
                """
                select compaction_id,
                       thread_id,
                       source_message_ids,
                       summary_message_id,
                       summary_text,
                       reason,
                       model,
                       input_token_count,
                       output_token_count,
                       created_at::text,
                       metadata::text
                from agent_compactions
                where thread_id = %s
                order by created_at asc
                """,
                (thread_id,),
            )
            return [
                CompactionRecord(
                    compaction_id=row[0],
                    thread_id=row[1],
                    source_message_ids=list(row[2] or []),
                    summary_message_id=row[3],
                    summary_text=row[4],
                    reason=row[5],
                    model=row[6],
                    input_token_count=row[7],
                    output_token_count=row[8],
                    created_at=str(row[9]),
                    metadata=_decode_json(row[10]),
                )
                for row in cursor.fetchall()
            ]
        finally:
            cursor.close()

    def get_thread_head(self, thread_id: str) -> Optional[dict[str, Any]]:
        cursor = self.connection.cursor()
        try:
            cursor.execute(
                """
                select model_context_message_ids,
                       full_history_message_ids,
                       compacted_from_message_ids,
                       step_ids,
                       revision,
                       metadata::text
                from agent_thread_heads
                where thread_id = %s
                """,
                (thread_id,),
            )
            row = cursor.fetchone()
            if not row:
                return None
            return {
                "model_context_message_ids": list(row[0] or []),
                "full_history_message_ids": list(row[1] or []),
                "compacted_from_message_ids": list(row[2] or []),
                "step_ids": list(row[3] or []),
                "revision": row[4],
                "metadata": _decode_json(row[5]),
            }
        finally:
            cursor.close()

    def _load_head_for_update(
        self, cursor: Any, thread_id: str
    ) -> Optional[tuple[Any, ...]]:
        cursor.execute(
            """
            select revision,
                   model_context_message_ids,
                   full_history_message_ids,
                   compacted_from_message_ids,
                   state::text,
                   metadata::text
            from agent_thread_heads
            where thread_id = %s
            for update
            """,
            (thread_id,),
        )
        return cursor.fetchone()

    def _insert_messages(
        self,
        cursor: Any,
        thread_id: str,
        messages: list[dict[str, Any]],
    ) -> list[str]:
        message_ids = []
        for message in messages:
            message_id = new_id("msg")
            self._insert_message(cursor, thread_id, message_id, message)
            message_ids.append(message_id)
        return message_ids

    def _insert_message(
        self,
        cursor: Any,
        thread_id: str,
        message_id: str,
        message: dict[str, Any],
    ) -> None:
        cursor.execute(
            """
            insert into agent_messages (
              message_id,
              thread_id,
              role,
              message,
              artifact_refs
            ) values (%s, %s, %s, %s::jsonb, %s::jsonb)
            """,
            (
                message_id,
                thread_id,
                message.get("role"),
                json.dumps(message),
                json.dumps(_message_artifact_refs(message)),
            ),
        )

    def _load_messages(
        self,
        cursor: Any,
        thread_id: str,
        message_ids: list[str],
    ) -> list[dict[str, Any]]:
        if not message_ids:
            return []
        cursor.execute(
            """
            select message_id, message::text
            from agent_messages
            where thread_id = %s and message_id = any(%s)
            """,
            (thread_id, message_ids),
        )
        messages_by_id = {row[0]: _decode_json(row[1]) for row in cursor.fetchall()}
        missing = [
            message_id for message_id in message_ids if message_id not in messages_by_id
        ]
        if missing:
            raise KeyError(f"Missing message rows for ids: {missing}")
        return [messages_by_id[message_id] for message_id in message_ids]


def _message_artifact_refs(message: dict[str, Any]) -> list[dict[str, Any]]:
    from .artifacts import collect_artifactized_fields

    refs = list(message.get("artifact_refs", []))
    refs.extend(collect_artifactized_fields(message))
    deduped: list[dict[str, Any]] = []
    seen = set()
    for ref in refs:
        artifact_ref = ref.get("artifact_ref") if isinstance(ref, dict) else None
        artifact_id = (
            artifact_ref.get("artifact_id") if isinstance(artifact_ref, dict) else None
        )
        key = artifact_id or json.dumps(ref, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(copy.deepcopy(ref))
    return ensure_json_serializable(deduped)


def _state_payload_without_messages(state: RunState) -> dict[str, Any]:
    payload = state.to_dict()
    payload["messages"] = []
    return payload


def _state_from_payload(
    payload: str | dict[str, Any], messages: list[dict[str, Any]]
) -> RunState:
    data = _decode_json(payload)
    data["messages"] = copy.deepcopy(messages)
    return RunState.from_dict(data)


def _stored_from_row(
    thread_id: str,
    state_payload: dict[str, Any],
    messages: list[dict[str, Any]],
    row: tuple[Any, ...],
) -> StoredRunState:
    return StoredRunState(
        thread_id=thread_id,
        state=_state_from_payload(state_payload, messages),
        revision=row[0],
        created_at=str(row[1]),
        updated_at=str(row[2]),
        metadata=_decode_json(row[3]),
    )


def _decode_json(value: Any) -> Any:
    if value is None:
        return {}
    if isinstance(value, str):
        return json.loads(value)
    return copy.deepcopy(value)


def _step_ids(state: RunState) -> list[str]:
    return [step.id for step in state.steps]


def _replace_ordered_subsequence(
    values: list[str],
    source: list[str],
    replacement: list[str],
) -> list[str]:
    if not source:
        raise ValueError("source must not be empty")
    for index in range(0, len(values) - len(source) + 1):
        if values[index : index + len(source)] == source:
            return [*values[:index], *replacement, *values[index + len(source) :]]
    raise ValueError("source_message_ids must be a contiguous subset of model context")


def _shared_message_prefix_length(
    old_messages: list[dict[str, Any]],
    new_messages: list[dict[str, Any]],
) -> int:
    count = 0
    for old_message, new_message in zip(old_messages, new_messages):
        if old_message != new_message:
            break
        count += 1
    return count


def _message_text(message: dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    return json.dumps(content, sort_keys=True)
