from __future__ import annotations

import copy
from typing import Any, Optional

from .artifact_store import ArtifactRef, ArtifactStore

ARTIFACT_CONTENT_TYPE = "artifact_ref"
DEFAULT_ARTIFACT_THRESHOLD_CHARS = 20_000
DEFAULT_ARTIFACT_PREVIEW_CHARS = 4_000
DEFAULT_ARTIFACT_FIELD_NAMES = frozenset(
    {"content", "diff", "patch", "stderr", "stdout"}
)


def dehydrate_messages(
    messages: list[dict[str, Any]],
    artifact_store: Optional[ArtifactStore],
    *,
    threshold_chars: int = DEFAULT_ARTIFACT_THRESHOLD_CHARS,
    preview_chars: int = DEFAULT_ARTIFACT_PREVIEW_CHARS,
) -> list[dict[str, Any]]:
    if artifact_store is None:
        return copy.deepcopy(messages)
    return [
        _dehydrate_message(
            message,
            artifact_store,
            threshold_chars=threshold_chars,
            preview_chars=preview_chars,
        )
        for message in messages
    ]


def hydrate_messages(
    messages: list[dict[str, Any]],
    artifact_store: Optional[ArtifactStore],
) -> list[dict[str, Any]]:
    if artifact_store is None:
        return copy.deepcopy(messages)
    return [_hydrate_message(message, artifact_store) for message in messages]


def artifactize_value(
    value: Any,
    artifact_store: Optional[ArtifactStore],
    *,
    threshold_chars: int = DEFAULT_ARTIFACT_THRESHOLD_CHARS,
    preview_chars: int = DEFAULT_ARTIFACT_PREVIEW_CHARS,
    field_names: frozenset[str] = DEFAULT_ARTIFACT_FIELD_NAMES,
    metadata: Optional[dict[str, Any]] = None,
) -> Any:
    """Replace selected large string fields with artifact refs.

    This is meant for persisted state and trace payloads. The tool execution path can
    still return the full value to the model and use this helper only for side-channel
    observability data.
    """
    if artifact_store is None:
        return copy.deepcopy(value)
    return _artifactize_value(
        value,
        artifact_store,
        threshold_chars=threshold_chars,
        preview_chars=preview_chars,
        field_names=field_names,
        field_path="",
        metadata=metadata or {},
    )


def hydrate_value(value: Any, artifact_store: Optional[ArtifactStore]) -> Any:
    if artifact_store is None:
        return copy.deepcopy(value)
    if _is_artifact_content(value):
        ref = ArtifactRef.from_dict(value["artifact_ref"])
        return artifact_store.get(ref).text()
    if isinstance(value, list):
        return [hydrate_value(item, artifact_store) for item in value]
    if isinstance(value, dict):
        return {key: hydrate_value(item, artifact_store) for key, item in value.items()}
    return copy.deepcopy(value)


def collect_artifactized_fields(value: Any) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    _collect_artifactized_fields(value, refs)
    return refs


def _dehydrate_message(
    message: dict[str, Any],
    artifact_store: ArtifactStore,
    *,
    threshold_chars: int,
    preview_chars: int,
) -> dict[str, Any]:
    dehydrated = copy.deepcopy(message)
    content = dehydrated.get("content")
    if isinstance(content, str) and len(content) > threshold_chars:
        ref = artifact_store.put(
            content,
            media_type="text/plain; charset=utf-8",
            metadata={
                "kind": "message_content",
                "role": dehydrated.get("role"),
            },
        )
        dehydrated["content"] = {
            "type": ARTIFACT_CONTENT_TYPE,
            "preview": content[:preview_chars],
            "artifact_ref": ref.to_dict(),
        }
    elif isinstance(content, (dict, list)):
        dehydrated["content"] = artifactize_value(
            content,
            artifact_store,
            threshold_chars=threshold_chars,
            preview_chars=preview_chars,
            metadata={
                "kind": "message_content_field",
                "role": dehydrated.get("role"),
            },
        )
    return dehydrated


def _hydrate_message(
    message: dict[str, Any], artifact_store: ArtifactStore
) -> dict[str, Any]:
    hydrated = copy.deepcopy(message)
    hydrated["content"] = hydrate_value(hydrated.get("content"), artifact_store)
    return hydrated


def _is_artifact_content(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and value.get("type") == ARTIFACT_CONTENT_TYPE
        and isinstance(value.get("artifact_ref"), dict)
    )


def _artifactize_value(
    value: Any,
    artifact_store: ArtifactStore,
    *,
    threshold_chars: int,
    preview_chars: int,
    field_names: frozenset[str],
    field_path: str,
    metadata: dict[str, Any],
) -> Any:
    if _is_artifact_content(value):
        return copy.deepcopy(value)
    if isinstance(value, list):
        return [
            _artifactize_value(
                item,
                artifact_store,
                threshold_chars=threshold_chars,
                preview_chars=preview_chars,
                field_names=field_names,
                field_path=f"{field_path}[{index}]",
                metadata=metadata,
            )
            for index, item in enumerate(value)
        ]
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            next_path = f"{field_path}.{key}" if field_path else str(key)
            if (
                isinstance(item, str)
                and key in field_names
                and len(item) > threshold_chars
            ):
                ref = artifact_store.put(
                    item,
                    media_type="text/plain; charset=utf-8",
                    metadata={
                        **copy.deepcopy(metadata),
                        "kind": metadata.get("kind", "artifactized_field"),
                        "field": key,
                        "field_path": next_path,
                    },
                )
                result[key] = {
                    "type": ARTIFACT_CONTENT_TYPE,
                    "preview": item[:preview_chars],
                    "artifact_ref": ref.to_dict(),
                }
            else:
                result[key] = _artifactize_value(
                    item,
                    artifact_store,
                    threshold_chars=threshold_chars,
                    preview_chars=preview_chars,
                    field_names=field_names,
                    field_path=next_path,
                    metadata=metadata,
                )
        return result
    return copy.deepcopy(value)


def _collect_artifactized_fields(value: Any, refs: list[dict[str, Any]]) -> None:
    if _is_artifact_content(value):
        refs.append(copy.deepcopy(value))
        return
    if isinstance(value, list):
        for item in value:
            _collect_artifactized_fields(item, refs)
        return
    if isinstance(value, dict):
        for item in value.values():
            _collect_artifactized_fields(item, refs)
