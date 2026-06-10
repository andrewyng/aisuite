from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Protocol

from .types import ensure_json_serializable
from .utils import new_id, now


@dataclass(kw_only=True)
class ArtifactRef:
    artifact_id: str
    uri: str
    media_type: str
    size_bytes: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return ensure_json_serializable(
            {
                "artifact_id": self.artifact_id,
                "uri": self.uri,
                "media_type": self.media_type,
                "size_bytes": self.size_bytes,
                "metadata": copy.deepcopy(self.metadata),
            }
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ArtifactRef":
        return cls(
            artifact_id=data["artifact_id"],
            uri=data["uri"],
            media_type=data["media_type"],
            size_bytes=data["size_bytes"],
            metadata=copy.deepcopy(data.get("metadata", {})),
        )


@dataclass(kw_only=True)
class Artifact:
    ref: ArtifactRef
    data: bytes
    created_at: str = ""

    def text(self, encoding: str = "utf-8") -> str:
        return self.data.decode(encoding)


class ArtifactStore(Protocol):
    def put(
        self,
        data: bytes | str,
        *,
        media_type: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> ArtifactRef: ...

    def get(self, ref: ArtifactRef | str) -> Artifact: ...

    def delete(self, ref: ArtifactRef | str) -> None: ...


class InMemoryArtifactStore:
    def __init__(self):
        self._artifacts: dict[str, Artifact] = {}

    def put(
        self,
        data: bytes | str,
        *,
        media_type: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> ArtifactRef:
        payload = _to_bytes(data)
        artifact_id = new_id("artifact")
        ref = ArtifactRef(
            artifact_id=artifact_id,
            uri=f"memory://{artifact_id}",
            media_type=media_type,
            size_bytes=len(payload),
            metadata=_artifact_metadata(payload, metadata),
        )
        self._artifacts[artifact_id] = Artifact(
            ref=ArtifactRef.from_dict(ref.to_dict()),
            data=payload,
            created_at=now(),
        )
        return ref

    def get(self, ref: ArtifactRef | str) -> Artifact:
        artifact_id = _artifact_id(ref)
        try:
            artifact = self._artifacts[artifact_id]
        except KeyError as exc:
            raise KeyError(f"Artifact {artifact_id!r} not found.") from exc
        return Artifact(
            ref=ArtifactRef.from_dict(artifact.ref.to_dict()),
            data=bytes(artifact.data),
            created_at=artifact.created_at,
        )

    def delete(self, ref: ArtifactRef | str) -> None:
        self._artifacts.pop(_artifact_id(ref), None)


class FileArtifactStore:
    def __init__(self, root: str | Path = ".aisuite/artifacts"):
        self.root = Path(root)

    def put(
        self,
        data: bytes | str,
        *,
        media_type: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> ArtifactRef:
        payload = _to_bytes(data)
        artifact_id = new_id("artifact")
        artifact_dir = self.root / artifact_id
        artifact_dir.mkdir(parents=True, exist_ok=False)
        data_path = artifact_dir / "data"
        meta_path = artifact_dir / "metadata.json"
        ref = ArtifactRef(
            artifact_id=artifact_id,
            uri=f"artifact://{artifact_id}",
            media_type=media_type,
            size_bytes=len(payload),
            metadata=_artifact_metadata(payload, metadata),
        )
        data_path.write_bytes(payload)
        meta_path.write_text(
            json.dumps(
                {
                    "ref": ref.to_dict(),
                    "created_at": now(),
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return ref

    def get(self, ref: ArtifactRef | str) -> Artifact:
        artifact_id = _artifact_id(ref)
        artifact_dir = self.root / artifact_id
        data_path = artifact_dir / "data"
        meta_path = artifact_dir / "metadata.json"
        if not data_path.exists() or not meta_path.exists():
            raise KeyError(f"Artifact {artifact_id!r} not found.")
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        return Artifact(
            ref=ArtifactRef.from_dict(metadata["ref"]),
            data=data_path.read_bytes(),
            created_at=metadata.get("created_at", ""),
        )

    def delete(self, ref: ArtifactRef | str) -> None:
        artifact_id = _artifact_id(ref)
        artifact_dir = self.root / artifact_id
        try:
            (artifact_dir / "data").unlink()
        except FileNotFoundError:
            pass
        try:
            (artifact_dir / "metadata.json").unlink()
        except FileNotFoundError:
            pass
        try:
            artifact_dir.rmdir()
        except FileNotFoundError:
            pass
        except OSError:
            pass


def _artifact_id(ref: ArtifactRef | str) -> str:
    if isinstance(ref, ArtifactRef):
        return ref.artifact_id
    if ref.startswith("artifact://"):
        return ref.removeprefix("artifact://")
    if ref.startswith("memory://"):
        return ref.removeprefix("memory://")
    return ref


def _to_bytes(data: bytes | str) -> bytes:
    if isinstance(data, bytes):
        return data
    return data.encode("utf-8")


def _artifact_metadata(
    payload: bytes,
    metadata: Optional[dict[str, Any]],
) -> dict[str, Any]:
    return ensure_json_serializable(
        {
            **copy.deepcopy(metadata or {}),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    )
