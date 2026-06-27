"""Persona manifest — parse + validate a persona definition.

Format: YAML frontmatter (identity + capability declaration) followed by a markdown body that
is the system prompt. `persona ⊇ skill` — the same frontmatter-markdown shape as SKILL.md, with
more structured fields. Parsing is strict: an invalid manifest raises ``ManifestError`` rather
than silently producing a broken persona (a third-party persona must fail loudly).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

VALID_FAMILIES = {"code", "knowledge"}
VALID_WORKSPACES = {"git", "deliverable", "none"}
VALID_MODES = {"discuss", "plan", "interactive", "custom", "auto"}


class ManifestError(ValueError):
    """A persona manifest is malformed or references unknown capabilities/values."""


@dataclass
class PersonaManifest:
    id: str
    name: str
    system_prompt: str
    icon: str = ""
    tagline: str = ""
    description: str = ""
    tools: list[str] = field(default_factory=list)
    family: str = "knowledge"  # "code" | "knowledge"
    workspace: str = "deliverable"  # "git" | "deliverable" | "none"
    messaging: bool = False
    connectors: bool = False
    default_permission_mode: str = "interactive"
    recommended_models: list[str] = field(default_factory=list)
    skills: list[str] = field(default_factory=list)
    mcp: list[str] = field(default_factory=list)
    builtin: bool = False
    source: Optional[str] = None  # where it was loaded from (path / url), for provenance

    @property
    def needs_workspace(self) -> bool:
        return self.workspace != "none"

    def to_agent(self):
        """Materialize the runtime Agent (prompt + catalog-expanded tools + traits)."""
        from ..agents.base import Agent
        from ..catalog import expand

        tool_ids = list(self.tools)
        factory = (lambda ctx: expand(tool_ids, ctx)) if tool_ids else None
        return Agent(
            name=self.id,
            title=self.name,
            system_prompt=self.system_prompt,
            needs_workspace=self.needs_workspace,
            tool_factory=factory,
            family=self.family,
            messaging=self.messaging,
            connectors=self.connectors,
        )


def _split_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    if not text.startswith("---"):
        raise ManifestError("manifest must start with a YAML frontmatter block (---)")
    end = text.find("\n---", 3)
    if end == -1:
        raise ManifestError("unterminated frontmatter block (missing closing ---)")
    raw = text[3:end]
    body = text[end + 4 :].lstrip("\n")
    try:
        meta = yaml.safe_load(raw) or {}
    except yaml.YAMLError as e:  # pragma: no cover - exercised via parse error path
        raise ManifestError(f"invalid YAML frontmatter: {e}") from e
    if not isinstance(meta, dict):
        raise ManifestError("frontmatter must be a mapping of key: value")
    return meta, body


def _strlist(meta: dict, key: str) -> list[str]:
    val = meta.get(key, [])
    if val is None:
        return []
    if isinstance(val, str):
        return [v.strip() for v in val.split(",") if v.strip()]
    if isinstance(val, list):
        return [str(v).strip() for v in val if str(v).strip()]
    raise ManifestError(f"`{key}` must be a list or comma-separated string")


def parse_manifest(
    text: str, *, fallback_id: Optional[str] = None, builtin: bool = False, source: Optional[str] = None
) -> PersonaManifest:
    meta, body = _split_frontmatter(text)

    persona_id = str(meta.get("id") or fallback_id or "").strip()
    if not persona_id:
        raise ManifestError("manifest needs an `id` (or a filename to derive one from)")
    if not body.strip():
        raise ManifestError(f"persona {persona_id!r} has no body (the system prompt)")

    family = str(meta.get("family", "knowledge")).strip().lower()
    if family not in VALID_FAMILIES:
        raise ManifestError(f"persona {persona_id!r}: family must be one of {sorted(VALID_FAMILIES)}")

    workspace = str(meta.get("workspace", "deliverable")).strip().lower()
    if workspace not in VALID_WORKSPACES:
        raise ManifestError(
            f"persona {persona_id!r}: workspace must be one of {sorted(VALID_WORKSPACES)}"
        )

    mode = str(meta.get("default_permission_mode", "interactive")).strip().lower()
    if mode not in VALID_MODES:
        raise ManifestError(
            f"persona {persona_id!r}: default_permission_mode must be one of {sorted(VALID_MODES)}"
        )

    tools = _strlist(meta, "tools")
    _validate_tools(persona_id, tools)

    return PersonaManifest(
        id=persona_id,
        name=str(meta.get("name") or persona_id).strip(),
        system_prompt=body.strip(),
        icon=str(meta.get("icon", "")).strip(),
        tagline=str(meta.get("tagline", "")).strip(),
        description=str(meta.get("description", "")).strip(),
        tools=tools,
        family=family,
        workspace=workspace,
        messaging=bool(meta.get("messaging", False)),
        connectors=bool(meta.get("connectors", False)),
        default_permission_mode=mode,
        recommended_models=_strlist(meta, "recommended_models"),
        skills=_strlist(meta, "skills"),
        mcp=_strlist(meta, "mcp"),
        builtin=builtin,
        source=source,
    )


def _validate_tools(persona_id: str, tools: list[str]) -> None:
    # Imported here to avoid a module-load cycle (catalog imports agents.base).
    from ..catalog import CATALOG

    unknown = [t for t in tools if t not in CATALOG]
    if unknown:
        raise ManifestError(
            f"persona {persona_id!r} references unknown tool capabilities: {unknown}. "
            f"Known: {sorted(CATALOG)}"
        )


def load_manifest_file(path: str | Path, *, builtin: bool = False) -> PersonaManifest:
    p = Path(path)
    return parse_manifest(
        p.read_text(encoding="utf-8"),
        fallback_id=p.stem,
        builtin=builtin,
        source=str(p),
    )
