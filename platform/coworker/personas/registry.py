"""Persona registry — the installed personas + their lifecycle state.

Unifies two sources behind one `id → Agent` resolver: the core surfaces (Code / Chat /
Cowork) wrap their existing agent builders (exact prompts preserved), and markdown manifests
(Ops today; third-party dirs in Phase 2) load through ``PersonaManifest``. Lifecycle —
installed → enabled → surfaced, plus a default — is persisted to a small JSON file.

A session is born from exactly one persona (recorded as ``SessionRecord.agent``); resolving an
id always returns its Agent even if the persona was later disabled, so live sessions keep
working. Disable/surface only affect what the *new-session* picker offers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from ..agents.base import Agent
from ..agents.chat import chat_agent
from ..agents.code import CODE_CAPABILITIES, code_agent
from ..agents.cowork import COWORK_CAPABILITIES, cowork_agent
from .manifest import PersonaManifest, load_manifest_file

DEFAULT_PERSONA_ID = "cowork"


@dataclass
class PersonaState:
    enabled: bool = True
    surfaced: bool = True


@dataclass
class PersonaEntry:
    id: str
    name: str
    icon: str = ""
    tagline: str = ""
    needs_workspace: bool = True
    builtin: bool = True
    family: str = "knowledge"
    tools: list[str] = field(default_factory=list)
    _builder: Optional[Callable[[], Agent]] = None
    manifest: Optional[PersonaManifest] = None

    def agent(self) -> Agent:
        if self._builder is not None:
            return self._builder()
        assert self.manifest is not None
        return self.manifest.to_agent()


class PersonaRegistry:
    def __init__(
        self,
        *,
        builtin_dir: Optional[str | Path] = None,
        extra_dirs: Optional[list[str | Path]] = None,
        state_path: Optional[str | Path] = None,
    ) -> None:
        self.state_path = Path(state_path) if state_path else None
        self._entries: dict[str, PersonaEntry] = {}
        self._enabled: dict[str, bool] = {}
        self._surfaced: dict[str, bool] = {}
        self._default = DEFAULT_PERSONA_ID
        self._load_builtin(builtin_dir)
        for d in extra_dirs or []:
            self._load_dir(d, builtin=False)
        self._load_state()

    # -- loading ----------------------------------------------------------------
    def _register_builder(
        self, id, name, icon, tagline, builder, needs_workspace, family, tools
    ) -> None:
        self._entries[id] = PersonaEntry(
            id=id,
            name=name,
            icon=icon,
            tagline=tagline,
            needs_workspace=needs_workspace,
            builtin=True,
            family=family,
            tools=list(tools),
            _builder=builder,
        )

    def _load_builtin(self, builtin_dir: Optional[str | Path]) -> None:
        # Core surfaces keep their exact prompts via the existing builders; order = sidebar order.
        self._register_builder(
            "code", "Code", "code", "Work in a codebase — files, git, shell",
            code_agent, True, "code", CODE_CAPABILITIES,
        )
        self._register_builder(
            "chat", "Chat", "chat", "Quick questions — no workspace",
            chat_agent, False, "knowledge", [],
        )
        self._register_builder(
            "cowork", "Coworker", "cowork", "Produce a deliverable — research, analysis, scripts",
            cowork_agent, True, "knowledge", COWORK_CAPABILITIES,
        )
        # Markdown-backed built-ins (Ops, …) — dogfood the manifest path.
        d = Path(builtin_dir) if builtin_dir else Path(__file__).parent / "builtin"
        self._load_dir(d, builtin=True)

    def _load_dir(self, directory: str | Path, *, builtin: bool) -> None:
        d = Path(directory)
        if not d.is_dir():
            return
        for md in sorted(d.glob("*.md")):
            m = load_manifest_file(md, builtin=builtin)
            self._entries[m.id] = PersonaEntry(
                id=m.id,
                name=m.name,
                icon=m.icon,
                tagline=m.tagline,
                needs_workspace=m.needs_workspace,
                builtin=builtin,
                family=m.family,
                tools=list(m.tools),
                manifest=m,
            )

    def _load_state(self) -> None:
        if self.state_path and self.state_path.is_file():
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
            self._enabled = dict(data.get("enabled", {}))
            self._surfaced = dict(data.get("surfaced", {}))
            self._default = data.get("default", DEFAULT_PERSONA_ID)

    def save(self) -> None:
        if not self.state_path:
            return
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(
            json.dumps(
                {
                    "enabled": self._enabled,
                    "surfaced": self._surfaced,
                    "default": self._default,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    # -- queries ----------------------------------------------------------------
    def ids(self) -> list[str]:
        return list(self._entries)

    def get(self, persona_id: str) -> Optional[PersonaEntry]:
        return self._entries.get(persona_id)

    def is_enabled(self, persona_id: str) -> bool:
        return self._enabled.get(persona_id, True)

    def is_surfaced(self, persona_id: str) -> bool:
        return self._surfaced.get(persona_id, True)

    def default_id(self) -> str:
        # The configured default if it's enabled, else cowork if present, else any enabled one.
        if self._default in self._entries and self.is_enabled(self._default):
            return self._default
        if DEFAULT_PERSONA_ID in self._entries and self.is_enabled(DEFAULT_PERSONA_ID):
            return DEFAULT_PERSONA_ID
        for pid in self._entries:
            if self.is_enabled(pid):
                return pid
        return DEFAULT_PERSONA_ID

    def agent(self, persona_id: Optional[str]) -> Agent:
        """Resolve a persona id to its Agent. Unknown ids fall back to the default persona;
        a known-but-disabled id still resolves (live sessions keep working)."""
        entry = self._entries.get(persona_id or "")
        if entry is None:
            entry = self._entries.get(self.default_id())
        if entry is None:
            raise KeyError(f"no persona to resolve for {persona_id!r}")
        return entry.agent()

    def sidebar(self) -> list[dict]:
        """Session surfaces for the new-session picker: enabled AND surfaced, in order."""
        out = []
        for e in self._entries.values():
            if self.is_enabled(e.id) and self.is_surfaced(e.id):
                out.append(
                    {
                        "name": e.id,
                        "title": e.name,
                        "needs_workspace": e.needs_workspace,
                        "icon": e.icon,
                        "tagline": e.tagline,
                        "default": e.id == self.default_id(),
                    }
                )
        return out

    def list_all(self) -> list[dict]:
        """Every installed persona + its lifecycle state — for the Personas settings panel."""
        return [
            {
                "id": e.id,
                "name": e.name,
                "icon": e.icon,
                "tagline": e.tagline,
                "needs_workspace": e.needs_workspace,
                "builtin": e.builtin,
                "family": e.family,
                "tools": e.tools,
                "enabled": self.is_enabled(e.id),
                "surfaced": self.is_surfaced(e.id),
                "default": e.id == self.default_id(),
            }
            for e in self._entries.values()
        ]

    # -- mutations --------------------------------------------------------------
    def set_enabled(self, persona_id: str, enabled: bool) -> None:
        if persona_id not in self._entries:
            raise KeyError(persona_id)
        self._enabled[persona_id] = bool(enabled)
        self.save()

    def set_surfaced(self, persona_id: str, surfaced: bool) -> None:
        if persona_id not in self._entries:
            raise KeyError(persona_id)
        self._surfaced[persona_id] = bool(surfaced)
        self.save()

    def set_default(self, persona_id: str) -> None:
        if persona_id not in self._entries:
            raise KeyError(persona_id)
        self._default = persona_id
        self._enabled[persona_id] = True  # a default must be enabled
        self.save()


# -- module singleton (used by agents.get_agent / list_agents) ------------------
_singleton: Optional[PersonaRegistry] = None


def get_registry() -> PersonaRegistry:
    global _singleton
    if _singleton is None:
        from ..secrets import state_dir

        _singleton = PersonaRegistry(state_path=state_dir() / "personas.json")
    return _singleton


def set_registry(registry: PersonaRegistry) -> None:
    """Install a registry as the process singleton (the manager does this with its data dir)."""
    global _singleton
    _singleton = registry
