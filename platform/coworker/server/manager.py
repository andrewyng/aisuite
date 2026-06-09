"""Session manager — owns engines (one per session), stores, and the provider.

Each session is bound to a workspace folder (Code requires one). Storage is a single DB
under a data dir (global for the real server, per-workspace for tests), so recents and
sessions span folders.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Optional

from ..agent import build_engine
from ..agents import get_agent
from ..audit import AuditStore
from ..conversations import ConversationStore, title_from
from ..engine import Approver, TurnEngine
from ..agents import myhelper_agent
from ..automation import Scheduler, TaskRun, TaskStore
from ..connectors import (
    Gateway,
    SUPERAGENT_MESSAGING_NOTE,
    SuperAgent,
    connect_connector,
    connector_list,
    disconnect_connector,
    load_settings,
    make_adapter,
    update_connector_tools,
)
from ..connectors.browser_automation import browser_close_session, browser_state, browser_take_screenshot
from ..mcp import (
    MCPManager,
    build_callables,
    delete_global_server,
    load_mcp_servers,
    patch_global_server,
    put_global_server,
    read_global,
)
from ..memory import MemoryStore, Scope, SQLiteMemoryStore
from ..permissions import Mode
from ..agents import list_agents as _list_agents
from ..providers import (
    ProviderClient,
    ProviderRouter,
    get_descriptor,
    provider_descriptors,
)
from ..secrets import SecretStore
from ..sessions import SessionRecord
from ..skills import SkillLoader

_SCOPES = {s.value for s in Scope}


class SessionManager:
    def __init__(
        self,
        *,
        workspace: Optional[str | Path] = None,  # default/seed workspace (e.g. --cwd)
        data_dir: Optional[str | Path] = None,
        model: str = "gpt-5.5",
        mode: Mode = Mode.INTERACTIVE,
        provider: Optional[ProviderClient] = None,
    ) -> None:
        self.default_workspace = str(Path(workspace).expanduser().resolve()) if workspace else None
        self.model = model
        self.mode = mode
        self.provider = provider

        if data_dir is not None:
            base = Path(data_dir).expanduser()
        elif self.default_workspace is not None:
            base = Path(self.default_workspace) / ".coworker"
        else:
            base = Path.home() / ".config" / "coworker"
        base.mkdir(parents=True, exist_ok=True)

        self.memory_store: MemoryStore = SQLiteMemoryStore(base / "coworker.db")
        self.audit_store = AuditStore(base / "coworker.db")
        self.session_store = ConversationStore(base)
        self.session_store.canonicalize_workspaces()  # collapse /tmp vs /private/tmp etc.
        if self.default_workspace:
            self.session_store.touch_workspace(self.default_workspace)
        self._engines: dict[str, TurnEngine] = {}
        self.secrets = SecretStore()
        # No explicit provider injected → route by the model's `provider:` prefix (OpenAI default,
        # Ollama, …). Tests inject a provider directly and bypass the router. The same router is
        # shared by every engine and the `/v1/chat/completions` proxy.
        if self.provider is None:
            self.provider = ProviderRouter(self.secrets, default_provider="openai")
        self.mcp = MCPManager()
        self.gateway: Optional[Gateway] = None
        self.superagent: Optional[SuperAgent] = None
        self._data_base = base
        # Desktop/UI prefs (default model, onboarding state) — not secrets; a plain JSON file.
        self._prefs = self._load_prefs()
        if self._prefs.get("default_model"):
            self.model = self._prefs["default_model"]
        # GUI super-agent surface: connected clients (send callbacks) + pending approval.
        self._sa_clients: set[Any] = set()
        self._sa_pending: Optional[asyncio.Future] = None
        # Automation: scheduled tasks store + the tick scheduler (started in the lifespan).
        self.task_store = TaskStore(base / "automation.db")
        self.scheduler = Scheduler(self.task_store, self._run_scheduled_task)

    # -- workspaces -------------------------------------------------------------
    def open_workspace(self, path: str, *, create: bool = False) -> dict[str, Any]:
        resolved = Path(path).expanduser()
        if resolved.exists() and not resolved.is_dir():
            return {"path": str(resolved), "ok": False, "error": "not a directory"}
        if not resolved.exists():
            if not create:
                return {"path": str(resolved), "ok": False, "error": "folder does not exist"}
            try:
                resolved.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                return {"path": str(resolved), "ok": False, "error": str(exc)}
        resolved = resolved.resolve()
        self.session_store.touch_workspace(str(resolved))
        return {"path": str(resolved), "ok": True, "git_branch": _git_branch(resolved)}

    def recent_workspaces(self) -> list[dict[str, Any]]:
        out = []
        for path in self.session_store.recent_workspaces():
            p = Path(path)
            out.append({"path": path, "name": p.name, "exists": p.is_dir()})
        return out

    def resolve_workspace(self, requested: Optional[str]) -> Optional[str]:
        if requested:
            p = Path(requested).expanduser()
            if p.is_dir():
                return str(p.resolve())
            return None
        return self.default_workspace

    # -- engines ----------------------------------------------------------------
    def engine_workspace(
        self, session_id: str, *, workspace: Optional[str] = None, agent: str = "code"
    ) -> Optional[str]:
        """The workspace `get_engine` would bind — for prepping MCP tools beforehand."""
        record = self.session_store.load(session_id)
        if record:
            return record.workspace or None
        ag = get_agent(agent or "code")
        return self.resolve_workspace(workspace) if ag.needs_workspace else None

    def get_engine(
        self,
        session_id: str,
        *,
        workspace: Optional[str] = None,
        agent: str = "code",
        approver: Optional[Approver] = None,
        extra_tools: Optional[list[Any]] = None,
    ) -> Optional[TurnEngine]:
        engine = self._engines.get(session_id)
        if engine is not None:
            if approver is not None:
                engine.approver = approver
            return engine

        record = self.session_store.load(session_id)
        agent_name = (record.agent if record else agent) or "code"
        ag = get_agent(agent_name)

        if record:
            ws = record.workspace or None
            model, mode, messages = record.model, Mode(record.mode), record.messages
        else:
            ws = self.resolve_workspace(workspace) if ag.needs_workspace else None
            model, mode, messages = self.model, self.mode, None

        if ag.needs_workspace and (not ws or not Path(ws).is_dir()):
            return None  # Code requires a valid folder; Chat does not

        if ws:
            self.session_store.touch_workspace(ws)
        engine = build_engine(
            agent=ag,
            workspace=ws,
            model=model,
            mode=mode,
            approver=approver,
            provider=self.provider,
            memory_store=self.memory_store,
            messages=messages,
            extra_tools=extra_tools,
            secrets=self.secrets,
            task_store=self.task_store,
            session_id=session_id,
            audit_sink=self.audit_store.append,
        )
        self._engines[session_id] = engine
        return engine

    # -- MCP --------------------------------------------------------------------
    async def prepare_mcp_tools(
        self, session_id: str, *, workspace: Optional[str] = None, agent: str = "code"
    ) -> list[Any]:
        """Connect enabled MCP servers (global + workspace) and return their tool callables.

        Called from the async WS handler before `get_engine`; no-op if the engine is already
        built (its MCP tools are attached). Servers that fail to connect are skipped.
        """
        if session_id in self._engines:
            return []
        ws = self.engine_workspace(session_id, workspace=workspace, agent=agent)
        loop = asyncio.get_running_loop()
        out: list[Any] = []
        for server in load_mcp_servers(ws, secrets=self.secrets):
            if not server.enabled:
                continue
            try:
                conn = await self.mcp.ensure(server)
            except Exception:  # bad command / unreachable url — skip, don't break the session
                continue
            out.extend(
                build_callables(
                    server,
                    conn.tools,
                    lambda tool, args, name=server.name: self.mcp.call(name, tool, args),
                    loop,
                )
            )
        return out

    def list_mcp(self) -> list[dict[str, Any]]:
        """Servers from the global config + connection status (does not connect)."""
        out = []
        for name, raw in read_global().items():
            connected = name in self.mcp._conns
            out.append(
                {
                    "name": name,
                    "enabled": bool(raw.get("enabled", True)),
                    "transport": "http" if (raw.get("url") or str(raw.get("type", "")).lower() in {"http", "sse", "streamable-http"}) else "stdio",
                    "requires_approval": bool(raw.get("requires_approval", True)),
                    "status": "connected" if connected else ("disabled" if not raw.get("enabled", True) else "configured"),
                    "tool_count": len(self.mcp._conns[name].tools) if connected else None,
                    "config": _redact(raw),
                }
            )
        return out

    def add_mcp(self, name: str, config: dict[str, Any]) -> dict[str, Any]:
        put_global_server(name, config)
        return {"ok": True, "name": name}

    def patch_mcp(self, name: str, changes: dict[str, Any]) -> dict[str, Any]:
        ok = patch_global_server(name, changes)
        return {"ok": ok, "name": name}

    def delete_mcp(self, name: str) -> dict[str, Any]:
        ok = delete_global_server(name)
        return {"ok": ok, "name": name}

    async def mcp_tools(self, name: str) -> dict[str, Any]:
        """Connect one server and list its tools (name + description)."""
        for server in load_mcp_servers(self.default_workspace, secrets=self.secrets):
            if server.name == name:
                try:
                    conn = await self.mcp.ensure(server)
                except Exception as exc:
                    return {"name": name, "ok": False, "error": str(exc), "tools": []}
                return {
                    "name": name,
                    "ok": True,
                    "tools": [{"name": t.name, "description": getattr(t, "description", "")} for t in conn.tools],
                }
        return {"name": name, "ok": False, "error": "unknown server", "tools": []}

    async def reload_mcp(self) -> dict[str, Any]:
        """Drop live MCP connections so new sessions reconnect with fresh config."""
        await self.mcp.aclose()
        return {"ok": True}

    # -- connectors -------------------------------------------------------------
    def list_connectors(self) -> list[dict[str, Any]]:
        return connector_list(self.secrets)

    def connect_connector(self, name: str, fields: dict[str, Any]) -> dict[str, Any]:
        # validates the token by a live API call (sync httpx) — run off the event loop
        return connect_connector(self.secrets, name, fields)

    def disconnect_connector(self, name: str) -> dict[str, Any]:
        return disconnect_connector(self.secrets, name)

    def update_connector_tools(self, name: str, enabled: dict[str, Any]) -> dict[str, Any]:
        return update_connector_tools(self.secrets, name, enabled)

    def list_audit(
        self,
        *,
        limit: int = 100,
        session_id: Optional[str] = None,
        connector: Optional[str] = None,
        tool: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        return self.audit_store.list(limit=limit, session_id=session_id, connector=connector, tool=tool)

    def browser_state(self) -> dict[str, Any]:
        return browser_state()

    def browser_screenshot(self) -> dict[str, Any]:
        return browser_take_screenshot()

    def browser_close(self) -> dict[str, Any]:
        return browser_close_session()

    def list_artifacts(self, session_id: str) -> list[dict[str, Any]]:
        record = self.session_store.load(session_id)
        workspace = record.workspace if record else self.default_workspace
        if not workspace:
            return []
        root = Path(workspace).expanduser().resolve()
        if not root.is_dir():
            return []
        out: list[dict[str, Any]] = []
        suffixes = {".md", ".markdown", ".html", ".htm", ".txt", ".json", ".csv", ".py", ".js", ".ts", ".tsx", ".css", ".png", ".jpg", ".jpeg", ".webp", ".gif"}
        for path in root.rglob("*"):
            try:
                rel = path.relative_to(root)
                if any(part.startswith(".") or part in {"node_modules", "target", "dist", "__pycache__"} for part in rel.parts):
                    continue
                if not path.is_file() or path.suffix.lower() not in suffixes:
                    continue
                st = path.stat()
                out.append(
                    {
                        "path": str(rel),
                        "name": path.name,
                        "kind": _artifact_kind(path),
                        "size": st.st_size,
                        "modified_at": st.st_mtime,
                    }
                )
            except OSError:
                continue
        out.sort(key=lambda a: a["modified_at"], reverse=True)
        return out[:80]

    def read_artifact(self, session_id: str, path: str) -> dict[str, Any]:
        record = self.session_store.load(session_id)
        workspace = record.workspace if record else self.default_workspace
        if not workspace:
            return {"ok": False, "error": "no workspace"}
        root = Path(workspace).expanduser().resolve()
        target = (root / path).expanduser().resolve()
        try:
            target.relative_to(root)
        except ValueError:
            return {"ok": False, "error": "path escapes workspace"}
        if not target.is_file():
            return {"ok": False, "error": "not found"}
        kind = _artifact_kind(target)
        if kind == "image":
            import base64

            mime = {
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".webp": "image/webp",
                ".gif": "image/gif",
            }.get(target.suffix.lower(), "application/octet-stream")
            data = base64.b64encode(target.read_bytes()).decode("ascii")
            return {"ok": True, "path": path, "kind": kind, "data_url": f"data:{mime};base64,{data}"}
        try:
            text = target.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return {"ok": False, "error": "binary file cannot be previewed"}
        return {"ok": True, "path": path, "kind": kind, "content": text[:500000], "truncated": len(text) > 500000}

    # -- web search -------------------------------------------------------------
    def get_web_search(self) -> dict[str, Any]:
        from ..config import load_config
        from ..web import provider_names

        profile = self.secrets.get("web_search:default") or {}
        provider = profile.get("provider") or load_config().web_search_provider or "duckduckgo"
        return {
            "provider": provider,
            "has_key": bool(profile.get("api_key")),
            "providers": provider_names(),
        }

    def set_web_search(self, provider: str, api_key: Optional[str] = None) -> dict[str, Any]:
        from ..web import provider_names

        if provider not in provider_names():
            return {"ok": False, "error": f"unknown provider: {provider}"}
        profile: dict[str, Any] = {"provider": provider}
        if api_key:
            profile["api_key"] = api_key
        self.secrets.put("web_search:default", profile)
        return {"ok": True, "provider": provider}

    # -- model providers (OpenAI, Ollama, …) ------------------------------------
    def get_providers(self) -> list[dict[str, Any]]:
        """Descriptor + per-provider status for the Settings UI. Never returns secret values;
        non-secret field values (e.g. the Ollama base URL) ARE returned so the form can prefill."""
        import os

        out: list[dict[str, Any]] = []
        for d in provider_descriptors():
            profile = self.secrets.get(f"provider:{d.name}") or {}
            if d.name == "openai":
                configured = bool(os.environ.get("OPENAI_API_KEY")) or bool(profile.get("api_key"))
            elif d.needs_key:
                configured = bool(profile.get("api_key"))
            else:
                configured = True  # keyless (Ollama) — usable out of the box
            values = {
                f.key: profile.get(f.key)
                for f in d.fields
                if not f.secret and profile.get(f.key)
            }
            out.append({
                **d.to_dict(),
                "configured": configured,
                "values": values,
                "suggested_models": self._suggested_models(d.name),
            })
        return out

    def _suggested_models(self, name: str) -> list[str]:
        """Bare model-name suggestions for the 'add model' form (datalist), per provider.
        OpenAI → the built-in list; Ollama → live `/api/tags` (best-effort)."""
        if name == "openai":
            return list(self.KNOWN_MODELS)
        if name == "ollama":
            return [m.split(":", 1)[-1] for m in self._ollama_models()]
        return []

    def set_provider(self, name: str, fields: Optional[dict[str, Any]]) -> dict[str, Any]:
        """Store a provider's config in its `provider:<name>` SecretStore profile and rebuild
        its cached client. Merges provided fields into any existing profile."""
        d = get_descriptor(name)
        if d is None:
            return {"ok": False, "error": f"unknown provider: {name}"}
        fields = fields or {}
        profile = dict(self.secrets.get(f"provider:{name}") or {})
        for f in d.fields:
            if f.key not in fields:
                continue
            val = fields.get(f.key)
            if isinstance(val, str):
                val = val.strip()
            if val:
                profile[f.key] = val
            elif not f.required:
                profile.pop(f.key, None)
        missing = [f.label for f in d.fields if f.required and not profile.get(f.key)]
        if missing:
            return {"ok": False, "error": "missing: " + ", ".join(missing)}
        self.secrets.put(f"provider:{name}", profile)
        self._refresh_provider(name)
        # Convenience: if the provider recommends a model and it's actually available, add it to
        # the curated list so it shows up in the composer right after configuring the provider.
        rec = d.recommended_model
        if rec and rec in self._suggested_models(name):
            self.add_model(f"{name}:{rec}")
        return {"ok": True, "provider": name, "recommended_model": rec}

    # -- settings / prefs (model API key, default model, onboarding) -------------
    KNOWN_MODELS = ["gpt-5.5", "gpt-4o", "gpt-4o-mini", "o3-mini", "deepseek-chat"]

    def _prefs_path(self) -> Path:
        return self._data_base / "prefs.json"

    def _load_prefs(self) -> dict[str, Any]:
        try:
            return json.loads(self._prefs_path().read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def _save_prefs(self) -> None:
        self._prefs_path().write_text(json.dumps(self._prefs, indent=2), encoding="utf-8")

    def _ollama_models(self) -> list[str]:
        """Live list of models pulled into the configured Ollama server (via its native
        `/api/tags`), as `ollama:<name>` so they're directly selectable. Empty if Ollama isn't
        configured or unreachable — best-effort, never raises."""
        profile = self.secrets.get("provider:ollama")
        if not profile:
            return []
        base = (profile.get("base_url") or "http://localhost:11434").strip().rstrip("/")
        if base.endswith("/v1"):
            base = base[: -len("/v1")]
        try:
            import httpx

            data = httpx.get(base + "/api/tags", timeout=2.0).json()
            return [f"ollama:{m['name']}" for m in data.get("models", []) if m.get("name")]
        except Exception:
            return []

    def _curated_models(self) -> list[str]:
        """The user-curated model list shown in the composer's selector. Persisted in prefs;
        defaults to the built-in OpenAI models on first run. The active default model is always
        included so it stays selectable."""
        models = self._prefs.get("models")
        if not isinstance(models, list) or not models:
            models = list(self.KNOWN_MODELS)
        return list(dict.fromkeys([self.model, *models]))

    def add_model(self, model: str) -> dict[str, Any]:
        """Add a model id (e.g. `gpt-4o`, `ollama:qwen2.5-coder:32b`) to the curated list."""
        model = (model or "").strip()
        if not model:
            return {"ok": False, "error": "empty model"}
        models = self._prefs.get("models")
        if not isinstance(models, list):
            models = list(self.KNOWN_MODELS)
        if model not in models:
            models.append(model)
        self._prefs["models"] = models
        self._save_prefs()
        return {"ok": True, **self.get_settings()}

    def remove_model(self, model: str) -> dict[str, Any]:
        """Remove a model id from the curated list."""
        models = self._prefs.get("models")
        if not isinstance(models, list):
            models = list(self.KNOWN_MODELS)
        self._prefs["models"] = [m for m in models if m != model]
        self._save_prefs()
        return {"ok": True, **self.get_settings()}

    def get_settings(self) -> dict[str, Any]:
        """Model-access + UI status. Never returns the key; `source` says where it comes from."""
        import os

        env_key = bool(os.environ.get("OPENAI_API_KEY"))
        stored = bool((self.secrets.get("provider:openai") or {}).get("api_key"))
        return {
            "provider": "openai",
            "model": self.model,
            "models": self._curated_models(),
            "has_key": env_key or stored,
            "source": "env" if env_key else ("store" if stored else None),
            "onboarded": bool(self._prefs.get("onboarded")),
            "surfaces": self._surfaces(),
        }

    def _surfaces(self) -> dict[str, bool]:
        """Which session surfaces are shown in the sidebar. Cowork is always on; Chat and Code
        are opt-in (default off) so a new user sees Cowork only."""
        return {
            "cowork": True,
            "chat": bool(self._prefs.get("show_chat", False)),
            "code": bool(self._prefs.get("show_code", False)),
        }

    def set_surfaces(self, chat: Optional[bool] = None, code: Optional[bool] = None) -> dict[str, Any]:
        """Toggle Chat/Code visibility (Cowork is always shown). Persisted in prefs."""
        if chat is not None:
            self._prefs["show_chat"] = bool(chat)
        if code is not None:
            self._prefs["show_code"] = bool(code)
        self._save_prefs()
        return {"ok": True, "surfaces": self._surfaces()}

    def set_model_key(self, api_key: str) -> dict[str, Any]:
        """Persist the model API key to the SecretStore (0600). The new provider client is
        built lazily on the next turn, so it picks the key up without a restart."""
        api_key = (api_key or "").strip()
        if not api_key:
            return {"ok": False, "error": "empty api key"}
        self.secrets.put("provider:openai", {"type": "api_key", "api_key": api_key})
        self._refresh_provider("openai")  # rebuild the OpenAI client with the new key
        return {"ok": True, **self.get_settings()}

    def set_default_model(self, model: str) -> dict[str, Any]:
        """Set + persist the default model for new sessions (the UI pre-selects it)."""
        model = (model or "").strip()
        if not model:
            return {"ok": False, "error": "empty model"}
        self.model = model
        self._prefs["default_model"] = model
        self._save_prefs()
        return {"ok": True, **self.get_settings()}

    def set_onboarded(self, value: bool = True) -> dict[str, Any]:
        """Record that first-run setup is complete (so it isn't shown again)."""
        self._prefs["onboarded"] = bool(value)
        self._save_prefs()
        return {"ok": True, "onboarded": bool(value)}

    # -- gateway / super-agent (inbound messaging) ------------------------------
    SUPERAGENT_SESSION_ID = "__superagent__"

    def _superagent_config_path(self) -> Path:
        return self._data_base / "superagent.json"

    def get_superagent_config(self) -> dict[str, Any]:
        try:
            return json.loads(self._superagent_config_path().read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def _superagent_workspace(self) -> Path:
        cfg = self.get_superagent_config()
        if cfg.get("workspace"):
            ws = Path(cfg["workspace"]).expanduser()
        else:
            ws = self._data_base / "superagent"
        ws.mkdir(parents=True, exist_ok=True)
        return ws.resolve()

    def set_superagent_workspace(self, path: str) -> dict[str, Any]:
        resolved = Path(path).expanduser()
        try:
            resolved.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            return {"ok": False, "error": str(exc)}
        return self._update_superagent_config(workspace=str(resolved.resolve()))

    def set_superagent_name(self, name: str) -> dict[str, Any]:
        name = (name or "").strip()
        if not name:
            return {"ok": False, "error": "name required"}
        return self._update_superagent_config(name=name)

    def _update_superagent_config(self, **changes: Any) -> dict[str, Any]:
        cfg = self.get_superagent_config()
        cfg.update(changes)
        self._superagent_config_path().write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        return {"ok": True, **changes, "restart_required": self.gateway is not None}

    def superagent_status(self) -> dict[str, Any]:
        from ..connectors import is_authorized as _is_auth
        from ..connectors.base import SessionSource

        connectors = []
        for name in ("telegram", "slack"):
            profile = self.secrets.get(f"{name}:default") or {}
            if not profile.get("bot_token"):
                continue
            allowed = list(profile.get("allowed_users") or [])
            allowed_set = set(allowed)
            recent = self.gateway.recent_senders(name) if self.gateway else []
            for r in recent:
                r["authorized"] = r.get("user_id") in allowed_set
            connectors.append(
                {
                    "name": name,
                    "account": profile.get("account"),
                    "listening": bool(self.gateway and name in self.gateway._adapters),
                    "allowed_users": allowed,
                    "recent": recent,
                }
            )
        return {
            "name": self.get_superagent_config().get("name") or "MyHelper",
            "workspace": str(self._superagent_workspace()),
            "running": self.gateway is not None,
            "connectors": connectors,
        }

    def allow_user(self, name: str, user_id: str) -> dict[str, Any]:
        return self._set_allowed(name, user_id, add=True)

    def disallow_user(self, name: str, user_id: str) -> dict[str, Any]:
        return self._set_allowed(name, user_id, add=False)

    def _set_allowed(self, name: str, user_id: str, *, add: bool) -> dict[str, Any]:
        user_id = str(user_id).strip()
        if not user_id:
            return {"ok": False, "error": "user_id required"}
        profile = self.secrets.get(f"{name}:default")
        if not profile:
            return {"ok": False, "error": "connector not connected"}
        allowed = set(profile.get("allowed_users") or [])
        allowed.add(user_id) if add else allowed.discard(user_id)
        profile["allowed_users"] = sorted(allowed)
        self.secrets.put(f"{name}:default", profile)
        # reflect into the live gateway so it takes effect without a restart
        if self.gateway is not None and name in self.gateway.settings:
            self.gateway.settings[name].allowed_users = set(allowed)
        return {"ok": True, "allowed_users": sorted(allowed)}

    def _build_superagent_engine(self) -> TurnEngine:
        ws = self._superagent_workspace()
        record = self.session_store.load(self.SUPERAGENT_SESSION_ID)
        name = self.get_superagent_config().get("name") or "MyHelper"
        engine = build_engine(
            agent=myhelper_agent(name),
            workspace=ws,
            model=self.model,
            mode=self.mode,
            approver=self._sa_approver,  # risky tools prompt the GUI surface (if watching)
            provider=self.provider,
            memory_store=self.memory_store,
            messages=record.messages if record else None,
            secrets=self.secrets,
            task_store=self.task_store,
            session_id=self.SUPERAGENT_SESSION_ID,
            audit_sink=self.audit_store.append,
        )
        engine.permissions.allow_tool_for_session("send_message")  # replies always go through
        engine.messages[0]["content"] += SUPERAGENT_MESSAGING_NOTE
        return engine

    async def start_gateway(self) -> list[str]:
        """Build the always-on super-agent (always) + gateway, and start enabled listeners.

        The super-agent runs even with no connector so the GUI surface can use it; adapters
        connect only for enabled connectors. Returns the platforms whose listeners came up.
        """
        settings = load_settings(self.secrets)
        engine = self._build_superagent_engine()
        self.superagent = SuperAgent(
            engine,
            on_saved=lambda: self.save(self.SUPERAGENT_SESSION_ID, engine),
            on_event=self._sa_broadcast,
        )
        self.gateway = Gateway(
            secrets=self.secrets, settings=settings, handler=self.superagent.on_message
        )
        for platform, st in settings.items():
            if not st.enabled:
                continue
            profile = self.secrets.get(f"{platform}:default") or {}
            adapter = make_adapter(platform, profile)
            if adapter is not None:
                self.gateway.register(adapter)
        self.superagent.start()
        self.scheduler.start()  # tick scheduler for automations (independent of connectors)
        return await self.gateway.start()

    async def stop_gateway(self) -> None:
        if self.gateway is not None:
            await self.gateway.stop()
            self.gateway = None
        if self.superagent is not None:
            await self.superagent.stop()
            self.superagent = None

    # -- GUI super-agent surface ------------------------------------------------
    def sa_register(self, send_cb: Any) -> None:
        self._sa_clients.add(send_cb)

    def sa_unregister(self, send_cb: Any) -> None:
        self._sa_clients.discard(send_cb)

    async def _sa_broadcast(self, message: dict) -> None:
        for cb in list(self._sa_clients):
            try:
                await cb(message)
            except Exception:
                self._sa_clients.discard(cb)

    async def _sa_approver(self, request) -> Any:
        from ..engine import ApprovalOutcome

        if not self._sa_clients:  # nobody watching → stay safe (deny writes/shell)
            return ApprovalOutcome.DENY
        self._sa_pending = asyncio.get_running_loop().create_future()
        try:
            decision = await asyncio.wait_for(self._sa_pending, timeout=300)
        except asyncio.TimeoutError:
            return ApprovalOutcome.DENY
        finally:
            self._sa_pending = None
        try:
            return ApprovalOutcome(decision)
        except ValueError:
            return ApprovalOutcome.DENY

    def sa_resolve_approval(self, decision: str) -> None:
        if self._sa_pending is not None and not self._sa_pending.done():
            self._sa_pending.set_result(decision)

    def sa_transcript(self) -> list[dict[str, Any]]:
        if self.superagent is not None:
            return self.superagent.engine.messages
        record = self.session_store.load(self.SUPERAGENT_SESSION_ID)
        return record.messages if record else []

    async def sa_user_message(self, text: str) -> bool:
        """Inject a message from the local GUI owner into the super-agent's thread."""
        if self.superagent is None:
            return False
        from ..connectors.base import MessageEvent, SessionSource

        event = MessageEvent(
            text=text,
            source=SessionSource(platform="gui", chat_id="gui", user_id="owner", user_name="you"),
        )
        await self.superagent.on_message(event)
        return True

    async def aclose(self) -> None:
        await self.scheduler.stop()
        await self.stop_gateway()
        await self.mcp.aclose()
        self.audit_store.close()

    # -- automation (scheduled tasks) -------------------------------------------
    def _scheduled_approver(self, task):
        from ..engine import ApprovalOutcome
        from ..permissions import WRITE_TOOLS

        allowed = set(task.always_allowed_tools)

        async def approver(request):
            # Unattended: auto-allow the deliverable writes (path-scoped to the task workspace)
            # + anything in the per-task "Always allowed" set; deny new consequential actions.
            if request.tool_name in WRITE_TOOLS or request.tool_name in allowed:
                return ApprovalOutcome.ONCE
            return ApprovalOutcome.DENY

        return approver

    def _build_task_engine(self, task, *, session_id: str) -> TurnEngine:
        ag = get_agent(task.agent)
        Path(task.workspace).mkdir(parents=True, exist_ok=True)
        engine = build_engine(
            agent=ag,
            workspace=task.workspace,
            model=task.model or self.model,
            mode=Mode.INTERACTIVE,
            approver=self._scheduled_approver(task),
            provider=self.provider,
            memory_store=self.memory_store,
            secrets=self.secrets,
            task_store=self.task_store,
            session_id=session_id,
            audit_sink=self.audit_store.append,
        )
        for tool in task.always_allowed_tools:
            engine.permissions.allow_tool_for_session(tool)
        return engine

    async def _run_scheduled_task(self, task, trigger: str) -> TaskRun:
        run = TaskRun(task_id=task.id, trigger=trigger)  # __post_init__ sets run.session_id
        self.task_store.add_run(run)  # mark "running"
        # Each run is a real, persisted conversation thread: it runs the instructions under its
        # own session id, then saves the transcript. The user can reopen that session and ask a
        # follow-up — the scheduled agent is no longer fire-and-forget.
        engine = self._build_task_engine(task, session_id=run.session_id)
        # The first turn is the task itself; a marker prefix makes the transcript read as a
        # scheduled run rather than a raw prompt.
        opening = f"⏰ Scheduled run — {task.title}\n\n{task.instructions}"
        try:
            async for _event in engine.run(opening):
                pass
            run.result_text = _last_assistant_text(engine.messages)
            run.artifacts = _recent_files(task.workspace, since=run.started_at)
            run.status = "ok"
            if task.notify_on_completion:
                await self._notify_task_done(task, run)
        except Exception as exc:
            run.status, run.error = "error", str(exc)
        finally:
            run.finished_at = _epoch()
            # Persist the run as a continuable session + keep the live engine for an immediate
            # follow-up; record the run (now carrying its session_id).
            try:
                self.save(run.session_id, engine)
                self._engines[run.session_id] = engine
            except Exception:
                pass
            self.task_store.add_run(run)
        return run

    async def _notify_task_done(self, task, run: TaskRun) -> None:
        summary = (run.result_text or "").strip()[:280]
        await self._sa_broadcast(
            {"type": "task_done", "data": {"task": task.title, "id": task.id, "text": summary, "run_id": run.run_id}}
        )
        if task.notify_target:
            from ..connectors.base import parse_target
            from ..connectors.senders import DEFAULT_SENDERS

            try:
                platform, chat_id, thread = parse_target(task.notify_target)
                sender = DEFAULT_SENDERS.get(platform)
                creds = self.secrets.get(f"{platform}:default") or {}
                if sender and creds.get("bot_token"):
                    await asyncio.to_thread(
                        sender, creds["bot_token"], chat_id, f"✓ {task.title}\n\n{summary}", thread
                    )
            except Exception:
                pass

    # -- automation REST --------------------------------------------------------
    def list_automations(self) -> dict[str, Any]:
        return {"tasks": [t.public() for t in self.task_store.list()]}

    def get_automation(self, task_id: str) -> dict[str, Any]:
        task = self.task_store.get(task_id)
        if task is None:
            return {"error": "not found"}
        return {"task": task.public(), "runs": [r.to_dict() for r in self.task_store.runs(task_id)]}

    def update_automation(self, task_id: str, changes: dict[str, Any]) -> dict[str, Any]:
        task = self.task_store.get(task_id)
        if task is None:
            return {"ok": False, "error": "not found"}
        if "enabled" in changes:
            task.enabled = bool(changes["enabled"])
        if changes.get("instructions") is not None:
            task.instructions = changes["instructions"]
        if changes.get("title") is not None:
            task.title = changes["title"]
        if changes.get("cron") is not None:
            from croniter import croniter

            if not croniter.is_valid(changes["cron"]):
                return {"ok": False, "error": "invalid cron"}
            task.schedule.cron, task.schedule.kind = changes["cron"], "cron"
        self.task_store.save(task)
        return {"ok": True, "task": task.public()}

    def delete_automation(self, task_id: str) -> dict[str, Any]:
        return {"ok": self.task_store.delete(task_id), "id": task_id}

    def prepare_manual_run(self, task_id: str) -> dict[str, Any]:
        """Create a 'running' manual run and return its session, so the GUI can open it and
        drive the task LIVE over the normal session WS (you watch the agent + follow up). The
        automatic scheduler path stays headless (`_run_scheduled_task`)."""
        task = self.task_store.get(task_id)
        if task is None:
            return {"ok": False, "error": "not found"}
        Path(task.workspace).mkdir(parents=True, exist_ok=True)
        run = TaskRun(task_id=task.id, trigger="manual")  # status "running", session_id auto
        self.task_store.add_run(run)
        return {
            "ok": True,
            "run_id": run.run_id,
            "session_id": run.session_id,
            "workspace": task.workspace,
            "agent": task.agent,
            "prompt": task.instructions,
        }

    def finalize_manual_run(self, task_id: str, run_id: str) -> dict[str, Any]:
        """Mark a manual run complete once its first turn finished (the WS already saved the
        session). Pulls result text + artifacts from the persisted transcript/workspace."""
        run = next((r for r in self.task_store.runs(task_id) if r.run_id == run_id), None)
        task = self.task_store.get(task_id)
        if run is None or task is None:
            return {"ok": False, "error": "not found"}
        if run.status == "running":
            record = self.session_store.load(run.session_id)
            run.result_text = _last_assistant_text(record.messages) if record else None
            run.artifacts = _recent_files(task.workspace, since=run.started_at)
            run.status = "ok"
            run.finished_at = _epoch()
            self.task_store.add_run(run)
            task.last_run, task.last_status = run.finished_at, "ok"
            task.run_count += 1
            self.task_store.save(task)
        return {"ok": True, "run": run.to_dict()}

    def save(self, session_id: str, engine: TurnEngine) -> None:
        executor = getattr(engine, "executor", None)
        workspace = os.path.realpath(str(executor.cwd)) if executor else ""
        self.session_store.save(
            SessionRecord(
                session_id=session_id,
                workspace=workspace,
                model=engine.model,
                mode=engine.permissions.mode.value,
                messages=engine.messages,
                title=title_from(engine.messages),
                agent=getattr(engine, "agent_name", "code"),
            )
        )

    def session_messages(self, session_id: str) -> list[dict[str, Any]]:
        record = self.session_store.load(session_id)
        return record.messages if record else []

    def rename_session(self, session_id: str, title: str) -> dict[str, Any]:
        if session_id.startswith("__"):
            return {"ok": False, "error": "internal sessions cannot be renamed"}
        ok = self.session_store.rename(session_id, title)
        return {"ok": ok, "session_id": session_id, "title": " ".join((title or "").split())[:120]}

    def delete_session(self, session_id: str) -> dict[str, Any]:
        if session_id.startswith("__"):
            return {"ok": False, "error": "internal sessions cannot be deleted here"}
        engine = self._engines.pop(session_id, None)
        if engine is not None:
            try:
                engine.interrupt()
            except Exception:
                pass
        ok = self.session_store.delete(session_id)
        return {"ok": ok, "session_id": session_id}

    # -- provider proxy ---------------------------------------------------------
    def provider_complete(self, model, messages, tools=None):
        return self.provider.complete(model=model, messages=messages, tools=tools)

    def _refresh_provider(self, name: Optional[str] = None) -> None:
        """Drop the router's cached client(s) so the next turn rebuilds with fresh config.
        No-op for an injected non-router provider (tests)."""
        invalidate = getattr(self.provider, "invalidate", None)
        if callable(invalidate):
            invalidate(name)

    # -- read models ------------------------------------------------------------
    def list_sessions(self, workspace: Optional[str] = None) -> list[dict[str, Any]]:
        ws = self.resolve_workspace(workspace) if workspace else None
        return [
            {
                "session_id": r.session_id,
                "title": r.title or "New session",
                "workspace": r.workspace,
                "agent": r.agent,
                "model": r.model,
                "mode": r.mode,
                "updated_at": r.updated_at,
                "messages": r.message_count,
            }
            for r in self.session_store.list(workspace=ws)
            if not r.session_id.startswith("__")  # hide superagent/task-run internal threads
        ]

    def list_agents(self) -> list[dict[str, Any]]:
        return _list_agents()

    def list_skills(self) -> list[dict[str, Any]]:
        loader = SkillLoader([Path.home() / ".config" / "coworker" / "skills"])
        return loader.catalog()

    def list_memory(self) -> list[dict[str, Any]]:
        return [
            {"id": m.id, "scope": m.scope.value, "content": m.content}
            for m in self.memory_store.list()
        ]

    def add_memory(self, content: str, scope: str = "workspace", workspace: Optional[str] = None) -> dict[str, Any]:
        chosen = Scope(scope) if scope in _SCOPES else Scope.WORKSPACE
        ws = self.resolve_workspace(workspace) if chosen is Scope.WORKSPACE else None
        item = self.memory_store.add(content, scope=chosen, workspace=ws)
        return {"id": item.id, "scope": item.scope.value, "content": item.content}


def _epoch() -> float:
    import time

    return time.time()


def _last_assistant_text(messages: list[dict[str, Any]]) -> Optional[str]:
    for msg in reversed(messages or []):
        if msg.get("role") == "assistant" and msg.get("content"):
            return msg["content"]
    return None


def _recent_files(workspace: str, *, since: float, limit: int = 20) -> list[str]:
    """Files in the task workspace modified during the run — the run's artifacts."""
    out: list[str] = []
    root = Path(workspace)
    if not root.is_dir():
        return out
    for path in root.rglob("*"):
        if any(part.startswith(".") for part in path.relative_to(root).parts):
            continue
        try:
            if path.is_file() and path.stat().st_mtime >= since - 1:
                out.append(str(path.relative_to(root)))
        except OSError:
            continue
        if len(out) >= limit:
            break
    return out


def _artifact_kind(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".md", ".markdown"}:
        return "markdown"
    if suffix in {".html", ".htm"}:
        return "html"
    if suffix in {".png", ".jpg", ".jpeg", ".webp", ".gif"}:
        return "image"
    if suffix in {".py", ".js", ".ts", ".tsx", ".css", ".json"}:
        return "code"
    return "text"


def _redact(raw: dict[str, Any]) -> dict[str, Any]:
    """Copy of a server config safe to return over REST — env/header values masked."""
    out = dict(raw)
    for key in ("env", "headers"):
        if isinstance(out.get(key), dict):
            out[key] = {k: ("***" if v else v) for k, v in out[key].items()}
    return out


def _git_branch(path: Path) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=path,
            capture_output=True,
            text=True,
            timeout=3,
        )
        branch = result.stdout.strip()
        return branch or None
    except (OSError, subprocess.SubprocessError):
        return None
