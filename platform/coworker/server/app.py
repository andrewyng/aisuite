"""FastAPI app — OpenAI-compatible endpoint + WS session API + REST.

The control plane every surface (GUI/IDE/messaging) rides on. The WS carries the engine
event stream and the approval channel; `/v1/chat/completions` is the OpenAI-compatible
proxy so any OpenAI-format client can use the runtime as a backend.
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from ..attachments import build_user_content
from ..engine import ApprovalOutcome
from ..inbox import VIS_INBOX, VIS_INLINE, args_preview
from ..permissions import Mode
from ..providers import AssistantTurn
from .manager import SessionManager


def create_app(manager: SessionManager) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        try:
            live = (
                await manager.start_gateway()
            )  # start messaging listeners (if configured)
            if live:
                print(f"[coworker] messaging gateway live: {', '.join(live)}")
        except Exception:  # never let a bad connector stop the server
            import traceback

            traceback.print_exc()
        yield
        await manager.aclose()  # stop gateway + close MCP connections on shutdown

    app = FastAPI(title="coworker", version="0.0.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # local-first; tighten when remote exposure lands
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.manager = manager

    @app.get("/v1/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "default_workspace": manager.default_workspace,
            "model": manager.model,
        }

    @app.get("/v1/agents")
    def agents() -> dict[str, Any]:
        return {"agents": manager.list_agents()}

    @app.get("/v1/personas")
    def personas() -> dict[str, Any]:
        return {"personas": manager.personas.list_all()}

    @app.get("/v1/inbox")
    def inbox(session_id: str = "", state: str = "") -> dict[str, Any]:
        from dataclasses import asdict

        # The cross-session Inbox list shows only Unattended (inbox-visibility) items; a per-session
        # query returns inline ones too, so the answer-in-context card sees parked attended prompts.
        items = manager.inbox.list(
            session_id=session_id or None,
            state=state or None,
            visibility=None if session_id else VIS_INBOX,
        )
        # Enrich with the originating session's context so the Inbox is self-contained — the
        # "go to session" chip needs title/agent/workspace without depending on a (possibly stale)
        # client-side session list, and can link straight to it.
        out: list[dict[str, Any]] = []
        for i in items:
            d = asdict(i)
            rec = manager.session_store.load(i.session_id)
            d["session_title"] = (rec.title if rec else None) or i.session_id
            d["session_agent"] = rec.agent if rec else None
            d["session_workspace"] = rec.workspace if rec else None
            d["session_exists"] = rec is not None
            out.append(d)
        return {"items": out}

    @app.post("/v1/inbox/{item_id}/resolve")
    async def resolve_inbox_item(item_id: str, body: dict) -> dict[str, Any]:
        # Idempotent + first-responder-wins: ok=False means it was already resolved elsewhere.
        # Routes through resolve_inbox so a restart-orphaned prompt durably resumes its turn.
        ok = await manager.resolve_inbox(item_id, str(body.get("resolution", "deny")))
        return {"ok": ok}

    @app.get("/v1/subscriptions")
    def subscriptions() -> dict[str, Any]:
        # Global view-only list: each (session → channel) subscription, enriched with the session's
        # title/agent and the channel its Inbox routes OUT to (so an inbound/outbound collision on
        # the same channel is visible).
        out: list[dict[str, Any]] = []
        for sub in manager.subscriptions.all():
            rec = manager.session_store.load(sub.session_id)
            agent = rec.agent if rec else ""
            routing = manager._routing_targets(sub.session_id, agent or "cowork")
            out.append(
                {
                    "session_id": sub.session_id,
                    "session_title": (rec.title if rec else None) or sub.session_id,
                    "agent": agent,
                    "channel": sub.channel,
                    "routing_target": routing[0] if routing else None,
                    "collision": bool(routing and sub.channel in routing),
                }
            )
        return {"subscriptions": out}

    @app.get("/v1/channels/recent")
    def recent_channels() -> dict[str, Any]:
        # The picker's "recently-seen" source: channels the bot has received messages from.
        return {"channels": manager.channel_buffer.channels()}

    @app.get("/v1/unrouted")
    def unrouted() -> dict[str, Any]:
        # Dead-letter view: inbound messages with no destination + background-turn failures.
        return {"items": manager.unrouted.list()}

    @app.post("/v1/subscriptions")
    def subscribe(body: dict) -> dict[str, Any]:
        from ..subscriptions import resolve_channel

        session_id = str(body.get("session_id", "")).strip()
        addr = resolve_channel(str(body.get("channel", "")))
        if not session_id or not addr or ":" not in addr:
            return {"ok": False, "error": "need a session_id and a channel"}
        manager.subscriptions.subscribe(session_id, addr)
        return {"ok": True, "channel": addr}

    @app.post("/v1/subscriptions/remove")
    def unsubscribe(body: dict) -> dict[str, Any]:
        from ..subscriptions import resolve_channel

        session_id = str(body.get("session_id", "")).strip()
        addr = resolve_channel(str(body.get("channel", "")))
        removed = manager.subscriptions.unsubscribe(session_id, addr)
        return {"ok": True, "removed": removed}

    @app.get("/v1/inbox/reconcile")
    def reconcile_inbox(session_id: str) -> dict[str, Any]:
        # Called when a session resumes attended control (surface pending + recap inline).
        return manager.inbox.reconcile_on_resume(session_id)

    @app.get("/v1/inbox/routing")
    def inbox_routing() -> dict[str, Any]:
        return {"bindings": manager.inbox_routing.bindings()}

    @app.post("/v1/inbox/routing/binding")
    def set_inbox_binding(body: dict) -> dict[str, Any]:
        name = str(body.get("name", "")).strip()
        if not name:
            return {"ok": False, "error": "binding needs a `name`"}
        manager.inbox_routing.set_binding(
            name,
            channel=body.get("channel") or None,
            target=str(body.get("target", "")),
        )
        return {"ok": True, "bindings": manager.inbox_routing.bindings()}

    @app.get("/v1/sessions/{session_id}/unattended")
    def get_unattended(session_id: str) -> dict[str, Any]:
        return {"unattended": manager.unattended.is_unattended(session_id)}

    @app.post("/v1/sessions/{session_id}/unattended")
    def set_unattended(session_id: str, body: dict) -> dict[str, Any]:
        # The GUI gates the on-transition behind a one-tap confirm.
        on = bool(body.get("unattended"))
        manager.unattended.set(session_id, on)
        return {"ok": True, "session_id": session_id, "unattended": on}

    @app.get("/v1/sessions/{session_id}/connections")
    def session_connections(session_id: str) -> dict[str, Any]:
        # §6: the Sources drawer payload — effective-enabled connectors + recommended + ⚠ count.
        return manager.session_connections_view(session_id)

    @app.post("/v1/sessions/{session_id}/connections")
    def set_session_connection(session_id: str, body: dict) -> dict[str, Any]:
        # §6: a session override. `clear` drops the override (inherit the persona default again);
        # otherwise set an explicit on/off. Return the refreshed view so the drawer can re-render.
        body = body or {}
        connector = str(body.get("connector", "")).strip()
        if not connector:
            return {"ok": False, "error": "connector required"}
        if body.get("clear"):
            manager.session_connections.clear(session_id, connector)
        else:
            manager.session_connections.set(
                session_id, connector, bool(body.get("enabled", False))
            )
        return {"ok": True, "connections": manager.session_connections_view(session_id)}

    @app.post("/v1/personas/install")
    def install_persona(body: dict) -> dict[str, Any]:
        # Returns a consent summary per persona; they land disabled pending the user's approval
        # (then POST /v1/personas/{id} {enabled:true, surfaced:true}).
        reg = manager.personas
        try:
            if body.get("git_url"):
                summaries = reg.install_from_git(str(body["git_url"]))
            elif body.get("dir"):
                summaries = reg.install_from_dir(str(body["dir"]))
            else:
                return {"ok": False, "error": "provide a `dir` or `git_url`"}
        except Exception as e:  # surface manifest/clone errors to the caller
            return {"ok": False, "error": str(e)}
        return {"ok": True, "consent": summaries, "personas": reg.list_all()}

    @app.post("/v1/personas/{persona_id}")
    def update_persona(persona_id: str, body: dict) -> dict[str, Any]:
        reg = manager.personas
        try:
            if "enabled" in body:
                reg.set_enabled(persona_id, bool(body["enabled"]))
            if "surfaced" in body:
                reg.set_surfaced(persona_id, bool(body["surfaced"]))
            if body.get("default"):
                reg.set_default(persona_id)
        except KeyError:
            return {"ok": False, "error": f"unknown persona: {persona_id}"}
        return {"ok": True, "personas": reg.list_all()}

    @app.get("/v1/personas/{persona_id}")
    def persona_detail(persona_id: str) -> dict[str, Any]:
        # §5 detail page: identity + capabilities + recommends(+connected) + default connections.
        detail = manager.persona_detail(persona_id)
        if detail is None:
            return {"ok": False, "error": f"unknown persona: {persona_id}"}
        return detail

    @app.post("/v1/personas/{persona_id}/enable")
    def persona_enable(persona_id: str, body: dict) -> dict[str, Any]:
        # Dedicated §5/§8 route; delegates to the same registry toggle as POST /v1/personas/{id}.
        try:
            manager.personas.set_enabled(
                persona_id, bool((body or {}).get("enabled", True))
            )
        except KeyError:
            return {"ok": False, "error": f"unknown persona: {persona_id}"}
        return {"ok": True, "personas": manager.personas.list_all()}

    @app.post("/v1/personas/{persona_id}/connections")
    def persona_set_connection(persona_id: str, body: dict) -> dict[str, Any]:
        # §5: flip a persona-default connector on/off; re-reads so the client can refresh.
        body = body or {}
        connector = str(body.get("connector", "")).strip()
        if not connector:
            return {"ok": False, "error": "connector required"}
        return manager.set_persona_connection(
            persona_id, connector, bool(body.get("enabled", False))
        )

    @app.get("/v1/skills")
    def skills() -> dict[str, Any]:
        return {"skills": manager.list_skills()}

    @app.get("/v1/workspaces/recent")
    def recent_workspaces() -> dict[str, Any]:
        return {"workspaces": manager.recent_workspaces()}

    @app.post("/v1/workspaces/open")
    def open_workspace(body: dict) -> dict[str, Any]:
        return manager.open_workspace(
            body.get("path", ""), create=bool(body.get("create"))
        )

    @app.get("/v1/sessions")
    def sessions(workspace: str | None = None) -> dict[str, Any]:
        return {"sessions": manager.list_sessions(workspace)}

    @app.get("/v1/sessions/{session_id}/messages")
    def session_messages(session_id: str) -> dict[str, Any]:
        return {"messages": manager.session_messages(session_id)}

    @app.patch("/v1/sessions/{session_id}")
    def session_patch(session_id: str, body: dict) -> dict[str, Any]:
        body = body or {}
        if "pinned" in body or "archived" in body:
            return manager.set_session_flags(
                session_id,
                pinned=bool(body["pinned"]) if "pinned" in body else None,
                archived=bool(body["archived"]) if "archived" in body else None,
            )
        return manager.rename_session(session_id, str(body.get("title", "")))

    @app.delete("/v1/sessions/{session_id}")
    def session_delete(session_id: str) -> dict[str, Any]:
        return manager.delete_session(session_id)

    @app.get("/v1/sessions/{session_id}/roots")
    def session_roots(session_id: str) -> dict[str, Any]:
        return {"roots": manager.get_roots(session_id)}

    @app.post("/v1/sessions/{session_id}/roots")
    def session_add_root(session_id: str, body: dict) -> dict[str, Any]:
        body = body or {}
        return manager.add_root(
            session_id, str(body.get("path", "")), bool(body.get("writable", False))
        )

    @app.delete("/v1/sessions/{session_id}/roots")
    def session_remove_root(session_id: str, path: str) -> dict[str, Any]:
        return manager.remove_root(session_id, path)

    @app.get("/v1/sessions/{session_id}/artifacts")
    def session_artifacts(session_id: str) -> dict[str, Any]:
        return {"artifacts": manager.list_artifacts(session_id)}

    @app.get("/v1/sessions/{session_id}/artifacts/read")
    def session_artifact_read(session_id: str, path: str) -> dict[str, Any]:
        return manager.read_artifact(session_id, path)

    @app.post("/v1/sessions/{session_id}/artifacts/reveal")
    def session_artifact_reveal(session_id: str, body: dict) -> dict[str, Any]:
        body = body or {}
        return manager.reveal_artifact(
            session_id, str(body.get("path", "")), str(body.get("mode", "reveal"))
        )

    @app.get("/v1/memory")
    def memory() -> dict[str, Any]:
        return {"memory": manager.list_memory()}

    @app.post("/v1/memory")
    def add_memory(body: dict) -> dict[str, Any]:
        return manager.add_memory(
            body.get("content", ""), body.get("scope", "workspace")
        )

    @app.post("/v1/chat/completions")
    def chat_completions(body: dict) -> dict[str, Any]:
        model = body.get("model", manager.model)
        turn = manager.provider_complete(
            model, body.get("messages", []), body.get("tools")
        )
        return _openai_response(model, turn)

    # -- MCP servers ------------------------------------------------------------
    @app.get("/v1/mcp")
    def mcp_list() -> dict[str, Any]:
        return {"servers": manager.list_mcp()}

    @app.post("/v1/mcp")
    def mcp_add(body: dict) -> dict[str, Any]:
        name = body.get("name")
        config = body.get("config")
        if not name or not isinstance(config, dict):
            return {"ok": False, "error": "name and config required"}
        return manager.add_mcp(name, config)

    @app.patch("/v1/mcp/{name}")
    def mcp_patch(name: str, body: dict) -> dict[str, Any]:
        return manager.patch_mcp(name, body or {})

    @app.delete("/v1/mcp/{name}")
    def mcp_delete(name: str) -> dict[str, Any]:
        return manager.delete_mcp(name)

    @app.get("/v1/mcp/{name}/tools")
    async def mcp_tools(name: str) -> dict[str, Any]:
        return await manager.mcp_tools(name)

    @app.post("/v1/mcp/reload")
    async def mcp_reload() -> dict[str, Any]:
        return await manager.reload_mcp()

    # -- connectors (Slack / Telegram / …) --------------------------------------
    @app.get("/v1/connectors")
    def connectors_list() -> dict[str, Any]:
        return {"connectors": manager.list_connectors()}

    @app.post("/v1/connectors/{name}/connect")
    async def connector_connect(name: str, body: dict) -> dict[str, Any]:
        fields = body.get("fields") if isinstance(body, dict) else None
        # experimental connectors require the caller to explicitly acknowledge the risk notice
        acknowledged = bool(isinstance(body, dict) and body.get("acknowledge_risk"))
        # token validation does a blocking HTTP call → keep it off the event loop
        return await asyncio.to_thread(
            lambda: manager.connect_connector(
                name, fields or {}, acknowledged=acknowledged
            )
        )

    @app.post("/v1/connectors/{name}/disconnect")
    def connector_disconnect(name: str) -> dict[str, Any]:
        return manager.disconnect_connector(name)

    @app.patch("/v1/connectors/{name}/tools")
    def connector_tools_patch(name: str, body: dict) -> dict[str, Any]:
        enabled = (body or {}).get("enabled")
        if not isinstance(enabled, dict):
            return {"ok": False, "error": "enabled map required"}
        return manager.update_connector_tools(name, enabled)

    @app.post("/v1/connectors/{name}/allow")
    def connector_allow(name: str, body: dict) -> dict[str, Any]:
        return manager.allow_user(name, str(body.get("user_id", "")))

    @app.post("/v1/connectors/{name}/disallow")
    def connector_disallow(name: str, body: dict) -> dict[str, Any]:
        return manager.disallow_user(name, str(body.get("user_id", "")))

    # -- audit / browser observability ------------------------------------------
    @app.get("/v1/audit")
    def audit_list(
        limit: int = 100,
        session_id: str | None = None,
        connector: str | None = None,
        tool: str | None = None,
    ) -> dict[str, Any]:
        return {
            "events": manager.list_audit(
                limit=limit, session_id=session_id, connector=connector, tool=tool
            )
        }

    @app.get("/v1/browser/state")
    def browser_state_get() -> dict[str, Any]:
        return manager.browser_state()

    @app.post("/v1/browser/screenshot")
    def browser_screenshot_post() -> dict[str, Any]:
        return manager.browser_screenshot()

    @app.post("/v1/browser/close")
    def browser_close_post() -> dict[str, Any]:
        return manager.browser_close()

    # -- web search -------------------------------------------------------------
    @app.get("/v1/web-search")
    def web_search_get() -> dict[str, Any]:
        return manager.get_web_search()

    @app.post("/v1/web-search")
    def web_search_set(body: dict) -> dict[str, Any]:
        provider = (body or {}).get("provider", "")
        if not provider:
            return {"ok": False, "error": "provider required"}
        return manager.set_web_search(provider, (body or {}).get("api_key"))

    # -- model providers (OpenAI, Ollama, …) ------------------------------------
    @app.get("/v1/providers")
    def providers_get() -> list[dict[str, Any]]:
        return manager.get_providers()

    @app.post("/v1/providers")
    def providers_set(body: dict) -> dict[str, Any]:
        name = (body or {}).get("name", "")
        if not name:
            return {"ok": False, "error": "name required"}
        return manager.set_provider(name, (body or {}).get("fields"))

    @app.post("/v1/providers/verify")
    async def providers_verify(body: dict) -> dict[str, Any]:
        # Live read-only credential check (sync httpx) — run off the event loop.
        name = (body or {}).get("name", "") or "openai"
        return await asyncio.to_thread(
            manager.verify_provider, name, (body or {}).get("fields")
        )

    # -- settings (model API key) -----------------------------------------------
    @app.get("/v1/settings")
    def settings_get() -> dict[str, Any]:
        return manager.get_settings()

    @app.post("/v1/settings/model-key")
    def settings_set_model_key(body: dict) -> dict[str, Any]:
        return manager.set_model_key((body or {}).get("api_key", ""))

    @app.post("/v1/settings/default-model")
    def settings_set_default_model(body: dict) -> dict[str, Any]:
        return manager.set_default_model((body or {}).get("model", ""))

    @app.post("/v1/settings/models/add")
    def settings_models_add(body: dict) -> dict[str, Any]:
        return manager.add_model((body or {}).get("model", ""))

    @app.post("/v1/settings/models/remove")
    def settings_models_remove(body: dict) -> dict[str, Any]:
        return manager.remove_model((body or {}).get("model", ""))

    @app.post("/v1/settings/onboarded")
    def settings_set_onboarded(body: dict) -> dict[str, Any]:
        return manager.set_onboarded(bool((body or {}).get("value", True)))

    @app.post("/v1/settings/experimental-connectors")
    def settings_set_experimental(body: dict) -> dict[str, Any]:
        return manager.set_experimental_connectors(bool((body or {}).get("value")))

    @app.post("/v1/settings/surfaces")
    def settings_set_surfaces(body: dict) -> dict[str, Any]:
        b = body or {}
        return manager.set_surfaces(chat=b.get("chat"), code=b.get("code"))

    @app.post("/v1/settings/scratch-base")
    def settings_set_scratch_base(body: dict) -> dict[str, Any]:
        return manager.set_scratch_base(str((body or {}).get("path", "")))

    # -- direct-message routing -------------------------------------------------
    @app.get("/v1/messaging/dm-route")
    def dm_route_get() -> dict[str, Any]:
        return {"dm_session": manager.dm_session()}

    @app.post("/v1/messaging/dm-route")
    def dm_route_set(body: dict) -> dict[str, Any]:
        # A falsy session_id clears the designation (DMs then park as unrouted).
        return manager.set_dm_session((body or {}).get("session_id", ""))

    if os.environ.get("COWORKER_DEBUG_INJECT") == "1":
        # Dev-only (env-gated, localhost): feed a message through the real inbound path so the
        # messaging stack can be exercised without a live bot connection. Not registered otherwise.
        @app.post("/v1/_debug/inject_inbound")
        async def debug_inject_inbound(body: dict) -> dict[str, Any]:
            from ..connectors.base import MessageEvent, SessionSource

            event = MessageEvent(
                text=str((body or {}).get("text", "")),
                source=SessionSource(
                    platform=str(body.get("platform", "slack")),
                    chat_id=str(body.get("chat_id", "C0BD7KZ1AH5")),
                    user_id=str(body.get("user_id", "U07JK68S4BH")),
                    user_name=str(body.get("user_name", "tester")),
                    chat_type=str(body.get("chat_type", "channel")),
                ),
            )
            await manager._dispatch_inbound(event)
            return {"ok": True}

    # -- automations (scheduled tasks) ------------------------------------------
    @app.get("/v1/automations")
    def automations_list() -> dict[str, Any]:
        return manager.list_automations()

    @app.post("/v1/automations")
    def automations_create(body: dict) -> dict[str, Any]:
        return manager.create_automation(body or {})

    @app.get("/v1/automations/{task_id}")
    def automation_get(task_id: str) -> dict[str, Any]:
        return manager.get_automation(task_id)

    @app.patch("/v1/automations/{task_id}")
    def automation_update(task_id: str, body: dict) -> dict[str, Any]:
        return manager.update_automation(task_id, body or {})

    @app.delete("/v1/automations/{task_id}")
    def automation_delete(task_id: str) -> dict[str, Any]:
        return manager.delete_automation(task_id)

    @app.post("/v1/automations/{task_id}/run")
    def automation_run(task_id: str) -> dict[str, Any]:
        # Prepare a live manual run; the GUI opens the returned session and drives it.
        return manager.prepare_manual_run(task_id)

    @app.post("/v1/automations/{task_id}/runs/{run_id}/finalize")
    def automation_run_finalize(task_id: str, run_id: str) -> dict[str, Any]:
        return manager.finalize_manual_run(task_id, run_id)

    @app.websocket("/ws/session/{session_id}")
    async def ws_session(ws: WebSocket, session_id: str) -> None:
        await ws.accept()
        agent = ws.query_params.get("agent") or "code"

        # All four interactive prompts (approval / question / directory / plan) are parked as Inbox
        # items and awaited via inbox.wait — so they survive a dropped socket (redelivered on
        # reconnect) and can be resolved from any surface. `visibility` decides where they SHOW:
        # Unattended → the cross-session Inbox; attended → inline in this session only. The agent
        # stays blocked until the item is resolved (live WS response, REST, or a bound channel).
        def _visibility() -> str:
            return (
                VIS_INBOX
                if manager.unattended.is_unattended(session_id)
                else VIS_INLINE
            )

        async def _mirror(item) -> None:
            # Unattended items mirror to a bound channel as buttons (see mirror_inbox_item).
            await manager.mirror_inbox_item(item)

        def _route() -> str:
            return manager.inbox_routing.route_for(session_id, agent)

        async def approver(_request) -> ApprovalOutcome:
            # The engine has already emitted PERMISSION_REQUIRED (the live inline card). Park the
            # item so the answer can also come from the Inbox / a reconnect / after a restart.
            item = manager.inbox.add_approval(
                session_id,
                f"Run `{_request.tool_name}`?",
                body="\n".join(
                    p
                    for p in (
                        (getattr(_request, "reason", "") or "").strip(),
                        args_preview(getattr(_request, "arguments", None)),
                    )
                    if p
                ),
                inbox=_route(),
                visibility=_visibility(),
                tool_call_id=getattr(_request, "tool_call_id", None),
            )
            if (
                item.state == "pending"
            ):  # freshly raised (not a durable-resume re-raise)
                manager.persist_session(
                    session_id
                )  # the pending tool call is now on disk
                if item.visibility == VIS_INBOX:
                    await _mirror(item)
            resolution = await manager.inbox.wait(item.id)
            # Accept both vocabularies: the live card sends once/always_tool/always_command/deny;
            # the Inbox / a channel send allow/always/deny.
            try:
                return ApprovalOutcome(resolution)
            except ValueError:
                pass
            if resolution == "allow":
                return ApprovalOutcome.ONCE
            if resolution == "always":
                return ApprovalOutcome.ALWAYS_TOOL
            return ApprovalOutcome.DENY

        async def question_asker(args: dict, tool_call_id=None) -> dict:
            # ask_user (engine does NOT emit the event — we do, only when attended).
            item = manager.inbox.add_question(
                session_id,
                str(args.get("question", "")),
                inbox=_route(),
                visibility=_visibility(),
                options=list(args.get("options") or []),
                allow_text=bool(args.get("allow_text", True)),
                multi=bool(args.get("multi", False)),
                tool_call_id=tool_call_id,
            )
            if item.state == "pending":
                manager.persist_session(session_id)
                if item.visibility == VIS_INBOX:
                    await _mirror(item)
                else:
                    await ws.send_json(
                        {
                            "type": "question_requested",
                            "data": {
                                "question": item.title,
                                "options": item.options,
                                "allow_text": item.allow_text,
                                "multi": item.multi,
                                "header": str(args.get("header", "")),
                            },
                        }
                    )
            return {"answer": await manager.inbox.wait(item.id)}

        async def directory_requester(args: dict, tool_call_id=None) -> dict:
            # The engine has already emitted DIRECTORY_REQUESTED. Park, await, then apply the grant.
            item = manager.inbox.add_directory(
                session_id,
                "Grant access to a folder?",
                body=str(args.get("reason", "")),
                inbox=_route(),
                visibility=_visibility(),
                data={
                    "path": str(args.get("path", "")),
                    "writable": bool(args.get("writable", False)),
                },
                tool_call_id=tool_call_id,
            )
            if item.state == "pending":
                manager.persist_session(session_id)
                if item.visibility == VIS_INBOX:
                    await _mirror(item)
            resp = _parse_json(
                await manager.inbox.wait(item.id)
            )  # {granted, path, writable}
            if not resp.get("granted"):
                return {"granted": False, "reason": "the user declined the request"}
            path = (resp.get("path") or args.get("path") or "").strip()
            if not path:
                return {"granted": False, "error": "no directory was provided"}
            writable = bool(resp.get("writable", args.get("writable", False)))
            res = manager.add_root(session_id, path, writable)
            if not res.get("ok"):
                return {
                    "granted": False,
                    "error": res.get("error", "could not grant access"),
                }
            primary = next(
                (
                    r
                    for r in res.get("roots", [])
                    if r.get("path")
                    and Path(r["path"]).expanduser().resolve()
                    == Path(path).expanduser().resolve()
                ),
                None,
            )
            return {
                "granted": True,
                "path": (primary or {}).get("path", path),
                "writable": writable,
            }

        async def plan_approver(_args: dict, tool_call_id=None) -> dict:
            # The engine has already emitted PLAN_PROPOSED. Park, await the verdict.
            item = manager.inbox.add_plan(
                session_id,
                "Approve the plan?",
                body=str(_args.get("plan", "")),
                inbox=_route(),
                visibility=_visibility(),
                tool_call_id=tool_call_id,
            )
            if item.state == "pending":
                manager.persist_session(session_id)
                if item.visibility == VIS_INBOX:
                    await _mirror(item)
            resp = _parse_json(
                await manager.inbox.wait(item.id)
            )  # {approved, mode, feedback}
            if not resp.get("approved"):
                return {
                    "approved": False,
                    "feedback": resp.get("feedback") or "the user rejected the plan",
                }
            return {"approved": True, "mode": resp.get("mode") or "interactive"}

        def _resolve_pending(resolution: str) -> None:
            # Live WS responses resolve THE session's single pending prompt (one at a time, since the
            # agent blocks). Reconnect / Inbox resolve by id via REST instead.
            pend = manager.inbox.pending(session_id)
            if pend:
                manager.inbox.resolve(pend[0].id, resolution)

        workspace = ws.query_params.get("workspace")
        mcp_tools = await manager.prepare_mcp_tools(
            session_id, workspace=workspace, agent=agent
        )
        engine = manager.get_engine(
            session_id,
            workspace=workspace,
            agent=agent,
            approver=approver,
            extra_tools=mcp_tools,
            directory_requester=directory_requester,
            plan_approver=plan_approver,
            question_asker=question_asker,
        )
        if engine is None:
            await ws.send_json(
                {
                    "type": "error",
                    "data": {
                        "error": "no valid workspace — choose a project folder first"
                    },
                }
            )
            await ws.close()
            return
        await ws.send_json(
            {
                "type": "ready",
                "data": {
                    "session_id": session_id,
                    "agent": getattr(engine, "agent_name", "code"),
                    "model": engine.model,
                    "mode": engine.permissions.mode.value,
                    "workspace": (
                        str(getattr(engine, "executor").cwd)
                        if getattr(engine, "executor", None)
                        else None
                    ),
                },
            }
        )

        # Checkpoint events: persist mid-turn so a crash/quit can't eat the conversation.
        # turn_start = the user message just landed (a brand-new session gets its row here,
        # not at connect — empty never-used sessions shouldn't appear in Recents);
        # permission_required/directory_requested = parked indefinitely on the user;
        # iteration_end = a model response + its tool results completed.
        _CHECKPOINTS = {
            "turn_start",
            "permission_required",
            "directory_requested",
            "plan_proposed",
            "iteration_end",
        }

        async def run_turn(content) -> None:
            manager.mark_running(
                session_id
            )  # busy → self-wakes steer instead of colliding
            try:
                async for event in engine.run(content):
                    # Broadcast to every socket viewing this session (this socket included — it's a
                    # registered client), so a second view of the same session stays in sync too.
                    await manager.broadcast_session(
                        session_id, {"type": event.type.value, "data": event.data}
                    )
                    if event.type.value in _CHECKPOINTS:
                        manager.save(session_id, engine)
            finally:
                manager.mark_idle(session_id)
                manager.save(session_id, engine)
                await manager.broadcast_session(
                    session_id, {"type": "turn_done", "data": {}}
                )

        # This socket is now a live view of the session; background turns (channel delivery,
        # self-wake, durable resume) broadcast here too, not just locally driven run_turns.
        manager.register_session_client(session_id, ws.send_json)
        try:
            while True:
                message = await ws.receive_json()
                kind = message.get("type")
                if kind == "approval":
                    _resolve_pending(message.get("decision", "deny"))
                elif kind == "directory_response":
                    _resolve_pending(
                        json.dumps(
                            {
                                "granted": bool(message.get("granted")),
                                "path": message.get("path", ""),
                                "writable": bool(message.get("writable", False)),
                            }
                        )
                    )
                elif kind == "plan_response":
                    _resolve_pending(
                        json.dumps(
                            {
                                "approved": bool(message.get("approved")),
                                "mode": message.get("mode", "interactive"),
                                "feedback": message.get("feedback", ""),
                            }
                        )
                    )
                elif kind == "question_response":
                    _resolve_pending(str(message.get("answer", "")))
                elif kind == "interrupt":
                    engine.request_interrupt()
                elif kind == "set_mode":
                    try:
                        engine.permissions.mode = Mode(message.get("mode"))
                    except ValueError:
                        pass
                elif kind == "set_model":
                    model = message.get("model")
                    if model:
                        engine.model = model
                elif kind == "user_message":
                    text = (message.get("text") or "").strip()
                    attachments = message.get("attachments") or []
                    if text or attachments:
                        content = build_user_content(text, attachments)
                        asyncio.create_task(run_turn(content))
        except WebSocketDisconnect:
            pass
        finally:
            manager.unregister_session_client(session_id, ws.send_json)

    return app


def _parse_json(s: str) -> dict[str, Any]:
    """Parse a structured Inbox resolution (directory/plan carry their reply as a JSON string)."""
    try:
        v = json.loads(s) if s else {}
        return v if isinstance(v, dict) else {}
    except Exception:
        return {}


def _openai_response(model: str, turn: AssistantTurn) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": turn.text or ""}
    if turn.tool_calls:
        message["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
            }
            for tc in turn.tool_calls
        ]
    return {
        "id": "chatcmpl-" + uuid.uuid4().hex[:12],
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": turn.finish_reason or "stop",
            }
        ],
    }
