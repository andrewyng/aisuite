"""Tests for the messaging connector core (C2 increment 1): targets, the send_message
tool, settings/authorization, and the gateway inbound loop — all offline via FakeAdapter.
"""

from __future__ import annotations

import asyncio

import pytest

from coworker.connectors import (
    ConnectorSettings,
    FakeAdapter,
    Gateway,
    MessageEvent,
    SessionSource,
    format_target,
    is_authorized,
    make_send_message_tool,
    parse_target,
)
from coworker.connectors.base import SendResult
from coworker.secrets import SecretStore


# -- target tokens -------------------------------------------------------------
def test_target_round_trip():
    assert format_target("telegram", "12345") == "telegram:12345"
    assert format_target("slack", "C1", "168.9") == "slack:C1:168.9"
    assert parse_target("telegram:12345") == ("telegram", "12345", None)
    assert parse_target("slack:C1:168.9") == ("slack", "C1", "168.9")


def test_target_invalid():
    for bad in ("", "telegram", "telegram:", ":123"):
        with pytest.raises(ValueError):
            parse_target(bad)


def test_session_source_target_and_label():
    s = SessionSource(
        platform="telegram", chat_id="42", user_name="Alice", chat_type="dm"
    )
    assert s.target == "telegram:42"
    assert "Alice" in s.label() and "telegram" in s.label()


def test_message_tagged_text_carries_reply_handle():
    s = SessionSource(
        platform="slack", chat_id="C9", user_name="Bob", chat_type="channel"
    )
    ev = MessageEvent(text="ship it", source=s)
    tag = ev.tagged_text()
    assert "reply→slack:C9" in tag and "ship it" in tag


# -- send_message tool ---------------------------------------------------------
def _fake_senders(record):
    def sender(token, chat_id, text, thread_id=None):
        record.append(
            {"token": token, "chat_id": chat_id, "text": text, "thread_id": thread_id}
        )
        return SendResult(True, message_id="99")

    return {"telegram": sender, "slack": sender}


def test_send_message_success(tmp_path):
    secrets = SecretStore(tmp_path / "secrets.json")
    secrets.put("telegram:default", {"type": "token", "bot_token": "T0K"})
    record = []
    tool = make_send_message_tool(secrets, senders=_fake_senders(record))

    out = tool(target="telegram:12345", text="hello")
    assert out == {"ok": True, "message_id": "99", "target": "telegram:12345"}
    assert record == [
        {"token": "T0K", "chat_id": "12345", "text": "hello", "thread_id": None}
    ]
    # tool carries gating metadata + an explicit schema
    assert tool.__aisuite_tool_metadata__.requires_approval is True
    assert tool.__coworker_schema__["function"]["name"] == "send_message"


def test_send_message_missing_token(tmp_path):
    secrets = SecretStore(tmp_path / "secrets.json")
    tool = make_send_message_tool(secrets, senders=_fake_senders([]))
    assert "error" in tool(target="telegram:1", text="x")


def test_send_message_unknown_platform(tmp_path):
    tool = make_send_message_tool(
        SecretStore(tmp_path / "secrets.json"), senders=_fake_senders([])
    )
    assert "unknown platform" in tool(target="discord:1", text="x")["error"]


def test_send_message_bad_target(tmp_path):
    tool = make_send_message_tool(
        SecretStore(tmp_path / "secrets.json"), senders=_fake_senders([])
    )
    assert "error" in tool(target="nonsense", text="x")


# -- settings / authorization --------------------------------------------------
def test_is_authorized():
    s = ConnectorSettings(platform="telegram", allowed_users={"u1"})
    assert is_authorized(s, SessionSource("telegram", "c", user_id="u1"))
    assert not is_authorized(s, SessionSource("telegram", "c", user_id="u2"))
    # empty allowlist = nobody
    assert not is_authorized(
        ConnectorSettings("telegram"), SessionSource("telegram", "c", user_id="u1")
    )
    # allow_all opens it
    assert is_authorized(
        ConnectorSettings("telegram", allow_all=True),
        SessionSource("telegram", "c", user_id="x"),
    )


def test_load_settings_from_secretstore(tmp_path, monkeypatch):
    monkeypatch.delenv("TELEGRAM_ALLOWED_USERS", raising=False)
    secrets = SecretStore(tmp_path / "secrets.json")
    secrets.put(
        "telegram:default", {"type": "token", "bot_token": "T", "allowed_users": ["u1"]}
    )
    settings = __import__(
        "coworker.connectors.config", fromlist=["load_settings"]
    ).load_settings(secrets)
    assert settings["telegram"].enabled is True
    assert settings["telegram"].allowed_users == {"u1"}
    assert settings["slack"].enabled is False  # no token


def test_load_settings_env_allowlist(tmp_path, monkeypatch):
    monkeypatch.setenv("TELEGRAM_ALLOWED_USERS", "a, b ,c")
    secrets = SecretStore(tmp_path / "secrets.json")
    secrets.put("telegram:default", {"bot_token": "T"})
    settings = __import__(
        "coworker.connectors.config", fromlist=["load_settings"]
    ).load_settings(secrets)
    assert settings["telegram"].allowed_users == {"a", "b", "c"}


# -- gateway inbound loop (FakeAdapter) ----------------------------------------
async def test_gateway_dispatches_authorized():
    received: list[MessageEvent] = []

    async def handler(ev: MessageEvent) -> None:
        received.append(ev)

    settings = {"fake": ConnectorSettings("fake", enabled=True, allowed_users={"u1"})}
    gw = Gateway(settings=settings, handler=handler)
    fake = FakeAdapter()
    gw.register(fake)
    live = await gw.start()
    assert live == ["fake"] and fake.connected

    await fake.inject("hi", user_id="u1")
    assert len(received) == 1 and received[0].text == "hi"

    await fake.inject("nope", user_id="intruder")  # not in allowlist
    assert len(received) == 1  # dropped

    await gw.stop()
    assert not fake.connected


async def test_gateway_deliver_via_adapter():
    gw = Gateway(
        settings={"fake": ConnectorSettings("fake", enabled=True, allow_all=True)}
    )
    fake = FakeAdapter()
    gw.register(fake)
    result = await gw.deliver("fake:c9", "pong")
    assert result.ok
    assert fake.outbox == [{"chat_id": "c9", "text": "pong", "thread_id": None}]


async def test_gateway_full_echo_loop():
    """Inbound → handler replies via deliver → lands in the adapter outbox."""
    gw = Gateway(
        settings={"fake": ConnectorSettings("fake", enabled=True, allow_all=True)}
    )
    fake = FakeAdapter()

    async def echo(ev: MessageEvent) -> None:
        await gw.deliver(ev.source.target, f"echo: {ev.text}")

    gw.set_handler(echo)
    gw.register(fake)
    await fake.inject("ping", chat_id="c1", user_id="u1")
    assert fake.outbox == [{"chat_id": "c1", "text": "echo: ping", "thread_id": None}]


# -- engine integration: send_message appears only when a connector is configured ----
class _StubProvider:
    """Minimal ProviderClient stand-in (build_engine never calls it)."""

    def complete(self, **_kw):  # pragma: no cover - never invoked at build time
        from coworker.providers import AssistantTurn

        return AssistantTurn()

    def capabilities(self, _model):  # pragma: no cover
        from coworker.providers.base import ModelCapabilities

        return ModelCapabilities()

    def stream(self, **_kw):  # pragma: no cover
        from coworker.providers.base import StreamChunk

        yield StreamChunk(turn=self.complete())


def test_engine_connector_tools_are_cowork_scoped(tmp_path):
    from coworker.agent import build_engine
    from coworker.agents import chat_agent, code_agent, cowork_agent, myhelper_agent

    secrets = SecretStore(tmp_path / "secrets.json")
    eng = build_engine(agent=chat_agent(), provider=_StubProvider(), secrets=secrets)
    assert "send_message" not in eng.registry.names()  # no connector yet
    assert "browser_read_url" not in eng.registry.names()

    secrets.put("telegram:default", {"bot_token": "T"})
    chat = build_engine(agent=chat_agent(), provider=_StubProvider(), secrets=secrets)
    code = build_engine(
        agent=code_agent(),
        workspace=tmp_path,
        provider=_StubProvider(),
        secrets=secrets,
    )
    cowork = build_engine(
        agent=cowork_agent(),
        workspace=tmp_path,
        provider=_StubProvider(),
        secrets=secrets,
    )
    helper = build_engine(
        agent=myhelper_agent(),
        workspace=tmp_path,
        provider=_StubProvider(),
        secrets=secrets,
    )

    assert "send_message" not in chat.registry.names()
    assert "send_message" not in code.registry.names()
    assert "browser_read_url" not in chat.registry.names()
    assert "browser_read_url" not in code.registry.names()

    assert "send_message" in cowork.registry.names()
    assert "browser_read_url" in cowork.registry.names()
    assert "browser_open_url" in cowork.registry.names()
    assert "browser_click" in cowork.registry.names()
    assert "browser_type" in cowork.registry.names()
    assert "github_search" not in cowork.registry.names()
    assert "send_message" in helper.registry.names()
    assert "browser_read_url" not in helper.registry.names()
    assert "browser_open_url" not in helper.registry.names()

    assert cowork.registry.get("browser_open_url").metadata.requires_approval is True
    assert cowork.registry.get("browser_snapshot").metadata.requires_approval is True
    assert cowork.registry.get("browser_click").metadata.requires_approval is True
    assert cowork.registry.get("browser_type").metadata.requires_approval is True
    cowork.permissions.allow_tool_for_session("browser_click")
    decision = cowork.permissions.evaluate(
        "browser_click",
        {"target": "button"},
        cowork.registry.get("browser_click").metadata,
    )
    assert decision.needs_user is True

    secrets.put("github:default", {"token": "ghp_test", "enabled": True})
    cowork_with_github = build_engine(
        agent=cowork_agent(),
        workspace=tmp_path,
        provider=_StubProvider(),
        secrets=secrets,
    )
    assert "github_search" in cowork_with_github.registry.names()
    assert (
        cowork_with_github.registry.get("github_search").metadata.requires_approval
        is True
    )


# -- connector setup (descriptors / connect / disconnect / list) ---------------
def test_connector_list_descriptors(tmp_path):
    from coworker.connectors import connector_list

    by_name = {
        c["name"]: c for c in connector_list(SecretStore(tmp_path / "secrets.json"))
    }
    assert (
        by_name["telegram"]["two_way"] is True
        and by_name["telegram"]["connected"] is False
    )
    assert (
        by_name["gmail"]["available"] is True and by_name["gmail"]["connected"] is False
    )
    assert (
        by_name["browser"]["available"] is True
        and by_name["browser"]["connected"] is True
    )
    assert (
        by_name["github"]["available"] is True
        and by_name["github"]["connected"] is False
    )
    assert any(
        t["name"] == "browser_open_url" and t["requires_approval"]
        for t in by_name["browser"]["tools"]
    )
    # telegram exposes a bot_token field + setup instructions
    keys = {f["key"] for f in by_name["telegram"]["fields"]}
    assert "bot_token" in keys and by_name["telegram"]["instructions"]


def test_connector_list_connected_for_required_profiles(tmp_path):
    from coworker.connectors import (
        connect_connector,
        connector_list,
        update_connector_tools,
    )

    secrets = SecretStore(tmp_path / "secrets.json")
    assert (
        connect_connector(secrets, "github", {"token": "ghp_test"}, validate=False)[
            "ok"
        ]
        is True
    )
    assert (
        connect_connector(
            secrets,
            "jira",
            {
                "base_url": "https://example.atlassian.net",
                "email": "me@example.com",
                "api_token": "tok",
            },
            validate=False,
        )["ok"]
        is True
    )

    by_name = {c["name"]: c for c in connector_list(secrets)}
    assert (
        by_name["github"]["connected"] is True and by_name["github"]["enabled"] is True
    )
    assert by_name["jira"]["connected"] is True and by_name["jira"]["enabled"] is True

    assert (
        update_connector_tools(secrets, "github", {"github_search": False})["ok"]
        is True
    )
    by_name = {c["name"]: c for c in connector_list(secrets)}
    gh_tools = {t["name"]: t for t in by_name["github"]["tools"]}
    assert gh_tools["github_search"]["enabled"] is False
    assert gh_tools["github_get_issue"]["enabled"] is True


def test_connect_disconnect_no_validate(tmp_path):
    from coworker.connectors import (
        connect_connector,
        connector_list,
        disconnect_connector,
    )

    secrets = SecretStore(tmp_path / "secrets.json")
    res = connect_connector(
        secrets,
        "telegram",
        {"bot_token": "T0K", "allowed_users": "u1, u2"},
        validate=False,
    )
    assert res["ok"] is True
    profile = secrets.get("telegram:default")
    assert profile["bot_token"] == "T0K" and profile["allowed_users"] == ["u1", "u2"]
    assert profile["enabled"] is True

    listed = {c["name"]: c for c in connector_list(secrets)}["telegram"]
    assert (
        listed["connected"] is True
        and listed["enabled"] is True
        and listed["allowed_users"] == 2
    )

    assert disconnect_connector(secrets, "telegram")["ok"] is True
    assert secrets.get("telegram:default") is None


def test_connect_missing_required_field(tmp_path):
    from coworker.connectors import connect_connector

    secrets = SecretStore(tmp_path / "secrets.json")
    res = connect_connector(
        secrets, "slack", {"bot_token": "xoxb"}, validate=False
    )  # app_token missing
    assert res["ok"] is False and "missing" in res["error"]


def test_connect_validation_runs(tmp_path):
    from coworker.connectors import connect_connector
    from coworker.connectors.descriptors import ValidationResult, get_descriptor

    secrets = SecretStore(tmp_path / "secrets.json")
    desc = get_descriptor("telegram")
    orig = desc.validate
    desc.validate = lambda creds: ValidationResult(True, identity="@mybot")
    try:
        res = connect_connector(
            secrets, "telegram", {"bot_token": "T"}
        )  # validate=True
    finally:
        desc.validate = orig
    assert res == {"ok": True, "account": "@mybot"}
    assert secrets.get("telegram:default")["account"] == "@mybot"


def test_connectors_rest(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from coworker.connectors.descriptors import ValidationResult, get_descriptor
    from coworker.server.app import create_app
    from coworker.server.manager import SessionManager

    monkeypatch.setenv("COWORKER_STATE_DIR", str(tmp_path / "state"))
    desc = get_descriptor("telegram")
    monkeypatch.setattr(
        desc, "validate", lambda creds: ValidationResult(True, identity="@testbot")
    )

    client = TestClient(create_app(SessionManager(data_dir=tmp_path / "data")))

    listed = client.get("/v1/connectors").json()["connectors"]
    assert any(c["name"] == "telegram" for c in listed)

    r = client.post(
        "/v1/connectors/telegram/connect",
        json={"fields": {"bot_token": "T0K", "allowed_users": "u1"}},
    )
    assert r.json() == {"ok": True, "account": "@testbot"}

    tg = {c["name"]: c for c in client.get("/v1/connectors").json()["connectors"]}[
        "telegram"
    ]
    assert tg["connected"] is True and tg["account"] == "@testbot"
    # secrets never leak over REST
    assert "T0K" not in client.get("/v1/connectors").text

    assert client.post("/v1/connectors/telegram/disconnect").json()["ok"] is True
    assert {c["name"]: c for c in client.get("/v1/connectors").json()["connectors"]}[
        "telegram"
    ]["connected"] is False


# -- inbound: event mappers ----------------------------------------------------
def test_telegram_message_mapper():
    from types import SimpleNamespace

    from coworker.connectors import telegram_message_to_event

    msg = SimpleNamespace(
        text="hello",
        message_id=7,
        chat=SimpleNamespace(id=12345, type="private"),
        from_user=SimpleNamespace(id=99, full_name="Alice"),
        message_thread_id=None,
    )
    ev = telegram_message_to_event(msg)
    assert ev.text == "hello" and ev.source.target == "telegram:12345"
    assert ev.source.user_id == "99" and ev.source.chat_type == "dm"
    # non-text (e.g. a sticker) maps to None
    assert (
        telegram_message_to_event(
            SimpleNamespace(text=None, chat=SimpleNamespace(id=1, type="private"))
        )
        is None
    )


def test_slack_event_mapper_and_loop_guard():
    from coworker.connectors import slack_event_to_event

    ev = slack_event_to_event(
        {
            "text": "ship it",
            "channel": "C9",
            "user": "U1",
            "channel_type": "channel",
            "ts": "1.2",
        },
        "BOT",
    )
    assert (
        ev.text == "ship it"
        and ev.source.target == "slack:C9"
        and ev.source.chat_type == "channel"
    )
    # bot echo / edits / empty → dropped (reply-loop guard)
    assert slack_event_to_event({"text": "x", "user": "BOT"}, "BOT") is None
    assert slack_event_to_event({"text": "x", "bot_id": "B1"}, None) is None
    assert (
        slack_event_to_event({"subtype": "message_changed", "text": "x"}, None) is None
    )


def test_make_adapter():
    from coworker.connectors import SlackAdapter, TelegramAdapter, make_adapter

    assert isinstance(make_adapter("telegram", {"bot_token": "T"}), TelegramAdapter)
    assert isinstance(
        make_adapter("slack", {"bot_token": "x", "app_token": "y"}), SlackAdapter
    )
    assert make_adapter("slack", {"bot_token": "x"}) is None  # app_token missing
    assert make_adapter("telegram", {}) is None


# -- inbound: super-agent runner -----------------------------------------------
class _FakeEngine:
    def __init__(self):
        self.runs: list[str] = []
        self.steers: list[str] = []
        self.gate = None  # asyncio.Event to hold a turn "busy"

    async def run(self, text):
        self.runs.append(text)
        if self.gate is not None:
            await self.gate.wait()
        if False:  # make this an async generator that yields nothing
            yield

    def queue_steering(self, text):
        self.steers.append(text)


async def test_superagent_idle_starts_a_turn():
    from coworker.connectors import SuperAgent

    eng = _FakeEngine()
    sa = SuperAgent(eng)
    sa.start()
    src = SessionSource("fake", "c1", user_id="u1", user_name="t")
    await sa.on_message(MessageEvent("hi", src))
    for _ in range(50):
        if eng.runs:
            break
        await asyncio.sleep(0.01)
    assert eng.runs == [MessageEvent("hi", src).tagged_text()]
    await sa.stop()


async def test_superagent_busy_steers_into_active_turn():
    from coworker.connectors import SuperAgent

    eng = _FakeEngine()
    eng.gate = asyncio.Event()
    sa = SuperAgent(eng)
    sa.start()
    src = SessionSource("fake", "c1", user_id="u1")
    await sa.on_message(MessageEvent("first", src))
    for _ in range(50):
        if sa._running:
            break
        await asyncio.sleep(0.01)
    assert sa._running is True
    await sa.on_message(
        MessageEvent("second", src)
    )  # arrives mid-turn → steered, not queued
    assert eng.steers == [MessageEvent("second", src).tagged_text()]
    eng.gate.set()
    await asyncio.sleep(0.02)
    await sa.stop()


async def test_superagent_replies_via_send_message(tmp_path):
    """Inbound → super-agent runs a real engine → agent calls send_message → sender fires."""
    from coworker.connectors import SuperAgent, make_send_message_tool
    from coworker.engine import TurnEngine
    from coworker.permissions import PermissionEngine
    from coworker.providers import AssistantTurn, ProviderClient, ToolCall
    from coworker.providers.base import ModelCapabilities
    from coworker.tools import ToolRegistry

    sent: list[tuple] = []

    secrets = SecretStore(tmp_path / "secrets.json")
    secrets.put("telegram:default", {"bot_token": "T"})

    def fake_sender(token, chat_id, text, thread_id=None):
        sent.append((chat_id, text))
        return SendResult(True, message_id="1")

    registry = ToolRegistry()
    registry.register(
        make_send_message_tool(secrets, senders={"telegram": fake_sender})
    )

    class _Scripted(ProviderClient):
        def __init__(self):
            self._turns = [
                AssistantTurn(
                    tool_calls=[
                        ToolCall(
                            id="c1",
                            name="send_message",
                            arguments={"target": "telegram:999", "text": "hi back"},
                        )
                    ],
                    finish_reason="tool_calls",
                ),
                AssistantTurn(text="done", finish_reason="stop"),
            ]

        def complete(self, *, model, messages, tools=None, **s):
            return self._turns.pop(0)

        def capabilities(self, model):
            return ModelCapabilities()

    perms = PermissionEngine(workspace_root=tmp_path)
    perms.allow_tool_for_session("send_message")
    engine = TurnEngine(
        provider=_Scripted(), registry=registry, permissions=perms, model="x"
    )

    sa = SuperAgent(engine)
    sa.start()
    await sa.on_message(
        MessageEvent("ping", SessionSource("telegram", "999", user_id="u1"))
    )
    for _ in range(100):
        if sent:
            break
        await asyncio.sleep(0.01)
    assert sent == [("999", "hi back")]
    await sa.stop()


async def test_manager_start_gateway_wires_superagent(tmp_path, monkeypatch):
    """manager.start_gateway builds the super-agent + gateway and connects adapters."""
    import coworker.server.manager as mgr_mod
    from coworker.connectors import FakeAdapter, connect_connector
    from coworker.server.manager import SessionManager

    monkeypatch.setenv("COWORKER_STATE_DIR", str(tmp_path / "state"))
    manager = SessionManager(data_dir=tmp_path / "data", provider=_StubProvider())
    connect_connector(manager.secrets, "telegram", {"bot_token": "T"}, validate=False)

    fake = FakeAdapter()
    fake.platform = "telegram"  # so the gateway keys it under the enabled platform
    monkeypatch.setattr(
        mgr_mod,
        "make_adapter",
        lambda platform, profile: fake if platform == "telegram" else None,
    )

    live = await manager.start_gateway()
    assert live == ["telegram"] and fake.connected
    assert manager.superagent is not None and manager.gateway is not None
    # super-agent is a persistent Cowork engine that can reply
    assert "send_message" in manager.superagent.engine.registry.names()

    await manager.stop_gateway()
    assert fake.connected is False and manager.gateway is None


# -- chat-ID auto-capture + allowlist + super-agent config ---------------------
async def test_gateway_records_recent_senders():
    gw = Gateway(
        settings={"fake": ConnectorSettings("fake", enabled=True, allowed_users={"u1"})}
    )
    fake = FakeAdapter()
    gw.register(fake)
    await fake.inject(
        "hi", user_id="u2", user_name="Bob"
    )  # unauthorized → dropped but captured
    await fake.inject("yo", user_id="u1", user_name="Al")  # authorized
    recent = gw.recent_senders()
    assert [r["user_id"] for r in recent] == ["u1", "u2"]  # most-recent first
    assert recent[1]["user_name"] == "Bob"
    # same sender again de-dupes and moves to front
    await fake.inject("again", user_id="u2")
    assert [r["user_id"] for r in gw.recent_senders("fake")] == ["u2", "u1"]


def test_manager_allow_disallow(tmp_path, monkeypatch):
    from coworker.connectors import connect_connector
    from coworker.server.manager import SessionManager

    monkeypatch.setenv("COWORKER_STATE_DIR", str(tmp_path / "state"))
    m = SessionManager(data_dir=tmp_path / "data")
    connect_connector(m.secrets, "telegram", {"bot_token": "T"}, validate=False)

    assert m.allow_user("telegram", "12345")["allowed_users"] == ["12345"]
    assert m.secrets.get("telegram:default")["allowed_users"] == ["12345"]
    assert m.disallow_user("telegram", "12345")["allowed_users"] == []
    assert m.allow_user("slack", "x")["ok"] is False  # slack not connected


def test_manager_superagent_workspace_and_status(tmp_path, monkeypatch):
    from coworker.connectors import connect_connector
    from coworker.server.manager import SessionManager

    monkeypatch.setenv("COWORKER_STATE_DIR", str(tmp_path / "state"))
    m = SessionManager(data_dir=tmp_path / "data")

    st = m.superagent_status()
    assert (
        st["running"] is False
        and st["connectors"] == []
        and st["workspace"].endswith("superagent")
    )

    wsdir = tmp_path / "my-assistant"
    assert m.set_superagent_workspace(str(wsdir))["ok"] is True
    assert m.superagent_status()["workspace"] == str(wsdir.resolve())

    connect_connector(
        m.secrets, "telegram", {"bot_token": "T", "allowed_users": "1"}, validate=False
    )
    tg = m.superagent_status()["connectors"][0]
    assert tg["name"] == "telegram" and tg["allowed_users"] == ["1"]


async def test_superagent_status_surfaces_recent_for_capture(tmp_path, monkeypatch):
    import coworker.server.manager as mgr_mod
    from coworker.connectors import FakeAdapter, connect_connector
    from coworker.server.manager import SessionManager

    monkeypatch.setenv("COWORKER_STATE_DIR", str(tmp_path / "state"))
    m = SessionManager(data_dir=tmp_path / "data", provider=_StubProvider())
    connect_connector(
        m.secrets, "telegram", {"bot_token": "T", "allowed_users": "u1"}, validate=False
    )
    fake = FakeAdapter()
    fake.platform = "telegram"
    monkeypatch.setattr(
        mgr_mod, "make_adapter", lambda p, prof: fake if p == "telegram" else None
    )

    await m.start_gateway()
    await fake.inject(
        "hello", user_id="stranger", user_name="Eve"
    )  # unauthorized → captured
    tg = m.superagent_status()["connectors"][0]
    assert tg["listening"] is True
    assert (
        tg["recent"][0]["user_id"] == "stranger"
        and tg["recent"][0]["authorized"] is False
    )

    # allow them live, then the next status shows authorized
    m.allow_user("telegram", "stranger")
    await fake.inject("hello again", user_id="stranger", user_name="Eve")
    assert m.superagent_status()["connectors"][0]["recent"][0]["authorized"] is True
    await m.stop_gateway()


class _TextProvider:
    """Provider that answers every turn with one fixed assistant text (no network)."""

    def __init__(self, text):
        self.text = text

    def complete(self, *, model, messages, tools=None, **s):
        from coworker.providers import AssistantTurn

        return AssistantTurn(text=self.text, finish_reason="stop")

    def capabilities(self, model):
        from coworker.providers.base import ModelCapabilities

        return ModelCapabilities()

    def stream(self, *, model, messages, tools=None, **s):
        from coworker.providers.base import StreamChunk

        yield StreamChunk(turn=self.complete(model=model, messages=messages))


async def test_sa_gui_message_streams_to_client(tmp_path, monkeypatch):
    from coworker.server.manager import SessionManager

    monkeypatch.setenv("COWORKER_STATE_DIR", str(tmp_path / "state"))
    m = SessionManager(
        data_dir=tmp_path / "data", provider=_TextProvider("hello from SA")
    )
    await m.start_gateway()  # super-agent runs even with no connector configured
    assert m.superagent is not None

    got: list[dict] = []

    async def client(msg):
        got.append(msg)

    m.sa_register(client)
    assert await m.sa_user_message("hi there") is True
    for _ in range(100):
        if any(x["type"] == "assistant_message" for x in got):
            break
        await asyncio.sleep(0.01)
    texts = [x["data"].get("text") for x in got if x["type"] == "assistant_message"]
    assert "hello from SA" in texts
    await m.stop_gateway()


async def test_sa_approver_denies_when_unwatched_resolves_when_watched(
    tmp_path, monkeypatch
):
    from coworker.engine import ApprovalOutcome
    from coworker.server.manager import SessionManager

    monkeypatch.setenv("COWORKER_STATE_DIR", str(tmp_path / "state"))
    m = SessionManager(data_dir=tmp_path / "data", provider=_StubProvider())

    # No GUI client connected → risky actions are denied (safe when nobody's watching).
    assert await m._sa_approver(None) is ApprovalOutcome.DENY

    # With a client connected, the approver waits for the GUI's decision.
    m.sa_register(lambda msg: None)

    async def resolve_soon():
        await asyncio.sleep(0.02)
        m.sa_resolve_approval("once")

    asyncio.create_task(resolve_soon())
    assert await m._sa_approver(None) is ApprovalOutcome.ONCE


def test_superagent_rest(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from coworker.connectors import connect_connector
    from coworker.server.app import create_app
    from coworker.server.manager import SessionManager

    monkeypatch.setenv("COWORKER_STATE_DIR", str(tmp_path / "state"))
    m = SessionManager(data_dir=tmp_path / "data")
    connect_connector(m.secrets, "telegram", {"bot_token": "T"}, validate=False)
    client = TestClient(create_app(m))

    assert client.get("/v1/superagent").json()["running"] is False
    assert client.post("/v1/connectors/telegram/allow", json={"user_id": "999"}).json()[
        "allowed_users"
    ] == ["999"]
    assert client.get("/v1/superagent").json()["connectors"][0]["allowed_users"] == [
        "999"
    ]
    assert (
        client.post(
            "/v1/superagent/workspace", json={"path": str(tmp_path / "asst")}
        ).json()["ok"]
        is True
    )
