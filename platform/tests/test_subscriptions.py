"""Channel subscriptions: the store, the agent tools, and the gateway fan-out dispatch."""

import asyncio

import pytest

from coworker.connectors.base import MessageEvent, SessionSource
from coworker.subscriptions import (
    ChannelBuffer,
    SubscriptionStore,
    resolve_channel,
    subscription_tools,
)
from coworker.providers import ModelCapabilities, ProviderClient
from coworker.server.manager import SessionManager


class ScriptedProvider(ProviderClient):
    def __init__(self, turns):
        self._turns = list(turns)

    def complete(self, *, model, messages, tools=None, **settings):
        return self._turns.pop(0)

    def capabilities(self, model):
        return ModelCapabilities()


def test_resolve_channel():
    assert resolve_channel("<#C0123|alerts>") == "slack:C0123"  # Slack #mention token
    assert resolve_channel("slack:C0999") == "slack:C0999"  # already an address
    assert resolve_channel("C0777") == "slack:C0777"  # bare id → default platform
    assert resolve_channel("") == ""


def test_store_crud_and_persistence(tmp_path):
    p = tmp_path / "subs.json"
    st = SubscriptionStore(p)
    st.subscribe("s1", "slack:C1")
    st.subscribe("s2", "slack:C1")
    st.subscribe("s1", "slack:C2")
    assert {s.session_id for s in st.for_channel("slack:C1")} == {"s1", "s2"}
    assert {s.channel for s in st.for_session("s1")} == {"slack:C1", "slack:C2"}
    # idempotent subscribe (no duplicate)
    st.subscribe("s1", "slack:C1")
    assert len(st.for_channel("slack:C1")) == 2
    # persistence round-trip
    assert {(s.session_id, s.channel) for s in SubscriptionStore(p).all()} == {
        ("s1", "slack:C1"),
        ("s2", "slack:C1"),
        ("s1", "slack:C2"),
    }
    # explicit unsubscribe + session removal (the only implicit teardown)
    assert st.unsubscribe("s2", "slack:C1") is True
    assert st.unsubscribe("s2", "slack:C1") is False
    st.remove_session("s1")
    assert st.all() == []


def test_buffer_and_tools(tmp_path):
    st = SubscriptionStore(tmp_path / "subs.json")
    buf = ChannelBuffer()
    sub, unsub, lst, getmsgs = subscription_tools(
        st, "sess", buf, routing_targets=["slack:CINBOX"]
    )

    assert sub("<#C0123|alerts>")["subscribed"] == "slack:C0123"
    assert lst()["channels"] == ["slack:C0123"]
    # subscribing the channel the Inbox routes to warns (inbound vs outbound hygiene)
    assert "warning" in sub("slack:CINBOX")

    buf.record("slack:C0123", "bob", "deploy failed")
    buf.record("slack:C0123", "sue", "rolling back")
    msgs = getmsgs("slack:C0123", 5)["messages"]
    assert [m["text"] for m in msgs] == ["deploy failed", "rolling back"]

    assert unsub("slack:C0123")["was_subscribed"] is True
    assert "slack:C0123" not in lst()["channels"]


def _event(text, *, chat_type, chat_id="C1", user="bob"):
    return MessageEvent(
        text=text,
        source=SessionSource(
            platform="slack", chat_id=chat_id, user_name=user, chat_type=chat_type
        ),
    )


def _connect_slack(mgr):
    """Inbound delivery is gated on the connector being CONNECTED (§4.3). Tests used to pass
    by riding the developer's real Slack profile; with the isolated state dir (conftest) each
    test must connect its own."""
    mgr.secrets.put("slack:default", {"bot_token": "xoxb-test", "app_token": "xapp-test", "enabled": True})


def test_dispatch_fans_out_to_subscribers(tmp_path, monkeypatch):
    mgr = SessionManager(workspace=tmp_path, provider=ScriptedProvider([]))
    _connect_slack(mgr)
    delivered: list[tuple[str, str]] = []

    async def fake_deliver(session_id, message, *, source=None):
        delivered.append((session_id, message))

    monkeypatch.setattr(mgr, "deliver_to_session", fake_deliver)

    mgr.subscriptions.subscribe("sA", "slack:C1")
    mgr.subscriptions.subscribe("sB", "slack:C1")

    # a CHANNEL message → fan out to both subscribers, buffered
    asyncio.run(mgr._dispatch_inbound(_event("deploy failed", chat_type="channel")))
    assert {sid for sid, _ in delivered} == {"sA", "sB"}
    assert mgr.channel_buffer.recent("slack:C1")[-1]["text"] == "deploy failed"

    # a CHANNEL with no subscribers → buffered, nobody delivered
    delivered.clear()
    asyncio.run(
        mgr._dispatch_inbound(_event("noise", chat_type="channel", chat_id="C2"))
    )
    assert delivered == []
    assert mgr.channel_buffer.recent("slack:C2")[-1]["text"] == "noise"

    # a DM with no designated session → parked as unrouted, nobody delivered
    asyncio.run(mgr._dispatch_inbound(_event("hi there", chat_type="dm", chat_id="D1")))
    assert delivered == []
    assert mgr.unrouted.list()[0]["reason"] == "no DM session designated"

    # a DM with a designated session → delivered to it
    mgr.set_dm_session("sDM")
    asyncio.run(mgr._dispatch_inbound(_event("hello", chat_type="dm", chat_id="D1")))
    assert delivered[-1][0] == "sDM"


def test_subscriptions_endpoint_and_collision(tmp_path):
    from fastapi.testclient import TestClient
    from coworker.server import create_app

    mgr = SessionManager(workspace=tmp_path, provider=ScriptedProvider([]))
    mgr.subscriptions.subscribe("s1", "slack:C1")
    client = TestClient(create_app(mgr))
    subs = client.get("/v1/subscriptions").json()["subscriptions"]
    assert len(subs) == 1
    row = subs[0]
    assert row["session_id"] == "s1" and row["channel"] == "slack:C1"
    assert row["collision"] is False  # no Inbox routing bound → no collision
    # the per-session list field is present too
    sessions = client.get("/v1/sessions").json()["sessions"]
    assert all("subscriptions" in s for s in sessions)


def test_subscribe_unsubscribe_and_recent_endpoints(tmp_path):
    from fastapi.testclient import TestClient
    from coworker.server import create_app

    mgr = SessionManager(workspace=tmp_path, provider=ScriptedProvider([]))
    mgr.channel_buffer.record("slack:C9", "bob", "deploy failed")  # seeds the picker
    client = TestClient(create_app(mgr))

    assert [
        c["channel"] for c in client.get("/v1/channels/recent").json()["channels"]
    ] == ["slack:C9"]

    # subscribe via a Slack #mention token → resolved to the id
    r = client.post(
        "/v1/subscriptions", json={"session_id": "sZ", "channel": "<#C9|alerts>"}
    ).json()
    assert r["ok"] and r["channel"] == "slack:C9"
    assert [s.channel for s in mgr.subscriptions.for_session("sZ")] == ["slack:C9"]

    # unsubscribe
    r = client.post(
        "/v1/subscriptions/remove", json={"session_id": "sZ", "channel": "slack:C9"}
    ).json()
    assert r["ok"] and r["removed"] is True
    assert mgr.subscriptions.for_session("sZ") == []
