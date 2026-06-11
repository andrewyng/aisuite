"""Tests for MCP OAuth (authorization-code + PKCE) — token cache, refresh, consent
flow against a local loopback redirect, 401 retry, and config plumbing. No network:
token endpoints are httpx.MockTransport; the "browser" is a stub that fetches the
redirect URI itself.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import socket
import time
import urllib.parse

import httpx
import pytest

from coworker.mcp import load_mcp_servers
from coworker.mcp.oauth import (
    GOOGLE_AUTHORIZE_URL,
    GOOGLE_TOKEN_URL,
    OAuthBearer,
    OAuthError,
    OAuthSpec,
    OAuthTokenManager,
    _pkce_pair,
    build_authorize_url,
)
from coworker.secrets import SecretStore


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _store(tmp_path, monkeypatch) -> SecretStore:
    monkeypatch.setenv("COWORKER_STATE_DIR", str(tmp_path / "state"))
    return SecretStore()


def _spec(**overrides) -> OAuthSpec:
    base = dict(
        client_id="cid.apps.googleusercontent.com",
        client_secret="csecret",
        scopes=["https://www.googleapis.com/auth/calendar.events.readonly"],
    )
    base.update(overrides)
    return OAuthSpec(**base)


# -- pure pieces -----------------------------------------------------------------
def test_pkce_pair_is_s256():
    verifier, challenge = _pkce_pair()
    expected = (
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
        .rstrip(b"=")
        .decode()
    )
    assert challenge == expected
    assert "=" not in verifier and len(verifier) >= 43


def test_authorize_url_params():
    url = build_authorize_url(_spec(), state="st4te", challenge="ch4llenge")
    parsed = urllib.parse.urlsplit(url)
    query = {k: v[0] for k, v in urllib.parse.parse_qs(parsed.query).items()}
    assert url.startswith(GOOGLE_AUTHORIZE_URL)
    assert query["client_id"] == "cid.apps.googleusercontent.com"
    assert query["response_type"] == "code"
    assert query["state"] == "st4te"
    assert query["code_challenge"] == "ch4llenge"
    assert query["code_challenge_method"] == "S256"
    assert query["access_type"] == "offline"
    assert query["prompt"] == "consent"
    assert query["scope"].startswith("https://www.googleapis.com/auth/calendar")
    assert query["redirect_uri"].startswith("http://localhost:")


def test_spec_from_config_defaults_and_validation():
    spec = OAuthSpec.from_config({"client_id": "abc", "scopes": ["s1", "s2"]})
    assert spec.authorize_url == GOOGLE_AUTHORIZE_URL
    assert spec.token_url == GOOGLE_TOKEN_URL
    assert spec.client_secret == ""
    assert spec.scopes == ["s1", "s2"]
    with pytest.raises(ValueError):
        OAuthSpec.from_config({"scopes": ["s1"]})


# -- token cache / refresh ---------------------------------------------------------
async def test_cached_token_skips_network(tmp_path, monkeypatch):
    store = _store(tmp_path, monkeypatch)
    store.put(
        "mcp-oauth:cal",
        {"access_token": "cached", "refresh_token": "r", "expires": time.time() + 600},
    )

    def _no_network(request):  # pragma: no cover - failure path
        raise AssertionError("token endpoint should not be called")

    manager = OAuthTokenManager(
        "cal", _spec(), store, transport=httpx.MockTransport(_no_network)
    )
    assert await manager.get_token() == "cached"


async def test_refresh_preserves_refresh_token(tmp_path, monkeypatch):
    store = _store(tmp_path, monkeypatch)
    store.put(
        "mcp-oauth:cal",
        {"access_token": "stale", "refresh_token": "rtok", "expires": time.time() - 5},
    )
    seen: dict = {}

    def _token_endpoint(request):
        seen.update(urllib.parse.parse_qsl(request.content.decode()))
        return httpx.Response(200, json={"access_token": "fresh", "expires_in": 3600})

    manager = OAuthTokenManager(
        "cal", _spec(), store, transport=httpx.MockTransport(_token_endpoint)
    )
    assert await manager.get_token() == "fresh"
    assert seen["grant_type"] == "refresh_token"
    assert seen["refresh_token"] == "rtok"
    assert seen["client_secret"] == "csecret"
    saved = store.get("mcp-oauth:cal")
    assert saved["access_token"] == "fresh"
    assert saved["refresh_token"] == "rtok"  # kept although response omitted it
    assert saved["expires"] > time.time() + 3000


# -- interactive consent flow ------------------------------------------------------
async def test_interactive_flow_roundtrip(tmp_path, monkeypatch):
    store = _store(tmp_path, monkeypatch)
    spec = _spec(callback_port=_free_port())
    exchanged: dict = {}

    def _token_endpoint(request):
        exchanged.update(urllib.parse.parse_qsl(request.content.decode()))
        return httpx.Response(
            200,
            json={"access_token": "live", "refresh_token": "rnew", "expires_in": 3599},
        )

    def _browser(url: str):
        query = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(url).query))

        async def _consent():
            redirect = (
                f"{query['redirect_uri']}?code=authcode123&state={query['state']}"
            )
            async with httpx.AsyncClient() as client:
                response = await client.get(redirect)
            assert response.status_code == 200
            assert "close this tab" in response.text

        asyncio.get_running_loop().create_task(_consent())

    manager = OAuthTokenManager(
        "cal",
        spec,
        store,
        open_browser=_browser,
        transport=httpx.MockTransport(_token_endpoint),
        flow_timeout=10,
    )
    assert await manager.get_token() == "live"
    assert exchanged["grant_type"] == "authorization_code"
    assert exchanged["code"] == "authcode123"
    assert exchanged["code_verifier"]
    assert exchanged["redirect_uri"] == spec.redirect_uri
    saved = store.get("mcp-oauth:cal")
    assert saved["refresh_token"] == "rnew"


async def test_interactive_flow_rejects_bad_state_and_errors(tmp_path, monkeypatch):
    store = _store(tmp_path, monkeypatch)
    spec = _spec(callback_port=_free_port())

    def _browser(url: str):
        query = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(url).query))

        async def _consent():
            redirect = f"{query['redirect_uri']}?code=x&state=WRONG"
            async with httpx.AsyncClient() as client:
                response = await client.get(redirect)
            assert response.status_code == 400

        asyncio.get_running_loop().create_task(_consent())

    manager = OAuthTokenManager(
        "cal", spec, store, open_browser=_browser, flow_timeout=10
    )
    with pytest.raises(OAuthError, match="state mismatch"):
        await manager.get_token()


async def test_failed_refresh_falls_back_to_interactive(tmp_path, monkeypatch):
    store = _store(tmp_path, monkeypatch)
    spec = _spec(callback_port=_free_port())
    store.put(
        "mcp-oauth:cal",
        {"access_token": "stale", "refresh_token": "dead", "expires": time.time() - 5},
    )

    def _token_endpoint(request):
        data = dict(urllib.parse.parse_qsl(request.content.decode()))
        if data["grant_type"] == "refresh_token":
            return httpx.Response(400, json={"error": "invalid_grant"})
        return httpx.Response(200, json={"access_token": "anew", "expires_in": 3600})

    def _browser(url: str):
        query = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(url).query))

        async def _consent():
            redirect = f"{query['redirect_uri']}?code=c2&state={query['state']}"
            async with httpx.AsyncClient() as client:
                await client.get(redirect)

        asyncio.get_running_loop().create_task(_consent())

    manager = OAuthTokenManager(
        "cal",
        spec,
        store,
        open_browser=_browser,
        transport=httpx.MockTransport(_token_endpoint),
        flow_timeout=10,
    )
    assert await manager.get_token() == "anew"


# -- bearer auth hook ---------------------------------------------------------------
async def test_bearer_retries_once_on_401():
    class _FakeManager:
        def __init__(self):
            self.calls = []

        async def get_token(self, *, force_refresh: bool = False) -> str:
            self.calls.append(force_refresh)
            return "tok2" if force_refresh else "tok1"

    manager = _FakeManager()

    def _server(request):
        if request.headers["Authorization"] == "Bearer tok1":
            return httpx.Response(401)
        assert request.headers["Authorization"] == "Bearer tok2"
        return httpx.Response(200, json={"ok": True})

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(_server), auth=OAuthBearer(manager)
    ) as client:
        response = await client.get("https://mcp.example/v1")
    assert response.status_code == 200
    assert manager.calls == [False, True]


# -- config plumbing -----------------------------------------------------------------
def test_config_parses_oauth_block_with_var_resolution(tmp_path, monkeypatch):
    monkeypatch.setenv("COWORKER_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("GCP_CLIENT_SECRET", "resolved-secret")
    path = tmp_path / "state" / "mcp.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "calendar": {
                        "url": "https://calendarmcp.googleapis.com/mcp/v1",
                        "oauth": {
                            "client_id": "cid",
                            "client_secret": "${GCP_CLIENT_SECRET}",
                            "scopes": ["s"],
                        },
                    },
                    "plain": {"command": "echo"},
                }
            }
        ),
        encoding="utf-8",
    )
    servers = {s.name: s for s in load_mcp_servers(secrets=SecretStore())}
    assert servers["calendar"].transport == "http"
    assert servers["calendar"].oauth["client_secret"] == "resolved-secret"
    assert servers["plain"].oauth is None
