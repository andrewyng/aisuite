"""Sidecar loopback routes for OpenCoworker Cloud: /oauth/callback,
/auth/callback, /v1/cloud/*, connect-managed gating."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from coworker.server import SessionManager, create_app


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("COWORKER_STATE_DIR", str(tmp_path / "state"))
    manager = SessionManager(workspace=tmp_path)
    app = create_app(manager)
    with TestClient(app) as c:
        c.manager = manager
        yield c


def test_cloud_status_signed_out(client):
    body = client.get("/v1/cloud/status").json()
    assert body == {"signed_in": False, "account": "", "user_id": ""}


def test_connect_managed_requires_sign_in(client):
    body = client.post("/v1/connectors/gmail/connect-managed").json()
    assert not body["ok"]
    assert "not signed in" in body["error"]


def test_oauth_callback_writes_profile_and_returns_page(client):
    resp = client.post(
        "/oauth/callback",
        data={
            "provider": "google",
            "connector": "gmail",
            "connection_id": "conn_9",
            "access_token": "ya29.tok",
            "refresh_token": "1//r",
            "expires_in": "3599",
            "scope": "gmail.readonly",
            "account": "a@b.c",
            "app_state": "s",
        },
    )
    assert resp.status_code == 200
    assert "gmail connected" in resp.text

    profile = client.manager.secrets.get("gmail:default")
    assert profile["access_token"] == "ya29.tok"
    assert profile["managed"] is True
    assert profile["connection_id"] == "conn_9"

    listed = {c["name"]: c for c in client.manager.list_connectors()}
    assert listed["gmail"]["connected"]
    assert listed["gmail"]["account"] == "a@b.c"


def test_oauth_callback_error_shows_failure_page(client):
    resp = client.post(
        "/oauth/callback",
        data={"connector": "gmail", "error": "access_denied"},
    )
    assert resp.status_code == 400
    assert "access_denied" in resp.text
    assert client.manager.secrets.get("gmail:default") is None


def test_oauth_callback_rejects_unmanaged_connector(client):
    resp = client.post(
        "/oauth/callback",
        data={"connector": "github", "access_token": "x"},
    )
    assert resp.status_code == 400
    assert client.manager.secrets.get("github:default") is None


def test_auth_callback_rejects_unknown_state(client):
    resp = client.get("/auth/callback", params={"code": "c", "state": "forged"})
    assert resp.status_code == 400
    assert "Sign-in failed" in resp.text


def test_disconnect_works_signed_out(client):
    # manual profile, no cloud session: disconnect must not require the cloud
    client.manager.secrets.put("gmail:default", {"type": "oauth", "access_token": "t"})
    body = client.post("/v1/connectors/gmail/disconnect").json()
    assert body["ok"]
    assert client.manager.secrets.get("gmail:default") is None
