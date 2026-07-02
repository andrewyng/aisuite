"""OpenCoworker Cloud integration: sign-in, managed connect callback, refresh.

Everything is offline: Auth0 and the cloud broker are stubbed at the httpx
boundary. The invariants under test are the product promises — manual paste
works signed out, managed profiles are field-compatible with manual ones, and
manual profiles are never touched by cloud refresh.
"""

from __future__ import annotations

import time

import pytest

from coworker import cloud
from coworker.config import Config
from coworker.connectors.setup import (
    connect_connector,
    connector_list,
    managed_connect_connector,
)
from coworker.secrets import SecretStore


@pytest.fixture
def secrets(tmp_path, monkeypatch):
    monkeypatch.setenv("COWORKER_STATE_DIR", str(tmp_path / "state"))
    return SecretStore(path=tmp_path / "state" / "secrets.json")


@pytest.fixture
def config():
    return Config(
        cloud_base_url="https://cloud.test",
        cloud_auth_domain="tenant.auth0.test",
        cloud_client_id="client123",
        port=8765,
    )


class FakeResponse:
    def __init__(self, status_code=200, body=None):
        self.status_code = status_code
        self._body = body or {}

    def json(self):
        return self._body


# --- sign-in -------------------------------------------------------------------


def test_begin_login_builds_pkce_authorize_url(config):
    out = cloud.begin_login(config)
    url = out["authorize_url"]
    assert url.startswith("https://tenant.auth0.test/authorize?")
    assert "code_challenge_method=S256" in url
    assert "client_id=client123" in url
    assert "8765%2Fauth%2Fcallback" in url
    assert out["state"] in url


def test_complete_login_stores_tokens_and_account(secrets, config, monkeypatch):
    state = cloud.begin_login(config)["state"]

    def fake_post(url, **kwargs):
        assert url == "https://tenant.auth0.test/oauth/token"
        assert kwargs["data"]["code_verifier"]
        return FakeResponse(200, {"access_token": "at1", "refresh_token": "rt1", "expires_in": 3600})

    def fake_get(url, **kwargs):
        assert url == "https://cloud.test/v1/me"
        return FakeResponse(200, {"user": {"email": "a@b.c", "user_id": "usr_1"}})

    monkeypatch.setattr(cloud.httpx, "post", fake_post)
    monkeypatch.setattr(cloud.httpx, "get", fake_get)

    result = cloud.complete_login(secrets, config, "code1", state)
    assert result["ok"] and result["signed_in"]
    assert result["account"] == "a@b.c"
    profile = secrets.get(cloud.CLOUD_AUTH_PROFILE)
    assert profile["access_token"] == "at1"
    assert profile["refresh_token"] == "rt1"


def test_complete_login_rejects_unknown_state(secrets, config):
    assert not cloud.complete_login(secrets, config, "code", "forged-state")["ok"]


def test_logout_clears_session(secrets, config):
    secrets.put(cloud.CLOUD_AUTH_PROFILE, {"access_token": "x"})
    cloud.logout(secrets)
    assert cloud.status(secrets) == {"signed_in": False, "account": "", "user_id": ""}


# --- managed connect -------------------------------------------------------------


def test_begin_managed_connect_requires_sign_in(secrets, config):
    out = cloud.begin_managed_connect(secrets, config, "gmail")
    assert not out["ok"]
    assert "not signed in" in out["error"]


def test_managed_profile_is_field_compatible_with_manual(secrets):
    form = {
        "provider": "google",
        "connector": "gmail",
        "connection_id": "conn_1",
        "access_token": "ya29.x",
        "refresh_token": "1//r",
        "expires_in": "3599",
        "scope": "gmail.readonly",
        "account": "a@b.c",
    }
    result = managed_connect_connector(
        secrets, "gmail", cloud.managed_profile_from_callback(form)
    )
    assert result["ok"] and result["account"] == "a@b.c"

    listed = {c["name"]: c for c in connector_list(secrets)}
    gmail = listed["gmail"]
    assert gmail["connected"] and gmail["managed"] and gmail["managed_profile"]
    profile = secrets.get("gmail:default")
    assert profile["access_token"] == "ya29.x"  # same key manual paste writes
    assert profile["connection_id"] == "conn_1"


def test_managed_connect_rejected_for_unmanaged_connector(secrets):
    result = managed_connect_connector(secrets, "github", {"access_token": "x"})
    assert not result["ok"]


def test_manual_paste_still_works_and_is_not_managed(secrets):
    result = connect_connector(
        secrets, "gmail", {"access_token": "manual-token"}, validate=False
    )
    assert result["ok"]
    listed = {c["name"]: c for c in connector_list(secrets)}
    assert listed["gmail"]["connected"]
    assert not listed["gmail"]["managed_profile"]  # manual profile, managed capable
    assert listed["gmail"]["managed"]


# --- refresh ---------------------------------------------------------------------


def _signed_in(secrets):
    secrets.put(
        cloud.CLOUD_AUTH_PROFILE,
        {"access_token": "cloud-at", "expires": time.time() + 3600},
    )


def test_refresh_updates_expiring_managed_profile(secrets, config, monkeypatch):
    _signed_in(secrets)
    secrets.put(
        "gmail:default",
        {
            "type": "oauth",
            "managed": True,
            "provider": "google",
            "access_token": "old",
            "refresh_token": "1//r",
            "connection_id": "conn_1",
            "expires": time.time() - 10,
        },
    )

    def fake_post(url, **kwargs):
        assert url == "https://cloud.test/v1/oauth/google/refresh"
        assert kwargs["json"]["connection_id"] == "conn_1"
        return FakeResponse(200, {"access_token": "new", "expires_in": 3600})

    monkeypatch.setattr(cloud.httpx, "post", fake_post)
    cloud.ensure_fresh_connector_token(secrets, config, "gmail")
    assert secrets.get("gmail:default")["access_token"] == "new"


def test_refresh_never_touches_manual_profiles(secrets, config, monkeypatch):
    _signed_in(secrets)
    secrets.put("gmail:default", {"type": "oauth", "access_token": "manual"})

    def boom(url, **kwargs):  # any network call would be a bug
        raise AssertionError("manual profiles must not trigger cloud refresh")

    monkeypatch.setattr(cloud.httpx, "post", boom)
    cloud.ensure_fresh_connector_token(secrets, config, "gmail")
    assert secrets.get("gmail:default")["access_token"] == "manual"


def test_fresh_profile_not_refreshed(secrets, config, monkeypatch):
    _signed_in(secrets)
    secrets.put(
        "gmail:default",
        {
            "managed": True,
            "provider": "google",
            "access_token": "current",
            "refresh_token": "1//r",
            "expires": time.time() + 3600,
        },
    )
    monkeypatch.setattr(
        cloud.httpx, "post", lambda *a, **k: (_ for _ in ()).throw(AssertionError())
    )
    cloud.ensure_fresh_connector_token(secrets, config, "gmail")
    assert secrets.get("gmail:default")["access_token"] == "current"
