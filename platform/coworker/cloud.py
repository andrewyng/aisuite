"""OpenCoworker Cloud client: sign-in and managed one-click connectors.

Everything here is OPTIONAL. The app is fully functional signed out — manual
token paste stays available for every connector (and remains available after
sign-in too). Cloud sign-in only unlocks the one-click managed OAuth path and
the metadata conveniences that come with it.

Flows (ported from the proven `ocw_cli` reference in opencoworker-cloud):

- Sign-in: Auth0 Authorization Code + PKCE. The sidecar generates the PKCE
  pair, the browser signs in, Auth0 redirects to the sidecar's loopback
  `GET /auth/callback`, and the code is exchanged here. Cloud session tokens
  live in the SecretStore under `cloud:auth`.
- Managed connect: authenticated `POST /v1/oauth/{provider}/start` returns the
  provider authorize URL; the broker's callback page form-POSTs the token
  payload to the sidecar's loopback `POST /oauth/callback`; the profile is
  written locally. Connector tokens never touch cloud storage.
- Refresh: managed profiles (they have refresh_token + connection_id) renew
  through the broker just before expiry; manual profiles are never touched.
"""

from __future__ import annotations

import base64
import hashlib
import secrets as _secrets
import time
import urllib.parse
from typing import Any, Optional

import httpx

from .config import Config
from .secrets import SecretStore

CLOUD_AUTH_PROFILE = "cloud:auth"
LOGIN_SCOPES = "openid profile email offline_access"

# connector id (canonical, = descriptor name) -> broker provider key
PROVIDER_FOR_CONNECTOR = {
    "gmail": "google",
    "google_calendar": "google",
    "google_drive": "google",
    "slack": "slack",
    "notion": "notion",
    "hubspot": "hubspot",
}

# Pending PKCE verifiers keyed by OAuth state; in-process only. A login that
# outlives the sidecar process simply has to be restarted.
_pending_logins: dict[str, dict[str, float | str]] = {}
_PENDING_TTL = 600


def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _now() -> float:
    return time.time()


# --- sign-in -----------------------------------------------------------------


def begin_login(config: Config) -> dict[str, Any]:
    """Create a PKCE login and return the browser URL. The sidecar's
    GET /auth/callback completes it."""
    verifier = _b64url(_secrets.token_bytes(48))
    challenge = _b64url(hashlib.sha256(verifier.encode()).digest())
    state = _secrets.token_urlsafe(16)

    for key, pending in list(_pending_logins.items()):  # expire stale attempts
        if float(pending["created"]) < _now() - _PENDING_TTL:
            _pending_logins.pop(key, None)
    _pending_logins[state] = {"verifier": verifier, "created": _now()}

    redirect_uri = f"http://127.0.0.1:{config.port}/auth/callback"
    authorize_url = f"https://{config.cloud_auth_domain}/authorize?" + urllib.parse.urlencode(
        {
            "response_type": "code",
            "client_id": config.cloud_client_id,
            "redirect_uri": redirect_uri,
            "scope": LOGIN_SCOPES,
            "audience": config.cloud_audience,
            "state": state,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
        }
    )
    return {"authorize_url": authorize_url, "state": state}


def complete_login(
    secrets: SecretStore, config: Config, code: str, state: str
) -> dict[str, Any]:
    pending = _pending_logins.pop(state, None)
    if pending is None or float(pending["created"]) < _now() - _PENDING_TTL:
        return {"ok": False, "error": "unknown or expired sign-in attempt"}

    resp = httpx.post(
        f"https://{config.cloud_auth_domain}/oauth/token",
        data={
            "grant_type": "authorization_code",
            "client_id": config.cloud_client_id,
            "code": code,
            "code_verifier": pending["verifier"],
            "redirect_uri": f"http://127.0.0.1:{config.port}/auth/callback",
        },
        timeout=15,
    )
    if resp.status_code != 200:
        return {"ok": False, "error": "token exchange failed"}
    _store_cloud_tokens(secrets, resp.json())

    # Best-effort profile fetch so the GUI can show who is signed in.
    me = fetch_me(secrets, config)
    if me:
        profile = secrets.get(CLOUD_AUTH_PROFILE) or {}
        profile["account"] = me.get("user", {}).get("email") or ""
        profile["user_id"] = me.get("user", {}).get("user_id") or ""
        secrets.put(CLOUD_AUTH_PROFILE, profile)
    return {"ok": True, **status(secrets)}


def _store_cloud_tokens(secrets: SecretStore, token: dict) -> None:
    profile = secrets.get(CLOUD_AUTH_PROFILE) or {"type": "oauth", "enabled": True}
    profile["access_token"] = token.get("access_token", "")
    if token.get("refresh_token"):  # rotating refresh tokens: keep the newest
        profile["refresh_token"] = token["refresh_token"]
    profile["expires"] = _now() + int(token.get("expires_in") or 3600) - 60
    secrets.put(CLOUD_AUTH_PROFILE, profile)


def status(secrets: SecretStore) -> dict[str, Any]:
    profile = secrets.get(CLOUD_AUTH_PROFILE) or {}
    return {
        "signed_in": bool(profile.get("access_token")),
        "account": profile.get("account") or "",
        "user_id": profile.get("user_id") or "",
    }


def logout(secrets: SecretStore) -> dict[str, Any]:
    secrets.delete(CLOUD_AUTH_PROFILE)
    return {"ok": True, "signed_in": False}


def fresh_access_token(secrets: SecretStore, config: Config) -> Optional[str]:
    """Valid cloud session token, silently refreshed near expiry; None when
    signed out or the session can't be renewed (GUI shows "sign in again")."""
    profile = secrets.get(CLOUD_AUTH_PROFILE) or {}
    if not profile.get("access_token"):
        return None
    if float(profile.get("expires") or 0) > _now():
        return profile["access_token"]
    if not profile.get("refresh_token"):
        return None
    resp = httpx.post(
        f"https://{config.cloud_auth_domain}/oauth/token",
        data={
            "grant_type": "refresh_token",
            "client_id": config.cloud_client_id,
            "refresh_token": profile["refresh_token"],
        },
        timeout=15,
    )
    if resp.status_code != 200:
        return None
    _store_cloud_tokens(secrets, resp.json())
    return (secrets.get(CLOUD_AUTH_PROFILE) or {}).get("access_token")


def fetch_me(secrets: SecretStore, config: Config) -> Optional[dict]:
    token = fresh_access_token(secrets, config)
    if not token:
        return None
    try:
        resp = httpx.get(
            config.cloud_base_url.rstrip("/") + "/v1/me",
            headers={"Authorization": f"Bearer {token}"},
            timeout=15,
        )
    except httpx.HTTPError:
        return None
    return resp.json() if resp.status_code == 200 else None


# --- managed connectors --------------------------------------------------------


def begin_managed_connect(
    secrets: SecretStore, config: Config, connector: str
) -> dict[str, Any]:
    """Authenticated start: returns the provider consent URL for the browser.
    Requires sign-in — the manual token path stays available regardless."""
    provider = PROVIDER_FOR_CONNECTOR.get(connector)
    if provider is None:
        return {"ok": False, "error": f"{connector} has no managed OAuth path"}
    token = fresh_access_token(secrets, config)
    if not token:
        return {"ok": False, "error": "not signed in", "signed_in": False}

    app_state = _secrets.token_urlsafe(16)
    try:
        resp = httpx.post(
            config.cloud_base_url.rstrip("/") + f"/v1/oauth/{provider}/start",
            json={
                "connector": connector,
                "redirect": f"http://127.0.0.1:{config.port}/oauth/callback",
                "app_state": app_state,
            },
            headers={"Authorization": f"Bearer {token}"},
            timeout=15,
        )
    except httpx.HTTPError as exc:
        return {"ok": False, "error": f"cloud unreachable: {type(exc).__name__}"}
    if resp.status_code != 200:
        return {"ok": False, "error": f"start failed ({resp.status_code})"}
    return {"ok": True, "authorize_url": resp.json()["authorize_url"], "app_state": app_state}


def managed_profile_from_callback(form: dict[str, str]) -> dict[str, Any]:
    """Local connector profile from the broker's form-POST payload.

    Field-compatible with a manual paste (`access_token` etc.) so tools and
    gating treat both paths identically; the managed extras (refresh_token,
    connection_id) are what enable broker refresh and cloud disconnect.
    """
    return {
        "type": "oauth",
        "enabled": True,
        "managed": True,
        "access_token": form.get("access_token", ""),
        "refresh_token": form.get("refresh_token", ""),
        "expires": _now() + int(form.get("expires_in") or 3600) - 60,
        "scope": form.get("scope", ""),
        "connection_id": form.get("connection_id", ""),
        "provider": form.get("provider", ""),
        "account": form.get("account", ""),
    }


def refresh_managed_token(
    secrets: SecretStore, config: Config, connector: str
) -> Optional[dict[str, Any]]:
    """Renew a managed connector token through the broker. Returns the updated
    profile, or None if this profile can't be (or doesn't need to be) renewed
    that way. Manual profiles are never touched."""
    profile = secrets.get(f"{connector}:default") or {}
    if not (profile.get("managed") and profile.get("refresh_token")):
        return None
    provider = profile.get("provider") or PROVIDER_FOR_CONNECTOR.get(connector)
    token = fresh_access_token(secrets, config)
    if not provider or not token:
        return None
    try:
        resp = httpx.post(
            config.cloud_base_url.rstrip("/") + f"/v1/oauth/{provider}/refresh",
            json={
                "refresh_token": profile["refresh_token"],
                "connection_id": profile.get("connection_id", ""),
                "connector": connector,
            },
            headers={"Authorization": f"Bearer {token}"},
            timeout=20,
        )
    except httpx.HTTPError:
        return None
    if resp.status_code != 200:
        return None
    fresh = resp.json()
    profile["access_token"] = fresh.get("access_token", "")
    if fresh.get("refresh_token"):
        profile["refresh_token"] = fresh["refresh_token"]
    profile["expires"] = _now() + int(fresh.get("expires_in") or 3600) - 60
    secrets.put(f"{connector}:default", profile)
    return profile


def ensure_fresh_connector_token(
    secrets: SecretStore, config: Config, connector: str, *, leeway: int = 120
) -> None:
    """Refresh-on-expiry hook for connector tools: if this is a managed profile
    about to expire, renew it in place. No-op for manual profiles."""
    profile = secrets.get(f"{connector}:default") or {}
    if not profile.get("managed"):
        return
    expires = float(profile.get("expires") or 0)
    if expires and expires > _now() + leeway:
        return
    refresh_managed_token(secrets, config, connector)


def cloud_disconnect(secrets: SecretStore, config: Config, connector: str) -> None:
    """Best-effort: tell the cloud a managed connection is gone so its metadata
    flips to disconnected. Local deletion always proceeds regardless."""
    profile = secrets.get(f"{connector}:default") or {}
    connection_id = profile.get("connection_id")
    if not (profile.get("managed") and connection_id):
        return
    token = fresh_access_token(secrets, config)
    if not token:
        return
    try:
        httpx.post(
            config.cloud_base_url.rstrip("/") + f"/v1/connections/{connection_id}/disconnect",
            headers={"Authorization": f"Bearer {token}"},
            timeout=10,
        )
    except httpx.HTTPError:
        pass
