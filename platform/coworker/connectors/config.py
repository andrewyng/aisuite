"""Connector settings — which platforms are enabled + the inbound allowlist.

Tokens live in the SecretStore (profile `<platform>:default`); this module only carries
enablement + authorization. The allowlist is the inbound security guard: **empty = nobody**
(you must add your own user id), `allow_all` opens it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

from ..secrets import SecretStore
from .base import SessionSource

PLATFORMS: list[str] = ["telegram", "slack"]

# Per-platform credential key in the SecretStore profile that proves "connected enough to
# listen". None → the platform needs no stored credential (e.g. QR-paired bridges); a bare
# profile written by connect_connector is enough.
_CREDENTIAL_KEYS: dict[str, Optional[str]] = {
    "telegram": "bot_token",
    "slack": "bot_token",
}


def register_platform(
    name: str, *, credential_key: Optional[str] = "bot_token"
) -> None:
    """Register an extra two-way platform (used by the experimental package)."""
    if name not in PLATFORMS:
        PLATFORMS.append(name)
    _CREDENTIAL_KEYS[name] = credential_key


@dataclass
class ConnectorSettings:
    platform: str
    enabled: bool = False
    allowed_users: set[str] = field(default_factory=set)
    allow_all: bool = False


def is_authorized(settings: ConnectorSettings, source: SessionSource) -> bool:
    if settings.allow_all:
        return True
    uid = source.user_id
    return bool(uid) and uid in settings.allowed_users


def _csv(value: Optional[str]) -> set[str]:
    return {p.strip() for p in (value or "").split(",") if p.strip()}


def load_settings(
    secrets: Optional[SecretStore] = None,
) -> dict[str, ConnectorSettings]:
    """Per-platform settings from the SecretStore profile + env overrides.

    A platform is enabled when its token profile exists (and isn't explicitly disabled).
    Allowlist/allow-all come from the profile or `<PLATFORM>_ALLOWED_USERS` /
    `<PLATFORM>_ALLOW_ALL_USERS` env vars (env wins).
    """
    secrets = secrets or SecretStore()
    out: dict[str, ConnectorSettings] = {}
    for platform in PLATFORMS:
        profile = secrets.get(f"{platform}:default") or {}
        cred_key = _CREDENTIAL_KEYS.get(platform, "bot_token")
        has_cred = bool(profile.get(cred_key)) if cred_key else bool(profile)
        allowed = set(profile.get("allowed_users") or [])
        allowed |= _csv(os.environ.get(f"{platform.upper()}_ALLOWED_USERS"))
        allow_all = bool(profile.get("allow_all")) or os.environ.get(
            f"{platform.upper()}_ALLOW_ALL_USERS", ""
        ).lower() in ("1", "true", "yes")
        enabled = has_cred and profile.get("enabled", True)
        out[platform] = ConnectorSettings(
            platform=platform,
            enabled=enabled,
            allowed_users=allowed,
            allow_all=allow_all,
        )
    return out
