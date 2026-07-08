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

PLATFORMS = ("telegram", "slack")


@dataclass
class TeamAuth:
    """One workspace's inbound authorization (managed multi-workspace Slack).

    User/channel ids are workspace-scoped — a U… only means something inside its
    team — so each connected workspace carries its own allow-list.
    """

    allowed_users: set[str] = field(default_factory=set)
    allow_all: bool = False


@dataclass
class ConnectorSettings:
    platform: str
    enabled: bool = False
    allowed_users: set[str] = field(default_factory=set)
    allow_all: bool = False
    # Per-workspace auth, keyed by team_id (populated from `slack:team:*` profiles).
    # Only relay-mode Slack fills this; manual Socket Mode uses the flat fields above.
    teams: dict[str, TeamAuth] = field(default_factory=dict)


def is_authorized(settings: ConnectorSettings, source: SessionSource) -> bool:
    team_id = getattr(source, "team_id", None)
    if team_id:
        # Relay events carry their workspace; authorization is that team's list
        # alone. An unknown team means no install we know of — deny (park).
        team = settings.teams.get(team_id)
        if team is None:
            return False
        if team.allow_all:
            return True
        uid = source.user_id
        return bool(uid) and uid in team.allowed_users
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
        token = profile.get("bot_token")
        allowed = set(profile.get("allowed_users") or [])
        allowed |= _csv(os.environ.get(f"{platform.upper()}_ALLOWED_USERS"))
        allow_all = bool(profile.get("allow_all")) or os.environ.get(
            f"{platform.upper()}_ALLOW_ALL_USERS", ""
        ).lower() in ("1", "true", "yes")
        # Managed Slack relay carries no bot_token in the default profile (tokens
        # are per-team); it enables on `mode == "relay"` instead of on a token.
        if platform == "slack" and profile.get("mode") == "relay":
            enabled = bool(profile.get("enabled", True))
        else:
            enabled = bool(token) and profile.get("enabled", True)
        teams: dict[str, TeamAuth] = {}
        if platform == "slack":
            for team_id, team_profile in _slack_team_profiles(secrets):
                teams[team_id] = TeamAuth(
                    allowed_users=set(team_profile.get("allowed_users") or []),
                    allow_all=bool(team_profile.get("allow_all")),
                )
        out[platform] = ConnectorSettings(
            platform=platform,
            enabled=enabled,
            allowed_users=allowed,
            allow_all=allow_all,
            teams=teams,
        )
    return out


def _slack_team_profiles(secrets: SecretStore) -> list[tuple[str, dict]]:
    """(team_id, profile) for every managed-install workspace (`slack:team:*`)."""
    out: list[tuple[str, dict]] = []
    for meta in secrets.status():
        name = meta.get("profile", "")
        if not name.startswith("slack:team:"):
            continue
        team_id = name[len("slack:team:") :]
        profile = secrets.get(name)
        if team_id and profile:
            out.append((team_id, profile))
    return out
