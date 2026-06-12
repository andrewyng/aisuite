"""Connect / disconnect / list connectors — writes tokens to the SecretStore.

Pure functions over a SecretStore so they're testable without the server. `validate=False`
skips the network check (used by tests). Secrets are never returned — only status + the
public bot identity captured at connect time.
"""

from __future__ import annotations

from typing import Any

from ..secrets import SecretStore
from .descriptors import get_descriptor, list_descriptors
from .tool_defs import patch_tool_settings, tool_dicts

_EXPERIMENTAL_KEY = "experimental:settings"


def experimental_enabled(secrets: SecretStore) -> bool:
    """Whether the user has opted in to experimental (use-at-your-own-risk) connectors."""
    return bool((secrets.get(_EXPERIMENTAL_KEY) or {}).get("enabled"))


def set_experimental_enabled(secrets: SecretStore, value: bool) -> dict[str, Any]:
    secrets.put(_EXPERIMENTAL_KEY, {"enabled": bool(value)})
    return {"ok": True, "enabled": bool(value)}


def _profile_connected(descriptor, profile: dict[str, Any]) -> bool:
    if not descriptor.available:
        return False
    if descriptor.auth == "none":
        return True
    required = [
        f.key for f in descriptor.fields if f.required and f.key != "allowed_users"
    ]
    return bool(profile) and all(bool(profile.get(k)) for k in required)


def connector_list(secrets: SecretStore) -> list[dict[str, Any]]:
    show_experimental = experimental_enabled(secrets)
    out: list[dict[str, Any]] = []
    for d in list_descriptors():
        # Experimental connectors are invisible (not just disabled) until the user opts in;
        # hiding them here also drops their tools from engine builds via
        # _enabled_connector_tools, so flipping the setting off cuts access immediately.
        if d.experimental and not show_experimental:
            continue
        profile = secrets.get(f"{d.name}:default") or {}
        connected = _profile_connected(d, profile)
        out.append(
            {
                "name": d.name,
                "title": d.title,
                "icon": d.icon,
                "blurb": d.blurb,
                "auth": d.auth,
                "two_way": d.two_way,
                "available": d.available,
                "fields": [f.to_dict() for f in d.fields],
                "instructions": d.instructions,
                "connected": connected,
                "account": profile.get("account"),
                "enabled": bool(profile.get("enabled", True)) and connected,
                "allowed_users": len(profile.get("allowed_users") or []),
                "tools": tool_dicts(secrets, d.name),
                "experimental": d.experimental,
                "risk_notice": d.risk_notice,
            }
        )
    return out


def update_connector_tools(
    secrets: SecretStore, name: str, enabled: dict[str, Any]
) -> dict[str, Any]:
    if get_descriptor(name) is None:
        return {"ok": False, "error": "unknown connector"}
    return patch_tool_settings(secrets, name, enabled)


def connect_connector(
    secrets: SecretStore,
    name: str,
    fields: dict[str, Any],
    *,
    validate: bool = True,
    acknowledged: bool = False,
) -> dict[str, Any]:
    d = get_descriptor(name)
    if d is None or not d.available:
        return {"ok": False, "error": "unknown or unavailable connector"}
    if d.experimental:
        if not experimental_enabled(secrets):
            return {"ok": False, "error": "experimental connectors are disabled"}
        if not acknowledged:
            return {
                "ok": False,
                "error": "risk acknowledgment required",
                "risk_notice": d.risk_notice,
            }

    raw = {f.key: str(fields.get(f.key) or "").strip() for f in d.fields}
    missing = [f.label for f in d.fields if f.required and not raw.get(f.key)]
    if missing:
        return {"ok": False, "error": "missing: " + ", ".join(missing)}

    allowed = sorted(
        {u.strip() for u in raw.get("allowed_users", "").split(",") if u.strip()}
    )
    token_creds = {k: v for k, v in raw.items() if k != "allowed_users" and v}

    identity = None
    if validate and d.validate is not None:
        result = d.validate(token_creds)
        if not result.ok:
            return {"ok": False, "error": result.error or "validation failed"}
        identity = result.identity

    profile_type = (
        "oauth" if d.auth == "oauth" else "none" if d.auth == "none" else "token"
    )
    profile: dict[str, Any] = {"type": profile_type, "enabled": True, **token_creds}
    if any(f.key == "allowed_users" for f in d.fields):
        profile["allowed_users"] = allowed
    if identity:
        profile["account"] = identity
    secrets.put(f"{name}:default", profile)
    return {"ok": True, "account": identity}


def disconnect_connector(secrets: SecretStore, name: str) -> dict[str, Any]:
    return {"ok": secrets.delete(f"{name}:default")}
