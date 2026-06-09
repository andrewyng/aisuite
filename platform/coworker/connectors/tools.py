"""The `send_message` outbound tool — available to every agent.

Stateless: parses the `target` token, pulls the bot token from the SecretStore at call time
(never in the model's context), and dispatches via a swappable sender registry. Permission-
gated (`requires_approval=True` → asks outside Auto mode).
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import aisuite as ai

from ..secrets import SecretStore
from .base import parse_target
from .senders import DEFAULT_SENDERS, Sender

_SCHEMA = {
    "type": "function",
    "function": {
        "name": "send_message",
        "description": (
            "Send a message to a connected chat (Slack or Telegram). `target` is the "
            "reply handle from an inbound message (e.g. 'telegram:12345' or 'slack:C0123', "
            "optionally with a ':<thread>' suffix). Use this to actually reach a person — "
            "plain assistant text is not delivered anywhere."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "description": "Destination handle 'platform:chat_id[:thread]', e.g. 'telegram:12345'.",
                },
                "text": {"type": "string", "description": "The message text to send."},
            },
            "required": ["target", "text"],
        },
    },
}


def make_send_message_tool(
    secrets: SecretStore,
    *,
    senders: Optional[dict[str, Sender]] = None,
) -> Callable[..., Any]:
    """Build the `send_message` tool bound to a SecretStore (and optional sender registry)."""
    senders = senders if senders is not None else DEFAULT_SENDERS

    def send_message(target: str, text: str) -> dict[str, Any]:
        try:
            platform, chat_id, thread_id = parse_target(target)
        except ValueError as exc:
            return {"error": str(exc)}
        sender = senders.get(platform)
        if sender is None:
            return {"error": f"unknown platform: {platform}"}
        creds = secrets.get(f"{platform}:default") or {}
        token = creds.get("bot_token")
        if not token:
            return {"error": f"no bot token for {platform} — connect it first"}
        result = sender(token, chat_id, text, thread_id)
        if result.ok:
            return {"ok": True, "message_id": result.message_id, "target": target}
        return {"error": result.error or "send failed"}

    send_message.__name__ = "send_message"
    send_message.__doc__ = _SCHEMA["function"]["description"]
    send_message.__aisuite_tool_metadata__ = ai.ToolMetadata(
        name="send_message",
        category="messaging",
        risk_level="medium",
        capabilities=["messaging"],
        requires_approval=True,
    )
    send_message.__coworker_schema__ = _SCHEMA
    return send_message
