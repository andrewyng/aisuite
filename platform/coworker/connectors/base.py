"""Messaging connector core — the platform-agnostic adapter contract + value types.

Patterns borrowed from Hermes' gateway (read-only ref). An adapter connects to a platform
(Slack/Telegram), receives inbound messages and dispatches them via `handle_message`, and
can `send` outbound. Inbound identity is carried by `SessionSource`; a `target` token
(`platform:chat_id[:thread]`) is the opaque handle the agent passes back to reply.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Optional


class MessageType(str, Enum):
    TEXT = "text"
    COMMAND = "command"
    MEDIA = "media"


# -- target tokens -------------------------------------------------------------
def format_target(platform: str, chat_id: str, thread_id: Optional[str] = None) -> str:
    base = f"{platform}:{chat_id}"
    return f"{base}:{thread_id}" if thread_id else base


def parse_target(target: str) -> tuple[str, str, Optional[str]]:
    """`'platform:chat_id[:thread]'` -> (platform, chat_id, thread_id)."""
    parts = (target or "").split(":")
    if len(parts) < 2 or not parts[0] or not parts[1]:
        raise ValueError(
            f"invalid target {target!r} (expected 'platform:chat_id[:thread]')"
        )
    thread = ":".join(parts[2:]) if len(parts) > 2 else None
    return parts[0], parts[1], (thread or None)


# -- value types ---------------------------------------------------------------
@dataclass
class SessionSource:
    platform: str
    chat_id: str
    user_id: Optional[str] = None
    user_name: Optional[str] = None
    chat_type: str = "dm"  # "dm" | "group" | "channel"
    thread_id: Optional[str] = None

    @property
    def target(self) -> str:
        return format_target(self.platform, self.chat_id, self.thread_id)

    def label(self) -> str:
        who = self.user_name or self.user_id or "?"
        where = {"dm": "DM", "group": "group", "channel": "channel"}.get(
            self.chat_type, self.chat_type
        )
        return f"{self.platform} {where} · {who}"


@dataclass
class MessageEvent:
    text: str
    source: SessionSource
    message_id: Optional[str] = None
    message_type: MessageType = MessageType.TEXT
    reply_to_message_id: Optional[str] = None
    raw: Any = None

    def tagged_text(self) -> str:
        """How the message enters the super-agent thread: source + reply handle + text.

        The local GUI owner ('gui') is answered with plain assistant text (no `send_message`);
        messaging platforms carry a reply handle the agent passes back to `send_message`.
        """
        if self.source.platform == "gui":
            return f"[Owner, in the app]: {self.text}"
        return f"[{self.source.label()} | reply→{self.source.target}]: {self.text}"


@dataclass
class SendResult:
    ok: bool
    message_id: Optional[str] = None
    error: Optional[str] = None


MessageHandler = Callable[[MessageEvent], Awaitable[None]]


class BasePlatformAdapter(ABC):
    """One messaging platform. Subclasses implement connect/disconnect/send and call
    `handle_message` for inbound events."""

    platform: str = "base"

    def __init__(self) -> None:
        self._handler: Optional[MessageHandler] = None

    def set_message_handler(self, handler: MessageHandler) -> None:
        self._handler = handler

    @abstractmethod
    async def connect(self) -> bool:
        """Connect + start the inbound listener. True on success."""

    @abstractmethod
    async def disconnect(self) -> None:
        """Stop the listener and close connections."""

    @abstractmethod
    async def send(
        self, chat_id: str, text: str, *, thread_id: Optional[str] = None
    ) -> SendResult:
        """Send an outbound message."""

    async def handle_message(self, event: MessageEvent) -> None:
        if self._handler is not None:
            await self._handler(event)
