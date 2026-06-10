"""Gateway — owns the messaging adapters and routes inbound messages.

Lives inside the always-on `coworker-server` (started/stopped in its lifespan). On inbound:
enforce the per-platform allowlist, then hand the message to the registered handler (the
super-agent runner, wired in the next increment). Outbound replies go through the
`send_message` tool, not the gateway — so the gateway stays a thin inbound router here.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Optional

from ..secrets import SecretStore
from .base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageHandler,
    SendResult,
    parse_target,
)
from .config import ConnectorSettings, is_authorized, load_settings

logger = logging.getLogger("coworker.connectors")

_RECENT_CAP = 20  # most-recent distinct senders kept for chat-ID auto-capture


class Gateway:
    def __init__(
        self,
        *,
        secrets: Optional[SecretStore] = None,
        settings: Optional[dict[str, ConnectorSettings]] = None,
        handler: Optional[MessageHandler] = None,
    ) -> None:
        self.secrets = secrets or SecretStore()
        self.settings = (
            settings if settings is not None else load_settings(self.secrets)
        )
        self._handler = handler
        self._adapters: dict[str, BasePlatformAdapter] = {}
        # In-memory recent senders for chat-ID auto-capture (identity only, never persisted).
        self._recent: "OrderedDict[tuple[str, str], dict]" = OrderedDict()

    def set_handler(self, handler: MessageHandler) -> None:
        self._handler = handler

    def register(self, adapter: BasePlatformAdapter) -> None:
        adapter.set_message_handler(self._on_inbound)
        self._adapters[adapter.platform] = adapter

    async def _on_inbound(self, event: MessageEvent) -> None:
        self._record_recent(event)  # capture identity even from unauthorized senders
        settings = self.settings.get(event.source.platform)
        if settings is None or not is_authorized(settings, event.source):
            logger.info("dropping unauthorized inbound from %s", event.source.label())
            return
        if self._handler is not None:
            await self._handler(event)

    def _record_recent(self, event: MessageEvent) -> None:
        s = event.source
        if not s.user_id:
            return
        key = (s.platform, s.user_id)
        self._recent.pop(key, None)  # move to most-recent
        self._recent[key] = {
            "platform": s.platform,
            "user_id": s.user_id,
            "user_name": s.user_name,
            "chat_id": s.chat_id,
            "chat_type": s.chat_type,
            "target": s.target,
        }
        while len(self._recent) > _RECENT_CAP:
            self._recent.popitem(last=False)

    def recent_senders(self, platform: Optional[str] = None) -> list[dict]:
        """Most-recent-first list of who has messaged (for the allowlist UI)."""
        items = list(self._recent.values())[::-1]
        return [e for e in items if platform is None or e["platform"] == platform]

    async def start(self) -> list[str]:
        """Connect every enabled+registered adapter. Returns the platforms that came up."""
        live: list[str] = []
        for platform, settings in self.settings.items():
            if not settings.enabled:
                continue
            adapter = self._adapters.get(platform)
            if adapter is None:
                continue
            try:
                if await adapter.connect():
                    live.append(platform)
            except Exception:  # bad token / network — skip, don't break the server
                logger.exception("failed to connect %s adapter", platform)
        return live

    async def stop(self) -> None:
        for adapter in self._adapters.values():
            try:
                await adapter.disconnect()
            except Exception:
                logger.exception("error disconnecting %s adapter", adapter.platform)

    async def deliver(self, target: str, text: str) -> SendResult:
        """Send via a live adapter (used where the persistent connection is preferred)."""
        platform, chat_id, thread_id = parse_target(target)
        adapter = self._adapters.get(platform)
        if adapter is None:
            return SendResult(False, error=f"no adapter for {platform}")
        return await adapter.send(chat_id, text, thread_id=thread_id)

    def status(self) -> list[dict]:
        out = []
        for platform, settings in self.settings.items():
            out.append(
                {
                    "platform": platform,
                    "enabled": settings.enabled,
                    "connected": platform in self._adapters,
                    "allow_all": settings.allow_all,
                    "allowed_users": len(settings.allowed_users),
                }
            )
        return out
