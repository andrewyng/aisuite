"""Real inbound adapters — Telegram (long-poll) and Slack (Socket Mode).

The heavy SDKs are **lazy-imported inside `connect()`** so the module imports without them
and they're optional extras. Outbound reuses the stateless senders. The raw-event → MessageEvent
mappers are pure functions (testable with plain objects/dicts, no SDK).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

from .base import (
    BasePlatformAdapter,
    InteractionEvent,
    MessageEvent,
    SendResult,
    SessionSource,
)
from .senders import _send_slack, _send_slack_interactive, _send_telegram

logger = logging.getLogger("coworker.connectors")


# -- pure mappers --------------------------------------------------------------
def telegram_message_to_event(msg: Any) -> Optional[MessageEvent]:
    text = getattr(msg, "text", None)
    if not text:
        return None
    chat = msg.chat
    user = getattr(msg, "from_user", None)
    chat_type = (
        "dm"
        if str(getattr(chat, "type", "private")).lower().endswith("private")
        else "group"
    )
    thread = getattr(msg, "message_thread_id", None)
    source = SessionSource(
        platform="telegram",
        chat_id=str(chat.id),
        user_id=str(user.id) if user else None,
        user_name=getattr(user, "full_name", None) if user else None,
        chat_type=chat_type,
        thread_id=str(thread) if thread else None,
    )
    return MessageEvent(
        text=text, source=source, message_id=str(getattr(msg, "message_id", ""))
    )


def slack_event_to_event(
    event: dict, bot_user_id: Optional[str]
) -> Optional[MessageEvent]:
    # Skip bot echoes / message edits / joins etc. (reply-loop guard).
    if event.get("bot_id") or event.get("subtype"):
        return None
    if bot_user_id and event.get("user") == bot_user_id:
        return None
    text = event.get("text") or ""
    if not text:
        return None
    chat_type = "dm" if event.get("channel_type") == "im" else "channel"
    source = SessionSource(
        platform="slack",
        chat_id=str(event.get("channel", "")),
        user_id=event.get("user"),
        chat_type=chat_type,
        thread_id=event.get("thread_ts"),
    )
    return MessageEvent(text=text, source=source, message_id=event.get("ts"))


# -- adapters ------------------------------------------------------------------
class TelegramAdapter(BasePlatformAdapter):
    platform = "telegram"

    def __init__(self, token: str) -> None:
        super().__init__()
        self.token = token
        self._app = None

    async def connect(self) -> bool:
        try:
            from telegram.ext import Application, MessageHandler, filters
        except ImportError:
            logger.warning(
                "python-telegram-bot not installed — `pip install coworker[messaging]`"
            )
            return False

        self._app = Application.builder().token(self.token).build()

        async def _on_update(update, _context):
            event = telegram_message_to_event(update.effective_message)
            if event is not None:
                await self.handle_message(event)

        self._app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, _on_update)
        )
        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling(drop_pending_updates=True)
        logger.info("telegram adapter polling")
        return True

    async def disconnect(self) -> None:
        if self._app is None:
            return
        try:
            await self._app.updater.stop()
            await self._app.stop()
            await self._app.shutdown()
        finally:
            self._app = None

    async def send(
        self, chat_id: str, text: str, *, thread_id: Optional[str] = None
    ) -> SendResult:
        return _send_telegram(self.token, chat_id, text, thread_id)


class SlackAdapter(BasePlatformAdapter):
    platform = "slack"

    def __init__(self, bot_token: str, app_token: str) -> None:
        super().__init__()
        self.bot_token = bot_token
        self.app_token = app_token
        self._app = None
        self._socket = None
        self._task: Optional[asyncio.Task] = None
        self._bot_user_id: Optional[str] = None
        self._name_cache: dict[str, str] = {}  # user_id → display name (resolved once via users.info)

    async def connect(self) -> bool:
        try:
            from slack_bolt.adapter.socket_mode.async_handler import (
                AsyncSocketModeHandler,
            )
            from slack_bolt.async_app import AsyncApp
        except ImportError:
            logger.warning(
                "slack-bolt not installed — `pip install coworker[messaging]`"
            )
            return False

        self._app = AsyncApp(token=self.bot_token)
        try:
            auth = await self._app.client.auth_test()
            self._bot_user_id = auth.get("user_id")
        except Exception:
            logger.exception("slack auth_test failed")
            return False

        @self._app.event("message")
        async def _on_message(event, _say):
            mapped = slack_event_to_event(event, self._bot_user_id)
            if mapped is not None:
                # Slack message events carry only the user id; resolve a friendly name so recent
                # senders / the allow-list don't read "unknown".
                if not mapped.source.user_name:
                    mapped.source.user_name = await self._display_name(mapped.source.user_id)
                await self.handle_message(mapped)

        # Button clicks on interactive prompts (action_id `ocw_*`). Socket mode delivers these over
        # the same connection — no public endpoint, just "Interactivity" enabled in the Slack app.
        import re as _re

        @self._app.action(_re.compile(r"^ocw_"))
        async def _on_action(ack, body):
            await ack()
            actions = body.get("actions") or [{}]
            value = actions[0].get("value", "")
            user = (body.get("user") or {})
            channel = (body.get("channel") or {}).get("id", "")
            ts = (body.get("message") or {}).get("ts")
            await self.handle_interaction(
                InteractionEvent(
                    platform="slack",
                    chat_id=str(channel),
                    message_id=ts,
                    value=str(value),
                    user_name=user.get("username") or user.get("name"),
                )
            )

        self._socket = AsyncSocketModeHandler(self._app, self.app_token)
        self._task = asyncio.create_task(self._socket.start_async())
        logger.info("slack adapter connected (socket mode) as %s", self._bot_user_id)
        return True

    async def _display_name(self, uid: Optional[str]) -> Optional[str]:
        """Resolve a user id to a display name via users.info, cached. Best-effort: None on failure
        (the caller falls back to the id)."""
        if not uid:
            return None
        if uid in self._name_cache:
            return self._name_cache[uid]
        try:
            info = await self._app.client.users_info(user=uid)
            u = info.get("user") or {}
            prof = u.get("profile") or {}
            name = (
                prof.get("display_name")
                or prof.get("real_name")
                or u.get("real_name")
                or u.get("name")
            )
        except Exception:
            name = None
        if name:
            self._name_cache[uid] = name
        return name

    async def disconnect(self) -> None:
        if self._socket is not None:
            try:
                await self._socket.close_async()
            except Exception:
                pass
        if self._task is not None:
            self._task.cancel()
            self._task = None

    async def send(
        self, chat_id: str, text: str, *, thread_id: Optional[str] = None
    ) -> SendResult:
        return _send_slack(self.bot_token, chat_id, text, thread_id)

    async def send_interactive(
        self, chat_id: str, text: str, buttons, *, thread_id: Optional[str] = None
    ) -> SendResult:
        return _send_slack_interactive(self.bot_token, chat_id, text, buttons, thread_id)

    async def update_message(self, chat_id: str, message_id: str, text: str) -> None:
        """Replace a resolved prompt's buttons with a plain-text outcome ("✅ Approved by …")."""
        if self._app is None or not message_id:
            return
        try:
            await self._app.client.chat_update(channel=chat_id, ts=message_id, text=text, blocks=[])
        except Exception:
            logger.debug("slack chat_update failed", exc_info=True)


def make_adapter(platform: str, profile: dict) -> Optional[BasePlatformAdapter]:
    """Build the adapter for a connected platform from its SecretStore profile."""
    if platform == "telegram" and profile.get("bot_token"):
        return TelegramAdapter(profile["bot_token"])
    if platform == "slack" and profile.get("bot_token") and profile.get("app_token"):
        return SlackAdapter(profile["bot_token"], profile["app_token"])
    return None
