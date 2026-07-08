"""Managed-relay inbound adapter — the cloud-relay alternative to Socket Mode.

The desktop offers the user two ways to receive Slack:
- **Socket Mode** (`SlackAdapter`): manual bot + app tokens, one workspace, a
  direct WebSocket to Slack. No cloud involved.
- **Managed relay** (`SlackRelayAdapter`, here): "Add to Slack" OAuth, no tokens
  typed, *many* workspaces, events pushed from OpenCoworker Cloud over one
  authenticated WebSocket. Replies still go desktop → Slack Web API directly
  with the per-team bot token (the relay is inbound-only).

Both register on the gateway as platform ``slack`` and produce the same
``MessageEvent``/``InteractionEvent`` — downstream code doesn't care which mode
delivered a message. Managed-relay reply handles are **team-qualified**
(``slack:T…/C…``) so multi-workspace replies pick the right token (see
``slack_addr``).

The socket transport is injectable so the frame-handling logic is tested with a
fake relay (no live WebSocket); the default transport is a thin ``websockets``
client, lazy-imported like the Socket-Mode SDK.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from typing import Any, Awaitable, Callable, Optional, Protocol

from .adapters import _SLACK_MENTION_RE, slack_event_to_event
from .base import BasePlatformAdapter, InteractionEvent, SendResult, SessionSource
from .senders import _send_slack, _send_slack_interactive
from .slack_addr import qualify

logger = logging.getLogger("coworker.connectors")


class RelayTransport(Protocol):
    """One live connection to the cloud relay. Implementations lazy-import their
    WebSocket library; the frame contract is decoded JSON dicts."""

    async def open(self) -> None: ...
    async def recv(self) -> Optional[dict]:
        """Next frame, or None when the connection has closed."""
        ...
    async def close(self) -> None: ...


TransportFactory = Callable[[], RelayTransport]
TokenProvider = Callable[[], str]  # returns the current cloud sign-in JWT
# team_id, channel, count -> list of raw Slack message dicts (newest last)
HistoryFetcher = Callable[[str, str, int], Awaitable[list[dict]]]


class SlackRelayAdapter(BasePlatformAdapter):
    platform = "slack"

    _RECONNECT_DELAY = 2.0

    def __init__(
        self,
        relay_url: str,
        token_provider: TokenProvider,
        *,
        teams: Optional[dict[str, dict[str, Any]]] = None,
        transport_factory: Optional[TransportFactory] = None,
        history_fetcher: Optional[HistoryFetcher] = None,
        reconnect_delay: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.relay_url = relay_url
        self._token_provider = token_provider
        # team_id -> {"bot_token", "bot_user_id"}. Mutable: a `revoked` frame or a
        # new install updates it.
        self._teams: dict[str, dict[str, Any]] = dict(teams or {})
        self._transport_factory = transport_factory or self._default_transport_factory
        self._history_fetcher = history_fetcher
        self._reconnect_delay = (
            reconnect_delay if reconnect_delay is not None else self._RECONNECT_DELAY
        )
        self._transport: Optional[RelayTransport] = None
        self._task: Optional[asyncio.Task] = None
        self._closing = False
        self._connections = 0  # total successful opens; reconnects == connections-1
        self._dispatched = 0  # frames dispatched (observable for tests)
        self._progress = asyncio.Event()
        # Name resolution caches, keyed PER WORKSPACE — a U…/C… id only means
        # something inside its team, and resolution uses that team's bot token.
        self._names: dict[str, dict[str, str]] = {}  # team_id -> {uid: name}
        self._channels: dict[str, dict[str, str]] = {}  # team_id -> {cid: name}

    # -- lifecycle -----------------------------------------------------------
    async def connect(self) -> bool:
        self._closing = False
        self._transport = self._transport_factory()
        try:
            await self._transport.open()
        except Exception:
            logger.exception("relay connect failed")
            return False
        self._connections = 1
        self._task = asyncio.create_task(self._run())
        logger.info("slack adapter connected (managed relay), %d team(s)", len(self._teams))
        return True

    async def _run(self) -> None:
        """Read frames; on a dropped connection, reconnect (fresh transport) —
        the relay's own watchdog analogue on the desktop side."""
        while not self._closing:
            try:
                frame = await self._transport.recv() if self._transport else None
            except Exception:
                logger.exception("relay recv error")
                frame = None
            if frame is not None:
                try:
                    await self._dispatch(frame)
                except Exception:
                    logger.exception("relay frame dispatch failed")
                self._dispatched += 1
                self._progress.set()
                continue
            # Connection closed → reconnect unless we're shutting down.
            if self._closing:
                break
            await self._reconnect()

    async def _reconnect(self) -> None:
        try:
            await asyncio.sleep(self._reconnect_delay)
        except asyncio.CancelledError:
            return
        if self._closing:
            return
        self._transport = self._transport_factory()
        try:
            await self._transport.open()
            self._connections += 1
            logger.info("slack relay reconnected (#%d)", self._connections - 1)
        except Exception:
            logger.exception("relay reconnect failed — will retry")

    async def disconnect(self) -> None:
        self._closing = True
        if self._transport is not None:
            try:
                await self._transport.close()
            except Exception:
                pass
        if self._task is not None:
            self._task.cancel()
            self._task = None

    @property
    def reconnects(self) -> int:
        return max(0, self._connections - 1)

    async def wait_dispatched(self, at_least: int, timeout: float = 2.0) -> None:
        """Test helper: wait until at least N frames have been dispatched."""
        loop = asyncio.get_event_loop()
        deadline = loop.time() + timeout
        while self._dispatched < at_least:
            self._progress.clear()
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise TimeoutError(f"only {self._dispatched} frames dispatched (< {at_least})")
            try:
                await asyncio.wait_for(self._progress.wait(), timeout=remaining)
            except asyncio.TimeoutError:
                raise TimeoutError(f"only {self._dispatched} frames dispatched (< {at_least})")

    # -- team registry -------------------------------------------------------
    def set_team(self, team_id: str, bot_token: str, bot_user_id: Optional[str] = None) -> None:
        self._teams[team_id] = {"bot_token": bot_token, "bot_user_id": bot_user_id}

    def _bot_user_id(self, team_id: str) -> Optional[str]:
        return (self._teams.get(team_id) or {}).get("bot_user_id")

    def _bot_token(self, team_id: str) -> Optional[str]:
        return (self._teams.get(team_id) or {}).get("bot_token")

    # -- frame dispatch ------------------------------------------------------
    async def _dispatch(self, frame: dict) -> None:
        kind = frame.get("kind")
        if kind == "missed":
            await self._on_missed(frame)
            return
        if kind == "revoked":
            self._teams.pop(frame.get("team_id", ""), None)
            logger.info("slack relay team %s revoked — dropped", frame.get("team_id"))
            return
        if kind == "interactivity":
            await self._on_interactivity(frame)
            return
        # A routed Slack event.
        await self._on_event(frame)

    async def _on_event(self, frame: dict) -> None:
        await self._dispatch_slack_event(frame.get("team_id", ""), frame.get("event") or {})

    async def _dispatch_slack_event(self, team_id: str, event: dict) -> None:
        """Map a raw Slack event → MessageEvent, resolve display names via the
        per-team bot token, team-qualify the reply handle, and dispatch."""
        mapped = slack_event_to_event(event, self._bot_user_id(team_id))
        if mapped is None:
            return
        channel = mapped.source.chat_id  # bare channel id before qualification
        # Resolve friendly names with THIS workspace's bot token (cached per team),
        # mirroring the Socket-Mode adapter — so cards read "@ocw"/"Rohit"/"#ocw-test"
        # not raw U…/C… ids. Best-effort: ids fall through on failure.
        if not mapped.source.user_name:
            mapped.source.user_name = await self._display_name(team_id, mapped.source.user_id)
        if not mapped.source.chat_name:
            mapped.source.chat_name = await self._channel_name(team_id, channel)
        mapped.text = await self._resolve_mentions(team_id, mapped.text)
        # Team-qualify the reply handle so multi-workspace replies pick the right
        # per-team token.
        mapped.source.chat_id = qualify(team_id, channel)
        mapped.source.team_id = team_id
        await self.handle_message(mapped)

    async def _on_interactivity(self, frame: dict) -> None:
        interaction = frame.get("interaction") or {}
        actions = interaction.get("actions") or [{}]
        value = actions[0].get("value", "")
        user = interaction.get("user") or {}
        team_id = frame.get("team_id", "")
        channel = (interaction.get("channel") or {}).get("id", "")
        ts = (interaction.get("message") or {}).get("ts")
        await self.handle_interaction(
            InteractionEvent(
                platform="slack",
                chat_id=qualify(team_id, channel),
                message_id=ts,
                value=str(value),
                user_name=user.get("username") or user.get("name"),
            )
        )

    async def _on_missed(self, frame: dict) -> None:
        """A nudge: content was dropped (offline > TTL / overflow). Pull the
        recent channel history ourselves via the per-team bot token and replay
        the missed messages (spec §7 channel-context / nudge)."""
        team_id = frame.get("team_id", "")
        channel = frame.get("channel", "")
        count = int(frame.get("count", 0)) or 1
        if self._history_fetcher is None or not channel:
            return
        try:
            messages = await self._history_fetcher(team_id, channel, count)
        except Exception:
            logger.exception("relay nudge history fetch failed")
            return
        for raw in messages:
            await self._dispatch_slack_event(team_id, {**raw, "channel": channel})

    # -- name resolution (per workspace, via that team's bot token) ----------
    async def _slack_get(self, team_id: str, method: str, params: dict) -> Optional[dict]:
        """Call a Slack Web API read method with the team's bot token. Best-effort
        (None on any failure). `SLACK_API_URL` redirects to the fake in tests."""
        import httpx

        token = self._bot_token(team_id)
        if not token:
            return None
        base = os.environ.get("SLACK_API_URL", "https://slack.com/api/")
        try:
            async with httpx.AsyncClient(timeout=15) as http:
                resp = await http.get(
                    base + method, params=params,
                    headers={"Authorization": f"Bearer {token}"},
                )
            data = resp.json()
        except Exception:
            return None
        return data if data.get("ok") else None

    async def _display_name(self, team_id: str, uid: Optional[str]) -> Optional[str]:
        if not uid:
            return None
        cache = self._names.setdefault(team_id, {})
        if uid in cache:
            return cache[uid]
        data = await self._slack_get(team_id, "users.info", {"user": uid})
        u = (data or {}).get("user") or {}
        prof = u.get("profile") or {}
        name = prof.get("display_name") or prof.get("real_name") or u.get("real_name") or u.get("name")
        if name:
            cache[uid] = name
        return name

    async def _channel_name(self, team_id: str, cid: Optional[str]) -> Optional[str]:
        if not cid:
            return None
        cache = self._channels.setdefault(team_id, {})
        if cid in cache:
            return cache[cid]
        data = await self._slack_get(team_id, "conversations.info", {"channel": cid})
        chan = (data or {}).get("channel") or {}
        name = chan.get("name") or chan.get("name_normalized")
        if name:
            cache[cid] = name
        return name

    async def _resolve_mentions(self, team_id: str, text: str) -> str:
        """Rewrite `<@U…>` tokens to `@display-name` (cached). Best-effort."""
        out = text
        for uid in set(_SLACK_MENTION_RE.findall(text or "")):
            name = await self._display_name(team_id, uid)
            if name:
                out = re.sub(rf"<@{re.escape(uid)}(?:\|[^>]*)?>", f"@{name}", out)
        return out

    # -- outbound ------------------------------------------------------------
    async def send(
        self, chat_id: str, text: str, *, thread_id: Optional[str] = None
    ) -> SendResult:
        """Reply directly via the Slack Web API with the per-team bot token."""
        from .slack_addr import split

        team_id, _channel = split(chat_id)
        token = self._bot_token(team_id or "")
        if not token:
            return SendResult(False, error=f"no bot token for team {team_id}")
        return await asyncio.to_thread(_send_slack, token, chat_id, text, thread_id)

    async def send_interactive(
        self, chat_id: str, text: str, buttons, *, thread_id: Optional[str] = None
    ) -> SendResult:
        from .slack_addr import split

        team_id, _channel = split(chat_id)
        token = self._bot_token(team_id or "")
        if not token:
            return SendResult(False, error=f"no bot token for team {team_id}")
        return await asyncio.to_thread(
            _send_slack_interactive, token, chat_id, text, buttons, thread_id
        )

    # -- default transport ---------------------------------------------------
    def _default_transport_factory(self) -> RelayTransport:
        return _WebSocketsTransport(self.relay_url, self._token_provider)


class _WebSocketsTransport:
    """Real transport: an authenticated `websockets` client. Sends the cloud
    sign-in JWT in the Authorization header (the relay's $connect authorizer)."""

    def __init__(self, url: str, token_provider: TokenProvider) -> None:
        self._url = url
        self._token_provider = token_provider
        self._ws = None

    async def open(self) -> None:
        import websockets  # lazy: optional extra

        token = self._token_provider()
        self._ws = await websockets.connect(
            self._url, additional_headers={"Authorization": f"Bearer {token}"}
        )

    async def recv(self) -> Optional[dict]:
        import websockets

        if self._ws is None:
            return None
        try:
            raw = await self._ws.recv()
        except websockets.ConnectionClosed:
            return None
        try:
            return json.loads(raw)
        except (ValueError, TypeError):
            return None

    async def close(self) -> None:
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None
