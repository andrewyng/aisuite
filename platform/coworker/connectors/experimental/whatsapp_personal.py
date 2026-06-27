"""WhatsApp Personal adapter — EXPERIMENTAL, unofficial protocol, use at your own risk.

Two-way WhatsApp on a personal account through a supervised Node.js sidecar built on
Baileys (the unofficial WhatsApp Web protocol library). The sidecar pushes everything —
inbound messages, pairing QR codes, connection state — as NDJSON events on its stdout
pipe, which this adapter consumes directly; there is no inbound HTTP and no polling.
The sidecar's only listener is a loopback `POST /send`, kept so the stateless
`send_message` tool can deliver replies without a handle to the process. Pairing is by
QR code, so no credential is ever pasted, and npm dependencies install into the state
dir on first connect rather than shipping with the app.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import subprocess
import sys
from importlib import resources
from typing import Any, Optional

from ...secrets import state_dir
from ..base import BasePlatformAdapter, MessageEvent, SendResult, SessionSource

logger = logging.getLogger("coworker.connectors.experimental")

PLATFORM = "whatsapp_personal"
DEFAULT_PORT = 3941
_BRIDGE_FILES = ("bridge.js", "package.json")
_NPM_INSTALL_TIMEOUT = 600
_READY_TIMEOUT = 30.0


# -- pure mapper (testable without the sidecar) ----------------------------------
def bridge_event_to_message(event: dict) -> Optional[MessageEvent]:
    """Map a sidecar `message` event to a MessageEvent; anything else → None."""
    if event.get("event") != "message":
        return None
    text, chat = event.get("text") or "", event.get("chat") or ""
    if not text or not chat:
        return None
    source = SessionSource(
        platform=PLATFORM,
        chat_id=chat,
        user_id=str(event.get("sender") or "") or None,
        user_name=event.get("name") or None,
        chat_type="group" if event.get("group") else "dm",
    )
    return MessageEvent(text=text, source=source, message_id=str(event.get("id") or ""))


# -- stateless sender for the send_message tool ----------------------------------
def send_whatsapp_personal(
    port: str, chat_id: str, text: str, thread_id: Optional[str] = None
) -> SendResult:
    """POST to the sidecar's loopback send endpoint. `port` comes from the profile's
    bridge_port and may be empty (default port)."""
    import httpx

    url = f"http://127.0.0.1:{int(port or DEFAULT_PORT)}/send"
    try:
        data = httpx.post(url, json={"to": chat_id, "body": text}, timeout=30.0).json()
    except Exception as exc:
        return SendResult(False, error=f"bridge unreachable ({exc})")
    if data.get("sent"):
        return SendResult(True, message_id=str(data.get("id") or ""))
    return SendResult(False, error=data.get("error") or "whatsapp send failed")


# -- adapter ---------------------------------------------------------------------
class WhatsAppPersonalAdapter(BasePlatformAdapter):
    """Owns the sidecar: spawn on connect, consume its stdout event stream, kill on
    disconnect. Connection state and the pairing QR are cached from events, so
    `status()` answers without any network round-trip."""

    platform = PLATFORM

    def __init__(self, profile: Optional[dict] = None) -> None:
        super().__init__()
        profile = profile or {}
        self.port = int(profile.get("bridge_port") or DEFAULT_PORT)
        self.mode = "all" if str(profile.get("mode") or "").strip() == "all" else "self"
        # The sidecar runs out of the state dir, not the package: PyInstaller bundles
        # are ephemeral, and npm deps installed next to user state survive app updates.
        self.bridge_home = state_dir() / PLATFORM / "bridge"
        self.session_dir = state_dir() / PLATFORM / "session"
        self._proc: Optional[subprocess.Popen] = None
        self._reader_task: Optional[asyncio.Task] = None
        self._ready = asyncio.Event()
        self.state: dict[str, Any] = {"state": "down", "me": None, "qr": None}

    # -- event handling (pure-ish, unit-testable) --------------------------------
    async def handle_event_line(self, line: str) -> None:
        try:
            event = json.loads(line)
        except (ValueError, TypeError):
            return
        kind = event.get("event")
        if kind == "ready":
            self._ready.set()
        elif kind == "state":
            self.state["state"] = event.get("state")
            self.state["me"] = event.get("me")
            if event.get("state") == "open":
                self.state["qr"] = None
        elif kind == "qr":
            self.state["qr"] = event.get("qr")
        elif kind == "message":
            msg = bridge_event_to_message(event)
            if msg is not None:
                await self.handle_message(msg)

    # -- sidecar lifecycle --------------------------------------------------------
    @staticmethod
    def node_path() -> Optional[str]:
        return shutil.which("node")

    def _stage_bridge(self) -> None:
        """Copy the sidecar sources from the package into the state dir (always
        refreshed so app updates propagate; node_modules and the session persist)."""
        self.bridge_home.mkdir(parents=True, exist_ok=True)
        pkg = resources.files(__package__) / "whatsapp_bridge"
        for name in _BRIDGE_FILES:
            (self.bridge_home / name).write_bytes((pkg / name).read_bytes())

    def _deps_installed(self) -> bool:
        return (self.bridge_home / "node_modules" / "@whiskeysockets").is_dir()

    def _install_deps(self) -> bool:
        npm = shutil.which("npm")
        if not npm:
            logger.error("[%s] npm not found — install Node.js first", PLATFORM)
            return False
        logger.info("[%s] installing sidecar dependencies (first run)…", PLATFORM)
        result = subprocess.run(
            [npm, "install", "--omit=dev", "--no-audit", "--no-fund"],
            cwd=self.bridge_home,
            capture_output=True,
            timeout=_NPM_INSTALL_TIMEOUT,
        )
        if result.returncode != 0:
            logger.error(
                "[%s] npm install failed: %s",
                PLATFORM,
                result.stderr.decode(errors="replace")[-500:],
            )
        return result.returncode == 0

    def _spawn(self, node: str) -> subprocess.Popen:
        self.session_dir.mkdir(parents=True, exist_ok=True)
        kwargs: dict[str, Any] = {}
        if sys.platform != "win32":
            kwargs["start_new_session"] = True  # own process group → clean teardown
        return subprocess.Popen(
            [
                node,
                str(self.bridge_home / "bridge.js"),
                "--port",
                str(self.port),
                "--session",
                str(self.session_dir),
                "--mode",
                self.mode,
            ],
            cwd=self.bridge_home,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            **kwargs,
        )

    async def _read_events(self) -> None:
        """Consume the sidecar's stdout until it closes. Draining the pipe is also what
        keeps the sidecar from blocking on a full buffer."""
        proc = self._proc
        while proc is not None and proc.stdout is not None:
            line = await asyncio.to_thread(proc.stdout.readline)
            if not line:  # EOF → sidecar exited
                self.state.update({"state": "down", "qr": None})
                break
            await self.handle_event_line(line.decode(errors="replace"))

    async def connect(self) -> bool:
        node = self.node_path()
        if not node:
            logger.error("[%s] Node.js is required but not installed", PLATFORM)
            return False
        await asyncio.to_thread(self._stage_bridge)
        if not self._deps_installed():
            if not await asyncio.to_thread(self._install_deps):
                return False
        self._ready.clear()
        self._proc = self._spawn(node)
        self._reader_task = asyncio.create_task(self._read_events())
        try:
            await asyncio.wait_for(self._ready.wait(), timeout=_READY_TIMEOUT)
        except asyncio.TimeoutError:
            logger.error("[%s] sidecar did not become ready", PLATFORM)
            await self.disconnect()
            return False
        logger.info(
            "[%s] sidecar up on port %s (mode=%s)", PLATFORM, self.port, self.mode
        )
        return True

    async def status(self) -> dict[str, Any]:
        """Pairing/connection state for the UI, served from the cached event stream."""
        running = self._proc is not None and self._proc.poll() is None
        return {"running": running, **self.state}

    async def send(
        self, chat_id: str, text: str, *, thread_id: Optional[str] = None
    ) -> SendResult:
        return await asyncio.to_thread(
            send_whatsapp_personal, str(self.port), chat_id, text, thread_id
        )

    async def disconnect(self) -> None:
        if self._proc is not None:
            self._proc.terminate()
            try:
                await asyncio.to_thread(self._proc.wait, 5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None
        if self._reader_task is not None:
            self._reader_task.cancel()
            self._reader_task = None
        self.state.update({"state": "down", "qr": None})
