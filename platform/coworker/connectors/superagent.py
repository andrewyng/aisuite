"""The always-on super-agent runner — one continuous thread fed by a message queue.

Inbound messages from every connected platform interleave into a single conversation:
- **idle** → the queued message(s) start a `user` turn (tagged with source + reply handle);
- **busy** mid tool-loop → injected via `TurnEngine.queue_steering()` at the next loop
  boundary, so the running agent picks them up without breaking the agentic loop.

The agent replies only by calling the `send_message` tool (the gateway never auto-delivers
its plain text). A worker task drains the queue and drives the engine.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable, Optional

from .base import MessageEvent

logger = logging.getLogger("coworker.superagent")

SUPERAGENT_MESSAGING_NOTE = (
    "\n\nYou are reachable over messaging (Slack/Telegram). Each incoming message is shown "
    "tagged like '[telegram DM · Alice | reply→telegram:12345]: <text>'. To reply to a "
    "person you MUST call the send_message tool with target set to the reply→ handle from "
    "that message — your plain text is internal and is NOT delivered anywhere. New messages "
    "may arrive while you are working; handle them in turn. Only message users who have "
    "written to you."
)


class SuperAgent:
    """Owns one persistent engine + an inbound queue. `on_message` is the gateway handler.

    `on_event` (set by the server) receives every engine event so the GUI super-agent surface
    can render the live thread.
    """

    def __init__(
        self,
        engine,
        *,
        on_saved: Optional[Callable[[], None]] = None,
        on_event: Optional[Callable[[Any], Awaitable[None]]] = None,
    ) -> None:
        self.engine = engine
        self._on_saved = on_saved
        self._on_event = on_event
        self._queue: asyncio.Queue[MessageEvent] = asyncio.Queue()
        self._running = False
        self._task: Optional[asyncio.Task] = None

    def set_event_sink(
        self, on_event: Optional[Callable[[Any], Awaitable[None]]]
    ) -> None:
        self._on_event = on_event

    async def on_message(self, event: MessageEvent) -> None:
        """Gateway/GUI handler: steer into the active turn if busy, else enqueue a new turn."""
        if self._running:
            self.engine.queue_steering(event.tagged_text())
            logger.info(
                "steered message from %s into the active turn", event.source.label()
            )
            await self._emit(
                {
                    "type": "inbound",
                    "data": {"text": event.text, "source": event.source.label()},
                }
            )
        else:
            await self._queue.put(event)

    async def _emit(self, message: dict) -> None:
        if self._on_event is not None:
            try:
                await self._on_event(message)
            except Exception:
                logger.exception("super-agent event sink failed")

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._worker())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _worker(self) -> None:
        while True:
            event = await self._queue.get()
            batch = [event]
            while not self._queue.empty():  # coalesce anything already waiting
                batch.append(self._queue.get_nowait())
            text = "\n".join(e.tagged_text() for e in batch)
            self._running = True
            try:
                async for engine_event in self.engine.run(text):
                    await self._emit(
                        {"type": engine_event.type.value, "data": engine_event.data}
                    )
            except Exception:
                logger.exception("super-agent turn failed")
            finally:
                self._running = False
                if self._on_saved is not None:
                    try:
                        self._on_saved()
                    except Exception:
                        logger.exception("super-agent save failed")
