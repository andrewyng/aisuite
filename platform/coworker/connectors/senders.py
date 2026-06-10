"""Stateless outbound senders — one-shot HTTP POSTs, no SDK, no live connection.

These power the `send_message` tool (and the super-agent's replies). Both Telegram and
Slack outbound are simple HTTP calls, so we use a synchronous `httpx` client and avoid the
heavy SDKs (those are only needed for the inbound listeners). Sync fits the ToolRegistry's
`execute` contract (the engine runs it in a thread).

A `Sender` is `(token, chat_id, text, thread_id) -> SendResult`. The registry is swappable so
tests inject fakes — no network.
"""

from __future__ import annotations

from typing import Callable, Optional

from .base import SendResult

Sender = Callable[[str, str, str, Optional[str]], SendResult]

_TIMEOUT = 30.0


def _send_telegram(
    token: str, chat_id: str, text: str, thread_id: Optional[str] = None
) -> SendResult:
    import httpx

    payload: dict = {"chat_id": chat_id, "text": text}
    # Telegram's General forum topic is thread_id "1", which sendMessage rejects → omit it.
    if thread_id and thread_id != "1":
        try:
            payload["message_thread_id"] = int(thread_id)
        except ValueError:
            pass
    try:
        resp = httpx.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json=payload,
            timeout=_TIMEOUT,
        )
        data = resp.json()
    except Exception as exc:  # network / decode
        return SendResult(False, error=str(exc))
    if data.get("ok"):
        return SendResult(
            True, message_id=str(data.get("result", {}).get("message_id"))
        )
    return SendResult(False, error=data.get("description") or "telegram send failed")


def _send_slack(
    token: str, chat_id: str, text: str, thread_id: Optional[str] = None
) -> SendResult:
    import httpx

    payload: dict = {"channel": chat_id, "text": text}
    if thread_id:
        payload["thread_ts"] = thread_id
    try:
        resp = httpx.post(
            "https://slack.com/api/chat.postMessage",
            headers={"Authorization": f"Bearer {token}"},
            json=payload,
            timeout=_TIMEOUT,
        )
        data = resp.json()
    except Exception as exc:
        return SendResult(False, error=str(exc))
    if data.get("ok"):
        return SendResult(True, message_id=data.get("ts"))
    return SendResult(False, error=data.get("error") or "slack send failed")


DEFAULT_SENDERS: dict[str, Sender] = {
    "telegram": _send_telegram,
    "slack": _send_slack,
}
