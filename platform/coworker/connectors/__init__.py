"""Messaging connectors — Slack/Telegram adapters, the gateway, and the send_message tool."""

from __future__ import annotations

from .base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
    SessionSource,
    format_target,
    parse_target,
)
from .adapters import (
    SlackAdapter,
    TelegramAdapter,
    make_adapter,
    slack_event_to_event,
    telegram_message_to_event,
)
from .config import ConnectorSettings, is_authorized, load_settings
from .descriptors import ConnectorDescriptor, get_descriptor, list_descriptors
from .fake import FakeAdapter
from .gateway import Gateway
from .senders import DEFAULT_SENDERS
from .setup import (
    connect_connector,
    connector_list,
    disconnect_connector,
    update_connector_tools,
)
from .superagent import SUPERAGENT_MESSAGING_NOTE, SuperAgent
from .integration_tools import make_integration_tools
from .tools import make_send_message_tool
from .tool_defs import connector_for_tool

__all__ = [
    "BasePlatformAdapter",
    "MessageEvent",
    "MessageType",
    "SendResult",
    "SessionSource",
    "format_target",
    "parse_target",
    "ConnectorSettings",
    "is_authorized",
    "load_settings",
    "ConnectorDescriptor",
    "get_descriptor",
    "list_descriptors",
    "FakeAdapter",
    "Gateway",
    "DEFAULT_SENDERS",
    "connect_connector",
    "connector_list",
    "disconnect_connector",
    "update_connector_tools",
    "make_integration_tools",
    "make_send_message_tool",
    "connector_for_tool",
    "SlackAdapter",
    "TelegramAdapter",
    "make_adapter",
    "slack_event_to_event",
    "telegram_message_to_event",
    "SuperAgent",
    "SUPERAGENT_MESSAGING_NOTE",
]
