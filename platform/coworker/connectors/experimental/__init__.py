"""Experimental connectors — use-at-your-own-risk integrations, excluded from release builds.

Connectors in this package are hidden behind the experimental-connectors setting, require an
explicit per-connector risk acknowledgment to connect, and are stripped from official desktop
builds by packaging/coworker-server.spec (set COWORKER_EXPERIMENTAL=1 at build time to include
them in a self-built binary).

To add one: define a `ConnectorDescriptor` with a `risk_notice` that states the concrete
downside in plain language, append it to `EXPERIMENTAL_DESCRIPTORS`, and register its
platform/adapter/sender via the registries in config.py, adapters.py, and senders.py. The
`experimental` flag is forced on by the loader in descriptors.py regardless of what the
descriptor sets.
"""

from __future__ import annotations

from ..adapters import register_adapter_factory
from ..config import register_platform
from ..descriptors import ConnectorDescriptor, Field
from ..senders import register_sender
from .whatsapp_personal import (
    PLATFORM as _WA_PLATFORM,
    WhatsAppPersonalAdapter,
    send_whatsapp_personal,
)

WHATSAPP_PERSONAL = ConnectorDescriptor(
    name=_WA_PLATFORM,
    title="WhatsApp Personal",
    icon="◌",
    blurb="Two-way WhatsApp on your personal account via the unofficial WhatsApp Web protocol.",
    auth="qr",
    two_way=True,
    fields=[
        Field(
            "mode",
            "Mode",
            required=False,
            help="self (default): the agent lives in your message-yourself thread. all: every chat on the account.",
            placeholder="self",
        ),
        Field(
            "bridge_port",
            "Bridge port",
            required=False,
            help="Local port for the bridge process. Default 3941.",
            placeholder="3941",
        ),
        Field(
            "allowed_users",
            "Allowed user IDs",
            required=False,
            help="Comma-separated phone numbers (digits only) allowed to message the agent. Empty = nobody (self-chat mode only needs your own).",
            placeholder="14155550123",
        ),
    ],
    instructions=[
        "Requires Node.js on this machine — the bridge installs its own dependencies on first start.",
        "Use a SECONDARY phone number. Never pair your primary personal account.",
        "Connect here, then start the super-agent: the bridge starts and shows a QR code in the connector status.",
        "Scan it from WhatsApp → Settings → Linked devices → Link a device.",
        "In self mode, talk to the agent in your own message-yourself thread.",
    ],
    risk_notice=(
        "This connector speaks the unofficial WhatsApp Web protocol (Baileys). That violates "
        "WhatsApp's Terms of Service, and Meta detects and permanently bans accounts using it, "
        "without warning and without appeal. Only use a secondary number you can afford to "
        "lose. Nothing about this is endorsed by or affiliated with WhatsApp/Meta."
    ),
)

register_platform(_WA_PLATFORM, credential_key=None)
register_adapter_factory(_WA_PLATFORM, WhatsAppPersonalAdapter)
register_sender(
    _WA_PLATFORM,
    send_whatsapp_personal,
    credential_key="bridge_port",
    credential_required=False,
)

EXPERIMENTAL_DESCRIPTORS: list[ConnectorDescriptor] = [WHATSAPP_PERSONAL]
