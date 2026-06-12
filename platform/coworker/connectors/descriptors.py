"""Connector descriptors — data that drives the guided setup wizard.

Adding a connector is (mostly) data, not UI code: a descriptor declares its auth method,
the fields the user pastes, step-by-step instructions, and a `validate` that confirms the
token by a real API call (and returns the bot identity to show back). Designed so a managed
one-click OAuth (`auth="oauth"`) can slot in later for the cloud product without changing the
data model — only the connect action differs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class Field:
    key: str
    label: str
    secret: bool = False
    required: bool = True
    help: str = ""
    placeholder: str = ""

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "label": self.label,
            "secret": self.secret,
            "required": self.required,
            "help": self.help,
            "placeholder": self.placeholder,
        }


@dataclass
class ValidationResult:
    ok: bool
    identity: Optional[str] = (
        None  # e.g. "@mybot" — shown back to the user, never a secret
    )
    error: Optional[str] = None


@dataclass
class ConnectorDescriptor:
    name: str
    title: str
    icon: str
    blurb: str
    auth: str  # "bot_token" | "socket_app" | "oauth" | "token" | "api_token" | "none"
    two_way: bool
    fields: list[Field]
    instructions: list[str]
    available: bool = True  # False → shown as "soon"
    validate: Optional[Callable[[dict], ValidationResult]] = None


# -- validators (sync httpx, one-shot) -----------------------------------------
def _validate_telegram(creds: dict) -> ValidationResult:
    import httpx

    token = creds.get("bot_token", "")
    try:
        data = httpx.get(
            f"https://api.telegram.org/bot{token}/getMe", timeout=15
        ).json()
    except Exception as exc:
        return ValidationResult(False, error=str(exc))
    if data.get("ok"):
        return ValidationResult(
            True, identity="@" + str(data["result"].get("username", "bot"))
        )
    return ValidationResult(False, error=data.get("description") or "invalid bot token")


def _validate_email(creds: dict) -> ValidationResult:
    from .email_tools import validate_email_account

    ok, identity, error = validate_email_account(creds)
    return ValidationResult(ok, identity=identity or None, error=error or None)


def _validate_slack(creds: dict) -> ValidationResult:
    import httpx

    token = creds.get("bot_token", "")
    try:
        data = httpx.post(
            "https://slack.com/api/auth.test",
            headers={"Authorization": f"Bearer {token}"},
            timeout=15,
        ).json()
    except Exception as exc:
        return ValidationResult(False, error=str(exc))
    if data.get("ok"):
        return ValidationResult(
            True, identity=f"{data.get('team', '?')} / {data.get('user', 'bot')}"
        )
    return ValidationResult(False, error=data.get("error") or "invalid bot token")


_ALLOWED_FIELD = Field(
    key="allowed_users",
    label="Allowed user IDs",
    required=False,
    help="Comma-separated IDs allowed to message the bot. Leave empty, then DM the bot and use Capture.",
    placeholder="123456789",
)

DESCRIPTORS: list[ConnectorDescriptor] = [
    ConnectorDescriptor(
        name="telegram",
        title="Telegram",
        icon="✈",
        blurb="Two-way messaging with a Telegram bot.",
        auth="bot_token",
        two_way=True,
        fields=[
            Field(
                "bot_token",
                "Bot token",
                secret=True,
                help="From @BotFather.",
                placeholder="123456:ABC-DEF…",
            ),
            _ALLOWED_FIELD,
        ],
        instructions=[
            "Open Telegram and message @BotFather.",
            "Send /newbot and pick a name + username.",
            "Copy the HTTP API token it gives you and paste it below.",
            "After connecting, DM your new bot once, then use Capture to grab your user ID.",
        ],
        validate=_validate_telegram,
    ),
    ConnectorDescriptor(
        name="slack",
        title="Slack",
        icon="💬",
        blurb="Two-way messaging via a Slack app (Socket Mode).",
        auth="socket_app",
        two_way=True,
        fields=[
            Field(
                "bot_token",
                "Bot token",
                secret=True,
                help="Bot User OAuth Token.",
                placeholder="xoxb-…",
            ),
            Field(
                "app_token",
                "App token",
                secret=True,
                help="App-level token for Socket Mode.",
                placeholder="xapp-…",
            ),
            _ALLOWED_FIELD,
        ],
        instructions=[
            "Go to api.slack.com/apps → Create New App (from scratch).",
            "Settings → Socket Mode: enable it and generate an app-level token (xapp-) with connections:write.",
            "OAuth & Permissions: add bot scopes chat:write, app_mentions:read, im:history, channels:history.",
            "Install to workspace and copy the Bot User OAuth Token (xoxb-).",
            "Paste both tokens below and Connect, then invite the bot to a channel or DM it.",
        ],
        validate=_validate_slack,
    ),
    ConnectorDescriptor(
        name="email",
        title="Email (IMAP)",
        icon="✉",
        blurb="Read, search, and send mail from any IMAP account — Gmail, iCloud, Fastmail, or custom.",
        auth="app_password",
        two_way=False,
        fields=[
            Field("address", "Email address", placeholder="you@gmail.com"),
            Field(
                "app_password",
                "App password",
                secret=True,
                help="Gmail/iCloud: generate an app password (requires 2-step verification). Not your account password.",
            ),
            Field(
                "display_name",
                "Display name",
                required=False,
                help="Shown as the From name on sent mail.",
            ),
            Field(
                "imap_host",
                "IMAP host (advanced)",
                required=False,
                help="Only needed for providers we don't auto-detect.",
                placeholder="imap.example.com",
            ),
            Field(
                "imap_port", "IMAP port (advanced)", required=False, placeholder="993"
            ),
            Field(
                "smtp_host",
                "SMTP host (advanced)",
                required=False,
                placeholder="smtp.example.com",
            ),
            Field(
                "smtp_port", "SMTP port (advanced)", required=False, placeholder="587"
            ),
        ],
        instructions=[
            "Gmail: turn on 2-Step Verification, then create an app password at myaccount.google.com/apppasswords.",
            "iCloud: generate an app-specific password at account.apple.com → Sign-In and Security.",
            "Enter your address and the app password below. Gmail, iCloud, and Fastmail servers are detected automatically; for other providers fill in the IMAP/SMTP hosts.",
            "Note: Google Workspace and Microsoft 365 accounts often have IMAP or app passwords disabled by the org admin.",
        ],
        validate=_validate_email,
    ),
    ConnectorDescriptor(
        name="gmail",
        title="Gmail",
        icon="✉",
        blurb="Search, summarize, draft, and send email.",
        auth="oauth",
        two_way=False,
        fields=[
            Field(
                "access_token",
                "OAuth access token",
                secret=True,
                help="Google OAuth token with Gmail scopes.",
            ),
        ],
        instructions=[
            "Use a Google OAuth access token with Gmail readonly and send scopes.",
            "Paste the access token below. Managed sign-in will replace this manual step later.",
        ],
        available=True,
    ),
    ConnectorDescriptor(
        name="google_calendar",
        title="Google Calendar",
        icon="◷",
        blurb="Read availability, summarize schedules, and create events.",
        auth="oauth",
        two_way=False,
        fields=[
            Field(
                "access_token",
                "OAuth access token",
                secret=True,
                help="Google OAuth token with Calendar scopes.",
            ),
        ],
        instructions=[
            "Use a Google OAuth access token with Calendar read/write scopes.",
            "Paste the access token below. Managed sign-in will replace this manual step later.",
        ],
        available=True,
    ),
    ConnectorDescriptor(
        name="browser",
        title="Browser",
        icon="⌕",
        blurb="Let agents navigate, read, and act on websites with approval.",
        auth="none",
        two_way=False,
        fields=[],
        instructions=[
            "No setup required. Browser tools are available to Cowork sessions."
        ],
        available=True,
    ),
    ConnectorDescriptor(
        name="github",
        title="GitHub",
        icon="⌘",
        blurb="Work with issues, pull requests, repository files, and CI status.",
        auth="token",
        two_way=False,
        fields=[
            Field(
                "token",
                "Personal access token",
                secret=True,
                help="Fine-grained or classic GitHub token.",
            ),
        ],
        instructions=[
            "Create a GitHub personal access token with access to the target repositories.",
            "For write actions, include Issues or Pull Requests write permissions as needed.",
        ],
        available=True,
    ),
    ConnectorDescriptor(
        name="notion",
        title="Notion",
        icon="□",
        blurb="Search pages, summarize knowledge bases, and draft updates.",
        auth="token",
        two_way=False,
        fields=[
            Field(
                "token",
                "Integration token",
                secret=True,
                help="Internal integration secret from Notion.",
            ),
        ],
        instructions=[
            "Create a Notion internal integration and copy its secret.",
            "Share the relevant pages/databases with that integration.",
        ],
        available=True,
    ),
    ConnectorDescriptor(
        name="outlook",
        title="Outlook",
        icon="◎",
        blurb="Search, summarize, draft, and send Microsoft 365 email.",
        auth="oauth",
        two_way=False,
        fields=[
            Field(
                "access_token",
                "OAuth access token",
                secret=True,
                help="Microsoft Graph access token.",
            ),
        ],
        instructions=[
            "Use a Microsoft Graph OAuth access token with Mail and Calendar scopes.",
            "Paste the access token below. Managed sign-in will replace this manual step later.",
        ],
        available=True,
    ),
    ConnectorDescriptor(
        name="jira",
        title="Jira",
        icon="◆",
        blurb="Search, summarize, create, and update issues.",
        auth="api_token",
        two_way=False,
        fields=[
            Field(
                "base_url",
                "Atlassian site URL",
                secret=False,
                help="Example: https://example.atlassian.net",
            ),
            Field("email", "Account email", secret=False),
            Field("api_token", "API token", secret=True, help="Atlassian API token."),
        ],
        instructions=[
            "Create an Atlassian API token for your account.",
            "Paste your site URL, account email, and API token below.",
        ],
        available=True,
    ),
    ConnectorDescriptor(
        name="confluence",
        title="Confluence",
        icon="◫",
        blurb="Search spaces, read pages, and draft documentation.",
        auth="api_token",
        two_way=False,
        fields=[
            Field(
                "base_url",
                "Atlassian site URL",
                secret=False,
                help="Example: https://example.atlassian.net",
            ),
            Field("email", "Account email", secret=False),
            Field("api_token", "API token", secret=True, help="Atlassian API token."),
        ],
        instructions=[
            "Create an Atlassian API token for your account.",
            "Paste your site URL, account email, and API token below.",
        ],
        available=True,
    ),
    ConnectorDescriptor(
        name="zendesk",
        title="Zendesk",
        icon="◇",
        blurb="Search tickets, summarize customer context, and draft replies.",
        auth="api_token",
        two_way=False,
        fields=[
            Field(
                "subdomain",
                "Zendesk subdomain",
                secret=False,
                help="For example, 'acme' for acme.zendesk.com.",
            ),
            Field("email", "Agent email", secret=False),
            Field("api_token", "API token", secret=True),
        ],
        instructions=[
            "Create a Zendesk API token.",
            "Paste your subdomain, agent email, and API token below.",
        ],
        available=True,
    ),
]

_BY_NAME = {d.name: d for d in DESCRIPTORS}


def list_descriptors() -> list[ConnectorDescriptor]:
    return list(DESCRIPTORS)


def get_descriptor(name: str) -> Optional[ConnectorDescriptor]:
    return _BY_NAME.get(name)
