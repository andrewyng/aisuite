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
    # Experimental connectors are hidden unless the user enables them in settings, require an
    # explicit risk acknowledgment to connect, and ship in a separate package
    # (connectors/experimental/) that release builds exclude entirely.
    experimental: bool = False
    risk_notice: str = ""


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


def _validate_whoami(
    method: str,
    url: str,
    *,
    headers: dict,
    identity: Callable[[dict], str],
    json: Optional[dict] = None,
) -> ValidationResult:
    """Shared one-shot whoami check: 2xx + extractable identity, else a failure."""
    import httpx

    try:
        resp = httpx.request(method, url, headers=headers, json=json, timeout=15)
        data = resp.json()
    except Exception as exc:
        return ValidationResult(False, error=str(exc))
    if resp.status_code >= 400:
        detail = (
            (data.get("message") or data.get("error") or data.get("error_summary"))
            if isinstance(data, dict)
            else None
        )
        return ValidationResult(False, error=str(detail or f"HTTP {resp.status_code}"))
    try:
        return ValidationResult(True, identity=str(identity(data)))
    except Exception:
        return ValidationResult(False, error="unexpected response from API")


def _validate_linear(creds: dict) -> ValidationResult:
    return _validate_whoami(
        "POST",
        "https://api.linear.app/graphql",
        headers={
            "Authorization": creds.get("api_key", ""),
            "Content-Type": "application/json",
        },
        json={"query": "{ viewer { name } }"},
        identity=lambda d: d["data"]["viewer"]["name"],
    )


def _validate_gitlab(creds: dict) -> ValidationResult:
    base = str(creds.get("base_url") or "https://gitlab.com").rstrip("/")
    return _validate_whoami(
        "GET",
        f"{base}/api/v4/user",
        headers={"PRIVATE-TOKEN": creds.get("token", "")},
        identity=lambda d: "@" + d["username"],
    )


def _validate_discord(creds: dict) -> ValidationResult:
    return _validate_whoami(
        "GET",
        "https://discord.com/api/v10/users/@me",
        headers={"Authorization": f"Bot {creds.get('bot_token', '')}"},
        identity=lambda d: d["username"],
    )


def _validate_asana(creds: dict) -> ValidationResult:
    return _validate_whoami(
        "GET",
        "https://app.asana.com/api/1.0/users/me",
        headers={"Authorization": f"Bearer {creds.get('token', '')}"},
        identity=lambda d: d["data"]["name"],
    )


def _validate_hubspot(creds: dict) -> ValidationResult:
    return _validate_whoami(
        "GET",
        "https://api.hubapi.com/account-info/v3/details",
        headers={"Authorization": f"Bearer {creds.get('token', '')}"},
        identity=lambda d: f"portal {d['portalId']}",
    )


def _validate_dropbox(creds: dict) -> ValidationResult:
    return _validate_whoami(
        "POST",
        "https://api.dropboxapi.com/2/users/get_current_account",
        headers={"Authorization": f"Bearer {creds.get('access_token', '')}"},
        identity=lambda d: d["email"],
    )


def _quickbooks_host(creds: dict) -> str:
    env = str(creds.get("environment", "")).lower()
    return (
        "sandbox-quickbooks.api.intuit.com"
        if env.startswith("sand")
        else "quickbooks.api.intuit.com"
    )


def _validate_quickbooks(creds: dict) -> ValidationResult:
    realm = creds.get("realm_id", "")
    return _validate_whoami(
        "GET",
        f"https://{_quickbooks_host(creds)}/v3/company/{realm}/companyinfo/{realm}",
        headers={
            "Authorization": f"Bearer {creds.get('access_token', '')}",
            "Accept": "application/json",
        },
        identity=lambda d: d["CompanyInfo"]["CompanyName"],
    )


def _validate_box(creds: dict) -> ValidationResult:
    return _validate_whoami(
        "GET",
        "https://api.box.com/2.0/users/me",
        headers={"Authorization": f"Bearer {creds.get('access_token', '')}"},
        identity=lambda d: d["login"],
    )


def _validate_whatsapp(creds: dict) -> ValidationResult:
    return _validate_whoami(
        "GET",
        f"https://graph.facebook.com/v21.0/{creds.get('phone_number_id', '')}",
        headers={"Authorization": f"Bearer {creds.get('access_token', '')}"},
        identity=lambda d: d["display_phone_number"],
    )


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
    ConnectorDescriptor(
        name="linear",
        title="Linear",
        icon="⟋",
        blurb="Search, read, and create Linear issues.",
        auth="api_token",
        two_way=False,
        fields=[
            Field(
                "api_key",
                "API key",
                secret=True,
                help="Personal API key from Linear settings.",
                placeholder="lin_api_…",
            ),
        ],
        instructions=[
            "In Linear, open Settings → Security & access → Personal API keys.",
            "Create a key and paste it below.",
        ],
        validate=_validate_linear,
    ),
    ConnectorDescriptor(
        name="gitlab",
        title="GitLab",
        icon="▲",
        blurb="Work with issues and merge requests on GitLab.com or self-hosted.",
        auth="token",
        two_way=False,
        fields=[
            Field(
                "base_url",
                "GitLab URL",
                required=False,
                help="Leave empty for gitlab.com.",
                placeholder="https://gitlab.example.com",
            ),
            Field(
                "token",
                "Personal access token",
                secret=True,
                help="Token with read_api scope (api for write actions).",
                placeholder="glpat-…",
            ),
        ],
        instructions=[
            "Create a GitLab personal access token with the read_api scope (api for write actions).",
            "For self-hosted GitLab, enter your instance URL; leave empty for gitlab.com.",
        ],
        validate=_validate_gitlab,
    ),
    ConnectorDescriptor(
        name="discord",
        title="Discord",
        icon="✦",
        blurb="Read channels and send messages through a Discord bot.",
        auth="bot_token",
        two_way=False,
        fields=[
            Field(
                "bot_token",
                "Bot token",
                secret=True,
                help="From the Bot tab of your Discord application.",
            ),
        ],
        instructions=[
            "Go to discord.com/developers/applications → New Application → Bot.",
            "Copy the bot token and paste it below.",
            "Use the OAuth2 URL generator to invite the bot to your server with Read/Send Messages permissions.",
        ],
        validate=_validate_discord,
    ),
    ConnectorDescriptor(
        name="stripe",
        title="Stripe",
        icon="≋",
        blurb="Read-only access to customers, charges, and invoices.",
        auth="api_token",
        two_way=False,
        fields=[
            Field(
                "api_key",
                "Restricted API key",
                secret=True,
                help="Read-only restricted key recommended.",
                placeholder="rk_live_…",
            ),
        ],
        instructions=[
            "In the Stripe Dashboard, create a restricted API key with read access to Customers, Charges, and Invoices.",
            "Paste the key below. The connector only exposes read tools.",
        ],
    ),
    ConnectorDescriptor(
        name="asana",
        title="Asana",
        icon="⊙",
        blurb="Search tasks, read details, and create tasks.",
        auth="token",
        two_way=False,
        fields=[
            Field(
                "token",
                "Personal access token",
                secret=True,
                help="From the Asana developer console.",
            ),
        ],
        instructions=[
            "In Asana, open My Settings → Apps → Manage developer apps.",
            "Create a personal access token and paste it below.",
        ],
        validate=_validate_asana,
    ),
    ConnectorDescriptor(
        name="hubspot",
        title="HubSpot",
        icon="⊚",
        blurb="Search CRM contacts, companies, and deals; create contacts.",
        auth="token",
        two_way=False,
        fields=[
            Field(
                "token",
                "Private app token",
                secret=True,
                help="Access token of a HubSpot private app.",
                placeholder="pat-…",
            ),
        ],
        instructions=[
            "In HubSpot, go to Settings → Integrations → Private Apps and create an app.",
            "Grant CRM object read scopes (and crm.objects.contacts.write to create contacts).",
            "Copy the access token and paste it below.",
        ],
        validate=_validate_hubspot,
    ),
    ConnectorDescriptor(
        name="dropbox",
        title="Dropbox",
        icon="▣",
        blurb="Search, browse, and read files in Dropbox.",
        auth="oauth",
        two_way=False,
        fields=[
            Field(
                "access_token",
                "OAuth access token",
                secret=True,
                help="Dropbox token with files.metadata.read and files.content.read scopes.",
            ),
        ],
        instructions=[
            "Create an app in the Dropbox App Console with files.metadata.read and files.content.read scopes.",
            "Generate an access token and paste it below. Managed sign-in will replace this manual step later.",
        ],
        validate=_validate_dropbox,
    ),
    ConnectorDescriptor(
        name="box",
        title="Box",
        icon="▢",
        blurb="Search, browse, and read files in Box.",
        auth="oauth",
        two_way=False,
        fields=[
            Field(
                "access_token",
                "OAuth access token",
                secret=True,
                help="Box developer token or OAuth access token.",
            ),
        ],
        instructions=[
            "Create a Box app at app.box.com/developers/console.",
            "Generate a developer token (or OAuth access token) and paste it below. Managed sign-in will replace this manual step later.",
        ],
        validate=_validate_box,
    ),
    ConnectorDescriptor(
        name="whatsapp",
        title="WhatsApp",
        icon="◌",
        blurb="Send WhatsApp messages through Meta's official Cloud API (outbound only).",
        auth="token",
        two_way=False,
        fields=[
            Field(
                "access_token",
                "Access token",
                secret=True,
                help="From your Meta app's WhatsApp setup page (a system-user token for long-lived access).",
            ),
            Field(
                "phone_number_id",
                "Phone number ID",
                help="The Cloud API phone number ID (not the phone number itself).",
            ),
        ],
        instructions=[
            "Create a Meta app at developers.facebook.com and add the WhatsApp product.",
            "Copy the access token and the phone number ID from the API setup page.",
            "The free test number can message up to 5 verified recipients without business verification.",
            "Free-form messages only reach people who messaged your number in the last 24 hours; outside that window only approved templates are delivered.",
        ],
        validate=_validate_whatsapp,
    ),
    ConnectorDescriptor(
        name="quickbooks",
        title="QuickBooks",
        icon="◴",
        blurb="Read-only access to customers, invoices, and financial reports.",
        auth="oauth",
        two_way=False,
        fields=[
            Field(
                "access_token",
                "OAuth access token",
                secret=True,
                help="Intuit OAuth token with the com.intuit.quickbooks.accounting scope. Expires hourly.",
            ),
            Field(
                "realm_id",
                "Company ID (realm ID)",
                help="Shown during OAuth authorization and in the developer playground.",
            ),
            Field(
                "environment",
                "Environment",
                required=False,
                help="production (default) or sandbox.",
                placeholder="production",
            ),
        ],
        instructions=[
            "Create an app at developer.intuit.com and authorize it against your company (the OAuth playground works for testing).",
            "Copy the access token and the company ID (realm ID) and paste them below.",
            "Intuit access tokens expire after about an hour. Managed sign-in will replace this manual step later.",
        ],
        validate=_validate_quickbooks,
    ),
]

_BY_NAME = {d.name: d for d in DESCRIPTORS}


def register_descriptor(descriptor: ConnectorDescriptor) -> None:
    """Register an extra connector (used by the experimental package and tests)."""
    DESCRIPTORS.append(descriptor)
    _BY_NAME[descriptor.name] = descriptor


# Experimental connectors live in a separate package so release builds can exclude the code
# entirely (see packaging/coworker-server.spec). When the package is absent this is a no-op.
try:
    from .experimental import EXPERIMENTAL_DESCRIPTORS as _EXPERIMENTAL
except ImportError:
    _EXPERIMENTAL = []
for _exp in _EXPERIMENTAL:
    _exp.experimental = True  # enforced here, not trusted from the author
    register_descriptor(_exp)


def list_descriptors() -> list[ConnectorDescriptor]:
    return list(DESCRIPTORS)


def get_descriptor(name: str) -> Optional[ConnectorDescriptor]:
    return _BY_NAME.get(name)
