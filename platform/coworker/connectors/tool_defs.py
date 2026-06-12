"""Connector tool catalog and local enablement policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..secrets import SecretStore


@dataclass(frozen=True)
class ConnectorToolDef:
    connector: str
    name: str
    label: str
    kind: str
    description: str
    default_enabled: bool = True


TOOL_DEFS: tuple[ConnectorToolDef, ...] = (
    ConnectorToolDef(
        "browser",
        "browser_read_url",
        "Read public URL",
        "read",
        "Fetch readable text from a public URL.",
    ),
    ConnectorToolDef(
        "browser",
        "browser_open_url",
        "Open URL",
        "read",
        "Open a URL in the Playwright browser.",
    ),
    ConnectorToolDef(
        "browser",
        "browser_snapshot",
        "Snapshot page",
        "read",
        "Read page text and visible controls.",
    ),
    ConnectorToolDef(
        "browser",
        "browser_get_text",
        "Read page text",
        "read",
        "Read visible text from the current browser page.",
    ),
    ConnectorToolDef(
        "browser",
        "browser_click",
        "Click page",
        "write",
        "Click a visible browser element.",
    ),
    ConnectorToolDef(
        "browser",
        "browser_type",
        "Fill field",
        "write",
        "Type into or fill a browser field.",
    ),
    ConnectorToolDef(
        "browser",
        "browser_select",
        "Select option",
        "write",
        "Select a dropdown option.",
    ),
    ConnectorToolDef(
        "browser",
        "browser_upload_file",
        "Upload file",
        "write",
        "Upload a local file through a file input.",
    ),
    ConnectorToolDef(
        "browser", "browser_wait", "Wait", "read", "Wait for time or an element."
    ),
    ConnectorToolDef(
        "browser",
        "browser_screenshot",
        "Screenshot",
        "read",
        "Capture a browser screenshot.",
    ),
    ConnectorToolDef(
        "browser",
        "browser_close",
        "Close browser",
        "write",
        "Close the browser session.",
    ),
    ConnectorToolDef(
        "github",
        "github_search",
        "Search GitHub",
        "read",
        "Search issues, pull requests, or repositories.",
    ),
    ConnectorToolDef(
        "github",
        "github_get_issue",
        "Read issue",
        "read",
        "Read a GitHub issue or pull request.",
    ),
    ConnectorToolDef(
        "github",
        "github_create_issue",
        "Create issue",
        "write",
        "Create a GitHub issue.",
    ),
    ConnectorToolDef(
        "notion",
        "notion_search",
        "Search Notion",
        "read",
        "Search Notion pages and databases.",
    ),
    ConnectorToolDef(
        "notion", "notion_get_page", "Read page", "read", "Read a Notion page."
    ),
    ConnectorToolDef(
        "notion",
        "notion_create_page",
        "Create page",
        "write",
        "Create a Notion child page.",
    ),
    ConnectorToolDef(
        "email",
        "email_list_folders",
        "List folders",
        "read",
        "List mailbox folders and message counts.",
    ),
    ConnectorToolDef(
        "email",
        "email_search",
        "Search mail",
        "read",
        "Search the mailbox; returns envelopes, never marks messages read.",
    ),
    ConnectorToolDef(
        "email",
        "email_read",
        "Read message",
        "read",
        "Read one email's headers, body, and attachment list.",
    ),
    ConnectorToolDef(
        "email",
        "email_download_attachment",
        "Save attachment",
        "write",
        "Save one attachment into the session folder (requires approval).",
    ),
    ConnectorToolDef(
        "email",
        "email_send",
        "Send email",
        "write",
        "Send or reply to an email via SMTP (requires approval).",
    ),
    ConnectorToolDef(
        "gmail",
        "gmail_search_messages",
        "Search Gmail",
        "read",
        "Search Gmail messages.",
    ),
    ConnectorToolDef(
        "gmail", "gmail_get_message", "Read message", "read", "Read a Gmail message."
    ),
    ConnectorToolDef(
        "gmail",
        "gmail_send_email",
        "Send email",
        "write",
        "Send an email through Gmail.",
    ),
    ConnectorToolDef(
        "google_calendar",
        "gcal_list_events",
        "List events",
        "read",
        "List Google Calendar events.",
    ),
    ConnectorToolDef(
        "google_calendar",
        "gcal_create_event",
        "Create event",
        "write",
        "Create a Google Calendar event.",
    ),
    ConnectorToolDef(
        "outlook",
        "outlook_search_messages",
        "Search Outlook",
        "read",
        "Search Outlook messages.",
    ),
    ConnectorToolDef(
        "outlook",
        "outlook_send_mail",
        "Send mail",
        "write",
        "Send mail through Outlook.",
    ),
    ConnectorToolDef(
        "outlook",
        "outlook_list_events",
        "List events",
        "read",
        "List Outlook calendar events.",
    ),
    ConnectorToolDef(
        "outlook",
        "outlook_create_event",
        "Create event",
        "write",
        "Create an Outlook calendar event.",
    ),
    ConnectorToolDef(
        "jira", "jira_search_issues", "Search issues", "read", "Search Jira issues."
    ),
    ConnectorToolDef(
        "jira", "jira_get_issue", "Read issue", "read", "Read a Jira issue."
    ),
    ConnectorToolDef(
        "jira", "jira_create_issue", "Create issue", "write", "Create a Jira issue."
    ),
    ConnectorToolDef(
        "confluence",
        "confluence_search",
        "Search pages",
        "read",
        "Search Confluence pages.",
    ),
    ConnectorToolDef(
        "confluence",
        "confluence_get_page",
        "Read page",
        "read",
        "Read a Confluence page.",
    ),
    ConnectorToolDef(
        "confluence",
        "confluence_create_page",
        "Create page",
        "write",
        "Create a Confluence page.",
    ),
    ConnectorToolDef(
        "zendesk", "zendesk_search", "Search Zendesk", "read", "Search Zendesk."
    ),
    ConnectorToolDef(
        "zendesk", "zendesk_get_ticket", "Read ticket", "read", "Read a Zendesk ticket."
    ),
    ConnectorToolDef(
        "zendesk",
        "zendesk_create_ticket",
        "Create ticket",
        "write",
        "Create a Zendesk ticket.",
    ),
    ConnectorToolDef(
        "linear",
        "linear_search_issues",
        "Search issues",
        "read",
        "Search Linear issues.",
    ),
    ConnectorToolDef(
        "linear", "linear_get_issue", "Read issue", "read", "Read a Linear issue."
    ),
    ConnectorToolDef(
        "linear", "linear_list_teams", "List teams", "read", "List Linear teams."
    ),
    ConnectorToolDef(
        "linear",
        "linear_create_issue",
        "Create issue",
        "write",
        "Create a Linear issue.",
    ),
    ConnectorToolDef(
        "gitlab",
        "gitlab_search",
        "Search GitLab",
        "read",
        "Search projects, issues, or merge requests.",
    ),
    ConnectorToolDef(
        "gitlab", "gitlab_get_issue", "Read issue", "read", "Read a GitLab issue."
    ),
    ConnectorToolDef(
        "gitlab",
        "gitlab_get_merge_request",
        "Read merge request",
        "read",
        "Read a GitLab merge request.",
    ),
    ConnectorToolDef(
        "gitlab",
        "gitlab_create_issue",
        "Create issue",
        "write",
        "Create a GitLab issue.",
    ),
    ConnectorToolDef(
        "discord",
        "discord_list_channels",
        "List channels",
        "read",
        "List channels in a Discord server.",
    ),
    ConnectorToolDef(
        "discord",
        "discord_read_messages",
        "Read messages",
        "read",
        "Read recent Discord channel messages.",
    ),
    ConnectorToolDef(
        "discord",
        "discord_send_message",
        "Send message",
        "write",
        "Send a Discord channel message.",
    ),
    ConnectorToolDef(
        "stripe",
        "stripe_search_customers",
        "Search customers",
        "read",
        "Search Stripe customers.",
    ),
    ConnectorToolDef(
        "stripe",
        "stripe_list_charges",
        "List charges",
        "read",
        "List Stripe charges.",
    ),
    ConnectorToolDef(
        "stripe",
        "stripe_list_invoices",
        "List invoices",
        "read",
        "List Stripe invoices.",
    ),
    ConnectorToolDef(
        "asana",
        "asana_list_workspaces",
        "List workspaces",
        "read",
        "List Asana workspaces.",
    ),
    ConnectorToolDef(
        "asana", "asana_search_tasks", "Search tasks", "read", "Search Asana tasks."
    ),
    ConnectorToolDef(
        "asana", "asana_get_task", "Read task", "read", "Read an Asana task."
    ),
    ConnectorToolDef(
        "asana", "asana_create_task", "Create task", "write", "Create an Asana task."
    ),
    ConnectorToolDef(
        "hubspot",
        "hubspot_search",
        "Search CRM",
        "read",
        "Search HubSpot contacts, companies, deals, or tickets.",
    ),
    ConnectorToolDef(
        "hubspot",
        "hubspot_get_object",
        "Read record",
        "read",
        "Read a HubSpot CRM record.",
    ),
    ConnectorToolDef(
        "hubspot",
        "hubspot_create_contact",
        "Create contact",
        "write",
        "Create a HubSpot contact.",
    ),
    ConnectorToolDef(
        "dropbox", "dropbox_search", "Search files", "read", "Search Dropbox files."
    ),
    ConnectorToolDef(
        "dropbox",
        "dropbox_list_folder",
        "List folder",
        "read",
        "List a Dropbox folder.",
    ),
    ConnectorToolDef(
        "dropbox",
        "dropbox_read_file",
        "Read file",
        "read",
        "Read a text file from Dropbox.",
    ),
    ConnectorToolDef("box", "box_search", "Search files", "read", "Search Box files."),
    ConnectorToolDef(
        "box", "box_list_folder", "List folder", "read", "List a Box folder."
    ),
    ConnectorToolDef(
        "box", "box_read_file", "Read file", "read", "Read a text file from Box."
    ),
    ConnectorToolDef(
        "quickbooks",
        "quickbooks_query",
        "Query records",
        "read",
        "Run a QuickBooks Online query.",
    ),
    ConnectorToolDef(
        "quickbooks",
        "quickbooks_list_customers",
        "List customers",
        "read",
        "List QuickBooks customers.",
    ),
    ConnectorToolDef(
        "quickbooks",
        "quickbooks_list_invoices",
        "List invoices",
        "read",
        "List recent QuickBooks invoices.",
    ),
    ConnectorToolDef(
        "quickbooks",
        "quickbooks_get_report",
        "Run report",
        "read",
        "Run a QuickBooks financial report.",
    ),
    ConnectorToolDef(
        "whatsapp",
        "whatsapp_send_message",
        "Send message",
        "write",
        "Send a WhatsApp text message.",
    ),
    ConnectorToolDef(
        "whatsapp",
        "whatsapp_send_template",
        "Send template",
        "write",
        "Send an approved WhatsApp template message.",
    ),
)

TOOL_TO_CONNECTOR = {d.name: d.connector for d in TOOL_DEFS}
TOOLS_BY_CONNECTOR: dict[str, list[ConnectorToolDef]] = {}
for _def in TOOL_DEFS:
    TOOLS_BY_CONNECTOR.setdefault(_def.connector, []).append(_def)


def connector_for_tool(tool_name: str) -> str | None:
    return TOOL_TO_CONNECTOR.get(tool_name)


def load_tool_settings(secrets: SecretStore, connector: str) -> dict[str, bool]:
    raw = secrets.get(f"{connector}:tools") or {}
    enabled = raw.get("enabled") if isinstance(raw, dict) else None
    return {str(k): bool(v) for k, v in (enabled or {}).items()}


def tool_enabled(secrets: SecretStore, connector: str, tool_name: str) -> bool:
    overrides = load_tool_settings(secrets, connector)
    if tool_name in overrides:
        return overrides[tool_name]
    for tool in TOOLS_BY_CONNECTOR.get(connector, []):
        if tool.name == tool_name:
            return tool.default_enabled
    return False


def patch_tool_settings(
    secrets: SecretStore, connector: str, enabled: dict[str, Any]
) -> dict[str, Any]:
    known = {t.name for t in TOOLS_BY_CONNECTOR.get(connector, [])}
    if not known:
        return {"ok": False, "error": "unknown connector or no tools"}
    current = load_tool_settings(secrets, connector)
    for name, value in enabled.items():
        if name in known:
            current[name] = bool(value)
    secrets.put(f"{connector}:tools", {"enabled": current})
    return {"ok": True, "tools": current}


def tool_dicts(secrets: SecretStore, connector: str) -> list[dict[str, Any]]:
    overrides = load_tool_settings(secrets, connector)
    out = []
    for tool in TOOLS_BY_CONNECTOR.get(connector, []):
        out.append(
            {
                "name": tool.name,
                "label": tool.label,
                "kind": tool.kind,
                "description": tool.description,
                "enabled": bool(overrides.get(tool.name, tool.default_enabled)),
                "requires_approval": True,
            }
        )
    return out
