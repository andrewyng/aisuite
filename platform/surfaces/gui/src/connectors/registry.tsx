// Connector logo registry — maps a stable `logo` id (from a connector's API descriptor) to a
// small inline monochrome SVG that inherits `currentColor`, plus a human label. The brand color
// is deliberately NOT stored here: it comes from the API (`brand_color`) so the descriptor stays
// the single source of truth. Unknown / empty ids resolve to FALLBACK (a neutral plug glyph).
//
// The marks are intentionally simple, recognizable monochrome glyphs (not pixel-perfect brand
// logos): they paint with `currentColor`, so the surrounding ConnectorIcon / ConnectorBadge tints
// them with the connector's brand color. (Filename is `.tsx` because the entries are JSX — the
// spec's `registry.ts` can't hold JSX.)

// `JSX` is global with the react-jsx runtime + @types/react.
type LogoComponent = () => JSX.Element;

export interface ConnectorRegistryEntry {
  label: string;
  logo: LogoComponent;
}

const SlackLogo: LogoComponent = () => (
  <svg viewBox="0 0 122.8 122.8" width="100%" height="100%" fill="currentColor" aria-hidden="true">
    <path d="M25.8 77.6c0 7.1-5.8 12.9-12.9 12.9S0 84.7 0 77.6s5.8-12.9 12.9-12.9h12.9v12.9z" />
    <path d="M32.3 77.6c0-7.1 5.8-12.9 12.9-12.9s12.9 5.8 12.9 12.9v32.3c0 7.1-5.8 12.9-12.9 12.9s-12.9-5.8-12.9-12.9V77.6z" />
    <path d="M45.2 25.8c-7.1 0-12.9-5.8-12.9-12.9S38.1 0 45.2 0s12.9 5.8 12.9 12.9v12.9H45.2z" />
    <path d="M45.2 32.3c7.1 0 12.9 5.8 12.9 12.9s-5.8 12.9-12.9 12.9H12.9C5.8 58.1 0 52.3 0 45.2s5.8-12.9 12.9-12.9h32.3z" />
    <path d="M97 45.2c0-7.1 5.8-12.9 12.9-12.9s12.9 5.8 12.9 12.9-5.8 12.9-12.9 12.9H97V45.2z" />
    <path d="M90.5 45.2c0 7.1-5.8 12.9-12.9 12.9s-12.9-5.8-12.9-12.9V12.9C64.7 5.8 70.5 0 77.6 0s12.9 5.8 12.9 12.9v32.3z" />
    <path d="M77.6 97c7.1 0 12.9 5.8 12.9 12.9s-5.8 12.9-12.9 12.9-12.9-5.8-12.9-12.9V97h12.9z" />
    <path d="M77.6 90.5c-7.1 0-12.9-5.8-12.9-12.9s5.8-12.9 12.9-12.9h32.3c7.1 0 12.9 5.8 12.9 12.9s-5.8 12.9-12.9 12.9H77.6z" />
  </svg>
);

const TelegramLogo: LogoComponent = () => (
  <svg viewBox="0 0 24 24" width="100%" height="100%" fill="currentColor" aria-hidden="true">
    <path d="M21 4L3 11l5 2 2 5 3-3 4 3 4-14z" />
  </svg>
);

const GitHubLogo: LogoComponent = () => (
  <svg viewBox="0 0 24 24" width="100%" height="100%" fill="currentColor" aria-hidden="true">
    <path d="M12 1.5A10.5 10.5 0 001.5 12c0 4.6 3 8.6 7.2 10 .5.1.7-.2.7-.5v-1.7c-2.9.6-3.5-1.4-3.5-1.4-.5-1.2-1.2-1.5-1.2-1.5-.9-.6.1-.6.1-.6 1 .1 1.6 1 1.6 1 .9 1.6 2.4 1.1 3 .9.1-.7.4-1.1.7-1.4-2.3-.3-4.8-1.2-4.8-5.2 0-1.1.4-2.1 1-2.8-.1-.3-.5-1.4.1-2.8 0 0 .9-.3 2.8 1.1a9.6 9.6 0 015 0c1.9-1.3 2.8-1.1 2.8-1.1.6 1.4.2 2.5.1 2.8.7.7 1 1.7 1 2.8 0 4-2.5 4.9-4.8 5.2.4.3.7 1 .7 2v3c0 .3.2.6.7.5 4.2-1.4 7.2-5.4 7.2-10A10.5 10.5 0 0012 1.5z" />
  </svg>
);

const DatadogLogo: LogoComponent = () => (
  <svg
    viewBox="0 0 24 24"
    width="100%"
    height="100%"
    fill="none"
    stroke="currentColor"
    strokeWidth={2}
    strokeLinecap="round"
    strokeLinejoin="round"
    aria-hidden="true"
  >
    <path d="M3 12h3l2 6 4-14 2 8 2-3h5" />
  </svg>
);

const SalesforceLogo: LogoComponent = () => (
  <svg viewBox="0 0 24 24" width="100%" height="100%" fill="currentColor" aria-hidden="true">
    <path d="M10 7.5a3.2 3.2 0 015.7-1 3.6 3.6 0 014.8 3.4 3.4 3.4 0 01-1.2 6.6H6.5A3.5 3.5 0 016 9.6 3.2 3.2 0 0110 7.5z" />
  </svg>
);

const HubSpotLogo: LogoComponent = () => (
  <svg
    viewBox="0 0 24 24"
    width="100%"
    height="100%"
    fill="none"
    stroke="currentColor"
    strokeWidth={2}
    strokeLinecap="round"
    strokeLinejoin="round"
    aria-hidden="true"
  >
    <circle cx="8" cy="16" r="2.4" />
    <circle cx="17" cy="9" r="2.4" />
    <path d="M17 11.4V14M9.7 14.6l5.6-3.8M17 6.6V4" />
  </svg>
);

const PagerDutyLogo: LogoComponent = () => (
  <svg
    viewBox="0 0 24 24"
    width="100%"
    height="100%"
    fill="none"
    stroke="currentColor"
    strokeWidth={1.8}
    strokeLinecap="round"
    strokeLinejoin="round"
    aria-hidden="true"
  >
    <path d="M6 9a6 6 0 0 1 12 0c0 5 2 6 2 6H4s2-1 2-6z" />
    <path d="M10 20a2 2 0 0 0 4 0" />
  </svg>
);

const McpLogo: LogoComponent = () => (
  <svg
    viewBox="0 0 24 24"
    width="100%"
    height="100%"
    fill="none"
    stroke="currentColor"
    strokeWidth={1.8}
    strokeLinecap="round"
    strokeLinejoin="round"
    aria-hidden="true"
  >
    <circle cx="12" cy="12" r="2.2" />
    <circle cx="5" cy="6" r="1.8" />
    <circle cx="19" cy="6" r="1.8" />
    <circle cx="12" cy="20" r="1.8" />
    <path d="M10.3 10.6 6.4 7.3M13.7 10.6l3.9-3.3M12 14.2V18.2" />
  </svg>
);

const PlugLogo: LogoComponent = () => (
  <svg
    viewBox="0 0 24 24"
    width="100%"
    height="100%"
    fill="none"
    stroke="currentColor"
    strokeWidth={1.8}
    strokeLinecap="round"
    strokeLinejoin="round"
    aria-hidden="true"
  >
    <path d="M9 7V3M15 7V3M7 7h10v4a5 5 0 0 1-10 0V7zM12 16v5" />
  </svg>
);

/** Neutral fallback for unknown / empty logo ids. */
export const FALLBACK: ConnectorRegistryEntry = { label: "Connector", logo: PlugLogo };

export const CONNECTORS: Record<string, ConnectorRegistryEntry> = {
  slack: { label: "Slack", logo: SlackLogo },
  telegram: { label: "Telegram", logo: TelegramLogo },
  github: { label: "GitHub", logo: GitHubLogo },
  datadog: { label: "Datadog", logo: DatadogLogo },
  salesforce: { label: "Salesforce", logo: SalesforceLogo },
  hubspot: { label: "HubSpot", logo: HubSpotLogo },
  pagerduty: { label: "PagerDuty", logo: PagerDutyLogo },
  mcp: { label: "MCP", logo: McpLogo },
};

/**
 * Resolve a logo id to its registry entry plus the matched key. Unknown / empty ids return the
 * FALLBACK entry with key `"fallback"` (so callers and tests can distinguish a hit from a miss).
 */
export function resolveConnector(logo?: string): { key: string; entry: ConnectorRegistryEntry } {
  const id = (logo ?? "").trim();
  if (id && Object.prototype.hasOwnProperty.call(CONNECTORS, id)) {
    return { key: id, entry: CONNECTORS[id] };
  }
  return { key: "fallback", entry: FALLBACK };
}
