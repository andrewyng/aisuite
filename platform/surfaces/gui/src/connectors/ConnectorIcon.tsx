// <ConnectorIcon> / <ConnectorBadge> — render a connector's logo (from the registry, keyed by
// `logo` id) tinted with its `brand_color`. The brand color comes from the API data via the prop,
// so the descriptor is the single source of truth (no hardcoded per-connector CSS table). An
// unknown / empty `logo` id falls back to a neutral plug glyph while keeping the provided color.
//
// Mirrors the mock's `data-brand` → `var(--brand)` tinting, but `--brand` (and the derived
// `--brand-soft` used for the badge tint) are set inline from the prop rather than from CSS.

import type { CSSProperties } from "react";
import { resolveConnector } from "./registry";

export const NEUTRAL = "#6b7280"; // fallback gray, matches the descriptor default

/** Anything carrying the API connector visuals — the full `Connector` is structurally assignable. */
export interface ConnectorVisual {
  logo?: string;
  brand_color?: string;
  title?: string;
  label?: string;
}

/** Convert a `#rgb` / `#rrggbb` hex to an `rgba()` string at `alpha`; '' on malformed input. */
export function hexToRgba(hex: string, alpha: number): string {
  let h = hex.trim().replace(/^#/, "");
  if (h.length === 3) h = h.split("").map((c) => c + c).join("");
  if (!/^[0-9a-fA-F]{6}$/.test(h)) return "";
  const r = parseInt(h.slice(0, 2), 16);
  const g = parseInt(h.slice(2, 4), 16);
  const b = parseInt(h.slice(4, 6), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

/** Inline brand custom props: `--brand` (the logo color) + `--brand-soft` (a faint tinted bg). */
function brandVars(color: string): CSSProperties {
  const soft = hexToRgba(color, 0.12) || "var(--line)";
  return { ["--brand"]: color, ["--brand-soft"]: soft, color } as CSSProperties;
}

function visualColor(connector: ConnectorVisual): string {
  return (connector.brand_color || "").trim() || NEUTRAL;
}

function visualLabel(connector: ConnectorVisual, fallbackLabel: string, title?: string): string {
  return title ?? connector.label ?? connector.title ?? fallbackLabel;
}

/** Bare logo glyph, brand-tinted. */
export function ConnectorIcon({
  connector,
  size = 18,
  title,
}: {
  connector: ConnectorVisual;
  size?: number;
  title?: string;
}) {
  const color = visualColor(connector);
  const { key, entry } = resolveConnector(connector.logo);
  const Logo = entry.logo;
  const label = visualLabel(connector, entry.label, title);
  return (
    <span
      className="connector-icon"
      data-logo={key}
      data-brand={key}
      role="img"
      aria-label={label}
      title={label}
      style={{ ...brandVars(color), width: size, height: size }}
    >
      <Logo />
    </span>
  );
}

/** Rounded-square avatar: brand-soft background + brand-tinted glyph (mock's `.src-badge`). */
export function ConnectorBadge({
  connector,
  size = 32,
  title,
}: {
  connector: ConnectorVisual;
  size?: number;
  title?: string;
}) {
  const color = visualColor(connector);
  const { key, entry } = resolveConnector(connector.logo);
  const label = visualLabel(connector, entry.label, title);
  return (
    <span
      className="connector-badge"
      data-logo={key}
      data-brand={key}
      role="img"
      aria-label={label}
      title={label}
      style={{ ...brandVars(color), width: size, height: size }}
    >
      <ConnectorIcon connector={connector} size={Math.round(size * 0.56)} title={label} />
    </span>
  );
}
