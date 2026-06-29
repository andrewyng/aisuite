// SourcesDrawer — the per-session connections panel (§6, mock parity). Opened from the SourcesBar.
// Shows (when a persona is known) a why-connect blurb + "X of Y recommended connected" progress; a
// Connected list whose toggles set per-session overrides (POST /v1/sessions/{id}/connections — mute/
// unmute for THIS session only); a Recommended-connectors list (Connect → deep-link to Integrations);
// and a footer link to the global Integrations surface.
//
// The connection data (`conns`) and the connector index (`byName`, for real brand colors) are owned by
// the SourcesBar and passed down so toggling re-reads via `onReload`, keeping the bar's avatar stack +
// ⚠ count in sync. The persona blurb is fetched here, only when a `personaId` is provided.

import { useEffect, useState } from "react";
import { getPersonaDetail, setSessionConnection, type PersonaDetail, type SessionConnections } from "../api";
import { ConnectorBadge } from "../connectors/ConnectorIcon";
import { PersonaGlyph } from "./personaIcon";
import { Toggle } from "./Toggle";
import { labelFor, visualFor, type ConnectorMap } from "../connectors/visuals";

const SEC_H = "text-[11px] uppercase tracking-[0.05em] text-faint font-semibold";
const TAG_CORE =
  "text-[10px] px-1.5 py-0.5 rounded-full bg-warnSoft/70 text-warnInk border border-warnInk/15";
const BTN_ACCENT = "text-[12px] px-2.5 py-1.5 rounded-lg bg-accent text-white shrink-0";
const BTN_BORDERED =
  "text-[12px] px-2.5 py-1.5 rounded-lg border border-line bg-paper hover:border-lineStrong shrink-0";

export function SourcesDrawer({
  sessionId,
  personaId,
  conns,
  byName,
  onReload,
  onClose,
  onOpenIntegrations,
  onOpenPersona,
}: {
  sessionId: string;
  personaId?: string;
  conns: SessionConnections;
  byName: ConnectorMap;
  onReload: () => void;
  onClose: () => void;
  onOpenIntegrations?: () => void;
  onOpenPersona?: (id: string) => void;
}) {
  const [persona, setPersona] = useState<PersonaDetail | null>(null);

  useEffect(() => {
    if (!personaId) return;
    let live = true;
    getPersonaDetail(personaId)
      .then((d) => live && setPersona(d))
      .catch(() => {});
    return () => {
      live = false;
    };
  }, [personaId]);

  const toggleSession = async (connector: string, next: boolean) => {
    await setSessionConnection(sessionId, connector, next);
    onReload();
  };

  const { connected, recommended } = conns;
  const total = persona?.recommends.length ?? 0;
  const got = persona?.recommends.filter((r) => r.connected).length ?? 0;
  const pct = total > 0 ? Math.round((got / total) * 100) : 0;

  return (
    <div className="fixed inset-0 z-40">
      <div className="absolute inset-0 bg-black/30 backdrop-blur-[1px]" onClick={onClose} />
      <aside
        className="absolute right-0 top-0 h-full w-[420px] max-w-[92vw] bg-panel border-l border-line shadow-2xl flex flex-col"
        role="dialog"
        aria-label="Session connections"
      >
        <header className="px-4 h-12 shrink-0 flex items-center gap-2 border-b border-line">
          <span className="w-6 h-6 rounded-md bg-panel border border-line grid place-items-center text-[13px]">
            <PersonaGlyph icon={persona?.icon} size={13} />
          </span>
          <div className="min-w-0">
            <div className="text-[13px] font-semibold leading-tight truncate">
              {persona?.name || "Sources"}
            </div>
            <div className="text-[11px] text-faint leading-tight">Session connections</div>
          </div>
          <button
            className="ml-auto w-7 h-7 grid place-items-center rounded-md text-faint hover:text-ink hover:bg-paper"
            onClick={onClose}
            aria-label="Close"
          >
            ✕
          </button>
        </header>

        <div className="flex-1 overflow-y-auto hairline-scroll px-4 py-4 space-y-5">
          {/* persona why-connect blurb (when a persona is known) */}
          {persona && (
            <div className="rounded-xl2 border border-accent/25 bg-accentSoft/50 p-3.5">
              <div className="flex items-center gap-2 text-[13px] font-semibold">
                <span>⚡</span> Get the most out of {persona.name}
              </div>
              {persona.description && (
                <p className="text-[12.5px] text-muted mt-1.5 leading-relaxed">{persona.description}</p>
              )}
              {total > 0 && (
                <>
                  <div className="mt-2 h-1.5 rounded-full bg-panel overflow-hidden">
                    <div className="h-full bg-accent" style={{ width: `${pct}%` }} />
                  </div>
                  <div className="mt-1 text-[11px] text-faint">
                    {got} of {total} recommended connected
                  </div>
                </>
              )}
              {onOpenPersona && (
                <button
                  className="mt-2 inline-block text-[12px] font-medium text-accent hover:underline"
                  onClick={() => onOpenPersona(persona.id)}
                >
                  About this persona →
                </button>
              )}
            </div>
          )}

          {/* connected — each toggle is a per-session override (mute for this session only) */}
          <section>
            <div className="flex items-center justify-between mb-2">
              <div className={SEC_H}>Connected · {connected.length}</div>
              <div className="text-[10.5px] text-faint">enabled for this session</div>
            </div>
            <div className="space-y-1.5">
              {connected.length === 0 && (
                <div className="text-[12.5px] text-faint px-0.5 py-1">
                  No connectors enabled for this session.
                </div>
              )}
              {connected.map((c) => (
                <div
                  className="flex items-center gap-3 p-2.5 rounded-xl2 border border-line bg-paper"
                  key={c.connector}
                >
                  <ConnectorBadge connector={visualFor(c.connector, "connector", byName)} size={32} />
                  <div className="min-w-0 flex-1">
                    <div className="text-[13px] font-medium">{labelFor(c.connector, byName)}</div>
                    {c.detail && <div className="text-[12px] text-muted truncate">{c.detail}</div>}
                  </div>
                  <Toggle
                    checked={c.enabled}
                    onChange={(next) => toggleSession(c.connector, next)}
                    title="Enabled for this session — tap to mute here"
                  />
                </div>
              ))}
            </div>
            {connected.length > 0 && (
              <p className="text-[11px] text-faint mt-1.5">
                Turning one off mutes it for <b>this session only</b> — the connector stays connected
                and enabled for the persona.
              </p>
            )}
          </section>

          {/* recommended connectors (not yet connected) */}
          {recommended.length > 0 && (
            <section>
              <div className={`${SEC_H} mb-2`}>Recommended connectors</div>
              <div className="space-y-1.5">
                {recommended.map((r) => (
                  <div
                    className="flex items-start gap-3 p-2.5 rounded-xl2 border border-line"
                    key={r.connector}
                  >
                    <ConnectorBadge connector={visualFor(r.connector, "connector", byName)} size={32} />
                    <div className="min-w-0 flex-1">
                      <div className="flex items-center gap-2">
                        <span className="text-[13px] font-medium">{labelFor(r.connector, byName)}</span>
                        {r.tier === "core" && <span className={TAG_CORE}>core</span>}
                      </div>
                      <div className="text-[12px] text-muted">{r.reason}</div>
                    </div>
                    <button
                      className={r.tier === "core" ? BTN_ACCENT : BTN_BORDERED}
                      onClick={onOpenIntegrations}
                    >
                      Connect
                    </button>
                  </div>
                ))}
              </div>
            </section>
          )}
        </div>

        <footer className="px-4 py-3 border-t border-line">
          <button
            className="w-full text-[12.5px] text-accent font-medium hover:underline text-left"
            onClick={() => {
              onClose();
              onOpenIntegrations?.();
            }}
          >
            Manage all connectors (global) →
          </button>
        </footer>
      </aside>
    </div>
  );
}
