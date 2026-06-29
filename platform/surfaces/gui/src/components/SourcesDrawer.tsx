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
import { Toggle } from "./Toggle";
import { labelFor, visualFor, type ConnectorMap } from "../connectors/visuals";

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
    <div className="sources-drawer-overlay">
      <div className="sources-drawer-scrim" onClick={onClose} />
      <aside className="sources-drawer" role="dialog" aria-label="Session connections">
        <header className="drawer-head">
          <span className="drawer-head-icon">{persona?.icon || "🔌"}</span>
          <div className="drawer-head-text">
            <div className="drawer-head-title">{persona?.name || "Sources"}</div>
            <div className="drawer-head-sub">Session connections</div>
          </div>
          <button className="drawer-close" onClick={onClose} aria-label="Close">
            ✕
          </button>
        </header>

        <div className="drawer-scroll">
          {/* persona why-connect blurb (when a persona is known) */}
          {persona && (
            <div className="drawer-blurb">
              <div className="drawer-blurb-h">⚡ Get the most out of {persona.name}</div>
              {persona.description && <p>{persona.description}</p>}
              {total > 0 && (
                <>
                  <div className="drawer-progress">
                    <div style={{ width: `${pct}%` }} />
                  </div>
                  <div className="drawer-progress-label">
                    {got} of {total} recommended connected
                  </div>
                </>
              )}
              {onOpenPersona && (
                <button className="drawer-about-link" onClick={() => onOpenPersona(persona.id)}>
                  About this persona →
                </button>
              )}
            </div>
          )}

          {/* connected — each toggle is a per-session override (mute for this session only) */}
          <section>
            <div className="drawer-sec-h">
              <span>Connected · {connected.length}</span>
              <span className="drawer-sec-sub">enabled for this session</span>
            </div>
            <div className="drawer-list">
              {connected.length === 0 && (
                <div className="drawer-empty">No connectors enabled for this session.</div>
              )}
              {connected.map((c) => (
                <div className="drawer-conn-row" key={c.connector}>
                  <ConnectorBadge connector={visualFor(c.connector, "connector", byName)} size={32} />
                  <div className="drawer-conn-main">
                    <div className="drawer-conn-name">{labelFor(c.connector, byName)}</div>
                    {c.detail && <div className="drawer-conn-detail">{c.detail}</div>}
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
              <p className="drawer-foot-note">
                Turning one off mutes it for <b>this session only</b> — the connector stays connected
                and enabled for the persona.
              </p>
            )}
          </section>

          {/* recommended connectors (not yet connected) */}
          {recommended.length > 0 && (
            <section>
              <div className="drawer-sec-h">
                <span>Recommended connectors</span>
              </div>
              <div className="drawer-list">
                {recommended.map((r) => (
                  <div className="drawer-reco-row" key={r.connector}>
                    <ConnectorBadge connector={visualFor(r.connector, "connector", byName)} size={32} />
                    <div className="drawer-conn-main">
                      <div className="drawer-conn-name">
                        {labelFor(r.connector, byName)}
                        {r.tier === "core" && <span className="tag-core">core</span>}
                      </div>
                      <div className="drawer-conn-detail">{r.reason}</div>
                    </div>
                    <button
                      className={"btn sm" + (r.tier === "core" ? " reco-connect-primary" : "")}
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

        <footer className="drawer-foot">
          <button
            className="drawer-global-link"
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
