// SourcesBar — the slim per-session connections bar under the session title (§6). An icons-only
// avatar stack (one badge per effective-enabled connector) + a `⚠ N` attention badge (= count of the
// persona's recommended connectors not yet connected). Clicking opens the SourcesDrawer.
//
// Owns the connection data (GET /v1/sessions/{id}/connections) and the connector index (GET
// /v1/connectors, for real brand colors via visualFor) so the drawer's toggles re-read through
// `reload`, keeping the avatar stack + ⚠ count live. Renders nothing when a session has no connected
// connectors and nothing recommended (e.g. a brand-new/unsent session).

import { useCallback, useEffect, useState } from "react";
import { getConnectors, getSessionConnections, type SessionConnections } from "../api";
import { ConnectorIcon } from "../connectors/ConnectorIcon";
import { SourcesDrawer } from "./SourcesDrawer";
import { indexConnectors, labelFor, visualFor, type ConnectorMap } from "../connectors/visuals";

export function SourcesBar({
  sessionId,
  personaId,
  onOpenIntegrations,
  onOpenPersona,
}: {
  sessionId: string;
  personaId?: string;
  onOpenIntegrations?: () => void;
  onOpenPersona?: (id: string) => void;
}) {
  const [conns, setConns] = useState<SessionConnections | null>(null);
  const [byName, setByName] = useState<ConnectorMap>({});
  const [open, setOpen] = useState(false);

  const reload = useCallback(() => {
    getSessionConnections(sessionId)
      .then(setConns)
      .catch(() => setConns(null));
  }, [sessionId]);

  useEffect(() => {
    reload();
  }, [reload]);

  useEffect(() => {
    let live = true;
    getConnectors()
      .then((list) => live && setByName(indexConnectors(list)))
      .catch(() => {});
    return () => {
      live = false;
    };
  }, []);

  const connected = conns?.connected ?? [];
  const attention = conns?.attention ?? 0;

  // Nothing connected and nothing flagged → don't clutter the header.
  if (!conns || (connected.length === 0 && attention === 0)) return null;

  return (
    <div className="px-5 py-2 border-b border-line bg-panel flex items-center">
      <button
        className="sources-bar group inline-flex items-center gap-2.5 -ml-2 px-2 py-1 rounded-lg hover:bg-paper"
        onClick={() => setOpen(true)}
        title="Manage this session's connections"
      >
        <span className="text-[11px] uppercase tracking-[0.06em] text-faint font-semibold">Sources</span>
        {connected.length > 0 && (
          <span className="flex items-center">
            {connected.map((c, i) => (
              <span
                className={
                  "src-ava w-6 h-6 rounded-full bg-panel border border-line grid place-items-center ring-2 ring-paper" +
                  (i > 0 ? " -ml-1.5" : "")
                }
                key={c.connector}
                title={`${labelFor(c.connector, byName)}${c.detail ? ` · ${c.detail}` : ""}`}
              >
                <ConnectorIcon connector={visualFor(c.connector, "connector", byName)} size={14} />
              </span>
            ))}
          </span>
        )}
        {attention > 0 && (
          <span className="inline-flex items-center gap-1.5 text-[11px] text-faint group-hover:text-muted">
            <span
              className="inline-flex items-center gap-0.5 px-1.5 py-0.5 rounded-full bg-warnSoft text-warnInk border border-warnInk font-semibold"
              title={`${attention} recommended connection${attention === 1 ? "" : "s"} not yet connected`}
            >
              ⚠ {attention}
            </span>
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M6 9l6 6 6-6" />
            </svg>
          </span>
        )}
      </button>

      {open && (
        <SourcesDrawer
          sessionId={sessionId}
          personaId={personaId}
          conns={conns}
          byName={byName}
          onReload={reload}
          onClose={() => setOpen(false)}
          onOpenIntegrations={onOpenIntegrations}
          onOpenPersona={onOpenPersona}
        />
      )}
    </div>
  );
}
