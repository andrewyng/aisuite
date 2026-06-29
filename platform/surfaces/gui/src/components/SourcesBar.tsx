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
    <div className="sources-row">
      <button
        className="sources-bar"
        onClick={() => setOpen(true)}
        title="Manage this session's connections"
      >
        <span className="sources-label">Sources</span>
        {connected.length > 0 && (
          <span className="src-stack">
            {connected.map((c) => (
              <span
                className="src-ava"
                key={c.connector}
                title={`${labelFor(c.connector, byName)}${c.detail ? ` · ${c.detail}` : ""}`}
              >
                <ConnectorIcon connector={visualFor(c.connector, "connector", byName)} size={14} />
              </span>
            ))}
          </span>
        )}
        {attention > 0 && (
          <span
            className="attn-badge"
            title={`${attention} recommended connection${attention === 1 ? "" : "s"} not yet connected`}
          >
            ⚠ {attention}
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
