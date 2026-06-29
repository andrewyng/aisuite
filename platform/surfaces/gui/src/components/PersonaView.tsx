// PersonaView — the persona detail page (§5, mock parity). Identity header + Enable toggle, About,
// Built-in capabilities (tools), "Connections for full benefit" (manifest `recommends`, core/optional
// + reason + connect state), "New sessions get by default" (persona-default connection toggles), and a
// defaults footer (recommended models / default mode / workspace).
//
// Data: fetches GET /v1/personas/{id} on mount; also fetches /v1/connectors to thread real brand
// colors (Phase 1's `brand_color`) into the badges via visualFor(). Toggling a default connection
// POSTs /v1/personas/{id}/connections and applies the returned `default_connections` (re-read).
// Enabling/disabling POSTs /v1/personas/{id}/enable.

import { useEffect, useState } from "react";
import {
  getConnectors,
  getPersonaDetail,
  setPersonaConnection,
  setPersonaEnabled,
  type PersonaDetail,
} from "../api";
import { ConnectorBadge } from "../connectors/ConnectorIcon";
import { Icon } from "./Icon";
import { Toggle } from "./Toggle";
import { indexConnectors, labelFor, visualFor, type ConnectorMap } from "../connectors/visuals";

export function PersonaView({
  personaId,
  onBack,
  onOpenIntegrations,
}: {
  personaId: string;
  onBack?: () => void;
  onOpenIntegrations?: () => void;
}) {
  const [detail, setDetail] = useState<PersonaDetail | null>(null);
  const [byName, setByName] = useState<ConnectorMap>({});
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let live = true;
    setDetail(null);
    setError(null);
    getPersonaDetail(personaId)
      .then((d) => live && setDetail(d))
      .catch(() => live && setError("Could not load this persona."));
    getConnectors()
      .then((list) => live && setByName(indexConnectors(list)))
      .catch(() => {});
    return () => {
      live = false;
    };
  }, [personaId]);

  const toggleEnabled = async (next: boolean) => {
    setDetail((d) => (d ? { ...d, enabled: next } : d)); // optimistic
    const r = await setPersonaEnabled(personaId, next);
    if (!r.ok) getPersonaDetail(personaId).then(setDetail).catch(() => {});
  };

  const toggleDefault = async (connector: string, next: boolean) => {
    const r = await setPersonaConnection(personaId, connector, next);
    if (r.default_connections) {
      setDetail((d) => (d ? { ...d, default_connections: r.default_connections! } : d));
    } else {
      getPersonaDetail(personaId).then(setDetail).catch(() => {});
    }
  };

  const header = (
    <div className="persona-view-top">
      {onBack && (
        <button className="persona-back" onClick={onBack}>
          <Icon name="arrowLeft" size={14} /> Back
        </button>
      )}
      <span className="persona-view-crumb">Persona</span>
    </div>
  );

  if (error || !detail) {
    return (
      <main className="persona-view">
        {header}
        <div className="persona-view-empty">{error || "Loading…"}</div>
      </main>
    );
  }

  return (
    <main className="persona-view">
      {header}
      <div className="persona-view-scroll">
        <div className="persona-view-body">
          {/* identity + enable */}
          <header className="persona-id">
            <span className="persona-id-icon">{detail.icon || "🛠"}</span>
            <div className="persona-id-text">
              <h1>{detail.name}</h1>
              <p>{detail.tagline}</p>
            </div>
            <div className="persona-id-enable">
              <span>{detail.enabled ? "Enabled" : "Disabled"}</span>
              <Toggle checked={detail.enabled} onChange={toggleEnabled} title="Enable this persona" />
            </div>
          </header>

          {/* about */}
          {detail.description && (
            <section>
              <div className="persona-sec-h">About</div>
              <p className="persona-about">{detail.description}</p>
            </section>
          )}

          {/* tools */}
          {detail.tools.length > 0 && (
            <section>
              <div className="persona-sec-h">Built-in capabilities</div>
              <div className="persona-tools">
                {detail.tools.map((t) => (
                  <span className="persona-tool" key={t}>
                    {t}
                  </span>
                ))}
              </div>
            </section>
          )}

          {/* connections for full benefit (manifest recommends) */}
          {detail.recommends.length > 0 && (
            <section>
              <div className="persona-sec-h">Connections for full benefit</div>
              <p className="persona-sec-note">
                Declared by the persona — wire {detail.name} into these to unlock its full workflow.
              </p>
              <div className="persona-reco">
                {detail.recommends.map((r) => {
                  const isMcp = r.kind === "mcp";
                  return (
                    <div className="persona-reco-row" key={`${r.kind}:${r.ref}`}>
                      <ConnectorBadge connector={visualFor(r.ref, r.kind, byName)} size={32} />
                      <div className="persona-reco-main">
                        <div className="persona-reco-name">
                          <span>{labelFor(r.ref, byName)}</span>
                          {isMcp ? (
                            <span className="tag-mcp">MCP</span>
                          ) : r.tier === "core" ? (
                            <span className="tag-core">core</span>
                          ) : null}
                        </div>
                        <div className="persona-reco-reason">{r.reason}</div>
                      </div>
                      {r.connected ? (
                        <span className="reco-connected">
                          <span className="ok-dot" />
                          connected
                        </span>
                      ) : (
                        <button
                          className={"btn sm" + (r.tier === "core" && !isMcp ? " reco-connect-primary" : "")}
                          onClick={onOpenIntegrations}
                        >
                          {isMcp ? "Add" : "Connect"}
                        </button>
                      )}
                    </div>
                  );
                })}
              </div>
            </section>
          )}

          {/* persona-default connections (persona → session default) */}
          {detail.default_connections.length > 0 && (
            <section>
              <div className="persona-sec-h">New sessions get by default</div>
              <p className="persona-sec-note">
                When you start a {detail.name} session these are enabled automatically. You can still
                mute any of them per session.
              </p>
              <div className="persona-defs">
                {detail.default_connections.map((c) => (
                  <div
                    className="persona-def-row"
                    key={c.connector}
                    style={c.connected ? undefined : { opacity: 0.55 }}
                  >
                    <ConnectorBadge connector={visualFor(c.connector, "connector", byName)} size={32} />
                    <div className="persona-def-name">
                      {labelFor(c.connector, byName)}
                      {!c.connected && <span className="persona-def-hint"> · connect to enable</span>}
                    </div>
                    <Toggle
                      checked={c.enabled}
                      disabled={!c.connected}
                      onChange={(next) => toggleDefault(c.connector, next)}
                      title={c.connected ? "On by default for new sessions" : "Connect this first"}
                    />
                  </div>
                ))}
              </div>
            </section>
          )}

          {/* defaults footer */}
          <section className="persona-defaults">
            {detail.recommended_models.length > 0 && (
              <div>
                <span className="persona-defaults-k">Models</span> ·{" "}
                {detail.recommended_models.map((m, i) => (
                  <span key={m}>
                    <code>{m}</code>
                    {i < detail.recommended_models.length - 1 ? ", " : ""}
                  </span>
                ))}
              </div>
            )}
            {detail.default_permission_mode && (
              <div>
                <span className="persona-defaults-k">Default mode</span> · {detail.default_permission_mode}
              </div>
            )}
            {detail.workspace && (
              <div>
                <span className="persona-defaults-k">Workspace</span> · {detail.workspace}
              </div>
            )}
          </section>
        </div>
      </div>
    </main>
  );
}
