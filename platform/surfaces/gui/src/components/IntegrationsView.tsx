import { useEffect, useState } from "react";
import { getSubscriptions, type Subscription } from "../api";
import { ConnectorsTab, McpTab } from "./ManageModal";
import { Icon } from "./Icon";

// Combined "Integrations" surface: messaging connectors + MCP servers + channel subscriptions.
export function IntegrationsView() {
  return (
    <div className="main page-view">
      <div className="page-col">
        <div className="sa-view-head">
          <div className="sa-view-heading">
            <div className="sa-view-title"><Icon name="plug" size={21} /> Integrations</div>
            <div className="sa-view-sub">Connect the apps and tools OpenCoworker can use.</div>
          </div>
        </div>
        <div className="main-scroll">
          <div className="page-panel">
            <div className="sa-sub">Connectors</div>
            <ConnectorsTab />
            <div className="sa-sub" style={{ marginTop: 26 }}>
              MCP servers
            </div>
            <McpTab />
            <div className="sa-sub" style={{ marginTop: 26 }}>
              Channel subscriptions
            </div>
            <SubscriptionsTab />
          </div>
        </div>
      </div>
    </div>
  );
}

// Read-only: which sessions listen to which channels (inbound), and where each routes its Inbox
// (outbound). Subscriptions are created by the agent (it asks you via ask_user); GUI-managed
// subscribe/unsubscribe comes later.
function SubscriptionsTab() {
  const [subs, setSubs] = useState<Subscription[] | null>(null);
  useEffect(() => {
    const load = () => getSubscriptions().then(setSubs).catch(() => setSubs([]));
    load();
    const t = setInterval(load, 5000);
    return () => clearInterval(t);
  }, []);

  if (subs === null) return <div className="dim sub-empty">Loading…</div>;
  if (subs.length === 0)
    return (
      <div className="dim sub-empty">
        No channel subscriptions yet. Ask a coworker to watch a channel and it'll appear here.
      </div>
    );

  return (
    <div className="sub-table">
      <div className="sub-row sub-head">
        <span>Session</span>
        <span>Listens to (inbound)</span>
        <span>Inbox routes to (outbound)</span>
      </div>
      {subs.map((s, i) => (
        <div className="sub-row" key={i}>
          <span className="sub-session" title={s.session_title}>{s.session_title}</span>
          <span className="sub-chan">
            <Icon name="plug" size={12} /> {s.channel}
            {s.collision && (
              <span className="sub-warn" title="This channel is also your Inbox-routing target — inbound and outbound on one channel conflate broadcast with request/reply.">
                ⚠ collides with Inbox routing
              </span>
            )}
          </span>
          <span className="sub-route dim">{s.routing_target || "—"}</span>
        </div>
      ))}
    </div>
  );
}
