import { useEffect, useState } from "react";
import {
  getRecentChannels,
  getSessions,
  getSubscriptions,
  subscribeChannel,
  unsubscribeChannel,
  type RecentChannel,
  type Subscription,
} from "../api";
import type { SessionInfo } from "../types";
import { ConnectorsTab, McpTab } from "./ManageModal";
import { ChannelPicker } from "./SubscriptionsChip";
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

// Which sessions listen to which channels (inbound), and where each routes its Inbox (outbound).
// Subscriptions can be created by the agent (it asks you via ask_user) or added here directly.
function SubscriptionsTab() {
  const [subs, setSubs] = useState<Subscription[] | null>(null);
  const [sessions, setSessions] = useState<SessionInfo[]>([]);
  const [recent, setRecent] = useState<RecentChannel[]>([]);
  const [addSession, setAddSession] = useState("");
  const [addChannel, setAddChannel] = useState("");

  const load = () => {
    getSubscriptions().then(setSubs).catch(() => setSubs([]));
    getSessions().then(setSessions).catch(() => setSessions([]));
    getRecentChannels().then(setRecent).catch(() => setRecent([]));
  };
  useEffect(() => {
    load();
    const t = setInterval(load, 5000);
    return () => clearInterval(t);
  }, []);

  const real = sessions.filter((s) => !s.session_id.startsWith("__"));
  const add = async () => {
    if (!addSession || !addChannel.trim()) return;
    await subscribeChannel(addSession, addChannel.trim());
    setAddChannel("");
    load();
  };
  const remove = async (sessionId: string, channel: string) => {
    await unsubscribeChannel(sessionId, channel);
    load();
  };

  return (
    <div>
      {subs && subs.length > 0 ? (
        <div className="sub-table">
          <div className="sub-row sub-head">
            <span>Session</span>
            <span>Listens to (inbound)</span>
            <span>Inbox routes to (outbound)</span>
            <span />
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
              <button className="sub-pop-x" title="Unsubscribe" onClick={() => remove(s.session_id, s.channel)}>
                ×
              </button>
            </div>
          ))}
        </div>
      ) : (
        <div className="dim sub-empty">No channel subscriptions yet — add one below or ask a coworker to watch a channel.</div>
      )}

      <div className="sub-add">
        <select className="sub-add-session" value={addSession} onChange={(e) => setAddSession(e.target.value)}>
          <option value="">Choose a session…</option>
          {real.map((s) => (
            <option key={s.session_id} value={s.session_id}>
              {s.title || s.session_id}
            </option>
          ))}
        </select>
        <ChannelPicker value={addChannel} onChange={setAddChannel} recent={recent} onSubmit={add} />
        <button className="btn-primary sm" disabled={!addSession || !addChannel.trim()} onClick={add}>
          Subscribe
        </button>
      </div>
    </div>
  );
}
