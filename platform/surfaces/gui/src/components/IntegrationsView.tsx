import { useEffect, useState } from "react";
import {
  getDmRoute,
  getRecentChannels,
  getSessions,
  getSubscriptions,
  getUnrouted,
  setDmRoute,
  subscribeChannel,
  unsubscribeChannel,
  type RecentChannel,
  type Subscription,
  type UnroutedItem,
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
              Direct messages
            </div>
            <DmRouteTab />
            <div className="sa-sub" style={{ marginTop: 26 }}>
              Channel subscriptions
            </div>
            <SubscriptionsTab />
            <div className="sa-sub" style={{ marginTop: 26 }}>
              Unrouted &amp; failed messages
            </div>
            <UnroutedTab />
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

// Which session handles incoming DMs to the bot. None → DMs are parked in the Unrouted panel.
function DmRouteTab() {
  const [sessions, setSessions] = useState<SessionInfo[]>([]);
  const [dm, setDm] = useState<string>("");

  const load = () => {
    getSessions().then(setSessions).catch(() => setSessions([]));
    getDmRoute().then((s) => setDm(s || "")).catch(() => setDm(""));
  };
  useEffect(() => {
    load();
    const t = setInterval(load, 5000);
    return () => clearInterval(t);
  }, []);

  const real = sessions.filter((s) => !s.session_id.startsWith("__"));
  const choose = async (sessionId: string) => {
    setDm(sessionId);
    await setDmRoute(sessionId);
    load();
  };

  return (
    <div>
      <div className="dim sub-empty" style={{ marginBottom: 8 }}>
        Pick the session that handles direct messages to the bot. With none chosen, DMs are parked
        under “Unrouted &amp; failed messages” below.
      </div>
      <div className="sub-add">
        <select className="sub-add-session" value={dm} onChange={(e) => choose(e.target.value)}>
          <option value="">No session — park DMs</option>
          {real.map((s) => (
            <option key={s.session_id} value={s.session_id}>
              {s.title || s.session_id}
            </option>
          ))}
        </select>
      </div>
    </div>
  );
}

// Dead-letter view: inbound messages that had no destination (e.g. a DM with no session designated)
// and background turns that failed (e.g. a dead model). Read-only — for visibility/debugging.
function UnroutedTab() {
  const [items, setItems] = useState<UnroutedItem[] | null>(null);

  useEffect(() => {
    const load = () => getUnrouted().then(setItems).catch(() => setItems([]));
    load();
    const t = setInterval(load, 5000);
    return () => clearInterval(t);
  }, []);

  if (items && items.length === 0)
    return <div className="dim sub-empty">Nothing here — no dropped messages or failed turns.</div>;

  return (
    <div className="sub-table">
      <div className="sub-row sub-head unrouted-row">
        <span>When</span>
        <span>Source</span>
        <span>Reason</span>
        <span>Message</span>
      </div>
      {(items ?? []).map((it, i) => (
        <div className="sub-row unrouted-row" key={i}>
          <span className="dim">{new Date(it.ts * 1000).toLocaleString()}</span>
          <span className="sub-chan" title={it.sender}>{it.source}</span>
          <span className="sub-warn" title={it.reason}>{it.reason}</span>
          <span className="dim" title={it.text}>{it.text}</span>
        </div>
      ))}
    </div>
  );
}
