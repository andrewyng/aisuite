import { useEffect, useState } from "react";
import {
  getConnectors,
  getDmRoute,
  getInboxRouting,
  getRecentChannels,
  getSessions,
  getSubscriptions,
  getUnrouted,
  setDmRoute,
  setInboxBinding,
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

// The Integrations surface restructured (§7 / UX §6) into a left sub-nav so it's no longer one long
// scroll: Connectors · Messaging routing · Activity · MCP. The connector cards (with the Slack
// allow-list inline) live under Connectors; the DM-route / channel-subscriptions / approvals-routing
// controls move under Messaging routing; the Unrouted/dead-letter panel under Activity; MCP servers
// under MCP. No functionality is lost — the existing tab components are just regrouped.
type IntTab = "connectors" | "messaging" | "activity" | "mcp";

const INT_TABS: { key: IntTab; label: string; icon: "plug" | "chat" | "audit" | "code" }[] = [
  { key: "connectors", label: "Connectors", icon: "plug" },
  { key: "messaging", label: "Messaging routing", icon: "chat" },
  { key: "activity", label: "Activity", icon: "audit" },
  { key: "mcp", label: "MCP servers", icon: "code" },
];

export function IntegrationsView() {
  const [tab, setTab] = useState<IntTab>("connectors");
  // Sub-nav counts: how many connectors exist, and how many unrouted/failed items are parked (the
  // ⚠ N on Activity). Polled like the panels so the badges stay live.
  const [connCount, setConnCount] = useState<number | null>(null);
  const [activityCount, setActivityCount] = useState(0);

  useEffect(() => {
    const load = () => {
      getConnectors().then((cs) => setConnCount(cs.length)).catch(() => {});
      getUnrouted().then((u) => setActivityCount(u.length)).catch(() => setActivityCount(0));
    };
    load();
    const t = setInterval(load, 5000);
    return () => clearInterval(t);
  }, []);

  return (
    <div className="main page-view integrations-view">
      <nav className="int-subnav">
        <div className="int-subnav-title">
          <Icon name="plug" size={16} /> Integrations
        </div>
        {INT_TABS.map((t) => (
          <button
            key={t.key}
            className={"int-tab" + (tab === t.key ? " active" : "")}
            onClick={() => setTab(t.key)}
          >
            <span className="int-tab-label">
              <Icon name={t.icon} size={15} /> {t.label}
            </span>
            {t.key === "connectors" && connCount != null && (
              <span className="int-tab-count">{connCount}</span>
            )}
            {t.key === "activity" && activityCount > 0 && (
              <span className="int-tab-count warn">{activityCount}</span>
            )}
          </button>
        ))}
      </nav>

      <div className="int-main">
        <div className="main-scroll">
          <div className="int-panel">
            {tab === "connectors" ? (
              <>
                <PanelHead
                  title="Connectors"
                  sub="Apps and tools OpenCoworker can use. You bring the credentials for this local build."
                />
                <ConnectorsTab />
              </>
            ) : tab === "messaging" ? (
              <>
                <PanelHead
                  title="Messaging routing"
                  sub="How inbound messages reach sessions, and where Unattended approvals go out."
                />
                <div className="sa-sub">Channel subscriptions</div>
                <div className="dim int-sec-note">
                  Sessions that listen to a channel (inbound) and where each routes its Inbox (outbound).
                </div>
                <SubscriptionsTab />
                <div className="sa-sub" style={{ marginTop: 26 }}>
                  Direct messages
                </div>
                <DmRouteTab />
                <div className="sa-sub" style={{ marginTop: 26 }}>
                  Unattended approvals → channel
                </div>
                <InboxRoutingTab />
              </>
            ) : tab === "activity" ? (
              <>
                <PanelHead
                  title="Activity"
                  sub="Unrouted inbound and background-turn failures — nothing vanishes silently."
                />
                <UnroutedTab />
              </>
            ) : (
              <>
                <PanelHead
                  title="MCP servers"
                  sub="External tool servers (stdio or HTTP), shared across all agents."
                />
                <McpTab />
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function PanelHead({ title, sub }: { title: string; sub: string }) {
  return (
    <div className="int-panel-head">
      <h2 className="int-panel-title">{title}</h2>
      <p className="int-panel-sub">{sub}</p>
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

// Where an Unattended session's approvals/questions get mirrored as interactive buttons. Targets
// the "default" route (sessions fall back to it); pick a channel separate from any you subscribe to.
function InboxRoutingTab() {
  const [recent, setRecent] = useState<RecentChannel[]>([]);
  const [target, setTarget] = useState(""); // current default-binding address, e.g. "slack:C0123"
  const [draft, setDraft] = useState("");

  const load = () => {
    getRecentChannels().then(setRecent).catch(() => setRecent([]));
    getInboxRouting()
      .then((bs) => {
        const def = bs.find((b) => b.name === "default");
        setTarget(def?.channel ? `${def.channel}:${def.target}` : "");
      })
      .catch(() => setTarget(""));
  };
  useEffect(() => {
    load();
    const t = setInterval(load, 5000);
    return () => clearInterval(t);
  }, []);

  const save = async () => {
    const addr = draft.trim();
    if (!addr) return;
    // "slack:C0123" → channel="slack", target="C0123"; a bare id assumes slack.
    const [platform, id] = addr.includes(":") ? addr.split(":", 2) : ["slack", addr];
    await setInboxBinding("default", platform, id);
    setDraft("");
    load();
  };
  const clear = async () => {
    await setInboxBinding("default", null, "");
    load();
  };

  return (
    <div>
      <div className="dim sub-empty" style={{ marginBottom: 8 }}>
        Currently mirroring to: <strong>{target || "in-app Inbox only"}</strong>. Pick a channel the
        bot is in (and that you don't subscribe to) — approvals post there as Approve/Deny buttons.
      </div>
      <div className="sub-add">
        <ChannelPicker value={draft} onChange={setDraft} recent={recent} onSubmit={save} />
        <button className="btn-primary sm" disabled={!draft.trim()} onClick={save}>
          Set
        </button>
        {target && (
          <button className="link danger" onClick={clear}>
            clear
          </button>
        )}
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
