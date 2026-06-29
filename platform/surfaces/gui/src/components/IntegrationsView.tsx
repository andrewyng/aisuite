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
// under MCP. Restyled to the mock (redesign.html §view-integrations) with Tailwind utilities — no
// functionality is lost; the existing tab components are just regrouped and re-skinned.
type IntTab = "connectors" | "messaging" | "activity" | "mcp";

// Shared utility strings (mock parity — mirrors PersonaView's constants).
const CARD = "rounded-xl2 border border-line bg-panel";
const SELECT = "px-2.5 py-1.5 rounded-lg border border-line bg-paper text-[13px] text-ink";
const BTN_ACCENT_SM = "text-[12px] px-2.5 py-1 rounded-md bg-accent text-white disabled:opacity-50";

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
    <main className="flex-1 min-w-0 flex bg-paper">
      <nav className="w-[208px] shrink-0 border-r border-line bg-panel/40 px-3 py-4">
        <div className="px-2 text-[13.5px] font-semibold mb-3 flex items-center gap-2">
          <Icon name="plug" size={16} /> Integrations
        </div>
        {INT_TABS.map((t) => {
          const active = tab === t.key;
          return (
            <button
              key={t.key}
              className={
                "w-full text-left px-2.5 py-2 rounded-lg text-[13px] flex items-center justify-between " +
                (active
                  ? "bg-paper text-accent font-medium"
                  : "text-muted hover:bg-paper hover:text-ink")
              }
              onClick={() => setTab(t.key)}
            >
              <span className="flex items-center gap-2 min-w-0">
                <Icon name={t.icon} size={15} /> {t.label}
              </span>
              {t.key === "connectors" && connCount != null && (
                <span className={"text-[11px] shrink-0 " + (active ? "text-accent" : "text-faint")}>
                  {connCount}
                </span>
              )}
              {t.key === "activity" && activityCount > 0 && (
                <span className="text-[11px] text-white bg-warnInk/90 rounded-full px-1.5 leading-[15px] shrink-0">
                  {activityCount}
                </span>
              )}
            </button>
          );
        })}
      </nav>

      <div className="flex-1 min-w-0 overflow-y-auto hairline-scroll">
        <div className="max-w-4xl mx-auto px-7 py-6">
          {tab === "connectors" ? (
            <section>
              <PanelHead
                title="Connectors"
                sub="Apps and tools OpenCoworker can use. You bring the credentials for this local build."
              />
              <ConnectorsTab />
            </section>
          ) : tab === "messaging" ? (
            <section>
              <PanelHead
                title="Messaging routing"
                sub="How inbound messages reach sessions, and where Unattended approvals go out."
              />
              <SubscriptionsTab />
              <div className="grid grid-cols-2 gap-4">
                <DmRouteTab />
                <InboxRoutingTab />
              </div>
            </section>
          ) : tab === "activity" ? (
            <section>
              <PanelHead
                title="Activity"
                sub="Unrouted inbound and background-turn failures — nothing vanishes silently."
              />
              <UnroutedTab />
            </section>
          ) : (
            <section>
              <PanelHead
                title="MCP servers"
                sub="External tool servers (stdio or HTTP), shared across all agents."
              />
              <McpTab />
            </section>
          )}
        </div>
      </div>
    </main>
  );
}

function PanelHead({ title, sub }: { title: string; sub: string }) {
  return (
    <div className="mb-4">
      <h2 className="text-[18px] font-semibold tracking-tight">{title}</h2>
      <p className="text-[12.5px] text-muted mt-0.5">{sub}</p>
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
    <div className={CARD + " mb-4 overflow-hidden"}>
      <div className="px-4 py-3 border-b border-line flex items-center gap-2">
        <span className="text-muted shrink-0">
          <Icon name="plug" size={15} />
        </span>
        <span className="font-semibold text-[13.5px]">Channel subscriptions</span>
        <span className="text-[12px] text-muted">— sessions that listen to a channel (inbound)</span>
      </div>

      {subs && subs.length > 0 ? (
        <table className="w-full text-[13px]">
          <thead className="text-[11px] uppercase tracking-[0.04em] text-faint">
            <tr className="text-left">
              <th className="font-medium px-4 py-2">Session</th>
              <th className="font-medium px-4 py-2">Listens to</th>
              <th className="font-medium px-4 py-2">Inbox routes to</th>
              <th className="px-4 py-2" />
            </tr>
          </thead>
          <tbody>
            {subs.map((s, i) => (
              <tr className="border-t border-line" key={i}>
                <td className="px-4 py-2.5 truncate max-w-[12rem]" title={s.session_title}>
                  {s.session_title}
                </td>
                <td className="px-4 py-2.5">
                  <span className="inline-flex items-center gap-1.5">
                    <span className="text-muted shrink-0">
                      <Icon name="plug" size={13} />
                    </span>
                    {s.channel}
                  </span>
                  {s.collision && (
                    <span
                      className="ml-1.5 text-[11px] text-warnInk bg-warnSoft/70 border border-warnInk/15 rounded px-1.5 py-0.5"
                      title="This channel is also your Inbox-routing target — inbound and outbound on one channel conflate broadcast with request/reply."
                    >
                      ⚠ collides
                    </span>
                  )}
                </td>
                <td className="px-4 py-2.5 text-muted">{s.routing_target || "—"}</td>
                <td className="px-4 py-2.5 text-right">
                  <button
                    className="text-faint hover:text-danger"
                    title="Unsubscribe"
                    onClick={() => remove(s.session_id, s.channel)}
                  >
                    ×
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <div className="px-4 py-3 text-[12.5px] text-muted">
          No channel subscriptions yet — add one below or ask a coworker to watch a channel.
        </div>
      )}

      <div className="border-t border-line px-4 py-3 flex items-center gap-2 flex-wrap">
        <select
          className={SELECT}
          value={addSession}
          onChange={(e) => setAddSession(e.target.value)}
        >
          <option value="">Choose a session…</option>
          {real.map((s) => (
            <option key={s.session_id} value={s.session_id}>
              {s.title || s.session_id}
            </option>
          ))}
        </select>
        <ChannelPicker value={addChannel} onChange={setAddChannel} recent={recent} onSubmit={add} />
        <button className={BTN_ACCENT_SM} disabled={!addSession || !addChannel.trim()} onClick={add}>
          + Subscribe
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
    <div className={CARD + " p-4"}>
      <div className="font-semibold text-[13.5px] mb-1">Direct messages</div>
      <p className="text-[12px] text-muted mb-3">
        Session that handles DMs to the bot. Unrouted DMs are parked under Activity.
      </p>
      <div className="flex items-center gap-2">
        <span className="text-muted shrink-0">
          <Icon name="chat" size={16} />
        </span>
        <select className={"flex-1 " + SELECT} value={dm} onChange={(e) => choose(e.target.value)}>
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
    <div className={CARD + " p-4"}>
      <div className="font-semibold text-[13.5px] mb-1">Unattended approvals</div>
      <p className="text-[12px] text-muted mb-3">
        Channel where an Unattended session posts Approve/Deny buttons. Currently mirroring to{" "}
        <strong className="text-ink font-medium">{target || "in-app Inbox only"}</strong>.
      </p>
      <div className="flex items-center gap-2 flex-wrap">
        <span className="text-muted shrink-0">
          <Icon name="plug" size={16} />
        </span>
        <ChannelPicker value={draft} onChange={setDraft} recent={recent} onSubmit={save} />
        <button className={BTN_ACCENT_SM} disabled={!draft.trim()} onClick={save}>
          Set
        </button>
        {target && (
          <button className="text-[12px] text-danger/80 hover:text-danger" onClick={clear}>
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
    return (
      <div className={CARD + " p-4 text-[13px] text-muted"}>
        Nothing here — no dropped messages or failed turns.
      </div>
    );

  return (
    <div className={CARD + " overflow-hidden"}>
      <table className="w-full text-[13px]">
        <thead className="text-[11px] uppercase tracking-[0.04em] text-faint">
          <tr className="text-left">
            <th className="font-medium px-4 py-2">When</th>
            <th className="font-medium px-4 py-2">Source</th>
            <th className="font-medium px-4 py-2">Reason</th>
            <th className="font-medium px-4 py-2">Message</th>
          </tr>
        </thead>
        <tbody>
          {(items ?? []).map((it, i) => (
            <tr className="border-t border-line" key={i}>
              <td className="px-4 py-2.5 text-muted whitespace-nowrap">
                {new Date(it.ts * 1000).toLocaleString()}
              </td>
              <td className="px-4 py-2.5" title={it.sender}>
                {it.source}
              </td>
              <td className="px-4 py-2.5">
                <span className="text-warnInk" title={it.reason}>
                  {it.reason}
                </span>
              </td>
              <td className="px-4 py-2.5 text-muted truncate max-w-[16rem]" title={it.text}>
                {it.text}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
