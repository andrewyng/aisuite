import { useEffect, useMemo, useState } from "react";
import {
  getConnectors,
  getInbox,
  getInboxRouting,
  getPersonas,
  getRecentChannels,
  resolveInboxItem,
  setInboxBinding,
  type InboxItem,
  type Persona,
  type RecentChannel,
} from "../api";
import { Icon } from "./Icon";
import { InboxItemCard } from "./InboxItemCard";
import { shortPersonaName } from "../personaScope";
import { ChannelPicker } from "./SubscriptionsChip";

const ICON_FOR: Record<string, "diamond" | "chat" | "code"> = {
  cowork: "diamond",
  chat: "chat",
  code: "code",
};

const KIND_TABS: { key: string; label: string }[] = [
  { key: "all", label: "All" },
  { key: "approval", label: "Approvals" },
  { key: "question", label: "Questions" },
];

const CHIP = (active: boolean) =>
  "text-[11.5px] px-2.5 py-1 rounded-full border " +
  (active
    ? "border-accent text-accent bg-accentSoft"
    : "border-line text-muted hover:border-lineStrong");

// The Inbox: pending approvals / questions / notifications from across sessions, including
// unattended ones. Resolving here releases any agent suspended on the item. Each item links back
// to its originating session so you can see the context before answering. Items whose session
// was deleted are closed server-side (an orphaned prompt can never be answered), so everything
// listed here is actionable. Filters: by kind and by persona (owner ask, 2026-07-03).
export function InboxView({
  onOpenSession,
}: {
  onOpenSession: (sessionId: string, workspace: string, agent: string) => void;
}) {
  const [items, setItems] = useState<InboxItem[]>([]);
  const [personas, setPersonas] = useState<Persona[] | null>(null);
  const [routing, setRouting] = useState<string | null>(null); // e.g. "slack:C0123" or null
  const [slackConnected, setSlackConnected] = useState(false);
  const [configOpen, setConfigOpen] = useState(false);
  const [recent, setRecent] = useState<RecentChannel[]>([]);
  const [draft, setDraft] = useState("");
  const [kind, setKind] = useState<string>("all");
  const [personaFilter, setPersonaFilter] = useState<string>("all");

  const load = () => getInbox(undefined, "pending").then(setItems).catch(() => {});
  const loadRouting = () =>
    getInboxRouting()
      .then((bindings) => {
        const bound = bindings.find((b) => b.channel);
        setRouting(bound ? `${bound.channel}:${bound.target}` : null);
      })
      .catch(() => setRouting(null));
  useEffect(() => {
    load();
    loadRouting();
    getPersonas().then(setPersonas).catch(() => {});
    getConnectors()
      .then((cs) => setSlackConnected(!!cs.find((c) => c.name === "slack" && c.connected)))
      .catch(() => {});
    getRecentChannels().then(setRecent).catch(() => setRecent([]));
    const t = setInterval(load, 4000);
    return () => clearInterval(t);
  }, []);

  const saveRoute = async () => {
    const raw = draft.trim();
    if (!raw) return;
    // Slack-only for now (Teams / Telegram / WhatsApp later): a bare id is a slack channel.
    const [platform, id] = raw.includes(":") ? raw.split(":", 2) : ["slack", raw];
    await setInboxBinding("default", platform, id);
    setDraft("");
    setConfigOpen(false);
    loadRouting();
  };
  const clearRoute = async () => {
    await setInboxBinding("default", null, "");
    setConfigOpen(false);
    loadRouting();
  };

  const resolve = async (id: string, resolution: string) => {
    await resolveInboxItem(id, resolution);
    load();
  };

  // Personas that actually have pending items drive the filter chips (no empty chips).
  const personasWithItems = useMemo(() => {
    const ids = [...new Set(items.map((i) => i.session_agent).filter(Boolean))] as string[];
    return ids.map((id) => ({
      id,
      label: shortPersonaName(personas?.find((p) => p.id === id)?.name, id),
    }));
  }, [items, personas]);

  const visible = items.filter(
    (it) =>
      (kind === "all" || it.kind === kind) &&
      (personaFilter === "all" || it.session_agent === personaFilter),
  );

  // The originating-session chip: persona icon + session title, clickable to open that session.
  const sessionChip = (it: InboxItem) => {
    const exists = it.session_exists !== false;
    const p = personas?.find((x) => x.id === it.session_agent);
    const label = it.session_title || it.session_id;
    const icon = (p && ICON_FOR[p.icon]) || "diamond";
    const cls = `ico-${p?.icon || "cowork"}`;
    return (
      <button
        className="inbox-session-chip"
        title={exists ? `Open “${label}”` : "Session unavailable"}
        disabled={!exists}
        onClick={() =>
          exists && onOpenSession(it.session_id, it.session_workspace || "", it.session_agent || "cowork")
        }
      >
        <span className={"inbox-chip-ico " + cls}>
          <Icon name={icon} size={11} />
        </span>
        <span className="inbox-chip-label">{label}</span>
        {exists && <Icon name="chevronRight" size={13} className="inbox-chip-go" />}
      </button>
    );
  };

  return (
    <div className="main-scroll">
      <div className="page-col">
        <div className="sa-view-head">
          <div className="sa-view-title">Inbox</div>
          <div className="sa-view-sub dim">
            Approvals, questions, and notifications from your agents — including sessions running
            unattended.
          </div>
        </div>
        <div className="text-[12px] text-faint -mt-2 mb-4" data-testid="inbox-routing">
          {routing ? (
            <span className="inline-flex items-center gap-2 flex-wrap">
              <span>
                Also delivered to <span className="text-muted">{routing}</span> — replies there
                resolve items here.
              </span>
              <button className="text-accent hover:underline" onClick={() => setConfigOpen((v) => !v)}>
                Change
              </button>
              <button className="text-danger/80 hover:text-danger" onClick={clearRoute}>
                Stop
              </button>
            </span>
          ) : slackConnected ? (
            <span className="inline-flex items-center gap-2 flex-wrap">
              <span>Delivered here only.</span>
              <button
                className="text-accent hover:underline"
                data-testid="inbox-route-configure"
                onClick={() => setConfigOpen((v) => !v)}
              >
                Also send to a Slack channel →
              </button>
            </span>
          ) : (
            <>
              Delivered here only. Connect Slack (Integrations ▸ Connectors) to also get these in a
              channel — more platforms later.
            </>
          )}
          {configOpen && (
            <span className="flex items-center gap-2 mt-2">
              <ChannelPicker value={draft} onChange={setDraft} recent={recent} onSubmit={saveRoute} />
              <button
                className="text-[12px] px-2.5 py-1 rounded-md bg-accent text-white disabled:opacity-50"
                disabled={!draft.trim()}
                onClick={saveRoute}
              >
                Set
              </button>
            </span>
          )}
        </div>

        <div className="flex items-center gap-2 flex-wrap mb-4" data-testid="inbox-filters">
          {KIND_TABS.map((t) => (
            <button key={t.key} className={CHIP(kind === t.key)} onClick={() => setKind(t.key)}>
              {t.label}
            </button>
          ))}
          {personasWithItems.length > 1 && (
            <>
              <span className="w-px h-4 bg-line mx-1" />
              <button className={CHIP(personaFilter === "all")} onClick={() => setPersonaFilter("all")}>
                All coworkers
              </button>
              {personasWithItems.map((p) => (
                <button
                  key={p.id}
                  className={CHIP(personaFilter === p.id)}
                  onClick={() => setPersonaFilter(p.id)}
                >
                  {p.label}
                </button>
              ))}
            </>
          )}
        </div>

        {visible.length === 0 ? (
          <div className="manage-empty">
            {items.length === 0 ? "Nothing pending." : "Nothing pending for this filter."}
          </div>
        ) : null}

        <div className="space-y-4">
          {visible.map((it) => (
            <InboxItemCard key={it.id} item={it} onResolve={resolve} chip={sessionChip(it)} />
          ))}
        </div>
      </div>
    </div>
  );
}
