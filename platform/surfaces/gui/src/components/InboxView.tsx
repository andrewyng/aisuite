import { useEffect, useState } from "react";
import { getInbox, getPersonas, resolveInboxItem, type InboxItem, type Persona } from "../api";
import type { SessionInfo } from "../types";
import { Icon } from "./Icon";
import { InboxItemCard } from "./InboxItemCard";

const ICON_FOR: Record<string, "diamond" | "chat" | "code"> = {
  cowork: "diamond",
  chat: "chat",
  code: "code",
};

// The Inbox: pending approvals / questions / notifications from across sessions, including
// unattended ones. Resolving here releases any agent suspended on the item. Each item links back
// to its originating session so you can see the context before answering.
export function InboxView({
  sessions,
  onOpenSession,
}: {
  sessions: SessionInfo[];
  onOpenSession: (sessionId: string) => void;
}) {
  const [items, setItems] = useState<InboxItem[]>([]);
  const [personas, setPersonas] = useState<Persona[] | null>(null);

  const load = () => getInbox(undefined, "pending").then(setItems).catch(() => {});
  useEffect(() => {
    load();
    getPersonas().then(setPersonas).catch(() => {});
    const t = setInterval(load, 4000);
    return () => clearInterval(t);
  }, []);

  const resolve = async (id: string, resolution: string) => {
    await resolveInboxItem(id, resolution);
    load();
  };

  // The originating-session chip: persona icon + session title, clickable to open that session.
  const sessionChip = (sessionId: string) => {
    const s = sessions.find((x) => x.session_id === sessionId);
    const p = personas?.find((x) => x.id === s?.agent);
    const label = s?.title || sessionId;
    const icon = (p && ICON_FOR[p.icon]) || "diamond";
    const cls = `ico-${p?.icon || "cowork"}`;
    return (
      <button
        className="inbox-session-chip"
        title={s ? `Open “${label}”` : "Session unavailable"}
        disabled={!s}
        onClick={() => s && onOpenSession(sessionId)}
      >
        <span className={"inbox-chip-ico " + cls}>
          <Icon name={icon} size={11} />
        </span>
        <span className="inbox-chip-label">{label}</span>
        {s && <Icon name="chevronRight" size={13} className="inbox-chip-go" />}
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

        {items.length === 0 ? <div className="manage-empty">Nothing pending.</div> : null}

        {items.map((it) => (
          <InboxItemCard
            key={it.id}
            item={it}
            onResolve={resolve}
            chip={sessionChip(it.session_id)}
          />
        ))}
      </div>
    </div>
  );
}
