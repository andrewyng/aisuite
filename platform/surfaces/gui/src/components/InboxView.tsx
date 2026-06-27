import { useEffect, useState } from "react";
import { getInbox, resolveInboxItem, type InboxItem } from "../api";

// The Inbox: pending approvals / questions / notifications from across sessions, including
// unattended ones. Resolving here releases any agent suspended on the item.
export function InboxView() {
  const [items, setItems] = useState<InboxItem[]>([]);
  const [answer, setAnswer] = useState<Record<string, string>>({});

  const load = () => getInbox(undefined, "pending").then(setItems).catch(() => {});
  useEffect(() => {
    load();
    const t = setInterval(load, 4000);
    return () => clearInterval(t);
  }, []);

  const resolve = async (id: string, resolution: string) => {
    await resolveInboxItem(id, resolution);
    load();
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

        {items.length === 0 ? (
          <div className="manage-empty">Nothing pending.</div>
        ) : null}

        {items.map((it) => (
          <div key={it.id} className="inbox-item">
            <div className="inbox-item-kind dim">{it.kind}</div>
            <div className="inbox-item-title">{it.title}</div>
            {it.body ? <div className="dim inbox-item-body">{it.body}</div> : null}
            {it.kind === "approval" ? (
              <div className="inbox-item-actions">
                <button className="btn-primary sm" onClick={() => resolve(it.id, "allow")}>
                  Approve
                </button>
                <button className="btn sm" onClick={() => resolve(it.id, "deny")}>
                  Deny
                </button>
              </div>
            ) : it.kind === "question" ? (
              <div className="inbox-item-actions">
                <input
                  placeholder="Your answer…"
                  value={answer[it.id] || ""}
                  onChange={(e) =>
                    setAnswer((a) => ({ ...a, [it.id]: e.target.value }))
                  }
                />
                <button
                  className="btn-primary sm"
                  disabled={!(answer[it.id] || "").trim()}
                  onClick={() => resolve(it.id, answer[it.id])}
                >
                  Send
                </button>
              </div>
            ) : (
              <div className="inbox-item-actions">
                <button className="btn sm" onClick={() => resolve(it.id, "seen")}>
                  Dismiss
                </button>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
