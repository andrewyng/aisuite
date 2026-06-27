import { useState, type ReactNode } from "react";
import type { InboxItem } from "../api";

// One Inbox item, rendered identically in the Inbox list and inline in its own session view
// (answer-in-context). Resolving either place hits the same item id — first responder wins.
export function InboxItemCard({
  item,
  onResolve,
  chip,
  compact,
}: {
  item: InboxItem;
  onResolve: (id: string, resolution: string) => void;
  chip?: ReactNode; // optional "go to session" affordance (shown in the Inbox list, not inline)
  compact?: boolean;
}) {
  const [answer, setAnswer] = useState("");
  return (
    <div className={"inbox-item" + (compact ? " compact" : "")}>
      <div className="inbox-item-kind dim">{item.kind}</div>
      <div className="inbox-item-title">{item.title}</div>
      {item.body ? <div className="dim inbox-item-body">{item.body}</div> : null}
      {chip}
      {item.kind === "approval" ? (
        <div className="inbox-item-actions">
          <button className="btn-primary sm" onClick={() => onResolve(item.id, "allow")}>
            Approve
          </button>
          <button className="btn sm" onClick={() => onResolve(item.id, "deny")}>
            Deny
          </button>
        </div>
      ) : item.kind === "question" ? (
        <div className="inbox-item-actions">
          <input
            placeholder="Your answer…"
            value={answer}
            onChange={(e) => setAnswer(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && answer.trim()) onResolve(item.id, answer);
            }}
          />
          <button
            className="btn-primary sm"
            disabled={!answer.trim()}
            onClick={() => onResolve(item.id, answer)}
          >
            Send
          </button>
        </div>
      ) : (
        <div className="inbox-item-actions">
          <button className="btn sm" onClick={() => onResolve(item.id, "seen")}>
            Dismiss
          </button>
        </div>
      )}
    </div>
  );
}
