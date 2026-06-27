import { useState, type ReactNode } from "react";
import type { InboxItem } from "../api";

// One Inbox item, rendered identically in the Inbox list and inline in its own session view
// (answer-in-context). Resolving either place hits the same item id — first responder wins.
// Questions (ask_user) mirror Claude Code's AskUserQuestion: optional quick-reply options + an
// always-available free-text escape, with optional multi-select.
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
  const [selected, setSelected] = useState<string[]>([]);
  const options = item.options || [];
  const multi = !!item.multi;
  const allowText = item.allow_text !== false;

  const textRow = (placeholder: string) => (
    <div className="inbox-item-actions">
      <input
        placeholder={placeholder}
        value={answer}
        onChange={(e) => setAnswer(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && answer.trim()) onResolve(item.id, answer);
        }}
      />
      <button className="btn-primary sm" disabled={!answer.trim()} onClick={() => onResolve(item.id, answer)}>
        Send
      </button>
    </div>
  );

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
        <>
          {options.length > 0 && (
            <div className="inbox-options">
              {options.map((opt) => {
                const on = selected.includes(opt);
                return (
                  <button
                    key={opt}
                    className={"inbox-opt" + (on ? " on" : "")}
                    onClick={() => {
                      if (multi)
                        setSelected((s) => (on ? s.filter((x) => x !== opt) : [...s, opt]));
                      else onResolve(item.id, opt); // single-select resolves immediately
                    }}
                  >
                    {multi && <span className="inbox-opt-check">{on ? "✓" : ""}</span>}
                    {opt}
                  </button>
                );
              })}
            </div>
          )}
          {multi && options.length > 0 && (
            <div className="inbox-item-actions">
              <button
                className="btn-primary sm"
                disabled={!selected.length}
                onClick={() => onResolve(item.id, selected.join(", "))}
              >
                Send{selected.length ? ` (${selected.length})` : ""}
              </button>
            </div>
          )}
          {(allowText || options.length === 0) &&
            textRow(options.length ? "Or type your own answer…" : "Your answer…")}
        </>
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
