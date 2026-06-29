import { useState, type ReactNode } from "react";
import type { InboxItem } from "../api";

// One Inbox item, rendered identically in the Inbox list and inline in its own session view
// (answer-in-context). Resolving either place hits the same item id — first responder wins.
// Questions (ask_user) mirror Claude Code's AskUserQuestion: optional quick-reply options + an
// always-available free-text escape, with optional multi-select.

// Shared styles (mock parity — same language as SourcesDrawer/PersonaView).
const SEC = "text-[11px] uppercase tracking-[0.05em] text-faint font-semibold";
const BTN_PRIMARY =
  "px-3 py-1.5 rounded-lg bg-accent text-white text-[12.5px] font-medium hover:brightness-105 disabled:opacity-40 disabled:hover:brightness-100";
const BTN_BORDERED =
  "px-3 py-1.5 rounded-lg border border-line bg-paper text-[12.5px] hover:border-lineStrong";
const OPT_BASE =
  "inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full border text-[13px] transition-colors";
const OPT_OFF = "border-line bg-paper text-ink hover:border-accent hover:bg-accentSoft/50";
const OPT_ON = "border-accent bg-accentSoft text-accent font-medium";
const INPUT =
  "flex-1 min-w-0 rounded-lg bg-paper border border-line px-3 py-2 text-[13px] text-ink placeholder:text-faint outline-none focus:border-lineStrong";

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
    <div className="flex items-center gap-2 mt-2.5">
      <input
        className={INPUT}
        placeholder={placeholder}
        value={answer}
        onChange={(e) => setAnswer(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && answer.trim()) onResolve(item.id, answer);
        }}
      />
      <button className={BTN_PRIMARY} disabled={!answer.trim()} onClick={() => onResolve(item.id, answer)}>
        Send
      </button>
    </div>
  );

  return (
    <div
      className={
        compact
          ? "max-w-3xl mx-auto mb-2.5 rounded-xl2 border border-lineStrong bg-panel px-4 py-3.5 shadow-[0_8px_24px_rgba(0,0,0,0.08)]"
          : "mb-2.5 rounded-xl2 border border-line bg-panel px-3.5 py-3"
      }
    >
      <div className={SEC}>{item.kind}</div>
      <div className="text-[15px] font-semibold mt-0.5 leading-snug">{item.title}</div>
      {item.body ? <div className="text-[13px] text-muted mt-1 whitespace-pre-wrap">{item.body}</div> : null}
      {chip}
      {item.kind === "approval" ? (
        <div className="flex items-center gap-2 mt-2.5">
          <button className={BTN_PRIMARY} onClick={() => onResolve(item.id, "allow")}>
            Approve
          </button>
          <button className={BTN_BORDERED} onClick={() => onResolve(item.id, "deny")}>
            Deny
          </button>
        </div>
      ) : item.kind === "question" ? (
        <>
          {options.length > 0 && (
            <div className="flex flex-wrap gap-2 mt-2.5">
              {options.map((opt) => {
                const on = selected.includes(opt);
                return (
                  <button
                    key={opt}
                    className={OPT_BASE + " " + (on ? OPT_ON : OPT_OFF)}
                    onClick={() => {
                      if (multi)
                        setSelected((s) => (on ? s.filter((x) => x !== opt) : [...s, opt]));
                      else onResolve(item.id, opt); // single-select resolves immediately
                    }}
                  >
                    {multi && on && <span className="text-accent text-[11px] leading-none">✓</span>}
                    {opt}
                  </button>
                );
              })}
            </div>
          )}
          {multi && options.length > 0 && (
            <div className="mt-2.5">
              <button
                className={BTN_PRIMARY}
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
      ) : item.kind === "directory" ? (
        <div className="flex items-center gap-2 mt-2.5">
          <button
            className={BTN_PRIMARY}
            disabled={!item.data?.path}
            title={item.data?.path || "No folder was suggested"}
            onClick={() =>
              onResolve(
                item.id,
                JSON.stringify({ granted: true, path: item.data?.path || "", writable: !!item.data?.writable }),
              )
            }
          >
            {item.data?.path ? "Grant" : "Grant (no folder)"}
          </button>
          <button className={BTN_BORDERED} onClick={() => onResolve(item.id, JSON.stringify({ granted: false }))}>
            Deny
          </button>
        </div>
      ) : item.kind === "plan" ? (
        <div className="flex items-center gap-2 mt-2.5">
          <button
            className={BTN_PRIMARY}
            onClick={() => onResolve(item.id, JSON.stringify({ approved: true, mode: "interactive" }))}
          >
            Approve
          </button>
          <button
            className={BTN_BORDERED}
            onClick={() => onResolve(item.id, JSON.stringify({ approved: false, feedback: "" }))}
          >
            Reject
          </button>
        </div>
      ) : (
        <div className="flex items-center gap-2 mt-2.5">
          <button className={BTN_BORDERED} onClick={() => onResolve(item.id, "seen")}>
            Dismiss
          </button>
        </div>
      )}
    </div>
  );
}
