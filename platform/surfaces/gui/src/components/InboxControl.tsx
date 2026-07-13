import { useEffect, useState } from "react";
import { getUnattended, setUnattended } from "../api";
import { Icon } from "./Icon";
import { Toggle } from "./Toggle";

// The composer's Inbox control (replaces the old "Unattended" pill). An inbox icon + up-chevron that
// opens a small popover: "Send to Inbox" toggle (= run unattended — approvals/questions route to the
// Inbox so the agent keeps working) and an "Inbox settings…" link. The icon goes accent when ON, so
// it doubles as the state indicator.
export function InboxControl({
  sessionId,
  onChange,
  onOpenSettings,
}: {
  sessionId: string;
  onChange?: (on: boolean) => void;
  onOpenSettings?: () => void;
}) {
  const [on, setOn] = useState(false);
  const [open, setOpen] = useState(false);

  useEffect(() => {
    let alive = true;
    getUnattended(sessionId)
      .then((v) => alive && setOn(v))
      .catch(() => {});
    return () => {
      alive = false;
    };
  }, [sessionId]);

  const toggle = async (next: boolean) => {
    await setUnattended(sessionId, next);
    setOn(next);
    onChange?.(next);
  };

  return (
    <div className="relative">
      <button
        className={
          "inline-flex items-center gap-1 px-1.5 py-1 rounded-md hover:bg-paper shrink-0 " +
          (on ? "text-accent" : "text-muted hover:text-ink")
        }
        onClick={() => setOpen((v) => !v)}
        aria-haspopup="menu"
        aria-expanded={open}
        title={on ? "Sending approvals to the Inbox — the agent works unattended" : "Inbox routing"}
      >
        <Icon name="inbox" size={16} />
        <Icon
          name="chevronDown"
          size={11}
          className={"text-faint transition-transform " + (open ? "" : "rotate-180")}
        />
      </button>

      {open && (
        <>
          <div className="fixed inset-0 z-30" onClick={() => setOpen(false)} />
          <div className="absolute z-40 bottom-full mb-1 left-0 min-w-[230px] rounded-xl border border-line bg-panel shadow-2xl p-1.5">
            <div className="flex items-center gap-2 px-2 py-1.5">
              <span className="flex-1 min-w-0">
                <span className="block text-[13px] text-ink">Send to Inbox</span>
                <span className="block text-[11px] text-faint leading-snug">
                  Approvals &amp; questions go to the Inbox; the agent keeps working.
                </span>
              </span>
              <Toggle checked={on} onChange={toggle} title="Send approvals to the Inbox" />
            </div>
            <div className="my-1 border-t border-line" />
            <button
              className="w-full flex items-center gap-2.5 px-2 py-1.5 rounded-lg text-[13px] text-left hover:bg-paper"
              onClick={() => {
                setOpen(false);
                onOpenSettings?.();
              }}
            >
              <Icon name="gear" size={15} className="shrink-0 text-muted" /> Inbox settings…
            </button>
          </div>
        </>
      )}
    </div>
  );
}
