import { useEffect, useRef, useState } from "react";
import {
  getRecentChannels,
  subscribeChannel,
  unsubscribeChannel,
  type RecentChannel,
} from "../api";
import { Icon } from "./Icon";

// A channel input with a popover of recently-seen channels (the "recent list + type-the-id"
// picker). Free typing is allowed (a slack:C0123 address or a channel Copy-link URL). The
// popover is hand-rolled, NOT a <datalist>: WKWebView (the macOS desktop shell) doesn't render
// datalist suggestions at all, so the native path would silently show nothing on Mac.
export function ChannelPicker({
  value,
  onChange,
  recent,
  onSubmit,
}: {
  value: string;
  onChange: (v: string) => void;
  recent: RecentChannel[];
  onSubmit?: () => void;
}) {
  const [open, setOpen] = useState(false);
  const wrap = useRef<HTMLDivElement | null>(null);
  useEffect(() => {
    if (!open) return;
    const onDoc = (e: MouseEvent) => {
      if (wrap.current && !wrap.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", onDoc);
    return () => document.removeEventListener("mousedown", onDoc);
  }, [open]);

  // Filter as the user types (address or last-message text); the full list shows on focus.
  const q = value.trim().toLowerCase();
  const options = recent.filter(
    (c) =>
      !q ||
      c.channel.toLowerCase().includes(q) ||
      (c.last_text || "").toLowerCase().includes(q),
  );

  return (
    <div className="relative flex-1 min-w-0" ref={wrap}>
      <input
        className="chan-input w-full"
        placeholder="slack:C0123 or channel link"
        value={value}
        onChange={(e) => {
          onChange(e.target.value);
          setOpen(true);
        }}
        onFocus={() => setOpen(true)}
        onKeyDown={(e) => {
          if (e.key === "Escape") setOpen(false);
          if (e.key === "Enter" && onSubmit) {
            setOpen(false);
            onSubmit();
          }
        }}
      />
      {open && options.length > 0 && (
        <div
          className="absolute left-0 right-0 top-full mt-1 z-40 rounded-xl border border-line bg-panel shadow-lg py-1 max-h-56 overflow-y-auto"
          role="listbox"
          data-testid="channel-suggestions"
        >
          {options.map((c) => (
            <button
              key={c.channel}
              role="option"
              className="block w-full text-left px-3 py-1.5 hover:bg-paper"
              onMouseDown={(e) => {
                // mousedown (not click) so the pick lands before the input's blur
                e.preventDefault();
                onChange(c.channel);
                setOpen(false);
              }}
            >
              <span className="text-[12.5px] text-ink">{c.channel}</span>
              {c.last_text && (
                <span className="block text-[11px] text-faint truncate">
                  {c.last_from ? `${c.last_from}: ` : ""}
                  {c.last_text}
                </span>
              )}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

// The per-session "connections" chip in the composer head: shows how many channels this session
// listens to, and opens a popover to add (picker) / remove (×) — the per-session manage surface.
export function SubscriptionsChip({
  sessionId,
  channels,
  onChanged,
}: {
  sessionId: string;
  channels: string[];
  onChanged: () => void;
}) {
  const [open, setOpen] = useState(false);
  const [recent, setRecent] = useState<RecentChannel[]>([]);
  const [draft, setDraft] = useState("");
  const ref = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!open) return;
    getRecentChannels().then(setRecent).catch(() => setRecent([]));
    const onDoc = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", onDoc);
    return () => document.removeEventListener("mousedown", onDoc);
  }, [open]);

  const add = async () => {
    const c = draft.trim();
    if (!c) return;
    await subscribeChannel(sessionId, c);
    setDraft("");
    onChanged();
  };
  const remove = async (c: string) => {
    await unsubscribeChannel(sessionId, c);
    onChanged();
  };

  return (
    <div className="sub-chip-wrap" ref={ref}>
      <button
        className={"wschip sub-chip" + (open ? " active" : "")}
        title="Channels this session listens to"
        onClick={() => setOpen((v) => !v)}
      >
        <Icon name="plug" size={12} /> {channels.length || "+"}
      </button>
      {open && (
        <div className="sub-pop" onMouseDown={(e) => e.stopPropagation()}>
          <div className="sub-pop-head">Channels this session listens to</div>
          {channels.length === 0 ? (
            <div className="dim sub-pop-empty">Not subscribed to any channel.</div>
          ) : (
            channels.map((c) => (
              <div className="sub-pop-row" key={c}>
                <span className="sub-pop-chan">{c}</span>
                <button className="sub-pop-x" title="Unsubscribe" onClick={() => remove(c)}>
                  ×
                </button>
              </div>
            ))
          )}
          <div className="sub-pop-add">
            <ChannelPicker value={draft} onChange={setDraft} recent={recent} onSubmit={add} />
            <button className="btn-primary sm" disabled={!draft.trim()} onClick={add}>
              Add
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
