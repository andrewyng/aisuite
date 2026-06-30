import { useEffect, useLayoutEffect, useRef, useState, type ReactNode } from "react";
import type { Attachment } from "../types";
import { readFile } from "../attach";
import { Dropdown, type Option } from "./Dropdown";
import { Icon } from "./Icon";

const PERMISSION_OPTIONS: Option[] = [
  { value: "discuss", label: "Discuss", description: "Chat and explore — no edits or commands" },
  { value: "plan", label: "Plan", description: "Explore read-only, propose a plan for approval, then build" },
  { value: "interactive", label: "Ask for approval", description: "Ask before edits and commands" },
  { value: "auto", label: "Full access", description: "Run everything without asking" },
  { value: "custom", label: "Custom", description: "Use auto-allow rules from config.toml" },
];

// Fallback list when the server hasn't supplied one yet; the live list (incl. detected Ollama
// models) arrives via the `models` prop.
const MODEL_VALUES = ["gpt-5.5", "gpt-4o", "gpt-4o-mini", "o3-mini"];

// Drop the provider prefix for display (anthropic:claude-opus-4-8 → claude-opus-4-8); full id on hover.
const shortModel = (m: string) => (m.includes(":") ? m.split(":").slice(1).join(":") : m);

// Identify an attachment by name + payload size so duplicates (e.g. the same file picked twice,
// or a prefill applied twice) collapse to one chip.
const attKey = (a: Attachment) =>
  a.kind === "image" ? `i:${a.name}:${a.data_url?.length ?? 0}` : `t:${a.name}:${a.text?.length ?? 0}`;
const mergeAttachments = (cur: Attachment[], add: Attachment[]): Attachment[] => {
  const seen = new Set(cur.map(attKey));
  return [...cur, ...add.filter((a) => !seen.has(attKey(a)))].slice(0, 8);
};

interface Props {
  mode: string;
  model: string;
  models?: string[];
  running: boolean;
  connected: boolean;
  // False when the default model's provider has no key — the composer shows a "connect a model"
  // banner and routes sends to setup (preserving the draft) instead of dropping them.
  modelReady?: boolean;
  onConnectModel?: () => void;
  onSend: (text: string, attachments?: Attachment[]) => void;
  onInterrupt: () => void;
  onModeChange: (mode: string) => void;
  onModelChange: (model: string) => void;
  // When set (Code/Cowork), a workspace control is shown in the row.
  workspace?: string;
  branch?: string | null;
  onPickWorkspace?: () => void;
  // When set (orphan Cowork), the directory manager (icon-only folder popover) replaces the single
  // workspace folder button.
  rootsSlot?: ReactNode;
  approvalSlot?: ReactNode;
  // Contents of the "More" menu (Unattended now; Plugins/etc. later).
  moreSlot?: ReactNode;
  // Whether the session is Unattended — drives the inline indicator (the toggle itself lives in More).
  unattended?: boolean;
  // Push text + attachments into the composer (e.g. a start-panel task card). The `nonce` makes
  // repeated identical prefills re-apply; the user can still edit before sending.
  prefill?: { text: string; attachments?: Attachment[]; nonce: number };
  // Changes when the active conversation changes; clears any unsent draft.
  resetKey?: string;
  // Surface-specific hint shown in the empty textarea.
  placeholder?: string;
}

export function Composer(props: Props) {
  const [text, setText] = useState("");
  const [attachments, setAttachments] = useState<Attachment[]>([]);
  const [dragging, setDragging] = useState(false);
  const [attachMenuOpen, setAttachMenuOpen] = useState(false);
  const [moreMenuOpen, setMoreMenuOpen] = useState(false);
  const fileInput = useRef<HTMLInputElement | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);

  useLayoutEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    const max = parseFloat(getComputedStyle(el).lineHeight || "22") * 4;
    const next = Math.min(el.scrollHeight, max);
    el.style.height = `${Math.max(next, 24)}px`;
    el.style.overflowY = el.scrollHeight > max ? "auto" : "hidden";
  }, [text]);

  // Apply a prefill (text + attachments) pushed from outside, then focus the composer. Applied at
  // most once per nonce (a ref guards against StrictMode/re-render double-fires), and attachments
  // are de-duplicated so the same file never lands twice.
  const appliedNonce = useRef<number>(-1);
  useEffect(() => {
    const p = props.prefill;
    if (!p || p.nonce === appliedNonce.current) return;
    appliedNonce.current = p.nonce;
    setText(p.text);
    if (p.attachments?.length) setAttachments((cur) => mergeAttachments(cur, p.attachments!));
    textareaRef.current?.focus();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [props.prefill?.nonce]);

  // Clear the draft when the conversation changes, so a half-typed message / picked file doesn't
  // bleed from one session into another.
  useEffect(() => {
    setText("");
    setAttachments([]);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [props.resetKey]);

  const addFiles = async (files: FileList | File[]) => {
    const next = (await Promise.all(Array.from(files).map(readFile))).filter(Boolean) as Attachment[];
    if (next.length) setAttachments((a) => mergeAttachments(a, next));
  };

  // The "+" menu offers typed shortcuts; each just narrows the OS picker's filter.
  const pickFiles = (accept: string) => {
    setAttachMenuOpen(false);
    if (fileInput.current) {
      fileInput.current.accept = accept;
      fileInput.current.click();
    }
  };

  const needsModel = props.modelReady === false;

  const submit = () => {
    const t = text.trim();
    if ((!t && attachments.length === 0) || props.running) return;
    // No model connected: keep the draft (don't drop it) and send the user to setup instead.
    if (needsModel) {
      props.onConnectModel?.();
      return;
    }
    props.onSend(t, attachments);
    setText("");
    setAttachments([]);
  };

  const onKey = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  };

  const onPaste = (e: React.ClipboardEvent) => {
    const imgs = Array.from(e.clipboardData.items)
      .filter((it) => it.kind === "file" && it.type.startsWith("image/"))
      .map((it) => it.getAsFile())
      .filter(Boolean) as File[];
    if (imgs.length) {
      e.preventDefault();
      addFiles(imgs);
    }
  };

  const available = props.models && props.models.length ? props.models : MODEL_VALUES;
  const modelOptions: Option[] = Array.from(new Set([props.model, ...available])).map((m) => ({
    value: m,
    label: shortModel(m),
  }));

  const iconBtn =
    "w-7 h-7 grid place-items-center rounded-md text-muted hover:text-ink hover:bg-paper shrink-0";

  return (
    <div className="composer-wrap px-6 pb-5 pt-1">
      {props.approvalSlot}

      {/* Attachments preview — a strip ABOVE the input box (mock/Claude-style). */}
      {attachments.length > 0 && (
        <div className="max-w-3xl mx-auto mb-1.5 flex flex-wrap gap-2">
          {attachments.map((a, i) => (
            <AttachChip key={i} a={a} onRemove={() => setAttachments((all) => all.filter((_, j) => j !== i))} />
          ))}
        </div>
      )}

      <div
        className={
          "composer max-w-3xl mx-auto rounded-2xl border border-line bg-panel shadow-sm" +
          (dragging ? " dragging" : "")
        }
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          if (e.dataTransfer.files.length) addFiles(e.dataTransfer.files);
        }}
      >
        <textarea
          ref={textareaRef}
          className="w-full block px-3.5 pt-3.5 pb-1.5 text-[14.5px]"
          placeholder={props.placeholder || "Ask the coworker…  (drop or paste images)"}
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={onKey}
          onPaste={onPaste}
          rows={1}
        />

        {/* Single control row: + attach · folder · mode · More …(right)… model · send */}
        <div className="px-2.5 pb-2.5 pt-1 flex items-center gap-1.5">
          {/* + attach menu */}
          <div className="relative">
            <button
              className={iconBtn + (attachMenuOpen ? " bg-paper text-ink" : "")}
              title="Attach"
              aria-label="Attach"
              onClick={() => setAttachMenuOpen((v) => !v)}
            >
              <Icon name="plus" size={17} />
            </button>
            {attachMenuOpen && (
              <>
                <div className="fixed inset-0 z-30" onClick={() => setAttachMenuOpen(false)} />
                <div className="absolute z-40 bottom-full mb-1 left-0 min-w-[180px] rounded-xl border border-line bg-panel shadow-2xl py-1.5">
                  {attachItem("image", "Photo or image", () => pickFiles("image/*"))}
                  {attachItem("file", "PDF", () => pickFiles("application/pdf,.pdf"))}
                  {attachItem(
                    "fileCode",
                    "Other files",
                    () => pickFiles("text/*,.md,.csv,.json,.yaml,.yml,.log,.py,.ts,.tsx,.js,.rs,.go,.toml"),
                  )}
                </div>
              </>
            )}
          </div>
          <input
            ref={fileInput}
            type="file"
            multiple
            style={{ display: "none" }}
            onChange={(e) => {
              if (e.target.files) addFiles(e.target.files);
              e.target.value = "";
            }}
          />

          {/* folder: cowork = directory manager (rootsSlot); else a single-folder button (icon only,
              path on hover). Chat has no workspace → nothing. */}
          {props.rootsSlot ??
            (props.workspace !== undefined ? (
              <button
                className="wschip wschip-icon"
                onClick={props.onPickWorkspace}
                title={props.workspace || "Choose a folder"}
              >
                <Icon name="folder" size={15} />
                <Icon name="chevronDown" size={11} className="edit" />
              </button>
            ) : null)}
          {props.branch && (
            <span className="wsbranch">
              <Icon name="branch" size={13} /> {props.branch}
            </span>
          )}

          {/* permission mode — only for agents that touch files/commands */}
          {props.workspace !== undefined && (
            <Dropdown value={props.mode} options={PERMISSION_OPTIONS} onChange={props.onModeChange} />
          )}

          {/* More menu (Unattended now; Plugins later) */}
          {props.moreSlot && (
            <div className="relative">
              <button
                className={
                  "inline-flex items-center gap-1 px-2 py-1 rounded-md text-[12.5px] text-muted hover:text-ink hover:bg-paper " +
                  (moreMenuOpen ? "bg-paper text-ink" : "")
                }
                onClick={() => setMoreMenuOpen((v) => !v)}
                aria-haspopup="menu"
                aria-expanded={moreMenuOpen}
              >
                More
                <Icon
                  name="chevronDown"
                  size={12}
                  className={"text-faint transition-transform " + (moreMenuOpen ? "" : "rotate-180")}
                />
              </button>
              {moreMenuOpen && (
                <>
                  <div className="fixed inset-0 z-30" onClick={() => setMoreMenuOpen(false)} />
                  {/* No auto-close on inner click — items here (e.g. the Unattended toggle's
                      confirm step) manage their own state; closing is via the backdrop. */}
                  <div className="absolute z-40 bottom-full mb-1 left-0 min-w-[240px] rounded-xl border border-line bg-panel shadow-2xl p-2 flex flex-col gap-1.5">
                    {props.moreSlot}
                  </div>
                </>
              )}
            </div>
          )}

          {/* Unattended indicator — the toggle lives in More; this keeps the state glanceable. */}
          {props.unattended && (
            <span
              className="inline-flex items-center gap-1 text-[11.5px] text-accent font-medium"
              title="This session is running unattended — approvals route to the Inbox"
            >
              <span className="w-1.5 h-1.5 rounded-full bg-accent" /> Unattended
            </span>
          )}

          <span className="ml-auto" />

          {/* model (right) */}
          {needsModel ? (
            <button
              className="pill model-warn chip"
              onClick={() => props.onConnectModel?.()}
              title="Connect a model"
              aria-label="No model connected — connect a model"
            >
              <span className="pill-label">No model</span>
              <span className="model-warn-ico" aria-hidden>⚠</span>
            </button>
          ) : (
            <Dropdown value={props.model} options={modelOptions} onChange={props.onModelChange} align="right" className="chip" />
          )}

          {/* send / stop */}
          {props.running ? (
            <button className="btn danger" onClick={props.onInterrupt}>
              ⏹ Stop
            </button>
          ) : (
            <button
              className="w-8 h-8 rounded-full bg-accent text-white grid place-items-center hover:brightness-105 disabled:opacity-40 disabled:hover:brightness-100 shrink-0"
              onClick={submit}
              disabled={!props.connected}
              title={needsModel ? "Connect a model to send" : undefined}
              aria-label="Send"
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                <path d="M12 19V5M5 12l7-7 7 7" />
              </svg>
            </button>
          )}
        </div>
      </div>
      <div className="statusline">
        <span>
          <span
            className={"dot " + (needsModel || !props.connected ? "off" : props.running ? "running" : "idle")}
          />
          &nbsp;
          {needsModel
            ? "needs setup — connect a model to send"
            : !props.connected
              ? "disconnected"
              : props.running
                ? "working…"
                : "ready"}
        </span>
        <span>Enter to send · Shift+Enter for newline</span>
      </div>
    </div>
  );
}

// A row in the "+" attach menu.
function attachItem(icon: "image" | "file" | "fileCode", label: string, onClick: () => void) {
  return (
    <button
      className="w-full flex items-center gap-2.5 px-3 py-1.5 text-[13px] text-left hover:bg-paper"
      onClick={onClick}
    >
      <Icon name={icon} size={15} className="shrink-0 text-muted" /> {label}
    </button>
  );
}

function AttachChip({ a, onRemove }: { a: Attachment; onRemove: () => void }) {
  return (
    <div className={"attach-chip" + (a.kind === "image" ? " img" : "")}>
      {a.kind === "image" ? (
        <img src={a.data_url} alt={a.name} />
      ) : (
        <>
          <Icon name="file" size={13} />
          <span className="attach-name">{a.name}</span>
        </>
      )}
      <button className="attach-x" onClick={onRemove} title="Remove">
        ✕
      </button>
    </div>
  );
}
