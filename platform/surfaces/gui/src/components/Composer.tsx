import { useLayoutEffect, useRef, useState, type ReactNode } from "react";
import type { Attachment } from "../types";
import { Dropdown, type Option } from "./Dropdown";
import { Icon } from "./Icon";

const PERMISSION_OPTIONS: Option[] = [
  { value: "plan", label: "Read-only", description: "Suggest changes — don't edit files or run commands" },
  { value: "interactive", label: "Ask for approval", description: "Ask before edits and commands" },
  { value: "auto", label: "Full access", description: "Run everything without asking" },
  { value: "custom", label: "Custom", description: "Use auto-allow rules from config.toml" },
];

// Fallback list when the server hasn't supplied one yet; the live list (incl. detected Ollama
// models) arrives via the `models` prop.
const MODEL_VALUES = ["gpt-5.5", "gpt-4o", "gpt-4o-mini", "o3-mini", "deepseek-chat"];

const MAX_BYTES = 10 * 1024 * 1024; // skip files larger than ~10MB
const TEXT_RE = /\.(txt|md|markdown|csv|tsv|json|ya?ml|log|ini|toml|py|js|ts|tsx|jsx|rs|go|java|c|h|cpp|sh|html?|css|sql|xml)$/i;

function readFile(file: File): Promise<Attachment | null> {
  const isImage = file.type.startsWith("image/");
  const isText = file.type.startsWith("text/") || TEXT_RE.test(file.name);
  if ((!isImage && !isText) || file.size > MAX_BYTES) return Promise.resolve(null);
  return new Promise((resolve) => {
    const reader = new FileReader();
    reader.onerror = () => resolve(null);
    reader.onload = () =>
      resolve(
        isImage
          ? { kind: "image", name: file.name || "image", mime: file.type, data_url: String(reader.result) }
          : { kind: "text", name: file.name || "file.txt", mime: file.type, text: String(reader.result) },
      );
    if (isImage) reader.readAsDataURL(file);
    else reader.readAsText(file);
  });
}

interface Props {
  mode: string;
  model: string;
  models?: string[];
  running: boolean;
  connected: boolean;
  onSend: (text: string, attachments?: Attachment[]) => void;
  onInterrupt: () => void;
  onModeChange: (mode: string) => void;
  onModelChange: (model: string) => void;
  // When set (Code/Cowork), a workspace chip is shown inside the composer.
  workspace?: string;
  branch?: string | null;
  onPickWorkspace?: () => void;
  approvalSlot?: ReactNode;
}

export function Composer(props: Props) {
  const [text, setText] = useState("");
  const [attachments, setAttachments] = useState<Attachment[]>([]);
  const [dragging, setDragging] = useState(false);
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

  const addFiles = async (files: FileList | File[]) => {
    const next = (await Promise.all(Array.from(files).map(readFile))).filter(Boolean) as Attachment[];
    if (next.length) setAttachments((a) => [...a, ...next].slice(0, 8));
  };

  const submit = () => {
    const t = text.trim();
    if ((!t && attachments.length === 0) || props.running) return;
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
    label: m,
  }));

  const wsName = props.workspace ? props.workspace.split("/").filter(Boolean).pop() : "";

  return (
    <div className="composer-wrap">
      {props.approvalSlot}
      <div
        className={"composer" + (dragging ? " dragging" : "")}
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
        {props.workspace !== undefined && (
          <div className="composer-head">
            <button className="wschip" onClick={props.onPickWorkspace} title={props.workspace}>
              <Icon name="folder" size={14} />
              <span className="wsname">{wsName || "Choose folder"}</span>
              <Icon name="pencil" size={12} className="edit" />
            </button>
            {props.branch && (
              <span className="wsbranch">
                <Icon name="branch" size={13} /> {props.branch}
              </span>
            )}
          </div>
        )}

        {attachments.length > 0 && (
          <div className="attach-row">
            {attachments.map((a, i) => (
              <AttachChip key={i} a={a} onRemove={() => setAttachments((all) => all.filter((_, j) => j !== i))} />
            ))}
          </div>
        )}

        <textarea
          ref={textareaRef}
          placeholder="Ask the coder to build, fix, or explain…  (drop or paste images)"
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={onKey}
          onPaste={onPaste}
          rows={1}
        />
        <div className="composer-bar">
          <button className="icon-btn" title="Attach image or file" onClick={() => fileInput.current?.click()}>
            <Icon name="plus" size={16} />
          </button>
          <input
            ref={fileInput}
            type="file"
            multiple
            accept="image/*,text/*,.md,.csv,.json,.yaml,.yml,.log,.py,.ts,.tsx,.js,.rs,.go,.toml"
            style={{ display: "none" }}
            onChange={(e) => {
              if (e.target.files) addFiles(e.target.files);
              e.target.value = "";
            }}
          />
          {/* Permission modes only matter for agents that touch files/commands (Code/Cowork). */}
          {props.workspace !== undefined && (
            <Dropdown value={props.mode} options={PERMISSION_OPTIONS} onChange={props.onModeChange} />
          )}
          <Dropdown value={props.model} options={modelOptions} onChange={props.onModelChange} />
          <span className="spacer" />
          {props.running ? (
            <button className="btn danger" onClick={props.onInterrupt}>
              ⏹ Stop
            </button>
          ) : (
            <button className="send" onClick={submit} disabled={!props.connected}>
              ↑
            </button>
          )}
        </div>
      </div>
      <div className="statusline">
        <span>
          <span className={"dot " + (!props.connected ? "off" : props.running ? "running" : "idle")} />
          &nbsp;{!props.connected ? "disconnected" : props.running ? "working…" : "ready"}
        </span>
        <span>Enter to send · Shift+Enter for newline</span>
      </div>
    </div>
  );
}

function AttachChip({ a, onRemove }: { a: Attachment; onRemove: () => void }) {
  return (
    <div className={"attach-chip" + (a.kind === "image" ? " img" : "")}>
      {a.kind === "image" ? (
        <img src={a.data_url} alt={a.name} />
      ) : (
        <>
          <Icon name="folder" size={13} />
          <span className="attach-name">{a.name}</span>
        </>
      )}
      <button className="attach-x" onClick={onRemove} title="Remove">
        ✕
      </button>
    </div>
  );
}
