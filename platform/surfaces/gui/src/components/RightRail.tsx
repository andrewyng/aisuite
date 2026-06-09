import { useEffect, useState, type ReactNode } from "react";
import {
  closeBrowser,
  getArtifacts,
  getBrowserState,
  readArtifact,
  takeBrowserScreenshot,
  type ArtifactContent,
  type ArtifactInfo,
  type BrowserState,
} from "../api";
import type { TodoItem } from "../types";
import { Icon } from "./Icon";

type Panel = "progress" | "browser" | "artifacts";

interface Props {
  active: boolean;
  sessionId: string;
  refreshKey: number;
  toolNames: string[];
  todo: TodoItem[];
  running: boolean;
  onHide: () => void;
}

export function RightRail({ active, sessionId, refreshKey, toolNames, todo, running, onHide }: Props) {
  const [open, setOpen] = useState<Record<Panel, boolean>>({
    progress: true,
    browser: false,
    artifacts: true,
  });
  const [browser, setBrowser] = useState<BrowserState | null>(null);
  const [artifacts, setArtifacts] = useState<ArtifactInfo[]>([]);
  const [selected, setSelected] = useState<ArtifactInfo | null>(null);
  const [content, setContent] = useState<ArtifactContent | null>(null);

  const refreshBrowser = () => getBrowserState().then(setBrowser).catch(() => setBrowser(null));
  const refreshArtifacts = () => getArtifacts(sessionId).then(setArtifacts).catch(() => setArtifacts([]));

  useEffect(() => {
    if (!active) return;
    refreshBrowser();
    refreshArtifacts();
  }, [active, sessionId, refreshKey]);

  useEffect(() => {
    setContent(null);
    if (!selected) return;
    readArtifact(sessionId, selected.path).then(setContent).catch(() => setContent(null));
  }, [selected?.path, sessionId]);

  const reloadSelected = () => {
    if (!selected) return Promise.resolve();
    setContent(null);
    return readArtifact(sessionId, selected.path).then(setContent).catch(() => setContent(null));
  };

  const browserActive = !!(browser?.open || browser?.last_action || browser?.last_error || browser?.screenshot_data_url);
  if (!active) return null;

  return (
    <aside className={"right-rail" + (selected ? " artifact-mode" : "")}>
      {selected ? (
        <ArtifactViewer
          artifact={selected}
          content={content}
          onReload={reloadSelected}
          onBack={() => setSelected(null)}
          onClose={() => {
            setSelected(null);
            onHide();
          }}
        />
      ) : (
        <>
          <RailSection title="Progress" open={open.progress} onToggle={() => setOpen({ ...open, progress: !open.progress })}>
            <ProgressSummary running={running} toolNames={toolNames} todo={todo} />
          </RailSection>

          {(browserActive || open.browser) && (
            <RailSection title="Browser" open={open.browser || browserActive} onToggle={() => setOpen({ ...open, browser: !open.browser })}>
              <BrowserMini state={browser} onRefresh={refreshBrowser} />
            </RailSection>
          )}

          <RailSection
            title={`Artifacts${artifacts.length ? ` (${artifacts.length})` : ""}`}
            open={open.artifacts}
            onToggle={() => setOpen({ ...open, artifacts: !open.artifacts })}
            action={<button className="rail-mini-btn" onClick={(e) => { e.stopPropagation(); refreshArtifacts(); }} title="Refresh artifacts"><Icon name="refresh" size={13} /></button>}
          >
            {artifacts.length === 0 ? (
              <div className="rail-muted">No previewable files yet.</div>
            ) : (
              <div className="artifact-list">
                {artifacts.slice(0, 16).map((a) => (
                  <button className="artifact-row" key={a.path} onClick={() => setSelected(a)}>
                    <span className={`artifact-kind ${a.kind}`}>{a.kind}</span>
                    <span className="artifact-name">
                      {a.name}
                      <span className="artifact-row-meta">{formatBytes(a.size)} · {formatTime(a.modified_at)}</span>
                    </span>
                    <span className="artifact-open">Open</span>
                  </button>
                ))}
              </div>
            )}
          </RailSection>

        </>
      )}
    </aside>
  );
}

function ProgressSummary({ running, toolNames, todo }: { running: boolean; toolNames: string[]; todo: TodoItem[] }) {
  if (todo.length) {
    return (
      <div className="rail-todo-list">
        {todo.map((item, index) => (
          <div className={"rail-todo " + item.status} key={index}>
            <span className="rail-todo-mark" />
            <span>{item.content}</span>
          </div>
        ))}
        {running && (
          <div className="rail-muted">
            {toolNames.length ? `${toolNames.length} tool call${toolNames.length === 1 ? "" : "s"} so far.` : "Working..."}
          </div>
        )}
      </div>
    );
  }
  if (running) {
    return (
      <div className="rail-muted">
        Working on this task{toolNames.length ? ` with ${toolNames.length} tool call${toolNames.length === 1 ? "" : "s"} so far.` : "."}
      </div>
    );
  }
  return (
    <div className="rail-muted">
      For longer multi-step tasks, progress will appear here while Cowork plans, uses tools, waits for approval, and produces artifacts.
    </div>
  );
}

function RailSection({
  title,
  open,
  onToggle,
  children,
  action,
}: {
  title: string;
  open: boolean;
  onToggle: () => void;
  children: ReactNode;
  action?: ReactNode;
}) {
  return (
    <section className="rail-section">
      <div className="rail-section-head">
        <button className="rail-section-toggle" onClick={onToggle}>
          <Icon name={open ? "chevronDown" : "chevronRight"} size={14} className="rail-chev" />
          <span>{title}</span>
        </button>
        {action}
      </div>
      {open && <div className="rail-section-body">{children}</div>}
    </section>
  );
}

function BrowserMini({ state, onRefresh }: { state: BrowserState | null; onRefresh: () => void }) {
  const snap = async () => {
    await takeBrowserScreenshot();
    onRefresh();
  };
  const close = async () => {
    await closeBrowser();
    onRefresh();
  };
  if (!state) return <div className="rail-muted">Browser state unavailable.</div>;
  return (
    <div className="browser-mini">
      <div className="rail-muted">{state.open ? state.url || "Open page" : "Closed"}</div>
      <div className="rail-muted">{state.last_action || "No browser action yet"} {state.last_result ? `- ${state.last_result}` : ""}</div>
      {state.last_error && <div className="rail-error">{state.last_error}</div>}
      {state.screenshot_data_url ? <img className="browser-shot" src={state.screenshot_data_url} /> : null}
      <div className="rail-actions">
        <button className="btn" onClick={onRefresh}>Refresh</button>
        <button className="btn" onClick={snap}>Shot</button>
        <button className="btn danger" onClick={close}>Close</button>
      </div>
    </div>
  );
}

function ArtifactViewer({
  artifact,
  content,
  onReload,
  onBack,
  onClose,
}: {
  artifact: ArtifactInfo;
  content: ArtifactContent | null;
  onReload: () => Promise<void>;
  onBack: () => void;
  onClose: () => void;
}) {
  const [reloadKey, setReloadKey] = useState(0);
  const isHtml = content?.kind === "html" && !content.error;

  return (
    <div className="artifact-viewer">
      <div className="artifact-head">
        <button className="artifact-icon-btn" onClick={onBack} aria-label="Back to artifacts" title="Back">
          <Icon name="arrowLeft" size={16} />
        </button>
        <div className="artifact-heading">
          <div className="artifact-title"><span>Artifacts</span><span className="artifact-sep">/</span><span>{artifact.name}</span></div>
          <div className="artifact-path">{artifact.path}</div>
        </div>
        <div className="rail-actions">
          {isHtml && (
            <button
              className="artifact-icon-btn"
              onClick={async () => {
                await onReload();
                setReloadKey((k) => k + 1);
              }}
              aria-label="Reload preview"
              title="Reload"
            >
              <Icon name="refresh" size={16} />
            </button>
          )}
          <button className="artifact-icon-btn" onClick={() => navigator.clipboard?.writeText(artifact.path)} aria-label="Copy path" title="Copy path">
            <Icon name="copy" size={16} />
          </button>
          <button className="artifact-icon-btn" onClick={onClose} aria-label="Hide artifact viewer" title="Hide">
            <Icon name="panelClose" size={16} />
          </button>
        </div>
      </div>
      <div className="artifact-preview">
        {!content ? (
          <div className="rail-muted">Loading...</div>
        ) : content.error ? (
          <div className="rail-error">{content.error}</div>
        ) : content.kind === "html" ? (
          <iframe
            key={`${artifact.path}-${reloadKey}`}
            sandbox="allow-scripts allow-same-origin"
            className="artifact-frame"
            srcDoc={content.content || ""}
          />
        ) : content.kind === "markdown" ? (
          <MarkdownPreview text={content.content || ""} />
        ) : content.kind === "image" ? (
          <img className="artifact-image" src={content.data_url} />
        ) : (
          <pre className="artifact-code">{content.content}</pre>
        )}
      </div>
    </div>
  );
}

function MarkdownPreview({ text }: { text: string }) {
  const blocks = parseMarkdownBlocks(text);
  return (
    <div className="markdown-preview">
      {blocks.map((block, i) => {
        if (block.type === "code") return <pre className="md-code" key={i}>{block.text}</pre>;
        const trimmed = block.text.trim();
        if (block.type === "h1") return <h1 key={i}>{trimmed}</h1>;
        if (block.type === "h2") return <h2 key={i}>{trimmed}</h2>;
        if (block.type === "h3") return <h3 key={i}>{trimmed}</h3>;
        if (block.type === "quote") return <blockquote key={i}>{trimmed}</blockquote>;
        if (block.type === "ol") return <ol key={i}>{block.lines.map((line, j) => <li key={j}>{line}</li>)}</ol>;
        if (block.type === "ul") return <ul key={i}>{block.lines.map((line, j) => <li key={j}>{line}</li>)}</ul>;
        return <p key={i}>{trimmed}</p>;
      })}
    </div>
  );
}

function parseMarkdownBlocks(text: string): Array<{ type: string; text: string; lines: string[] }> {
  const out: Array<{ type: string; text: string; lines: string[] }> = [];
  const chunks = text.split(/\n{2,}/);
  for (const chunk of chunks) {
    const raw = chunk.trim();
    if (!raw) continue;
    if (raw.startsWith("```")) {
      out.push({ type: "code", text: raw.replace(/^```[a-zA-Z0-9_-]*\n?/, "").replace(/\n?```$/, ""), lines: [] });
    } else if (raw.startsWith("# ")) out.push({ type: "h1", text: raw.slice(2), lines: [] });
    else if (raw.startsWith("## ")) out.push({ type: "h2", text: raw.slice(3), lines: [] });
    else if (raw.startsWith("### ")) out.push({ type: "h3", text: raw.slice(4), lines: [] });
    else if (/^>\s/.test(raw)) out.push({ type: "quote", text: raw.split("\n").map((l) => l.replace(/^>\s?/, "")).join("\n"), lines: [] });
    else if (/^[-*]\s/m.test(raw)) out.push({ type: "ul", text: "", lines: raw.split("\n").map((l) => l.replace(/^[-*]\s+/, "")) });
    else if (/^\d+\.\s/m.test(raw)) out.push({ type: "ol", text: "", lines: raw.split("\n").map((l) => l.replace(/^\d+\.\s+/, "")) });
    else out.push({ type: "p", text: raw, lines: [] });
  }
  return out;
}

function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes)) return "";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatTime(epochSeconds: number): string {
  if (!epochSeconds) return "";
  return new Date(epochSeconds * 1000).toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit" });
}
