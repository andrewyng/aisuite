import { useEffect, useRef, useState } from "react";
import { getConnectors, getMcpServers } from "../api";
import type { Attachment } from "../types";
import { readFile } from "../attach";
import { useRoots } from "../useRoots";
import { AddFolderForm } from "./AddFolderForm";
import { Icon } from "./Icon";

// Empty-state for a fresh Cowork session. Deliberately quiet: a greeting, then two suggestion
// lists in the same visual language — "Set me up" (the lead-in to sharing folders and connecting
// mail/calendar/tools) and "Try a task". No cards or status panels: the temporary-space details
// live in the composer's directory chip; active integrations show as a muted suffix.

interface Task {
  ico: string;
  text: string;
  prompt: string;
  pickFile?: boolean; // open a file picker and attach the chosen file to the composer
}

const TASKS: Task[] = [
  { ico: "✦", text: "Research a topic and write a one-page brief", prompt: "Research this topic and write me a one-page brief: " },
  { ico: "▦", text: "Analyze a CSV and summarize the key trends", prompt: "Analyze this CSV and summarize the key trends.", pickFile: true },
  { ico: "✎", text: "Draft a project plan with milestones", prompt: "Draft a project plan with milestones for: " },
];

export function SessionIntro({
  sessionId,
  onOpenIntegrations,
  onPrefill,
}: {
  sessionId: string;
  onOpenIntegrations: () => void;
  onPrefill: (text: string, attachments?: Attachment[]) => void;
}) {
  const { roots, busy, error, addRoot } = useRoots(sessionId);
  const [active, setActive] = useState<string[]>([]);
  const [addingFolder, setAddingFolder] = useState(false);
  const fileInput = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    // Active integrations = connected+enabled connectors and enabled MCP servers.
    Promise.all([getConnectors().catch(() => []), getMcpServers().catch(() => [])]).then(([conns, mcp]) => {
      setActive([
        ...conns.filter((c) => c.connected && c.enabled).map((c) => c.title || c.name),
        ...mcp.filter((m) => m.enabled).map((m) => m.name),
      ]);
    });
  }, []);

  const shared = roots.filter((r) => !r.primary);

  const runTask = (t: Task) => {
    if (t.pickFile) fileInput.current?.click();
    else onPrefill(t.prompt);
  };
  const onFile = async (files: FileList | null) => {
    const f = files?.[0];
    if (!f) return;
    const att = await readFile(f);
    onPrefill("Analyze this CSV and summarize the key trends.", att ? [att] : undefined);
  };

  return (
    <div className="intro">
      <h1 className="greeting">
        <span className="mark">✳</span> What should we produce?
      </h1>
      <p className="intro-lede">
        I work in a private temporary space — share folders or connect your tools, and I can do more.
      </p>

      <div className="suggestions intro-setup">
        <div className="suggest-head">Set me up</div>
        <div className="suggest" onClick={() => setAddingFolder((v) => !v)}>
          <span className="ico"><Icon name="folderPlus" size={16} /></span>
          Give me access to a folder
          {shared.length > 0 && (
            <span className="suggest-hint">
              · {shared.length} folder{shared.length === 1 ? "" : "s"} shared
            </span>
          )}
        </div>
        {addingFolder && (
          <div className="intro-addfolder">
            <AddFolderForm
              startOpen
              busy={busy}
              onAdd={addRoot}
              onDismiss={() => setAddingFolder(false)}
            />
            {error && <div className="roots-err">{error}</div>}
          </div>
        )}
        <div className="suggest" onClick={onOpenIntegrations}>
          <span className="ico"><Icon name="plug" size={16} /></span>
          Connect Gmail, Calendar, Drive…
          {active.length > 0 && <span className="suggest-hint">· {active.join(" ✓ · ")} ✓</span>}
        </div>
      </div>

      <div className="suggestions">
        <div className="suggest-head">Try a task</div>
        {TASKS.map((t, i) => (
          <div className="suggest" key={i} onClick={() => runTask(t)}>
            <span className="ico">{t.ico}</span>
            {t.text}
            {t.pickFile && <span className="suggest-hint">· pick a file</span>}
          </div>
        ))}
        <input
          ref={fileInput}
          type="file"
          accept=".csv,.tsv,.txt,text/csv"
          style={{ display: "none" }}
          onChange={(e) => {
            onFile(e.target.files);
            e.target.value = "";
          }}
        />
      </div>
    </div>
  );
}
