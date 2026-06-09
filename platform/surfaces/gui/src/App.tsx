import { useCallback, useEffect, useRef, useState, type PointerEvent } from "react";
import {
  finalizeAutomationRun,
  getHealth,
  getRecentWorkspaces,
  getSessionMessages,
  getSessions,
  getSettings,
  getSuperagent,
  deleteSession,
  renameSession,
  runAutomation,
  Session,
  type RecentWorkspace,
  type SurfaceVisibility,
} from "./api";
import type { ApprovalDecision, Attachment, Item, SessionInfo, TodoItem, WsEvent } from "./types";
import { isTauri, startWindowDrag } from "./tauri";
import { Icon } from "./components/Icon";
import { Sidebar } from "./components/Sidebar";
import { Transcript } from "./components/Transcript";
import { Composer } from "./components/Composer";
import { FolderGate } from "./components/FolderGate";
import { ManageModal } from "./components/ManageModal";
import { Onboarding } from "./components/Onboarding";
import { SuperAgentView } from "./components/SuperAgentView";
import { ScheduledView } from "./components/ScheduledView";
import { RightRail } from "./components/RightRail";
import { IntegrationsView } from "./components/IntegrationsView";
import { AuditView } from "./components/AuditView";
import { ApprovalCard } from "./components/ApprovalCard";

const newId = () =>
  (crypto as any).randomUUID ? crypto.randomUUID().slice(0, 12) : Math.random().toString(36).slice(2, 14);

const SUGGESTIONS = [
  { ico: "⚙", text: "Run the test suite and summarize any failures." },
  { ico: "✦", text: "Read the project and give me a 5-bullet overview." },
  { ico: "↻", text: "Find and fix the failing build." },
];

const COWORK_SUGGESTIONS = [
  { ico: "✦", text: "Research a topic and write me a one-page brief." },
  { ico: "▦", text: "Analyze this CSV and summarize the key trends." },
  { ico: "✎", text: "Draft a project plan with milestones for…" },
];

const needsWorkspace = (a: string) => a === "code" || a === "cowork";
const LAST_SESSION_KEY = "coworker:last-session-by-agent:v1";

type LastSession = { sessionId: string; workspace: string; updatedAt: number };

function readLastSessions(): Record<string, LastSession> {
  try {
    const raw = localStorage.getItem(LAST_SESSION_KEY);
    return raw ? JSON.parse(raw) : {};
  } catch {
    return {};
  }
}

function rememberLastSession(agent: string, sessionId: string, workspace: string | null) {
  if (!agent || !sessionId) return;
  try {
    const all = readLastSessions();
    all[agent] = { sessionId, workspace: workspace || "", updatedAt: Date.now() };
    localStorage.setItem(LAST_SESSION_KEY, JSON.stringify(all));
  } catch {
    /* localStorage may be unavailable; session restore is best effort. */
  }
}

function sessionTs(s: SessionInfo): number {
  return Date.parse(s.updated_at || "") || Number(s.updated_at) || 0;
}

function resumeTargetForAgent(agent: string, sessions: SessionInfo[]): LastSession | null {
  const remembered = readLastSessions()[agent];
  if (remembered?.sessionId) {
    const live = sessions.find((s) => s.session_id === remembered.sessionId && s.agent === agent);
    if (live || remembered.workspace) {
      return {
        sessionId: remembered.sessionId,
        workspace: live?.workspace ?? remembered.workspace ?? "",
        updatedAt: live ? sessionTs(live) : remembered.updatedAt,
      };
    }
  }
  const recent = sessions
    .filter((s) => s.agent === agent && s.session_id && !s.session_id.startsWith("__"))
    .sort((a, b) => sessionTs(b) - sessionTs(a))[0];
  return recent ? { sessionId: recent.session_id, workspace: recent.workspace || "", updatedAt: sessionTs(recent) } : null;
}

function fallbackWorkspace(current: string | null, projects: RecentWorkspace[]): string {
  if (current) return current;
  const existing = projects.find((p) => p.exists);
  return existing?.path || projects[0]?.path || "";
}

export function App() {
  const [workspace, setWorkspace] = useState<string | null>(null);
  const [branch, setBranch] = useState<string | null>(null);
  const [showGate, setShowGate] = useState(false);
  const [agent, setAgent] = useState("cowork");
  const [model, setModel] = useState("gpt-5.5");
  const [models, setModels] = useState<string[]>([]);
  const [surfaces, setSurfaces] = useState<SurfaceVisibility>({ cowork: true, chat: false, code: false });
  const [mode, setMode] = useState("interactive");
  const [connected, setConnected] = useState(false);
  const [running, setRunning] = useState(false);
  const [items, setItems] = useState<Item[]>([]);
  const [streaming, setStreaming] = useState("");
  const [todo, setTodo] = useState<TodoItem[]>([]);
  const [sessions, setSessions] = useState<SessionInfo[]>([]);
  const [projects, setProjects] = useState<RecentWorkspace[]>([]);
  const [sessionId, setSessionId] = useState<string>(newId());
  const [gateCreate, setGateCreate] = useState(false);
  const [showManage, setShowManage] = useState(false);
  const [manageTab, setManageTab] = useState<"settings" | undefined>(undefined);
  const [surface, setSurface] = useState<"session" | "superagent" | "scheduled" | "integrations" | "audit">("session");
  const [helperName, setHelperName] = useState("MyHelper");
  const [browserRefreshKey, setBrowserRefreshKey] = useState(0);
  const [railHidden, setRailHidden] = useState(false);
  const [topbarMenuOpen, setTopbarMenuOpen] = useState(false);

  // The desktop tray's "Settings" item dispatches this on the window.
  useEffect(() => {
    const open = () => {
      setManageTab("settings");
      setShowManage(true);
    };
    window.addEventListener("coworker:open-settings", open);
    return () => window.removeEventListener("coworker:open-settings", open);
  }, []);

  // "Run setup again" (from Settings) re-opens the wizard.
  useEffect(() => {
    const open = () => {
      setShowManage(false);
      setOnboarding(true);
    };
    window.addEventListener("coworker:open-onboarding", open);
    return () => window.removeEventListener("coworker:open-onboarding", open);
  }, []);

  const sessionRef = useRef<Session | null>(null);
  const scrollRef = useRef<HTMLDivElement | null>(null);
  // A prompt to auto-send once the next session connects (used by "Run now").
  const pendingPromptRef = useRef<string | null>(null);
  // The in-flight manual run to finalize after its first turn ({taskId, runId, sessionId}).
  const activeRunRef = useRef<{ taskId: string; runId: string; sessionId: string } | null>(null);

  // Fetch ALL sessions + known projects so the sidebar can group them.
  const refreshSessions = useCallback(() => {
    getSessions().then(setSessions).catch(() => setSessions([]));
    getRecentWorkspaces().then(setProjects).catch(() => setProjects([]));
  }, []);

  // initial: adopt the server's seed workspace if any, else force the gate.
  // Retry health for a while: the desktop shell starts its sidecar in parallel, so the
  // server may not answer for a second or two. Only fall back to the gate once it's truly up.
  const [booting, setBooting] = useState(true);
  const [onboarding, setOnboarding] = useState(false);

  // On boot with no seeded workspace, reopen the last thing the user had — most recent
  // conversation (restores its folder + agent + transcript), else the most recent project
  // folder. Only a true first run (nothing to resume) falls through to the folder gate.
  const resumeLastOrGate = async () => {
    let loadedSessions: SessionInfo[] = [];
    try {
      loadedSessions = (await getSessions()).filter((s) => s.session_id && !s.session_id.startsWith("__"));
      setSessions(loadedSessions);
      const sess = loadedSessions;
      const ts = (s: SessionInfo) => Date.parse(s.updated_at || "") || Number(s.updated_at) || 0;
      const last = [...sess].sort((a, b) => ts(b) - ts(a))[0];
      if (last) {
        if (last.agent) setAgent(last.agent);
        if (last.workspace) {
          setWorkspace(last.workspace);
          setBranch(null);
        }
        try {
          setItems(itemsFromMessages(await getSessionMessages(last.session_id)));
        } catch {
          setItems([]);
        }
        setSessionId(last.session_id);
        setShowGate(false);
        return;
      }
    } catch {
      /* fall through */
    }
    try {
      const recents = await getRecentWorkspaces();
      setProjects(recents);
      const ws = recents.find((w) => w.exists) || recents[0];
      if (ws) {
        setWorkspace(ws.path);
        setShowGate(false);
        return;
      }
    } catch {
      /* fall through */
    }
    setShowGate(true); // nothing to resume → first-run folder gate
  };

  useEffect(() => {
    let cancelled = false;
    const attempt = (tries: number) => {
      getHealth()
        .then((h) => {
          if (cancelled) return;
          setBooting(false);
          setModel(h.model);
          // First-run setup wizard (desktop): show until the user completes/dismisses it.
          if (isTauri()) {
            getSettings()
              .then((s) => !cancelled && !s.onboarded && setOnboarding(true))
              .catch(() => {});
          }
          if (h.default_workspace) setWorkspace(h.default_workspace);
          else resumeLastOrGate();
        })
        .catch(() => {
          if (cancelled) return;
          if (tries <= 0) {
            setBooting(false);
            setShowGate(true);
          } else {
            setTimeout(() => attempt(tries - 1), 500);
          }
        });
    };
    attempt(40); // ~20s of 500ms retries
    return () => {
      cancelled = true;
    };
  }, []);

  const loadSettings = () =>
    getSettings()
      .then((s) => {
        setModels(s.models || []);
        if (s.surfaces) setSurfaces(s.surfaces);
      })
      .catch(() => {});

  useEffect(() => {
    refreshSessions();
    loadSettings(); // selectable models + which session surfaces are visible
    getSuperagent().then((s) => s?.name && setHelperName(s.name)).catch(() => {});
  }, [refreshSessions]);

  // If the active surface isn't visible (hidden in Settings, or a resumed session landed on a
  // hidden surface), fall back to Cowork (always visible). Watches both agent and surfaces so it
  // corrects regardless of which settled last.
  useEffect(() => {
    if ((agent === "chat" && !surfaces.chat) || (agent === "code" && !surfaces.code)) {
      switchAgent("cowork");
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [agent, surfaces]);

  useEffect(() => {
    if (surface === "session") rememberLastSession(agent, sessionId, workspace);
  }, [surface, agent, sessionId, workspace]);

  // (re)connect when workspace, session, or agent changes
  useEffect(() => {
    if (needsWorkspace(agent) && !workspace) return; // Code/Cowork need a folder (gate handles it)
    const handleEvent = (ev: WsEvent) => {
      const d = ev.data || {};
      switch (ev.type) {
        case "ready":
          setConnected(true);
          if (d.model) setModel(d.model);
          if (d.mode) setMode(d.mode);
          break;
        case "turn_start":
          setRunning(true);
          setStreaming("");
          break;
        case "assistant_delta":
          setStreaming((s) => s + (d.text || ""));
          break;
        case "assistant_message":
          if (d.text) setItems((p) => [...p, { kind: "assistant", text: d.text }]);
          setStreaming(""); // finalized into items (or empty tool-only turn)
          break;
        case "tool_proposed":
          if (d.name === "todo_write" && d.arguments?.items) setTodo(d.arguments.items);
          setItems((p) => [
            ...p,
            { kind: "tool", id: newId(), name: d.name, args: d.arguments, status: "…" },
          ]);
          break;
        case "permission_required":
          setItems((p) => [
            ...p,
            { kind: "approval", name: d.name, args: d.arguments, reason: d.reason, category: d.category },
          ]);
          break;
        case "tool_finished":
          setItems((p) => updateLastTool(p, d.name, d.status, d.result_preview || d.reason));
          if (String(d.name || "").startsWith("browser_")) setBrowserRefreshKey((k) => k + 1);
          break;
        case "turn_end":
          if (d.status === "max_iterations_exceeded")
            setItems((p) => [...p, { kind: "notice", tone: "warn", text: "Stopped: max iterations reached." }]);
          break;
        case "interrupted":
          setItems((p) => [...p, { kind: "notice", tone: "warn", text: "Interrupted." }]);
          break;
        case "error":
          setItems((p) => [...p, { kind: "notice", tone: "warn", text: "Error: " + (d.error || "unknown") }]);
          break;
        case "turn_done":
          setRunning(false);
          refreshSessions();
          // Finalize a manual run after its first turn completes (mark it ok in history).
          {
            const ar = activeRunRef.current;
            if (ar && ar.sessionId === sessionId) {
              activeRunRef.current = null;
              finalizeAutomationRun(ar.taskId, ar.runId).catch(() => {});
            }
          }
          break;
      }
    };

    const session = new Session(sessionId, workspace || "", agent, {
      onEvent: handleEvent,
      onOpen: () => {
        setConnected(true);
        // Auto-send the task prompt once a "Run now" session connects.
        const p = pendingPromptRef.current;
        if (p) {
          pendingPromptRef.current = null;
          setItems((prev) => [...prev, { kind: "user", text: p }]);
          sessionRef.current?.userMessage(p);
        }
      },
      onClose: () => setConnected(false),
    });
    sessionRef.current = session;
    return () => session.close();
  }, [sessionId, workspace, agent, refreshSessions]);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [items, streaming]);

  const send = (text: string, attachments?: Attachment[]) => {
    setItems((p) => [...p, { kind: "user", text, attachments }]);
    sessionRef.current?.userMessage(text, attachments);
  };
  const approve = (decision: ApprovalDecision) => {
    setItems((p) => resolveLastApproval(p, decision));
    sessionRef.current?.approve(decision);
  };
  const interrupt = () => sessionRef.current?.interrupt();
  const changeMode = (m: string) => {
    setMode(m);
    sessionRef.current?.setMode(m);
  };
  const changeModel = (m: string) => {
    setModel(m);
    sessionRef.current?.setModel(m);
  };

  const startNewSession = () => {
    setItems([]);
    setStreaming("");
    setTodo([]);
    setSessionId(newId());
  };
  const selectSession = async (id: string, ws: string, ag: string) => {
    setTodo([]);
    setStreaming("");
    setRunning(false);
    if (ag) setAgent(ag);
    if (!needsWorkspace(ag)) setShowGate(false);
    if (ws && ws !== workspace) {
      setWorkspace(ws); // switch project to the session's folder
      setBranch(null);
    }
    setSessionId(id);
    try {
      const messages = await getSessionMessages(id);
      setItems(itemsFromMessages(messages));
    } catch {
      setItems([]);
    }
  };
  const switchAgent = async (name: string) => {
    setSurface("session");
    if (name === agent) return;
    rememberLastSession(agent, sessionId, workspace);
    const knownSessions = sessions.length ? sessions : await getSessions().catch(() => []);
    const knownProjects = projects.length ? projects : await getRecentWorkspaces().catch(() => []);
    const target = resumeTargetForAgent(name, knownSessions);

    setAgent(name);
    setItems([]);
    setStreaming("");
    setTodo([]);
    setRunning(false);

    if (target) {
      const targetWorkspace = needsWorkspace(name) ? target.workspace || fallbackWorkspace(workspace, knownProjects) : "";
      if (targetWorkspace && targetWorkspace !== workspace) {
        setWorkspace(targetWorkspace);
        setBranch(null);
      }
      if (!needsWorkspace(name)) setShowGate(false);
      else if (targetWorkspace) setShowGate(false);
      else setShowGate(true);
      setSessionId(target.sessionId);
      try {
        setItems(itemsFromMessages(await getSessionMessages(target.sessionId)));
      } catch {
        setItems([]);
      }
      return;
    }

    const id = newId();
    const fallback = needsWorkspace(name) ? fallbackWorkspace(workspace, knownProjects) : "";
    if (fallback && fallback !== workspace) {
      setWorkspace(fallback);
      setBranch(null);
    }
    setSessionId(id);
    rememberLastSession(name, id, fallback);
    if (!needsWorkspace(name)) setShowGate(false);
    else setShowGate(!fallback);
  };
  const chooseWorkspace = (path: string, b?: string | null) => {
    setWorkspace(path);
    setBranch(b ?? null);
    setShowGate(false);
    setGateCreate(false);
    setItems([]);
    setStreaming("");
    setTodo([]);
    setSessionId(newId());
    getRecentWorkspaces().then(setProjects).catch(() => {});
  };
  const newProject = () => {
    setGateCreate(true);
    setShowGate(true);
  };
  const renameConversation = async (id: string, title: string) => {
    const res = await renameSession(id, title);
    if (res.ok) refreshSessions();
  };
  const deleteConversation = async (id: string) => {
    const res = await deleteSession(id);
    if (!res.ok) return;
    refreshSessions();
    if (id === sessionId) {
      setItems([]);
      setStreaming("");
      setTodo([]);
      setRunning(false);
      setSessionId(newId());
    }
  };

  // "Run now": prepare a manual run, open its session, and auto-send the task so the agent
  // runs LIVE in the main view; finalize it in history once the first turn finishes.
  const openRunSession = (sessionId: string, ws: string, ag: string) => {
    setSurface("session");
    setShowGate(false);
    selectSession(sessionId, ws, ag);
  };
  const runTaskNow = async (taskId: string) => {
    const r = await runAutomation(taskId);
    if (!r || !r.ok) return;
    pendingPromptRef.current = r.prompt;
    activeRunRef.current = { taskId, runId: r.run_id, sessionId: r.session_id };
    openRunSession(r.session_id, r.workspace, r.agent);
  };

  const idle = items.length === 0 && !streaming;
  const pendingApproval = [...items].reverse().find((i) => i.kind === "approval" && !i.resolved);
  const activeTitle = sessions.find((s) => s.session_id === sessionId)?.title || "New chat";

  const desktop = isTauri();
  const beginWindowDrag = (event: PointerEvent) => {
    if (!desktop || event.button !== 0) return;
    startWindowDrag();
  };

  if (booting) {
    return (
      <div className={"app boot-splash" + (desktop ? " tauri-overlay" : "")}>
        {desktop && (
          <div className="titlebar-drag" data-tauri-drag-region>
            <span className="titlebar-brand">
              <Icon name="sparkle" size={13} className="mark" /> coworker
            </span>
          </div>
        )}
        <div className="boot-mark">✳</div>
        <div className="boot-text">Starting coworker…</div>
      </div>
    );
  }

  return (
    <div className={"app" + (desktop ? " tauri-overlay" : "")}>
      {onboarding && (
        <Onboarding
          onDone={() => {
            setOnboarding(false);
            getHealth().then((h) => setModel(h.model)).catch(() => {});
            getSuperagent().then((s) => s?.name && setHelperName(s.name)).catch(() => {});
          }}
        />
      )}
      <Sidebar
        agent={agent}
        workspace={workspace || ""}
        model={model}
        mode={mode}
        surfaces={surfaces}
        sessions={sessions}
        projects={projects}
        activeSession={sessionId}
        onSwitchAgent={switchAgent}
        onNewSession={startNewSession}
        onSelectSession={selectSession}
        onNewProject={newProject}
        onRenameSession={renameConversation}
        onDeleteSession={deleteConversation}
        onManage={() => setShowManage(true)}
        onOpenSuperagent={() => setSurface("superagent")}
        onOpenScheduled={() => setSurface("scheduled")}
        onOpenIntegrations={() => setSurface("integrations")}
        onOpenAudit={() => setSurface("audit")}
        superagentActive={surface === "superagent"}
        scheduledActive={surface === "scheduled"}
        integrationsActive={surface === "integrations"}
        auditActive={surface === "audit"}
        helperName={helperName}
      />
      {surface === "superagent" ? (
        <SuperAgentView />
      ) : surface === "scheduled" ? (
        <ScheduledView onOpenRun={openRunSession} onRunNow={runTaskNow} />
      ) : surface === "integrations" ? (
        <IntegrationsView />
      ) : surface === "audit" ? (
        <AuditView />
      ) : (
      <div className={"main" + (surface === "session" && agent === "cowork" && !railHidden ? " rail-open" : "")}>
        <div className="main-topbar">
          <div className="main-title" onPointerDown={beginWindowDrag}>
            <span>{activeTitle}</span>
            <button
              className="title-menu-btn"
              onMouseDown={(e) => e.stopPropagation()}
              onClick={() => setTopbarMenuOpen((open) => !open)}
              aria-label="Conversation options"
              title="Conversation options"
            >
              <Icon name="moreHorizontal" size={16} />
            </button>
            {topbarMenuOpen && (
              <div className="title-menu" onMouseDown={(e) => e.stopPropagation()}>
                <button disabled>
                  <Icon name="pin" size={15} />
                  <span>Pin chat</span>
                </button>
                <button
                  onClick={() => {
                    setTopbarMenuOpen(false);
                    const next = window.prompt("Rename conversation", activeTitle);
                    if (next && next.trim() && next.trim() !== activeTitle) renameConversation(sessionId, next.trim());
                  }}
                >
                  <Icon name="pencil" size={15} />
                  <span>Rename chat</span>
                </button>
                <button disabled>
                  <Icon name="archive" size={15} />
                  <span>Archive chat</span>
                </button>
              </div>
            )}
          </div>
          <div className="main-drag-fill" onPointerDown={beginWindowDrag} />
          <div className="main-topbar-actions">
            {agent === "cowork" && (
              <button
                className="topbar-icon-btn"
                onMouseDown={(e) => e.stopPropagation()}
                onClick={() => setRailHidden((h) => !h)}
                aria-label={railHidden ? "Show side panel" : "Hide side panel"}
                title={railHidden ? "Show side panel" : "Hide side panel"}
              >
                <Icon name={railHidden ? "panelOpen" : "panelClose"} size={16} />
              </button>
            )}
          </div>
        </div>
        <div className={"main-workspace" + (railHidden ? " rail-hidden" : "")}>
          <div className="main-chat">
            <div className="main-scroll" ref={scrollRef}>
              {idle ? (
                <div className="hero">
                  <h1 className="greeting">
                    <span className="mark">✳</span>
                    {agent === "chat"
                      ? "How can I help?"
                      : agent === "cowork"
                        ? "What should we produce?"
                        : "Let's build something."}
                  </h1>
                  {needsWorkspace(agent) && (
                    <div className="suggestions">
                      <div className="suggest-head">Try a task</div>
                      {(agent === "cowork" ? COWORK_SUGGESTIONS : SUGGESTIONS).map((s, i) => (
                        <div className="suggest" key={i} onClick={() => workspace && send(s.text)}>
                          <span className="ico">{s.ico}</span>
                          {s.text}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              ) : (
                <>
                  <Transcript items={items} onApprove={approve} />
                  {running && !streaming && !lastItemIsAssistant(items) && <WaitingForAgent />}
                  {streaming && (
                    <div className="transcript">
                      <div className="bubble-assistant">
                        <div className="who">assistant</div>
                        {streaming}
                        <span className="stream-cursor">▍</span>
                      </div>
                    </div>
                  )}
                </>
              )}
            </div>

            <Composer
              mode={mode}
              model={model}
              models={models}
              running={running}
              connected={connected}
              onSend={send}
              onInterrupt={interrupt}
              onModeChange={changeMode}
              onModelChange={changeModel}
              workspace={needsWorkspace(agent) ? workspace || "" : undefined}
              branch={branch}
              onPickWorkspace={() => setShowGate(true)}
              approvalSlot={
                pendingApproval?.kind === "approval" ? (
                  <ApprovalCard item={pendingApproval} onApprove={approve} compact />
                ) : undefined
              }
            />
                  </div>
          <RightRail
            active={surface === "session" && agent === "cowork" && !railHidden}
            sessionId={sessionId}
            refreshKey={browserRefreshKey}
            toolNames={items.filter((i) => i.kind === "tool").map((i: any) => i.name)}
            todo={todo}
            running={running}
            onHide={() => setRailHidden(true)}
          />
        </div>
      </div>
      )}

      {showGate && surface === "session" && needsWorkspace(agent) && (
        <FolderGate
          create={gateCreate}
          onChoose={chooseWorkspace}
          onChat={() => switchAgent("chat")}
          onCancel={
            workspace
              ? () => {
                  setShowGate(false);
                  setGateCreate(false);
                }
              : undefined
          }
        />
      )}

      {showManage && (
        <ManageModal
          initialTab={manageTab}
          onClose={() => {
            setShowManage(false);
            setManageTab(undefined);
            loadSettings(); // pick up any Ollama model/URL or surface visibility just changed
          }}
        />
      )}
    </div>
  );
}

function itemsFromMessages(messages: any[]): Item[] {
  const items: Item[] = [];
  // Index tool results by tool_call_id so replayed tool rows can show their output
  // (the live view gets this from `tool_finished` events; on replay it's the `role:"tool"` msgs).
  const results: Record<string, string> = {};
  for (const m of messages || []) {
    if (m.role === "tool" && m.tool_call_id) {
      results[m.tool_call_id] =
        typeof m.content === "string" ? m.content : JSON.stringify(m.content);
    }
  }
  for (const m of messages || []) {
    if (m.role === "user") {
      const user = userItemFromContent(m.content);
      if (user.text || user.attachments?.length) items.push(user);
    } else if (m.role === "assistant") {
      if (m.content) items.push({ kind: "assistant", text: m.content });
      for (const tc of m.tool_calls || []) {
        let args: any = {};
        try {
          args = JSON.parse(tc.function?.arguments || "{}");
        } catch {
          args = {};
        }
        const preview = results[tc.id];
        items.push({ kind: "tool", id: tc.id, name: tc.function?.name, args, status: "ok", preview });
      }
    }
    // system messages are omitted; tool-result messages are folded into the tool row above
  }
  return items;
}

function userItemFromContent(content: any): Extract<Item, { kind: "user" }> {
  if (typeof content === "string") return { kind: "user", text: content };
  if (!Array.isArray(content)) return { kind: "user", text: "" };

  const text: string[] = [];
  const attachments: Attachment[] = [];
  for (const part of content) {
    if (!part || typeof part !== "object") continue;
    if (part.type === "text" && part.text) {
      text.push(String(part.text));
    } else if (part.type === "image_url") {
      const url = part.image_url?.url;
      if (typeof url === "string" && url.startsWith("data:image/")) {
        attachments.push({ kind: "image", name: "image", data_url: url });
      }
    }
  }
  return { kind: "user", text: text.join("\n\n"), attachments };
}

function lastItemIsAssistant(items: Item[]): boolean {
  for (let i = items.length - 1; i >= 0; i--) {
    const item = items[i];
    if (item.kind === "notice") continue;
    return item.kind === "assistant";
  }
  return false;
}

function WaitingForAgent() {
  return (
    <div className="waiting-transcript">
      <div className="waiting-row" aria-live="polite">
        <span className="waiting-spinner" />
        <span>Waiting for agent...</span>
      </div>
    </div>
  );
}

function updateLastTool(items: Item[], name: string, status: string, preview?: string): Item[] {
  const copy = [...items];
  for (let i = copy.length - 1; i >= 0; i--) {
    const it = copy[i];
    if (it.kind === "tool" && it.name === name && it.status === "…") {
      copy[i] = { ...it, status, preview };
      break;
    }
  }
  return copy;
}

function resolveLastApproval(items: Item[], decision: ApprovalDecision): Item[] {
  const copy = [...items];
  for (let i = copy.length - 1; i >= 0; i--) {
    const it = copy[i];
    if (it.kind === "approval" && !it.resolved) {
      copy[i] = { ...it, resolved: decision };
      break;
    }
  }
  return copy;
}
