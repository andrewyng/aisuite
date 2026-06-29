import { useCallback, useEffect, useRef, useState, type PointerEvent } from "react";
import {
  finalizeAutomationRun,
  getArtifacts,
  getHealth,
  getRecentWorkspaces,
  getSessionMessages,
  getSessions,
  getSettings,
  getPersonas,
  getInbox,
  getUnattended,
  resolveInboxItem,
  deleteSession,
  renameSession,
  runAutomation,
  setSessionFlags,
  Session,
  type InboxItem,
  type MessageSource,
  type Persona,
  type RecentWorkspace,
  type SurfaceVisibility,
} from "./api";
import type { ApprovalDecision, Attachment, Item, SessionInfo, TodoItem, WsEvent } from "./types";
import { itemsFromMessages } from "./itemsFromMessages";
import { InboxItemCard } from "./components/InboxItemCard";
import { isTauri, startWindowDrag } from "./tauri";
import { Icon } from "./components/Icon";
import { Sidebar } from "./components/Sidebar";
import { Transcript } from "./components/Transcript";
import { Composer } from "./components/Composer";
import { UnattendedToggle } from "./components/UnattendedToggle";
import { SubscriptionsChip } from "./components/SubscriptionsChip";
import { Markdown } from "./components/Markdown";
import { RootsBar } from "./components/RootsBar";
import { SessionIntro } from "./components/SessionIntro";
import { FolderGate } from "./components/FolderGate";
import { ManageModal } from "./components/ManageModal";
import { Onboarding } from "./components/Onboarding";
import { ScheduledView } from "./components/ScheduledView";
import { RightRail } from "./components/RightRail";
import { IntegrationsView } from "./components/IntegrationsView";
import { PersonaView } from "./components/PersonaView";
import { SourcesBar } from "./components/SourcesBar";
import { AuditView } from "./components/AuditView";
import { InboxView } from "./components/InboxView";
import { ApprovalCard } from "./components/ApprovalCard";
import { DirectoryRequestCard } from "./components/DirectoryRequestCard";
import { PlanCard } from "./components/PlanCard";

const newId = () =>
  (crypto as any).randomUUID ? crypto.randomUUID().slice(0, 12) : Math.random().toString(36).slice(2, 14);

const SUGGESTIONS = [
  { ico: "⚙", text: "Run the test suite and summarize any failures." },
  { ico: "✦", text: "Read the project and give me a 5-bullet overview." },
  { ico: "↻", text: "Find and fix the failing build." },
];

// Tools whose success means a new/changed file should show up under Artifacts right away.
const FILE_WRITE_TOOLS = new Set(["write_file", "apply_patch", "apply_unified_diff", "replace_in_file"]);

// Models sometimes pass todo items as bare strings instead of {content, status} objects (the
// backend tool normalizes them the same way; the GUI reads the raw proposal args, so mirror it).
function normalizeTodos(raw: unknown): TodoItem[] {
  if (!Array.isArray(raw)) return [];
  const statuses = new Set(["pending", "in_progress", "done"]);
  return raw.map((entry: any) => {
    if (entry && typeof entry === "object") {
      const status = entry.status === "completed" ? "done" : entry.status; // common model alias
      return {
        content: String(entry.content ?? ""),
        status: statuses.has(status) ? status : "pending",
      };
    }
    return { content: String(entry ?? ""), status: "pending" as const };
  });
}

// Fallbacks used only before the persona list loads (the in-component, family-aware
// needsWorkspace/gatesWorkspace consult the real persona once available).
const needsWorkspaceFallback = (a: string) => a === "code" || a === "cowork";
const gatesWorkspaceFallback = (a: string) => a === "code";
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
  const [manageTab, setManageTab] = useState<"settings" | "models" | undefined>(undefined);
  // Whether the default model's provider is actually configured (any provider). Drives the
  // composer's "No model connected" chip. Default true so we don't flash the chip before settings
  // load; corrected by loadSettings.
  const [modelReady, setModelReady] = useState(true);
  const [surface, setSurface] = useState<
    "session" | "scheduled" | "integrations" | "audit" | "inbox" | "persona"
  >("session");
  // The persona whose detail page is showing (surface === "persona"); empty falls back to the
  // active session's persona. Phase 5 wires the grouped-nav gear + "Manage personas…" entry points.
  const [personaViewId, setPersonaViewId] = useState<string>("");
  const [browserRefreshKey, setBrowserRefreshKey] = useState(0);
  const [railHidden, setRailHidden] = useState(false);
  // Count of files this Cowork conversation has produced — surfaces an "Artifacts (N)" button in
  // the topbar when the side panel is hidden, so produced files are never buried.
  const [artifactCount, setArtifactCount] = useState(0);
  const [topbarMenuOpen, setTopbarMenuOpen] = useState(false);
  // Inline rename in the topbar (window.prompt is a no-op in the desktop webview).
  const [renamingTitle, setRenamingTitle] = useState(false);
  const [renameDraft, setRenameDraft] = useState("");
  // A pending composer prefill (text + attachments) pushed from the session start panel.
  const [composerPrefill, setComposerPrefill] = useState<{ text: string; attachments?: Attachment[]; nonce: number }>();

  // Persona metadata drives workspace behavior by FAMILY, not by hardcoded id (so a DevOps/SecOps
  // code-family persona gates a folder like Code, and a knowledge persona starts orphan like Cowork).
  const [personas, setPersonas] = useState<Persona[] | null>(null);
  useEffect(() => {
    getPersonas().then(setPersonas).catch(() => {});
  }, []);
  const personaOf = (a: string) => personas?.find((p) => p.id === a);

  // Pending Inbox items for the ACTIVE session — surfaced inline above the composer so an
  // unattended session's blocking question/approval can be answered in context (resolving the
  // same item the Inbox shows; first responder wins).
  const [sessionInbox, setSessionInbox] = useState<InboxItem[]>([]);
  // Whether the active session is Unattended — when true, the agent's prompts route to the Inbox,
  // so we suppress the inline live cards (the Inbox / answer-in-context path shows them instead).
  // A ref too, because the WS event handler closes over stale state.
  const [unattended, setUnattendedState] = useState(false);
  const unattendedRef = useRef(false);
  const markUnattended = useCallback((on: boolean) => {
    unattendedRef.current = on;
    setUnattendedState(on);
  }, []);
  const resolveSessionInbox = async (id: string, resolution: string) => {
    await resolveInboxItem(id, resolution);
    getInbox(sessionId, "pending").then(setSessionInbox).catch(() => setSessionInbox([]));
    refreshSessions(); // attention badge should drop right away
  };
  // Shows a working-area chip / project grouping. Persona's needs_workspace; fallback before load.
  const needsWorkspace = (a: string) => personaOf(a)?.needs_workspace ?? needsWorkspaceFallback(a);
  // MUST pick a folder before starting (git-bound code family). Knowledge starts orphan — the
  // server auto-provisions a per-conversation scratch dir and reports it in the `ready` event.
  const gatesWorkspace = (a: string) => {
    const p = personaOf(a);
    return p ? p.family === "code" : gatesWorkspaceFallback(a);
  };

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
  // True once we've resumed a prior conversation on boot (drives the splash wording).
  const [resumedExisting, setResumedExisting] = useState(false);
  // Latched: keep the boot splash up until the restored session is actually CONNECTED (not just
  // until `booting` clears), so an early click can't land on a session that's still settling.
  const [uiReady, setUiReady] = useState(false);

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
        setResumedExisting(true);
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
      // Only auto-adopt a recent folder for gated surfaces (Code). Cowork starts orphan.
      if (gatesWorkspace(agent)) {
        const ws = recents.find((w) => w.exists) || recents[0];
        if (ws) {
          setWorkspace(ws.path);
          setShowGate(false);
          return;
        }
      }
    } catch {
      /* fall through */
    }
    setShowGate(gatesWorkspace(agent)); // only Code forces a first-run folder gate
  };

  useEffect(() => {
    let cancelled = false;
    const attempt = (tries: number) => {
      getHealth()
        .then(async (h) => {
          if (cancelled) return;
          setModel(h.model);
          // First-run setup wizard (desktop): show until the user completes/dismisses it.
          if (isTauri()) {
            getSettings()
              .then((s) => !cancelled && !s.onboarded && setOnboarding(true))
              .catch(() => {});
          }
          // Settle the active session BEFORE clearing `booting` (which unblocks the connection
          // effect). resumeLastOrGate is async — if we cleared `booting` first, the throwaway
          // initial sessionId would connect against an empty/stale workspace and the server
          // would provision a junk per-conversation scratch dir for it before resume could
          // flip to the real session. Cowork ignores default_workspace (a Code concept).
          if (h.default_workspace && gatesWorkspace(agent)) setWorkspace(h.default_workspace);
          else await resumeLastOrGate();
          if (!cancelled) setBooting(false);
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

  // Reveal the UI once boot has settled AND the restored session is connected (or we're showing
  // the folder gate). Latched, so later reconnects never flash the splash again.
  useEffect(() => {
    if (uiReady || booting) return;
    if (connected || showGate) setUiReady(true);
  }, [uiReady, booting, connected, showGate]);
  // Safety net: if the restored session never reports connected (backend slow/unreachable), reveal
  // the UI anyway. Boot already passed the health check, so a live connect is sub-second; this only
  // bites in the failure case, so keep it short.
  useEffect(() => {
    if (uiReady || booting) return;
    const t = setTimeout(() => setUiReady(true), 1500);
    return () => clearTimeout(t);
  }, [uiReady, booting]);

  const loadSettings = () =>
    getSettings()
      .then((s) => {
        setModels(s.models || []);
        setModelReady(s.model_ready);
        if (s.surfaces) setSurfaces(s.surfaces);
      })
      .catch(() => {});

  // Open Settings → Configure Models (from the composer's "No model connected" chip).
  const openModelSetup = () => {
    setManageTab("models");
    setShowManage(true);
  };

  useEffect(() => {
    refreshSessions();
    loadSettings(); // selectable models + which session surfaces are visible
  }, [refreshSessions]);

  // Poll the session list so the attention/liveness badges stay live and sessions created
  // out-of-band (unattended work, messaging, automations) appear without a manual refresh.
  useEffect(() => {
    const t = setInterval(refreshSessions, 5000);
    return () => clearInterval(t);
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
    if (booting) return; // wait until boot/resume settles the session before connecting
    if (gatesWorkspace(agent) && !workspace) return; // Code needs a folder (gate handles it)
    const handleEvent = (ev: WsEvent) => {
      const d = ev.data || {};
      switch (ev.type) {
        case "ready":
          setConnected(true);
          if (d.model) setModel(d.model);
          if (d.mode) setMode(d.mode);
          // Cowork: adopt the server-provisioned scratch dir (only when we don't already have one).
          if (d.workspace) setWorkspace((cur) => cur || d.workspace);
          break;
        case "turn_start":
          setRunning(true);
          setStreaming("");
          // Background-delivered turns (channel message, self-wake, durable resume) have no local
          // send(), so the triggering message isn't in `items` yet — surface it. A connector message
          // carries a structured `source` (§3.1) → render the rich card; otherwise a plain user item.
          // Foreground turns already appended it in send(); skip the duplicate.
          if (d.source?.connector) {
            const src = d.source as MessageSource;
            setItems((p) => {
              const last = p[p.length - 1];
              return last && last.kind === "connector" && last.source.ts === src.ts && last.source.text === src.text
                ? p
                : [...p, { kind: "connector", source: src }];
            });
          } else if (typeof d.input === "string" && d.input) {
            setItems((p) => {
              const last = p[p.length - 1];
              return last && last.kind === "user" && last.text === d.input
                ? p
                : [...p, { kind: "user", text: d.input as string }];
            });
          }
          break;
        case "assistant_delta":
          setStreaming((s) => s + (d.text || ""));
          break;
        case "assistant_message":
          if (d.text) setItems((p) => [...p, { kind: "assistant", text: d.text }]);
          setStreaming(""); // finalized into items (or empty tool-only turn)
          break;
        case "tool_proposed":
          if (d.name === "todo_write" && d.arguments?.items) setTodo(normalizeTodos(d.arguments.items));
          setItems((p) => [
            ...p,
            { kind: "tool", id: newId(), name: d.name, args: d.arguments, status: "…" },
          ]);
          break;
        case "permission_required":
          // Unattended → the backend parked it in the Inbox; don't also surface a live card.
          if (unattendedRef.current) break;
          setItems((p) => [
            ...p,
            { kind: "approval", name: d.name, args: d.arguments, reason: d.reason, category: d.category },
          ]);
          break;
        case "directory_requested":
          if (unattendedRef.current) break;
          setItems((p) => [
            ...p,
            { kind: "dirreq", reason: d.reason || "", path: d.path || "", writable: !!d.writable },
          ]);
          break;
        case "plan_proposed":
          if (unattendedRef.current) break;
          setItems((p) => [...p, { kind: "planreq", plan: d.plan || "" }]);
          break;
        case "question_requested":
          // ask_user in an attended session — answered inline (not routed to the Inbox).
          setItems((p) => [
            ...p,
            {
              kind: "question",
              question: d.question || "",
              options: d.options || [],
              allow_text: d.allow_text !== false,
              multi: !!d.multi,
            },
          ]);
          break;
        case "tool_finished":
          setItems((p) => updateLastTool(p, d.name, d.status, d.result_preview || d.reason));
          // Refresh the right rail when something it shows may have changed: browser state, or a
          // file write that should appear under Artifacts immediately (not only after the turn).
          if (String(d.name || "").startsWith("browser_") || FILE_WRITE_TOOLS.has(d.name)) {
            setBrowserRefreshKey((k) => k + 1);
          }
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
          // Catch-all artifact refresh: files created via shell or on a brand-new session (whose
          // record only exists after the first save) appear once the turn completes.
          setBrowserRefreshKey((k) => k + 1);
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
    // NOTE: `workspace` is intentionally NOT a dependency. Every real workspace change
    // (pick folder, select/switch session, new session) is paired with a `sessionId`
    // change, so the socket still reconnects when it should. The one workspace-only change
    // is the `ready` handler adopting the server's provisioned Cowork scratch dir — listing
    // `workspace` here made that adoption tear down and rebuild the socket immediately after
    // first connect, dropping the user's first message (the "send twice" bug). The scratch
    // dir is deterministic from `sessionId` server-side, so skipping that reconnect is safe.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [booting, sessionId, agent, refreshSessions]);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [items, streaming]);

  // Track produced-file count for the topbar "Artifacts" affordance (works even when the rail is
  // hidden, where the rail itself doesn't fetch). Cowork only; refreshes on file writes/turn end.
  useEffect(() => {
    if (agent !== "cowork" || surface !== "session") {
      setArtifactCount(0);
      return;
    }
    getArtifacts(sessionId).then((a) => setArtifactCount(a.length)).catch(() => {});
  }, [agent, surface, sessionId, browserRefreshKey]);

  // Keep the active session's pending Inbox items fresh (answer-in-context card). Loads on session
  // change + after each turn, plus a slow poll so an unattended agent's new question surfaces.
  useEffect(() => {
    if (surface !== "session") return;
    const load = () => {
      getInbox(sessionId, "pending").then(setSessionInbox).catch(() => setSessionInbox([]));
      getUnattended(sessionId).then(markUnattended).catch(() => markUnattended(false));
    };
    load();
    const t = setInterval(load, 4000);
    return () => clearInterval(t);
  }, [surface, sessionId, browserRefreshKey, markUnattended]);

  const send = (text: string, attachments?: Attachment[]) => {
    setItems((p) => [...p, { kind: "user", text, attachments }]);
    sessionRef.current?.userMessage(text, attachments);
  };
  const approve = (decision: ApprovalDecision) => {
    setItems((p) => resolveLastApproval(p, decision));
    sessionRef.current?.approve(decision);
  };
  const respondPlan = (approved: boolean, mode?: string, feedback?: string) => {
    setItems((p) => resolveLastPlan(p, approved ? "approved" : "rejected"));
    sessionRef.current?.respondPlan(approved, mode, feedback);
    if (approved && mode) setMode(mode); // the server flips the live engine to this mode
  };
  const respondDirectory = (granted: boolean, path?: string, writable?: boolean) => {
    setItems((p) => resolveLastDirReq(p, granted ? "granted" : "denied"));
    sessionRef.current?.respondDirectory(granted, path, writable);
  };
  const answerQuestion = (answer: string) => {
    setItems((p) => resolveLastQuestion(p, answer));
    sessionRef.current?.respondQuestion(answer);
  };
  const prefillComposer = (text: string, attachments?: Attachment[]) =>
    setComposerPrefill((p) => ({ text, attachments, nonce: (p?.nonce ?? 0) + 1 }));
  const interrupt = () => sessionRef.current?.interrupt();
  const changeMode = (m: string) => {
    setMode(m);
    sessionRef.current?.setMode(m);
  };
  const changeModel = (m: string) => {
    setModel(m);
    sessionRef.current?.setModel(m);
  };

  const startNewSession = (forAgent?: string) => {
    const target = forAgent || agent;
    setSurface("session"); // return to the conversation view if we were on a sub-view
    setItems([]);
    setStreaming("");
    setTodo([]);
    setRunning(false);
    // "New session" under a browsed persona switches to it (expand≠switch: the header alone
    // doesn't switch; this explicit action does).
    if (target !== agent) {
      setAgent(target);
      if (gatesWorkspace(target)) setShowGate(true);
      else setShowGate(false);
    }
    // Knowledge family: a new conversation starts fresh (orphan) — clear the workspace so the
    // server provisions a NEW scratch dir for the new session id. Code keeps its repo.
    if (!gatesWorkspace(target)) setWorkspace(null);
    setSessionId(newId());
  };
  // Inbox → session: the item carries its session's workspace/agent, so open it directly.
  const openSessionFromInbox = (sid: string, ws: string, ag: string) => selectSession(sid, ws, ag);
  const selectSession = async (id: string, ws: string, ag: string) => {
    setSurface("session"); // selecting a conversation always returns to the conversation view
    setTodo([]);
    setStreaming("");
    setRunning(false);
    if (ag) setAgent(ag);
    if (!gatesWorkspace(ag)) setShowGate(false);
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
      // Code falls back to a recent folder; Cowork resumes its scratch (target.workspace) or
      // starts orphan ("" → server provisions). Chat has no workspace.
      const targetWorkspace = gatesWorkspace(name)
        ? target.workspace || fallbackWorkspace(workspace, knownProjects)
        : needsWorkspace(name)
          ? target.workspace || ""
          : "";
      if (targetWorkspace && targetWorkspace !== workspace) {
        setWorkspace(targetWorkspace);
        setBranch(null);
      } else if (!targetWorkspace) {
        setWorkspace(null); // orphan cowork: clear so the next `ready` adopts a fresh scratch
      }
      if (!gatesWorkspace(name)) setShowGate(false);
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
    const fallback = gatesWorkspace(name) ? fallbackWorkspace(workspace, knownProjects) : "";
    if (fallback && fallback !== workspace) {
      setWorkspace(fallback);
      setBranch(null);
    } else if (!fallback && needsWorkspace(name)) {
      setWorkspace(null); // orphan cowork: server provisions a fresh scratch on connect
    }
    setSessionId(id);
    rememberLastSession(name, id, fallback);
    if (!gatesWorkspace(name)) setShowGate(false);
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
  const togglePinned = async (id: string, pinned: boolean) => {
    await setSessionFlags(id, { pinned });
    refreshSessions();
  };
  const toggleArchived = async (id: string, archived: boolean) => {
    await setSessionFlags(id, { archived });
    refreshSessions();
    // Archiving the open chat: leave it and start fresh (it moves to the Archived section).
    if (archived && id === sessionId) {
      setItems([]);
      setStreaming("");
      setTodo([]);
      setRunning(false);
      setSessionId(newId());
    }
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
  const pendingDirReq = [...items].reverse().find((i) => i.kind === "dirreq" && !i.resolved);
  const pendingPlan = [...items].reverse().find((i) => i.kind === "planreq" && !i.resolved);
  const pendingQuestion = [...items].reverse().find((i) => i.kind === "question" && !i.resolved);
  const activeInfo = sessions.find((s) => s.session_id === sessionId);
  const activeTitle = activeInfo?.title || "New chat";
  const commitTitleRename = () => {
    const next = renameDraft.trim();
    if (next && next !== activeTitle) renameConversation(sessionId, next);
    setRenamingTitle(false);
  };

  const desktop = isTauri();
  const beginWindowDrag = (event: PointerEvent) => {
    if (!desktop || event.button !== 0) return;
    startWindowDrag();
  };

  if (booting || !uiReady) {
    return (
      <div className={"app boot-splash" + (desktop ? " tauri-overlay" : "")}>
        {desktop && (
          <div className="titlebar-drag" data-tauri-drag-region>
            <span className="titlebar-brand">
              <Icon name="sparkle" size={13} className="mark" /> OpenCoworker
            </span>
          </div>
        )}
        <div className="boot-mark">✳</div>
        <div className="boot-text">{resumedExisting ? "Restoring your session…" : "Starting coworker…"}</div>
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
            loadSettings(); // pick up a model connected during setup (clears the composer chip)
          }}
        />
      )}
      <Sidebar
        agent={agent}
        workspace={workspace || ""}
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
        onTogglePin={togglePinned}
        onManage={() => setShowManage(true)}
        onOpenScheduled={() => setSurface("scheduled")}
        onOpenIntegrations={() => setSurface("integrations")}
        onOpenAudit={() => setSurface("audit")}
        onOpenInbox={() => setSurface("inbox")}
        scheduledActive={surface === "scheduled"}
        integrationsActive={surface === "integrations"}
        auditActive={surface === "audit"}
        inboxActive={surface === "inbox"}
      />
      {surface === "scheduled" ? (
        <ScheduledView onOpenRun={openRunSession} onRunNow={runTaskNow} />
      ) : surface === "integrations" ? (
        <IntegrationsView />
      ) : surface === "audit" ? (
        <AuditView />
      ) : surface === "inbox" ? (
        <InboxView onOpenSession={openSessionFromInbox} />
      ) : surface === "persona" ? (
        <PersonaView
          personaId={personaViewId || agent}
          onBack={() => setSurface("session")}
          onOpenIntegrations={() => setSurface("integrations")}
        />
      ) : (
      <div className={"main" + (surface === "session" && agent === "cowork" && !railHidden ? " rail-open" : "")}>
        <div className="main-topbar">
          <div className="main-title" onPointerDown={beginWindowDrag}>
            {renamingTitle ? (
              <input
                className="title-rename"
                value={renameDraft}
                autoFocus
                spellCheck={false}
                onPointerDown={(e) => e.stopPropagation()}
                onChange={(e) => setRenameDraft(e.target.value)}
                onBlur={commitTitleRename}
                onKeyDown={(e) => {
                  if (e.key === "Enter") commitTitleRename();
                  else if (e.key === "Escape") setRenamingTitle(false);
                }}
              />
            ) : (
              <span>{activeTitle}</span>
            )}
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
                <button
                  disabled={!activeInfo}
                  title={activeInfo ? undefined : "Send a message first"}
                  onClick={() => {
                    setTopbarMenuOpen(false);
                    togglePinned(sessionId, !activeInfo?.pinned);
                  }}
                >
                  <Icon name="pin" size={15} />
                  <span>{activeInfo?.pinned ? "Unpin chat" : "Pin chat"}</span>
                </button>
                <button
                  disabled={!activeInfo}
                  title={activeInfo ? undefined : "Send a message first"}
                  onClick={() => {
                    setTopbarMenuOpen(false);
                    setRenameDraft(activeTitle);
                    setRenamingTitle(true);
                  }}
                >
                  <Icon name="pencil" size={15} />
                  <span>Rename chat</span>
                </button>
                <button
                  disabled={!activeInfo}
                  title={activeInfo ? undefined : "Send a message first"}
                  onClick={() => {
                    setTopbarMenuOpen(false);
                    toggleArchived(sessionId, !activeInfo?.archived);
                  }}
                >
                  <Icon name="archive" size={15} />
                  <span>{activeInfo?.archived ? "Unarchive chat" : "Archive chat"}</span>
                </button>
              </div>
            )}
          </div>
          <div className="main-drag-fill" onPointerDown={beginWindowDrag} />
          <div className="main-topbar-actions">
            {agent !== "chat" && (
              <button
                className="topbar-icon-btn"
                onMouseDown={(e) => e.stopPropagation()}
                onClick={() => {
                  setPersonaViewId(agent);
                  setSurface("persona");
                }}
                aria-label="About this persona"
                title="About this persona"
              >
                <Icon name="sliders" size={16} />
              </button>
            )}
            {agent === "cowork" && railHidden && artifactCount > 0 && (
              <button
                className="topbar-artifacts-btn"
                onMouseDown={(e) => e.stopPropagation()}
                onClick={() => setRailHidden(false)}
                title="Show files this conversation produced"
              >
                <Icon name="file" size={14} />
                <span>Artifacts</span>
                <span className="topbar-artifacts-count">{artifactCount}</span>
              </button>
            )}
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
        {agent !== "chat" && (
          <SourcesBar
            sessionId={sessionId}
            personaId={agent}
            onOpenIntegrations={() => setSurface("integrations")}
            onOpenPersona={(id) => {
              setPersonaViewId(id);
              setSurface("persona");
            }}
          />
        )}
        <div className={"main-workspace" + (railHidden ? " rail-hidden" : "")}>
          <div className="main-chat">
            <div className="main-scroll" ref={scrollRef}>
              {idle ? (
                agent === "cowork" ? (
                  <SessionIntro
                    sessionId={sessionId}
                    onOpenIntegrations={() => setSurface("integrations")}
                    onPrefill={prefillComposer}
                  />
                ) : (
                  <div className="hero">
                    <h1 className="greeting">
                      <span className="mark">✳</span>
                      {agent === "chat" ? "How can I help?" : "Let's build something."}
                    </h1>
                    {needsWorkspace(agent) && (
                      <div className="suggestions">
                        <div className="suggest-head">Try a task</div>
                        {SUGGESTIONS.map((s, i) => (
                          <div className="suggest" key={i} onClick={() => workspace && send(s.text)}>
                            <span className="ico">{s.ico}</span>
                            {s.text}
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                )
              ) : (
                <>
                  <Transcript items={items} onApprove={approve} />
                  {running && !streaming && !lastItemIsAssistant(items) && <WaitingForAgent />}
                  {streaming && (
                    <div className="transcript">
                      <div className="bubble-assistant">
                        <div className="who">assistant</div>
                        <Markdown text={streaming} />
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
              modelReady={modelReady}
              onConnectModel={openModelSetup}
              onSend={send}
              onInterrupt={interrupt}
              onModeChange={changeMode}
              onModelChange={changeModel}
              workspace={needsWorkspace(agent) ? workspace || "" : undefined}
              branch={branch}
              onPickWorkspace={() => setShowGate(true)}
              rootsSlot={agent === "cowork" ? <RootsBar sessionId={sessionId} /> : undefined}
              unattendedSlot={
                agent !== "chat" ? (
                  <>
                    <UnattendedToggle sessionId={sessionId} onChange={markUnattended} />
                    <SubscriptionsChip
                      sessionId={sessionId}
                      channels={activeInfo?.subscriptions || []}
                      onChanged={refreshSessions}
                    />
                  </>
                ) : undefined
              }
              prefill={composerPrefill}
              resetKey={sessionId}
              placeholder={
                agent === "code"
                  ? "Ask the coder to build, fix, or explain…  (drop or paste images)"
                  : agent === "chat"
                    ? "Ask anything…  (drop or paste images)"
                    : "Ask the coworker…  (drop or paste images)"
              }
              approvalSlot={
                // Live inline cards are for ATTENDED sessions only; when Unattended the prompt is
                // parked in the Inbox and surfaced via the answer-in-context card below.
                !unattended && pendingPlan?.kind === "planreq" ? (
                  <PlanCard item={pendingPlan} onRespond={respondPlan} />
                ) : !unattended && pendingDirReq?.kind === "dirreq" ? (
                  <DirectoryRequestCard item={pendingDirReq} onRespond={respondDirectory} />
                ) : !unattended && pendingApproval?.kind === "approval" ? (
                  <ApprovalCard item={pendingApproval} onApprove={approve} compact />
                ) : !unattended && pendingQuestion?.kind === "question" ? (
                  // Live ask_user in an attended session — answer inline (reuses the Inbox card UI).
                  <InboxItemCard
                    item={{
                      id: "live-question",
                      session_id: sessionId,
                      kind: "question",
                      title: pendingQuestion.question,
                      body: "",
                      state: "pending",
                      resolution: null,
                      inbox: "default",
                      created_at: "",
                      resolved_at: null,
                      options: pendingQuestion.options,
                      allow_text: pendingQuestion.allow_text,
                      multi: pendingQuestion.multi,
                    }}
                    onResolve={(_id, answer) => answerQuestion(answer)}
                    compact
                  />
                ) : sessionInbox[0] ? (
                  // Unattended session blocked on an Inbox item — answer it in context.
                  <InboxItemCard item={sessionInbox[0]} onResolve={resolveSessionInbox} compact />
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
          />
        </div>
      </div>
      )}

      {showGate && surface === "session" && gatesWorkspace(agent) && (
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

function resolveLastDirReq(items: Item[], resolved: "granted" | "denied"): Item[] {
  const copy = [...items];
  for (let i = copy.length - 1; i >= 0; i--) {
    const it = copy[i];
    if (it.kind === "dirreq" && !it.resolved) {
      copy[i] = { ...it, resolved };
      break;
    }
  }
  return copy;
}

function resolveLastPlan(items: Item[], resolved: "approved" | "rejected"): Item[] {
  const copy = [...items];
  for (let i = copy.length - 1; i >= 0; i--) {
    const it = copy[i];
    if (it.kind === "planreq" && !it.resolved) {
      copy[i] = { ...it, resolved };
      break;
    }
  }
  return copy;
}

function resolveLastQuestion(items: Item[], answer: string): Item[] {
  const copy = [...items];
  for (let i = copy.length - 1; i >= 0; i--) {
    const it = copy[i];
    if (it.kind === "question" && !it.resolved) {
      copy[i] = { ...it, resolved: answer };
      break;
    }
  }
  return copy;
}
