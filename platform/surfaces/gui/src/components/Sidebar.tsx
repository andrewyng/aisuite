import { useEffect, useMemo, useState } from "react";
import {
  getPersonas,
  getSettings,
  PERSONAS_CHANGED,
  setNavLayout,
  type Persona,
  type RecentWorkspace,
  type SurfaceVisibility,
} from "../api";
import type { SessionInfo } from "../types";
import { isProjectScoped, shortPersonaName } from "../personaScope";
import { Icon, type IconName } from "./Icon";
import { PersonaGlyph, personaGlyph } from "./personaIcon";
import { SearchModal } from "./SearchModal";
import { baseName } from "../paths";

// Session surfaces shown as accordions, in display order. The surfaced personas drive this list
// (so third-party / Ops personas appear); the hardcoded set is the fallback before personas load.
const SURFACES: { key: string; label: string; icon: IconName; cls: string }[] = [
  { key: "cowork", label: "Coworker", icon: "diamond", cls: "ico-cowork" },
  { key: "chat", label: "Chat", icon: "chat", cls: "ico-chat" },
  { key: "code", label: "Code", icon: "code", cls: "ico-code" },
];

const surfaceFromPersona = (p: Persona) => ({
  key: p.id,
  label: shortPersonaName(p.name, p.id),
  icon: personaGlyph(p.icon, p.family),
  cls: `ico-${p.icon || "cowork"}`,
});

// Attention = Inbox items awaiting a session (an accent count that bubbles session → persona →
// footer Inbox — all views of the one Inbox queue, never a second list).
function AttnBadge({ n }: { n: number }) {
  if (!n) return null;
  return (
    <span
      className="text-[10px] font-semibold text-ink bg-faint/30 rounded-full px-1.5 leading-[15px] shrink-0"
      title={`${n} awaiting your attention`}
    >
      {n > 99 ? "99+" : n}
    </span>
  );
}

// Liveness = working (in-flight turn) / sleeping (a self-wake is pending). A count-less dot that
// never bubbles — it says "this is alive", not "this needs you".
function LiveDot({ state }: { state?: "working" | "sleeping" | "idle" }) {
  if (state !== "working" && state !== "sleeping") return null;
  return state === "working" ? (
    <span className="w-1.5 h-1.5 rounded-full bg-accent animate-pulse shrink-0" title="Working now" />
  ) : (
    <span
      className="w-1.5 h-1.5 rounded-full bg-faint/60 shrink-0"
      title="Sleeping (will wake itself)"
    />
  );
}

// A subscribed-connector presence dot (right edge of a row). Brand-colorless here — the sidebar
// isn't passed the connector registry — so it reads as a neutral "listening on a channel" dot.
function ConnectorDot({ subs }: { subs?: string[] }) {
  if (!subs || subs.length === 0) return null;
  return (
    <span
      className="w-1.5 h-1.5 rounded-full bg-faint shrink-0"
      data-brand={subs[0]}
      title={subs.join(", ")}
    />
  );
}

interface Props {
  agent: string;
  workspace: string;
  surfaces: SurfaceVisibility;
  sessions: SessionInfo[];
  projects: RecentWorkspace[];
  activeSession: string;
  onSwitchAgent: (agent: string) => void;
  onNewSession: (agent: string) => void;
  onSelectSession: (id: string, workspace: string, agent: string) => void;
  onNewProject: (persona: string) => void;
  onRenameSession: (id: string, title: string) => void;
  onDeleteSession: (id: string) => void;
  onArchiveSession: (id: string, archived: boolean) => void;
  onTogglePin: (id: string, pinned: boolean) => void;
  onManage: () => void;
  // Grouped-nav gear + New-session menu's "Manage personas…" entry points (§7).
  onOpenPersona: (id: string) => void;
  onManagePersonas: () => void;
  onOpenScheduled: () => void;
  onOpenIntegrations: () => void;
  onOpenAudit: () => void;
  onOpenInbox: () => void;
  scheduledActive: boolean;
  integrationsActive: boolean;
  auditActive: boolean;
  inboxActive: boolean;
  // Collapse controls (⌘B / hover-peek). `onCollapse` docks/undocks; `onPeekLeave` hides the
  // floating peek when the pointer leaves the panel.
  collapsed?: boolean;
  onCollapse?: () => void;
  onPeekLeave?: () => void;
}

// Codex-style compact age for project session rows: "now" / "5m" / "6h" / "3d" / "2w" / "4mo" / "2y".
const compactAge = (iso?: string | null): string => {
  if (!iso) return "";
  const then = Date.parse(iso);
  if (Number.isNaN(then)) return "";
  const secs = Math.max(0, Math.floor((Date.now() - then) / 1000));
  if (secs < 60) return "now";
  const mins = Math.floor(secs / 60);
  if (mins < 60) return `${mins}m`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h`;
  const days = Math.floor(hrs / 24);
  if (days < 7) return `${days}d`;
  const weeks = Math.floor(days / 7);
  if (days < 30) return `${weeks}w`;
  const months = Math.floor(days / 30);
  if (days < 365) return `${months}mo`;
  return `${Math.floor(days / 365)}y`;
};

// Sessions shown per group before "Show more" comes from Settings (sessions_peek, default 5).

export function Sidebar(props: Props) {
  const [searchModalOpen, setSearchModalOpen] = useState(false);
  const [appMenuOpen, setAppMenuOpen] = useState(false);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editValue, setEditValue] = useState("");
  // Two-step delete: first click arms the row (× → "Delete?"), second click deletes.
  // Archive is the primary way to put a conversation away — one click, reversible.
  const [confirmDelId, setConfirmDelId] = useState<string | null>(null);
  const [showArchived, setShowArchived] = useState(false);
  // Surfaced + enabled personas drive the surface list + family-aware behavior.
  // Refetched on the personas-changed event so an enable/install/delete in Settings
  // shows up here immediately (no page refresh).
  const [personas, setPersonas] = useState<Persona[] | null>(null);
  useEffect(() => {
    const load = () =>
      getPersonas()
        .then(setPersonas)
        .catch(() => setPersonas(null));
    load();
    window.addEventListener(PERSONAS_CHANGED, load);
    return () => window.removeEventListener(PERSONAS_CHANGED, load);
  }, []);
  const personaOf = (id: string) => personas?.find((p) => p.id === id);

  // Sidebar layout (§7): "grouped" = the per-persona accordion; "flat" = a single ungrouped list
  // (Pinned + Recent). Read the persisted preference on load (absent → grouped, the accordion),
  // write via setNavLayout when toggled (now flips flat-list ↔ accordion).
  const [layout, setLayout] = useState<"flat" | "grouped">("grouped");
  // Sessions shown per group before "Show more" — Settings ▸ Appearance ▸ Sidebar.
  const [peek, setPeek] = useState(5);
  useEffect(() => {
    getSettings()
      .then((s) => {
        setLayout(s.nav_layout === "flat" ? "flat" : "grouped");
        if (s.sessions_peek) setPeek(s.sessions_peek);
      })
      .catch(() => {});
  }, []);
  const setGroupBy = (next: "flat" | "grouped") => {
    setLayout(next);
    setNavLayout(next).catch(() => {});
  };
  // The RECENT-header group/filter popover (§20). Filter = show only these personas (empty = all).
  const [groupMenuOpen, setGroupMenuOpen] = useState(false);
  const [filterPersonas, setFilterPersonas] = useState<Set<string>>(new Set());
  const toggleFilterPersona = (id: string) =>
    setFilterPersonas((prev) => {
      const next = new Set(prev);
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });
  const personaVisible = (agent: string) =>
    filterPersonas.size === 0 || filterPersonas.has(agent);

  // Which accordion body is expanded. Decoupled from the active session (props.agent): expanding
  // a persona BROWSES its sessions without switching the chat area. Selecting a session or "New
  // session" is what switches (and re-opens that persona). Falls back to the active persona.
  const [openKey, setOpenKey] = useState<string | null>(props.agent);
  useEffect(() => setOpenKey(props.agent), [props.agent]);
  const browseKey = openKey ?? props.agent; // the persona whose sessions the body shows

  // Per-project collapse + "Show more". The active workspace's folder is open by default; toggling
  // any folder flips it (XOR). `projShowAll` lifts the peek cap for a given folder;
  // `personaShowAll` does the same for a (non-project) persona's flat session list.
  const [projToggled, setProjToggled] = useState<Set<string>>(new Set());
  const [projShowAll, setProjShowAll] = useState<Set<string>>(new Set());
  const [personaShowAll, setPersonaShowAll] = useState<Set<string>>(new Set());
  const toggleSet = (set: Set<string>, key: string) => {
    const next = new Set(set);
    next.has(key) ? next.delete(key) : next.add(key);
    return next;
  };

  // Pinned sessions across ALL personas — the cross-persona band at the top (manual pins only).
  const pinnedSessions = props.sessions.filter(
    (s) => s.pinned && !s.session_id.startsWith("__") && !s.archived,
  );
  const personaLabel = (agentId: string) => {
    const p = personas?.find((x) => x.id === agentId);
    return shortPersonaName(p?.name, agentId);
  };


  // A row in the bottom ⚙ "Settings & more" popup: closes the menu, then runs the destination.
  const appMenuItem = (icon: IconName, label: string, onClick: () => void, active?: boolean) => (
    <button
      className={
        "w-full flex items-center gap-2.5 px-3 py-1.5 text-[13px] text-left " +
        (active ? "text-ink bg-paper" : "hover:bg-paper")
      }
      onClick={() => {
        setAppMenuOpen(false);
        onClick();
      }}
    >
      <Icon name={icon} size={15} className="shrink-0 text-muted" /> {label}
    </button>
  );

  // Roll the per-session attention/liveness up to the persona header and the footer Inbox: the
  // accent count bubbles (sum), the liveness dot aggregates (working wins over sleeping).
  const attnByPersona = new Map<string, number>();
  const liveByPersona = new Map<string, "working" | "sleeping">();
  let totalAttention = 0;
  for (const s of props.sessions) {
    if (s.session_id.startsWith("__") || s.archived) continue;
    const a = s.attention || 0;
    if (a > 0) {
      attnByPersona.set(s.agent, (attnByPersona.get(s.agent) || 0) + a);
      totalAttention += a;
    }
    if (s.liveness === "working") liveByPersona.set(s.agent, "working");
    else if (s.liveness === "sleeping" && liveByPersona.get(s.agent) !== "working")
      liveByPersona.set(s.agent, "sleeping");
  }

  // Body data is keyed to the BROWSED persona (only one body renders at a time). Pinned sessions are
  // EXCLUDED here: they live in the cross-persona Pinned band only, so they don't repeat inside the
  // persona group / project list (matching the flat layout's Recent, which also drops pinned).
  const all = props.sessions.filter((s) => s.agent === browseKey && !s.session_id.startsWith("__"));
  const mine = all.filter((s) => !s.archived && !s.pinned);
  const archived = all.filter((s) => s.archived);
  // Only PROJECT-SCOPED personas group sessions by project (git-bound Code, project-bound Ops).
  // Scratch/deliverable conversations are orphan (each has its own per-conversation scratch dir),
  // so they list flat. Workspace-aware (not id-aware) — any git/project persona gets Projects.
  const workspaceSurface = isProjectScoped(personaOf(browseKey));

  // Search now lives in the SearchModal (Codex-style overlay), so the sidebar lists never filter
  // in place — these stay constant and the `.filter(matches)` / `normalizedQuery ? …` call sites
  // below are intentional no-ops kept to avoid churn.
  const normalizedQuery = "";
  const matches = (_s: SessionInfo) => true;

  // Recent = every non-pinned, non-archived, real session across ALL personas, newest first
  // (by updated_at; missing timestamps keep store order), search-filtered. Drives the flat layout.
  const recentSessions = [...props.sessions]
    .filter((s) => !s.archived && !s.session_id.startsWith("__") && !s.pinned)
    .filter((s) => personaVisible(s.agent))
    .filter(matches)
    .sort((a, b) => (b.updated_at || "").localeCompare(a.updated_at || ""));

  // A compact session row (mock §141 grouped/recent rows): one-line title + right-side indicators,
  // with the pin/rename/delete actions revealed on hover. Used in accordion bodies + grouped cards.
  const sessionRow = (s: SessionInfo, opts: { showTime?: boolean } = {}) => {
    const title = s.title || s.session_id;
    const editing = editingId === s.session_id;
    const active = s.session_id === props.activeSession;
    const commitRename = () => {
      const next = editValue.trim();
      if (next && next !== title) props.onRenameSession(s.session_id, next);
      setEditingId(null);
    };
    return (
      <div
        key={s.session_id}
        className={
          "group relative flex items-center gap-2 px-2 py-1.5 rounded-lg text-left cursor-pointer " +
          (active
            ? "bg-ink/[0.055] before:content-[''] before:absolute before:left-0 before:top-1.5 before:bottom-1.5 before:w-[3px] before:rounded-r-full before:bg-accent"
            : "hover:bg-panel")
        }
        onClick={() => {
          if (!editing) props.onSelectSession(s.session_id, s.workspace, s.agent);
        }}
        title={editing ? undefined : title}
      >
        {editing ? (
          <input
            className="flex-1 min-w-0 px-1.5 py-0.5 rounded-md bg-panel border border-accent text-[13px] text-ink outline-none"
            value={editValue}
            autoFocus
            onClick={(e) => e.stopPropagation()}
            onChange={(e) => setEditValue(e.target.value)}
            onBlur={commitRename}
            onKeyDown={(e) => {
              e.stopPropagation();
              if (e.key === "Enter") commitRename();
              else if (e.key === "Escape") setEditingId(null);
            }}
          />
        ) : (
          <>
            <span
              className={
                "min-w-0 flex-1 flex items-center gap-1.5 truncate text-[13px] " +
                (active ? "font-medium text-ink" : "text-ink")
              }
            >
              {s.pinned && <Icon name="pin" size={11} className="text-faint shrink-0" />}
              <span className="truncate">{title}</span>
            </span>
            <span className="flex items-center gap-1.5 shrink-0 group-hover:hidden">
              {opts.showTime && compactAge(s.updated_at) && (
                <span className="text-[11px] text-faint tabular-nums">{compactAge(s.updated_at)}</span>
              )}
              <LiveDot state={s.liveness} />
              <AttnBadge n={s.attention || 0} />
            </span>
            <span
              className="hidden group-hover:flex items-center gap-0.5 shrink-0"
              onClick={(e) => e.stopPropagation()}
            >
              <button
                title={s.pinned ? "Unpin" : "Pin to top"}
                className={
                  "w-5 h-5 grid place-items-center rounded hover:bg-paper " +
                  (s.pinned ? "text-accent" : "text-faint hover:text-ink")
                }
                onClick={() => props.onTogglePin(s.session_id, !s.pinned)}
              >
                <Icon name="pin" size={12} />
              </button>
              <button
                title="Rename"
                className="w-5 h-5 grid place-items-center rounded text-faint hover:text-ink hover:bg-paper"
                onClick={() => {
                  setEditingId(s.session_id);
                  setEditValue(title);
                }}
              >
                <Icon name="pencil" size={12} />
              </button>
              <button
                title={s.archived ? "Unarchive" : "Archive (reversible)"}
                className="w-5 h-5 grid place-items-center rounded text-faint hover:text-ink hover:bg-paper"
                onClick={() => props.onArchiveSession(s.session_id, !s.archived)}
              >
                <Icon name="archive" size={12} />
              </button>
              {confirmDelId === s.session_id ? (
                <button
                  title="Click to permanently delete"
                  className="px-1.5 h-5 grid place-items-center rounded bg-danger text-white text-[10.5px] font-medium"
                  onBlur={() => setConfirmDelId(null)}
                  onMouseLeave={() => setConfirmDelId(null)}
                  onClick={() => {
                    setConfirmDelId(null);
                    props.onDeleteSession(s.session_id);
                  }}
                >
                  Delete?
                </button>
              ) : (
                <button
                  title="Delete permanently"
                  className="w-5 h-5 grid place-items-center rounded text-faint hover:text-danger hover:bg-paper leading-none"
                  onClick={() => setConfirmDelId(s.session_id)}
                >
                  ×
                </button>
              )}
            </span>
          </>
        )}
      </div>
    );
  };

  // A 2-line card row (mock §141 list-flat): an optional persona icon tile + title + a
  // persona/channel subtitle + right-side indicators, with a pin/unpin button revealed on hover.
  // Shared by the flat layout's Pinned (no icon) and Recent (with icon) sections.
  const cardRow = (s: SessionInfo) => {
    const active = s.session_id === props.activeSession;
    const title = s.title || s.session_id;
    // Subtitle (deliberately quiet): just the persona label + the workspace basename for
    // project-scoped personas (a real folder). No channel/subscription detail — connectors show as
    // the dot indicator on the right, so the subtitle stays clean. Scratch/orphan personas omit the
    // workspace (it's an ugly per-conversation hash).
    const subParts = [personaLabel(s.agent)];
    if (s.workspace && isProjectScoped(personaOf(s.agent))) subParts.push(baseName(s.workspace));
    return (
      <div
        key={s.session_id}
        className={
          "group relative w-full flex items-center gap-2.5 px-2 py-2 rounded-lg cursor-pointer text-left " +
          (active
            ? "bg-ink/[0.055] before:content-[''] before:absolute before:left-0 before:top-1.5 before:bottom-1.5 before:w-[3px] before:rounded-r-full before:bg-accent"
            : "hover:bg-paper")
        }
        title={title}
        onClick={() => props.onSelectSession(s.session_id, s.workspace, s.agent)}
      >
        {/* No leading glyph on session rows — the persona shows in the subtitle (Rohit's call
            2026-07-07: the per-session icon read as noise in both grouped and chronological). */}
        <span className="min-w-0 flex-1">
          <span className="block truncate text-[13px] font-medium">{title}</span>
          <span className="block truncate text-[11px] text-faint">{subParts.join(" · ")}</span>
        </span>
        <span className="flex items-center gap-1.5 shrink-0">
          <ConnectorDot subs={s.subscriptions} />
          <LiveDot state={s.liveness} />
          <AttnBadge n={s.attention || 0} />
          <button
            className={
              "hidden group-hover:grid w-5 h-5 place-items-center rounded hover:bg-paper " +
              (s.pinned ? "text-accent" : "text-faint hover:text-ink")
            }
            title={s.pinned ? "Unpin" : "Pin to top"}
            onClick={(e) => {
              e.stopPropagation();
              props.onTogglePin(s.session_id, !s.pinned);
            }}
          >
            <Icon name="pin" size={11} />
          </button>
        </span>
      </div>
    );
  };

  // The cross-persona Pinned band (manual pins only) — icon-free rows. Appears in BOTH layouts
  // (flat list AND accordion), so it's factored here for reuse.
  const pinnedBand = () =>
    pinnedSessions.length > 0 ? (
      <div>
        <div className="px-1.5 text-[10.5px] uppercase tracking-[0.07em] text-faint font-semibold mb-1">
          Pinned
        </div>
        <div className="space-y-0.5">
          {pinnedSessions.map((s) => cardRow(s))}
        </div>
      </div>
    ) : null;

  // RECENT header with the group/filter control (§20) — the group toggle moved off the brand bar.
  // "Group by" flips the persona accordion ↔ chronological list; "Filter by coworker" narrows to
  // the checked personas (none checked = all shown).
  const recentHeader = () => {
    const filterPersonaList = (personas || []).filter(
      (p) => (p.enabled && p.surfaced) || agentsWithSessions.has(p.id),
    );
    return (
    <div className="relative flex items-center justify-between px-1.5 mb-1" data-testid="recent-header">
      <span className="text-[10.5px] uppercase tracking-[0.07em] text-faint font-semibold">
        Recent
      </span>
      <button
        className="w-6 h-6 grid place-items-center rounded-md text-faint hover:text-ink hover:bg-paper -mr-1"
        title="Group & filter conversations"
        aria-label="Group and filter conversations"
        onClick={() => setGroupMenuOpen((v) => !v)}
      >
        <Icon name="sliders" size={14} />
      </button>
      {groupMenuOpen && (
        <>
          <div className="fixed inset-0 z-40" onClick={() => setGroupMenuOpen(false)} />
          <div
            className="absolute right-0 top-7 z-50 w-56 rounded-xl border border-line bg-panel shadow-xl p-1.5"
            role="menu"
            data-testid="group-filter-menu"
          >
            <div className="px-2 pt-1 pb-1 text-[10.5px] uppercase tracking-[0.06em] text-faint font-semibold">
              Group by
            </div>
            {([["grouped", "Persona"], ["flat", "Chronological"]] as ["flat" | "grouped", string][]).map(
              ([key, label]) => (
                <button
                  key={key}
                  className="w-full flex items-center gap-2 px-2 py-1.5 rounded-lg text-[13px] text-left hover:bg-paper"
                  onClick={() => setGroupBy(key)}
                >
                  <span className="flex-1">{label}</span>
                  {layout === key && <span className="text-accent text-[12px]">✓</span>}
                </button>
              ),
            )}
            {filterPersonaList.length > 1 && (
              <>
                <div className="my-1 border-t border-line" />
                <div className="px-2 pt-1 pb-1 flex items-center justify-between">
                  <span className="text-[10.5px] uppercase tracking-[0.06em] text-faint font-semibold">
                    Filter by coworker
                  </span>
                  {filterPersonas.size > 0 && (
                    <button className="text-[11px] text-accent" onClick={() => setFilterPersonas(new Set())}>
                      Clear
                    </button>
                  )}
                </div>
                <div className="max-h-52 overflow-y-auto">
                  {filterPersonaList.map((p) => {
                    const checked = filterPersonas.has(p.id);
                    return (
                      <button
                        key={p.id}
                        className="w-full flex items-center gap-2 px-2 py-1.5 rounded-lg text-[13px] text-left hover:bg-paper"
                        onClick={() => toggleFilterPersona(p.id)}
                      >
                        <span
                          className={
                            "w-3.5 h-3.5 rounded border grid place-items-center shrink-0 text-white " +
                            (checked ? "bg-accent border-accent" : "border-line")
                          }
                        >
                          {checked && <span className="text-[9px] leading-none">✓</span>}
                        </span>
                        <span className="flex-1 truncate">{p.name}</span>
                      </button>
                    );
                  })}
                </div>
                <div className="px-2 pt-1 pb-0.5 text-[11px] text-faint leading-snug">
                  None checked shows all.
                </div>
              </>
            )}
          </div>
        </>
      )}
    </div>
    );
  };

  // Code/Cowork group by project; Chat is a flat recents list.
  const byProject = useMemo(() => {
    const grouped = new Map<string, SessionInfo[]>();
    for (const s of mine) {
      if (!grouped.has(s.workspace)) grouped.set(s.workspace, []);
      grouped.get(s.workspace)!.push(s);
    }
    return grouped;
  }, [mine]);

  const filteredByProject = useMemo(() => {
    const grouped = new Map<string, SessionInfo[]>();
    for (const [proj, list] of byProject) grouped.set(proj, list.filter(matches));
    return grouped;
  }, [byProject, normalizedQuery]);

  // Projects are tracked PER SURFACE: a folder appears under Code only if it has Code sessions,
  // under Cowork only if it has Cowork sessions (+ the currently-open folder). No cross-bleed.
  const projectOrder: string[] = [];
  const seen = new Set<string>();
  // Pin the active folder at top only when browsing the active persona (else it belongs elsewhere).
  if (props.workspace && browseKey === props.agent) {
    projectOrder.push(props.workspace);
    seen.add(props.workspace);
  }
  for (const s of mine) {
    if (s.workspace && !seen.has(s.workspace)) {
      seen.add(s.workspace);
      projectOrder.push(s.workspace);
    }
  }

  // Surfaced + enabled personas drive the surface list (default persona first); fall back to the
  // static set until loaded. A persona that has live sessions ALWAYS gets a section, surfaced or
  // not — every session must have a home in the grouped layout (a picker preference can hide the
  // persona from New Session, never orphan its conversations).
  const agentsWithSessions = new Set(
    props.sessions
      .filter((s) => !s.archived && !s.session_id.startsWith("__"))
      .map((s) => s.agent),
  );
  const visibleSurfaces = (
    personas
      ? personas
          .filter((p) => (p.enabled && p.surfaced) || agentsWithSessions.has(p.id))
          .sort((a, b) => Number(b.default) - Number(a.default)) // default leads
          .map(surfaceFromPersona)
      : SURFACES.filter(
          (s) => s.key === "cowork" || props.surfaces[s.key as keyof SurfaceVisibility],
        )
  ).filter((s) => personaVisible(s.key));

  const isCurrent = (key: string) => props.agent === key; // the active session's persona
  const isExpanded = (key: string) => openKey === key; // its body is open
  // Expand ≠ switch: clicking a header only browses (toggles the accordion). The chat area
  // changes only when a session is selected or "New session" is clicked.
  const onHeaderClick = (key: string) => setOpenKey((k) => (k === key ? null : key));

  // The expanded body for the active surface: a "New session" action, then the project-grouped
  // (or flat) session list, then the archived disclosure.
  const surfaceBody = () => {
    return (
      <div className="space-y-1 px-1.5 pb-2 pt-0.5">
        {/* Body is flush inside the expanded group's fill (provided by the wrapper) so the header +
            its sessions read as one connected block — clear where a group ends and the next begins. */}
        {/* No per-persona "New session" here — the top split button's ▾ already starts a session
            in any persona (it was redundant + the mock's grouped cards don't have it). */}
        {workspaceSurface ? (
          <>
            {/* Codex-style Projects: a "+" header affordance, then collapsible folders whose
                rows carry a right-aligned compact age and truncate to PROJECT_PEEK + "Show more". */}
            <div className="flex items-center justify-between px-1.5 pt-1">
              <span className="text-[10.5px] uppercase tracking-[0.07em] text-faint font-semibold">
                Projects
              </span>
              <button
                className="w-5 h-5 grid place-items-center rounded text-faint hover:text-ink hover:bg-panel"
                title="New project"
                aria-label="New project"
                onClick={() => props.onNewProject(browseKey)}
              >
                <Icon name="folderPlus" size={14} />
              </button>
            </div>
            <div className="space-y-0.5">
              {projectOrder.length === 0 && (
                <div className="px-2 py-1.5 text-[12px] text-faint leading-snug">
                  No projects yet — start one with the + above.
                </div>
              )}
              {projectOrder.map((proj) => {
                const list = filteredByProject.get(proj) || [];
                if (normalizedQuery && list.length === 0) return null; // hide non-matching folders while searching
                const isActive = proj === props.workspace;
                // Open the active project by default; if none is active (browsing from another
                // persona), open the most-recent folder so the accordion isn't all-collapsed.
                const activeInOrder = !!props.workspace && projectOrder.includes(props.workspace);
                const defaultOpen = isActive || (!activeInOrder && proj === projectOrder[0]);
                const open = !!normalizedQuery || defaultOpen !== projToggled.has(proj);
                const showAll = !!normalizedQuery || projShowAll.has(proj);
                const shown = showAll ? list : list.slice(0, peek);
                return (
                  <div key={proj}>
                    <div
                      className={
                        "flex items-center gap-1.5 px-1.5 py-1 rounded-lg cursor-pointer select-none hover:bg-panel " +
                        (isActive ? "text-ink" : "text-muted hover:text-ink")
                      }
                      onClick={() => setProjToggled((s) => toggleSet(s, proj))}
                      title={proj}
                    >
                      <Icon name="folder" size={15} className="shrink-0" />
                      <span
                        className={
                          "truncate min-w-0 text-[12.5px] " + (isActive ? "font-semibold" : "font-medium")
                        }
                      >
                        {baseName(proj)}
                      </span>
                      {/* Disclosure chevron sits AFTER the name (Codex parity), not leading the row. */}
                      <Icon
                        name={open ? "chevronDown" : "chevronRight"}
                        size={12}
                        className="text-faint shrink-0"
                      />
                    </div>
                    {open &&
                      (list.length > 0 ? (
                        // pl-[19px] aligns each session's name under the folder NAME (folder icon
                        // 15 + gap 6 + row px 6 − session px 8 = 19), per a clean-column layout.
                        <div className="space-y-0.5 pl-[19px]">
                          {shown.map((s) => sessionRow(s, { showTime: true }))}
                          {!showAll && list.length > peek && (
                            <button
                              className="px-2 py-1 text-[12px] text-faint hover:text-muted"
                              onClick={() => setProjShowAll((s) => toggleSet(s, proj))}
                            >
                              Show more ({list.length - peek})
                            </button>
                          )}
                        </div>
                      ) : (
                        <div className="px-2 py-1.5 pl-[19px] text-[12px] text-faint leading-snug">
                          No conversations in this project yet.
                        </div>
                      ))}
                  </div>
                );
              })}
            </div>
          </>
        ) : (
          <div className="space-y-0.5">
            {mine.filter(matches).length === 0 ? (
              <div className="px-2 py-1.5 text-[12px] text-faint leading-snug">
                {normalizedQuery ? "No matching conversations." : "No conversations yet."}
              </div>
            ) : (
              <>
                {(personaShowAll.has(browseKey)
                  ? mine.filter(matches)
                  : mine.filter(matches).slice(0, peek)
                ).map((s) => sessionRow(s))}
                {!personaShowAll.has(browseKey) && mine.filter(matches).length > peek && (
                  <button
                    className="px-2 py-1 text-[12px] text-faint hover:text-muted"
                    onClick={() => setPersonaShowAll((s) => toggleSet(s, browseKey))}
                  >
                    Show more ({mine.filter(matches).length - peek})
                  </button>
                )}
              </>
            )}
          </div>
        )}

        {archived.length > 0 && (
          <div className="mt-2 pt-1.5 border-t border-line">
            <button
              className="w-full flex items-center gap-1.5 px-1.5 py-1 rounded text-[12px] text-faint hover:text-muted"
              onClick={() => setShowArchived((v) => !v)}
            >
              <Icon name={showArchived ? "chevronDown" : "chevronRight"} size={13} className="shrink-0" />
              Archived ({archived.length})
            </button>
            {showArchived && (
              <div className="space-y-0.5 mt-0.5">{archived.filter(matches).map((s) => sessionRow(s))}</div>
            )}
          </div>
        )}
      </div>
    );
  };

  return (
    <div
      className="sidebar flex flex-col min-h-0 bg-panel border-r border-line"
      onMouseLeave={props.onPeekLeave}
    >
      {/* Header: collapse/pin control FIRST + wordmark. The pin sits at the same screen position
          as the collapsed reveal button (see .nav-pin-btn / .nav-reveal-btn in styles.css), so
          hovering the reveal peeks the nav and the pin lands right under the cursor — no travel.
          data-tauri-drag-region drags the window; on desktop the row clears the traffic lights. */}
      <div className="brand px-3.5 pt-2.5 pb-2 flex items-center gap-2" data-tauri-drag-region>
        {/* Collapse (dock) / pin the sidebar. ⌘B mirrors this. */}
        {props.onCollapse && (
          <button
            className="nav-pin-btn w-7 h-7 grid place-items-center rounded-md text-faint hover:text-ink hover:bg-paper shrink-0"
            title={props.collapsed ? "Dock sidebar (⌘B)" : "Collapse sidebar (⌘B)"}
            aria-label={props.collapsed ? "Dock sidebar" : "Collapse sidebar"}
            onClick={props.onCollapse}
          >
            <Icon name="sidebar" size={16} />
          </button>
        )}
        <div className="brand-wordmark text-[15px]">OpenCoworker</div>
      </div>

      {/* New session: split button — primary starts the last-used persona; ▾ picks a specific one. */}
      <NewSessionSplit
        personas={personas}
        current={props.agent}
        onNew={props.onNewSession}
        onManage={props.onManagePersonas}
      />

      {/* Search: a borderless nav-style entry (not a boxed input) that opens the command-palette
          SearchModal over the whole app. Matches the bottom-nav rows to reduce the boxy look. */}
      <div className="px-2.5 mt-1">
        <button
          className="w-full flex items-center gap-2.5 px-2.5 py-2 rounded-lg text-[13px] text-left text-muted hover:bg-paper hover:text-ink"
          onClick={() => setSearchModalOpen(true)}
        >
          <Icon name="search" size={15} className="shrink-0" /> Search
        </button>
      </div>

      {/* Scroll area: Pinned band + the RECENT header (with group/filter control), then the body —
          grouped (per-persona accordion) or flat (chronological list). */}
      <div className="flex-1 overflow-y-auto px-2.5 mt-3 pb-2">
        <div className="space-y-4">
          {pinnedBand()}
          <div>
            {recentHeader()}
            {layout === "grouped" ? (
            <div className="space-y-1.5">
              {visibleSurfaces.map((s) => {
                const expanded = isExpanded(s.key);
                return (
                  // When expanded, the wrapper carries the recessed fill so the header sits INSIDE
                  // the block with its sessions (one connected group). Collapsed = a plain row.
                  <div
                    key={s.key}
                    className={expanded ? "rounded-xl bg-paper/70 overflow-hidden" : ""}
                  >
                    <div
                      className={
                        "flex items-center gap-2.5 px-2 py-2 cursor-pointer select-none " +
                        (expanded
                          ? ""
                          : isCurrent(s.key)
                            ? "rounded-lg bg-paper"
                            : "rounded-lg hover:bg-paper")
                      }
                      onClick={() => onHeaderClick(s.key)}
                    >
                      <span
                        className={
                          "min-w-0 flex-1 truncate text-[13px] " +
                          (isCurrent(s.key) ? "font-semibold text-ink" : "font-medium text-ink")
                        }
                      >
                        {s.label}
                      </span>
                      <LiveDot state={liveByPersona.get(s.key)} />
                      <AttnBadge n={attnByPersona.get(s.key) || 0} />
                      {/* Persona configuration moved to Settings ▸ Personas (Rohit's call
                          2026-07-07) — the per-group gear read as clutter here. */}
                      <Icon
                        name={expanded ? "chevronDown" : "chevronRight"}
                        size={15}
                        className="text-faint shrink-0"
                      />
                    </div>
                    {expanded && surfaceBody()}
                  </div>
                );
              })}
            </div>
            ) : (
            <div className="space-y-0.5">
              {recentSessions.length === 0 ? (
                <div className="px-2 py-1.5 text-[12px] text-faint leading-snug">
                  {normalizedQuery ? "No matching conversations." : "No conversations yet."}
                </div>
              ) : (
                recentSessions.map((s) => cardRow(s))
              )}
            </div>
            )}
          </div>
        </div>
      </div>

      {/* Bottom: Inbox stays visible (its attention badge needs to be glanceable); the occasional
          destinations (Settings, Integrations, Automations, Activity) collapse into one ⚙ menu that
          opens upward — Codex/Claude-style — so the bottom isn't a stack of rows. */}
      <div className="px-2.5 py-2 border-t border-line space-y-0.5">
        <button
          className={
            "w-full flex items-center gap-2.5 px-2.5 py-2 rounded-lg text-[13px] text-left " +
            (props.inboxActive ? "bg-paper text-ink" : "hover:bg-paper")
          }
          onClick={props.onOpenInbox}
        >
          <Icon name="chat" size={15} className="shrink-0 text-muted" />
          <span className="flex-1">Inbox</span>
          <AttnBadge n={totalAttention} />
        </button>

        {/* The popup is anchored to THIS button (not the whole bottom block) so it opens directly
            above the trigger instead of floating detached above the Inbox row. */}
        <div className="relative">
          <button
            className={
              "w-full flex items-center gap-2.5 px-2.5 py-2 rounded-lg text-[13px] text-left " +
              (appMenuOpen || props.integrationsActive || props.scheduledActive || props.auditActive
                ? "bg-paper text-ink"
                : "hover:bg-paper")
            }
            onClick={() => setAppMenuOpen((v) => !v)}
            aria-haspopup="menu"
            aria-expanded={appMenuOpen}
          >
            <Icon name="gear" size={15} className="shrink-0 text-muted" />
            <span className="flex-1">Settings &amp; more</span>
            <Icon
              name="chevronDown"
              size={14}
              className={"text-faint shrink-0 transition-transform " + (appMenuOpen ? "" : "rotate-180")}
            />
          </button>

          {appMenuOpen && (
            <>
              <div className="fixed inset-0 z-30" onClick={() => setAppMenuOpen(false)} />
              <div className="absolute z-40 bottom-full left-0 right-0 mb-1 rounded-xl border border-line bg-panel shadow-2xl py-1">
                {props.workspace && (
                  <div
                    className="px-3 py-1.5 mb-1 text-[11px] text-faint truncate border-b border-line"
                    title={props.workspace}
                  >
                    {props.workspace}
                  </div>
                )}
                {appMenuItem("gear", "Settings", props.onManage)}
                {appMenuItem("plug", "Integrations", props.onOpenIntegrations, props.integrationsActive)}
                {appMenuItem("clock", "Automations", props.onOpenScheduled, props.scheduledActive)}
                {appMenuItem("audit", "Activity", props.onOpenAudit, props.auditActive)}
              </div>
            </>
          )}
        </div>
      </div>

      {searchModalOpen && (
        <SearchModal
          sessions={props.sessions}
          personas={personas ?? undefined}
          onSelect={(id, ws, ag) => {
            setSearchModalOpen(false);
            props.onSelectSession(id, ws, ag);
          }}
          onClose={() => setSearchModalOpen(false)}
        />
      )}
    </div>
  );
}

// New-session split button (§8): the primary action starts a session with the last-used persona
// (`current`); the ▾ opens a menu of the enabled personas (from /v1/personas) plus a "Manage
// personas…" entry. A plain custom split control — the pill-shaped Dropdown doesn't fit this shape.
function NewSessionSplit({
  personas,
  current,
  onNew,
  onManage,
}: {
  personas: Persona[] | null;
  current: string;
  onNew: (agent: string) => void;
  onManage: () => void;
}) {
  const [open, setOpen] = useState(false);
  const enabled = (personas || []).filter((p) => p.enabled);
  return (
    <div className="px-3 pt-2 relative">
      <div className="flex">
        <button
          className="newsplit-primary flex-1 text-left px-3 py-2 rounded-l-lg bg-accent text-white text-[13px] font-medium hover:opacity-95 flex items-center gap-2"
          onClick={() => onNew(current)}
        >
          <Icon name="plus" size={15} className="shrink-0" /> New session
        </button>
        <button
          className="px-2.5 rounded-r-lg bg-accent text-white border-l border-white/25 hover:opacity-95 flex items-center"
          title="Start with a specific persona"
          aria-label="Choose a persona"
          onClick={() => setOpen((v) => !v)}
        >
          <Icon name="chevronDown" size={13} />
        </button>
      </div>
      {open && (
        <>
          <div className="fixed inset-0 z-20" onClick={() => setOpen(false)} />
          <div className="newsplit-menu absolute left-3 right-3 mt-1 z-30 bg-panel border border-line rounded-xl2 shadow-xl p-1">
            <div className="px-2 py-1 text-[10.5px] uppercase tracking-[0.06em] text-faint font-semibold">
              Start a session as
            </div>
            {enabled.map((p) => (
              <button
                key={p.id}
                className="w-full flex items-center gap-2.5 px-2 py-1.5 rounded-lg hover:bg-paper text-left"
                onClick={() => {
                  setOpen(false);
                  onNew(p.id);
                }}
              >
                <span className="w-6 h-6 rounded-md bg-paper border border-line grid place-items-center text-muted shrink-0">
                  <PersonaGlyph icon={p.icon} family={p.family} size={12} />
                </span>
                <span className="min-w-0">
                  <span className="block text-[13px] font-medium truncate">
                    {shortPersonaName(p.name, p.id)}
                  </span>
                  {p.tagline && (
                    <span className="block text-[11px] text-muted truncate">{p.tagline}</span>
                  )}
                </span>
              </button>
            ))}
            <div className="border-t border-line mt-1 pt-1">
              <button
                className="w-full px-2 py-1.5 rounded-lg hover:bg-paper text-left text-[12.5px] text-muted"
                onClick={() => {
                  setOpen(false);
                  onManage();
                }}
              >
                Manage personas…
              </button>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
