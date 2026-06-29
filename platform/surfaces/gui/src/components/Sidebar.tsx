import { useEffect, useMemo, useState } from "react";
import {
  getPersonas,
  getSettings,
  setNavLayout,
  type Persona,
  type RecentWorkspace,
  type SurfaceVisibility,
} from "../api";
import type { SessionInfo } from "../types";
import { isProjectScoped } from "../personaScope";
import { Icon, type IconName } from "./Icon";
import { PersonaGlyph, personaGlyph } from "./personaIcon";

// Session surfaces shown as accordions, in display order. The surfaced personas drive this list
// (so third-party / Ops personas appear); the hardcoded set is the fallback before personas load.
const SURFACES: { key: string; label: string; icon: IconName; cls: string }[] = [
  { key: "cowork", label: "OpenCoworker", icon: "diamond", cls: "ico-cowork" },
  { key: "chat", label: "Chat", icon: "chat", cls: "ico-chat" },
  { key: "code", label: "Code", icon: "code", cls: "ico-code" },
];

const surfaceFromPersona = (p: Persona) => ({
  key: p.id,
  label: p.id === "cowork" ? "OpenCoworker" : p.name,
  icon: personaGlyph(p.icon, p.family),
  cls: `ico-${p.icon || "cowork"}`,
});

// Attention = Inbox items awaiting a session (an accent count that bubbles session → persona →
// footer Inbox — all views of the one Inbox queue, never a second list).
function AttnBadge({ n }: { n: number }) {
  if (!n) return null;
  return (
    <span
      className="text-[10px] font-semibold text-white bg-accent rounded-full px-1.5 leading-[15px] shrink-0"
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
}

const baseName = (p: string) => p.split("/").filter(Boolean).pop() || p;

export function Sidebar(props: Props) {
  const [query, setQuery] = useState("");
  const [searchOpen, setSearchOpen] = useState(false);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editValue, setEditValue] = useState("");
  const [showArchived, setShowArchived] = useState(false);
  // Surfaced + enabled personas drive the surface list + family-aware behavior.
  const [personas, setPersonas] = useState<Persona[] | null>(null);
  useEffect(() => {
    getPersonas()
      .then(setPersonas)
      .catch(() => setPersonas(null));
  }, []);
  const personaOf = (id: string) => personas?.find((p) => p.id === id);

  // Sidebar layout (§7): "grouped" = the per-persona accordion; "flat" = a single ungrouped list
  // (Pinned + Recent). Read the persisted preference on load (absent → grouped, the accordion),
  // write via setNavLayout when toggled (now flips flat-list ↔ accordion).
  const [layout, setLayout] = useState<"flat" | "grouped">("grouped");
  useEffect(() => {
    getSettings()
      .then((s) => setLayout(s.nav_layout === "flat" ? "flat" : "grouped"))
      .catch(() => {});
  }, []);
  const toggleLayout = () => {
    setLayout((cur) => {
      const next = cur === "grouped" ? "flat" : "grouped";
      setNavLayout(next).catch(() => {});
      return next;
    });
  };

  // Which accordion body is expanded. Decoupled from the active session (props.agent): expanding
  // a persona BROWSES its sessions without switching the chat area. Selecting a session or "New
  // session" is what switches (and re-opens that persona). Falls back to the active persona.
  const [openKey, setOpenKey] = useState<string | null>(props.agent);
  useEffect(() => setOpenKey(props.agent), [props.agent]);
  const browseKey = openKey ?? props.agent; // the persona whose sessions the body shows

  // Pinned sessions across ALL personas — the cross-persona band at the top (manual pins only).
  const pinnedSessions = props.sessions.filter(
    (s) => s.pinned && !s.session_id.startsWith("__") && !s.archived,
  );
  const personaLabel = (agentId: string) => {
    const p = personas?.find((x) => x.id === agentId);
    return p ? (p.id === "cowork" ? "OpenCoworker" : p.name) : agentId;
  };

  // A neutral persona icon tile (mock chrome): a panel chip with a hairline border + the persona's
  // resolved glyph (manifest icon → family default; emoji rendered as-is).
  const iconTile = (agentId: string, size = 13) => {
    const p = personas?.find((x) => x.id === agentId);
    return (
      <span className="w-6 h-6 rounded-md bg-panel border border-line grid place-items-center text-muted shrink-0">
        <PersonaGlyph icon={p?.icon} family={p?.family} size={size} />
      </span>
    );
  };

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

  // Body data is keyed to the BROWSED persona (only one body renders at a time).
  const all = props.sessions.filter((s) => s.agent === browseKey && !s.session_id.startsWith("__"));
  const mine = all.filter((s) => !s.archived);
  const archived = all.filter((s) => s.archived);
  // Only PROJECT-SCOPED personas group sessions by project (git-bound Code, project-bound Ops).
  // Scratch/deliverable conversations are orphan (each has its own per-conversation scratch dir),
  // so they list flat. Workspace-aware (not id-aware) — any git/project persona gets Projects.
  const workspaceSurface = isProjectScoped(personaOf(browseKey));

  const normalizedQuery = query.trim().toLowerCase();
  const matches = (s: SessionInfo) =>
    !normalizedQuery ||
    (s.title || s.session_id).toLowerCase().includes(normalizedQuery) ||
    s.session_id.toLowerCase().includes(normalizedQuery);

  // Recent = every non-pinned, non-archived, real session across ALL personas, newest first
  // (by updated_at; missing timestamps keep store order), search-filtered. Drives the flat layout.
  const recentSessions = [...props.sessions]
    .filter((s) => !s.archived && !s.session_id.startsWith("__") && !s.pinned)
    .filter(matches)
    .sort((a, b) => (b.updated_at || "").localeCompare(a.updated_at || ""));

  // A compact session row (mock §141 grouped/recent rows): one-line title + right-side indicators,
  // with the pin/rename/delete actions revealed on hover. Used in accordion bodies + grouped cards.
  const sessionRow = (s: SessionInfo) => {
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
          "group flex items-center gap-2 px-2 py-1.5 rounded-lg text-left cursor-pointer " +
          (active ? "bg-accentSoft/70 border border-accent/30" : "hover:bg-panel")
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
                title="Delete"
                className="w-5 h-5 grid place-items-center rounded text-faint hover:text-ink hover:bg-paper leading-none"
                onClick={() => {
                  if (window.confirm(`Delete "${title}"?`)) props.onDeleteSession(s.session_id);
                }}
              >
                ×
              </button>
            </span>
          </>
        )}
      </div>
    );
  };

  // A 2-line card row (mock §141 list-flat): an optional persona icon tile + title + a
  // persona/channel subtitle + right-side indicators, with a pin/unpin button revealed on hover.
  // Shared by the flat layout's Pinned (no icon) and Recent (with icon) sections.
  const cardRow = (s: SessionInfo, { showIcon }: { showIcon: boolean }) => {
    const active = s.session_id === props.activeSession;
    const title = s.title || s.session_id;
    // Subtitle: persona label, then channels if subscribed; otherwise the workspace basename ONLY
    // for project-scoped personas (a real folder). Scratch/orphan personas (Cowork, or an orphaned
    // persona) would otherwise show an ugly per-conversation scratch-dir hash, so we omit it.
    const subParts = [personaLabel(s.agent)];
    if (s.subscriptions?.length)
      subParts.push(
        s.subscriptions.length === 1 ? s.subscriptions[0] : `${s.subscriptions.length} channels`,
      );
    else if (s.workspace && isProjectScoped(personaOf(s.agent))) subParts.push(baseName(s.workspace));
    return (
      <div
        key={s.session_id}
        className={
          "group w-full flex items-center gap-2.5 px-2 py-2 rounded-lg cursor-pointer text-left " +
          (active ? "bg-accentSoft/60 border border-accent/30" : "hover:bg-paper")
        }
        title={title}
        onClick={() => props.onSelectSession(s.session_id, s.workspace, s.agent)}
      >
        {/* RECENT rows carry the persona glyph; PINNED rows stay icon-free (persona shows in the
            subtitle), per the mock + the pinned-band decision. */}
        {showIcon && iconTile(s.agent)}
        <span className="min-w-0 flex-1">
          <span className="block truncate text-[13px] font-medium">{title}</span>
          <span className="block truncate text-[11px] text-muted">{subParts.join(" · ")}</span>
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
          {pinnedSessions.map((s) => cardRow(s, { showIcon: false }))}
        </div>
      </div>
    ) : null;

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
  // static set until loaded.
  const visibleSurfaces = personas
    ? personas
        .filter((p) => p.enabled && p.surfaced)
        .sort((a, b) => Number(b.default) - Number(a.default)) // default leads
        .map(surfaceFromPersona)
    : SURFACES.filter(
        (s) => s.key === "cowork" || props.surfaces[s.key as keyof SurfaceVisibility],
      );

  const isCurrent = (key: string) => props.agent === key; // the active session's persona
  const isExpanded = (key: string) => openKey === key; // its body is open
  // Expand ≠ switch: clicking a header only browses (toggles the accordion). The chat area
  // changes only when a session is selected or "New session" is clicked.
  const onHeaderClick = (key: string) => setOpenKey((k) => (k === key ? null : key));

  // The expanded body for the active surface: a "New session" action, then the project-grouped
  // (or flat) session list, then the archived disclosure.
  const surfaceBody = () => {
    return (
      <div className="mt-0.5 space-y-1 pl-1">
        {/* No per-persona "New session" here — the top split button's ▾ already starts a session
            in any persona (it was redundant + the mock's grouped cards don't have it). */}
        {workspaceSurface ? (
          <>
            <div className="px-1.5 pt-1 text-[10.5px] uppercase tracking-[0.07em] text-faint font-semibold">
              Projects
            </div>
            <div className="space-y-0.5">
              <button
                className="w-full flex items-center gap-2 px-2 py-1.5 rounded-lg hover:bg-panel text-left text-[13px] text-muted hover:text-ink"
                onClick={() => props.onNewProject(browseKey)}
              >
                <Icon name="folderPlus" size={15} className="shrink-0" /> New project
              </button>
              {projectOrder.map((proj) => (
                <div className="space-y-0.5" key={proj}>
                  <div
                    className={
                      "flex items-center gap-2 px-1.5 pt-2 pb-1 text-[12px] " +
                      (proj === props.workspace ? "text-ink font-semibold" : "text-muted font-medium")
                    }
                    title={proj}
                  >
                    <Icon name="folder" size={15} className="shrink-0" />
                    <span className="truncate">{baseName(proj)}</span>
                  </div>
                  {(filteredByProject.get(proj) || []).length > 0 ? (
                    (filteredByProject.get(proj) || []).map(sessionRow)
                  ) : (
                    <div className="px-2 py-1.5 text-[12px] text-faint leading-snug">
                      {normalizedQuery ? "No matching conversations." : "No conversations in this project yet."}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </>
        ) : (
          <div className="space-y-0.5">
            {mine.filter(matches).length === 0 ? (
              <div className="px-2 py-1.5 text-[12px] text-faint leading-snug">
                {normalizedQuery ? "No matching conversations." : "No conversations yet."}
              </div>
            ) : (
              mine.filter(matches).map(sessionRow)
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
              <div className="space-y-0.5 mt-0.5">{archived.filter(matches).map(sessionRow)}</div>
            )}
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="sidebar flex flex-col min-h-0 bg-panel border-r border-line">
      {/* Header: accent check-logo tile + wordmark + flat↔grouped layout toggle (mock §102). */}
      <div className="brand px-3.5 pt-3.5 pb-2 flex items-center gap-2">
        <div className="w-6 h-6 rounded-md bg-accent/15 grid place-items-center text-accent shrink-0">
          <svg
            width={14}
            height={14}
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth={2.2}
            strokeLinecap="round"
            aria-hidden="true"
          >
            <path d="M5 13l4 4L19 7" />
          </svg>
        </div>
        <div className="font-semibold tracking-tight">OpenCoworker</div>
        {/* Layout toggle by the wordmark: flat (persona accordions) ↔ grouped (per-persona cards). */}
        <button
          className="ml-auto w-7 h-7 grid place-items-center rounded-md text-faint hover:text-ink hover:bg-paper"
          title={
            layout === "grouped"
              ? "Grouped by persona — tap for flat"
              : "Flat — tap to group by persona"
          }
          aria-label="Toggle sidebar layout"
          onClick={toggleLayout}
        >
          {layout === "grouped" ? (
            <svg width={16} height={16} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.8} aria-hidden="true">
              <rect x="3" y="4" width="18" height="6" rx="1.5" />
              <rect x="3" y="14" width="18" height="6" rx="1.5" />
            </svg>
          ) : (
            <svg width={16} height={16} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.8} strokeLinecap="round" aria-hidden="true">
              <path d="M3 5h18M3 12h18M3 19h18" />
            </svg>
          )}
        </button>
      </div>

      {/* New session: split button — primary starts the last-used persona; ▾ picks a specific one. */}
      <NewSessionSplit
        personas={personas}
        current={props.agent}
        onNew={props.onNewSession}
        onManage={props.onManagePersonas}
      />

      {/* Search (mock §133): a paper chip; opening it swaps in the live filter input. */}
      <div className="px-3 mt-1 flex items-center gap-2">
        {searchOpen ? (
          <div className="flex-1 flex items-center gap-2 px-2.5 py-1.5 rounded-lg bg-paper border border-line text-faint focus-within:border-lineStrong">
            <Icon name="search" size={14} className="shrink-0" />
            <input
              className="flex-1 min-w-0 bg-transparent outline-none text-[12.5px] text-ink placeholder:text-faint"
              placeholder="Search conversations"
              value={query}
              autoFocus
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Escape") {
                  setSearchOpen(false);
                  setQuery("");
                }
              }}
            />
            <button
              className="text-faint hover:text-ink text-[15px] leading-none shrink-0"
              title="Close search"
              onClick={() => {
                setSearchOpen(false);
                setQuery("");
              }}
            >
              ×
            </button>
          </div>
        ) : (
          <button
            className="flex-1 flex items-center gap-2 px-2.5 py-1.5 rounded-lg bg-paper border border-line text-faint hover:border-lineStrong text-left"
            onClick={() => setSearchOpen(true)}
          >
            <Icon name="search" size={14} className="shrink-0" />
            <span className="text-[12.5px]">Search</span>
          </button>
        )}
      </div>

      {/* Scroll area: grouped (Pinned band + per-persona accordion) or flat (Pinned + Recent list). */}
      <div className="flex-1 overflow-y-auto px-2.5 mt-3 pb-2">
        {layout === "grouped" ? (
          <div className="space-y-4">
            {pinnedBand()}
            <div className="space-y-0.5">
              {visibleSurfaces.map((s) => {
                const expanded = isExpanded(s.key);
                return (
                  <div key={s.key}>
                    <div
                      className={
                        "flex items-center gap-2.5 px-2 py-2 rounded-lg cursor-pointer select-none " +
                        (isCurrent(s.key) ? "bg-paper" : "hover:bg-paper")
                      }
                      onClick={() => onHeaderClick(s.key)}
                    >
                      {iconTile(s.key)}
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
                      <Icon
                        name={expanded ? "chevronDown" : "chevronRight"}
                        size={15}
                        className="text-faint shrink-0"
                      />
                      {/* Settings gear — the rightmost element; opens the persona page without
                          toggling the accordion (stops propagation). */}
                      <button
                        className="w-6 h-6 grid place-items-center rounded-md text-faint hover:text-ink hover:bg-paper shrink-0"
                        title={`About the ${s.label} persona`}
                        aria-label={`About the ${s.label} persona`}
                        onClick={(e) => {
                          e.stopPropagation();
                          props.onOpenPersona(s.key);
                        }}
                      >
                        <Icon name="sliders" size={14} />
                      </button>
                    </div>
                    {expanded && surfaceBody()}
                  </div>
                );
              })}
            </div>
          </div>
        ) : (
          <div className="space-y-4">
            {pinnedBand()}
            <div>
              <div className="px-1.5 text-[10.5px] uppercase tracking-[0.07em] text-faint font-semibold mb-1">
                Recent
              </div>
              <div className="space-y-0.5">
                {recentSessions.length === 0 ? (
                  <div className="px-2 py-1.5 text-[12px] text-faint leading-snug">
                    {normalizedQuery ? "No matching conversations." : "No conversations yet."}
                  </div>
                ) : (
                  recentSessions.map((s) => cardRow(s, { showIcon: true }))
                )}
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Bottom nav (mock §239): shared destinations. Keeps the app's existing routes. */}
      <div className="px-2.5 py-2 border-t border-line space-y-0.5">
        <button
          className={
            "w-full flex items-center gap-2.5 px-2.5 py-2 rounded-lg text-[13px] text-left " +
            (props.integrationsActive ? "bg-paper text-ink" : "hover:bg-paper")
          }
          onClick={props.onOpenIntegrations}
        >
          <Icon name="plug" size={15} className="shrink-0 text-muted" /> Integrations
        </button>
        <button
          className={
            "w-full flex items-center gap-2.5 px-2.5 py-2 rounded-lg text-[13px] text-left " +
            (props.scheduledActive ? "bg-paper text-ink" : "hover:bg-paper")
          }
          onClick={props.onOpenScheduled}
        >
          <Icon name="clock" size={15} className="shrink-0 text-muted" /> Automations
        </button>
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
        <button
          className={
            "w-full flex items-center gap-2.5 px-2.5 py-2 rounded-lg text-[13px] text-left " +
            (props.auditActive ? "bg-paper text-ink" : "hover:bg-paper")
          }
          onClick={props.onOpenAudit}
        >
          <Icon name="audit" size={15} className="shrink-0 text-muted" /> Activity
        </button>
      </div>

      {/* Footer (mock §254): the cwd path + a settings gear (→ Manage). */}
      <div className="px-3.5 py-2.5 border-t border-line text-[11.5px] text-muted flex items-center justify-between gap-2">
        <span className="truncate" title={props.workspace || undefined}>
          {props.workspace || ""}
        </span>
        <button
          className="text-faint hover:text-ink shrink-0"
          title="Manage"
          aria-label="Manage"
          onClick={props.onManage}
        >
          <svg width={15} height={15} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.8} aria-hidden="true">
            <circle cx="12" cy="12" r="3" />
            <path d="M19 12a7 7 0 00-.1-1l2-1.6-2-3.4-2.3 1a7 7 0 00-1.7-1L16.5 3h-4l-.4 2.4a7 7 0 00-1.7 1l-2.3-1-2 3.4 2 1.6a7 7 0 000 2l-2 1.6 2 3.4 2.3-1a7 7 0 001.7 1l.4 2.4h4l.4-2.4a7 7 0 001.7-1l2.3 1 2-3.4-2-1.6c.1-.3.1-.7.1-1z" />
          </svg>
        </button>
      </div>
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
                    {p.id === "cowork" ? "OpenCoworker" : p.name}
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
