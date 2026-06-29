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
import { Icon } from "./Icon";

// Session surfaces shown as accordions, in display order. The surfaced personas drive this list
// (so third-party / Ops personas appear); the hardcoded set is the fallback before personas load.
const SURFACES: { key: string; label: string; icon: "diamond" | "chat" | "code"; cls: string }[] = [
  { key: "cowork", label: "OpenCoworker", icon: "diamond", cls: "ico-cowork" },
  { key: "chat", label: "Chat", icon: "chat", cls: "ico-chat" },
  { key: "code", label: "Code", icon: "code", cls: "ico-code" },
];

const ICON_FOR: Record<string, "diamond" | "chat" | "code"> = {
  cowork: "diamond",
  chat: "chat",
  code: "code",
};
const surfaceFromPersona = (p: Persona) => ({
  key: p.id,
  label: p.id === "cowork" ? "OpenCoworker" : p.name,
  icon: ICON_FOR[p.icon] ?? "diamond",
  cls: `ico-${p.icon || "cowork"}`,
});

// Attention = Inbox items awaiting a session (an amber count that bubbles session → persona →
// footer Inbox — all views of the one Inbox queue, never a second list).
function AttnBadge({ n }: { n: number }) {
  if (!n) return null;
  return (
    <span className="attn-badge" title={`${n} awaiting your attention`}>
      {n > 99 ? "99+" : n}
    </span>
  );
}

// Liveness = working (in-flight turn) / sleeping (a self-wake is pending). A count-less dot that
// never bubbles — it says "this is alive", not "this needs you".
function LiveDot({ state }: { state?: "working" | "sleeping" | "idle" }) {
  if (state !== "working" && state !== "sleeping") return null;
  return (
    <span
      className={"live-dot live-" + state}
      title={state === "working" ? "Working now" : "Sleeping (will wake itself)"}
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
  onNewProject: () => void;
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
  const familyOf = (id: string) => personas?.find((p) => p.id === id)?.family;

  // Sidebar layout (§7): "flat" = the persona accordions; "grouped" = bounded per-persona cards.
  // Read the persisted preference on load (absent → flat), write via setNavLayout when toggled.
  const [layout, setLayout] = useState<"flat" | "grouped">("flat");
  useEffect(() => {
    getSettings()
      .then((s) => setLayout(s.nav_layout === "grouped" ? "grouped" : "flat"))
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
  const personaIcon = (agentId: string) => {
    const p = personas?.find((x) => x.id === agentId);
    return { icon: (p && ICON_FOR[p.icon]) || "diamond", cls: `ico-${p?.icon || "cowork"}` } as const;
  };

  // Roll the per-session attention/liveness up to the persona header and the footer Inbox: the
  // amber count bubbles (sum), the liveness dot aggregates (working wins over sleeping).
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
  // Only the CODE FAMILY groups sessions by project (git-bound). Knowledge-family conversations
  // are orphan (each has its own per-conversation scratch dir), so they list flat. Family-aware,
  // not id-aware — so a DevOps/SecOps code-family persona also gets Projects.
  const workspaceSurface = familyOf(browseKey) === "code";

  const normalizedQuery = query.trim().toLowerCase();
  const matches = (s: SessionInfo) =>
    !normalizedQuery ||
    (s.title || s.session_id).toLowerCase().includes(normalizedQuery) ||
    s.session_id.toLowerCase().includes(normalizedQuery);

  const sessionRow = (s: SessionInfo) => {
    const title = s.title || s.session_id;
    const editing = editingId === s.session_id;
    const commitRename = () => {
      const next = editValue.trim();
      if (next && next !== title) props.onRenameSession(s.session_id, next);
      setEditingId(null);
    };
    return (
      <div
        key={s.session_id}
        className={"session" + (s.session_id === props.activeSession ? " active" : "")}
        onClick={() => {
          if (!editing) props.onSelectSession(s.session_id, s.workspace, s.agent);
        }}
        title={editing ? undefined : title}
      >
        {editing ? (
          <input
            className="session-edit"
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
            <span className="session-title">
              {s.pinned && <Icon name="pin" size={11} className="session-pin" />}
              {title}
            </span>
            <span className="session-meta">
              <LiveDot state={s.liveness} />
              <AttnBadge n={s.attention || 0} />
            </span>
            <span className="session-actions" onClick={(e) => e.stopPropagation()}>
              <button
                title={s.pinned ? "Unpin" : "Pin to top"}
                className={s.pinned ? "pinned" : undefined}
                onClick={() => props.onTogglePin(s.session_id, !s.pinned)}
              >
                <Icon name="pin" size={12} />
              </button>
              <button
                title="Rename"
                onClick={() => {
                  setEditingId(s.session_id);
                  setEditValue(title);
                }}
              >
                <Icon name="pencil" size={12} />
              </button>
              <button
                title="Delete"
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

  // The expanded body for the active surface: action rows (New / Search / Integrations /
  // Automations) then the project-grouped (or flat) session list.
  const surfaceBody = () => {
    return (
      <div className="surf-body">
        <div className="surf-actions">
          <div className="surf-action" onClick={() => props.onNewSession(browseKey)}>
            <Icon name="plus" size={16} className="ico" /> New {browseKey === "chat" ? "chat" : "session"}
          </div>
        </div>

        {workspaceSurface ? (
          <>
            <div className="section-label">Projects</div>
            <div className="sessions">
              <div className="newbtn newbtn-secondary" onClick={props.onNewProject}>
                <Icon name="folderPlus" size={18} className="ico" /> New project
              </div>
              {projectOrder.map((proj) => (
                <div className="proj-group" key={proj}>
                  <div className={"proj-head" + (proj === props.workspace ? " current" : "")} title={proj}>
                    <Icon name="folder" size={18} className="ico" />
                    <span className="pname">{baseName(proj)}</span>
                  </div>
                  {(filteredByProject.get(proj) || []).length > 0 ? (
                    (filteredByProject.get(proj) || []).map(sessionRow)
                  ) : (
                    <div className="session-empty">
                      {normalizedQuery ? "No matching conversations." : "No conversations in this project yet."}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </>
        ) : (
          <>
            <div className="section-label">Recents</div>
            <div className="sessions">
              {mine.filter(matches).length === 0 ? (
                <div className="session-empty">
                  {normalizedQuery ? "No matching conversations." : "No conversations yet."}
                </div>
              ) : (
                mine.filter(matches).map(sessionRow)
              )}
            </div>
          </>
        )}

        {archived.length > 0 && (
          <div className="archived-block">
            <div className="archived-head" onClick={() => setShowArchived((v) => !v)}>
              <Icon name={showArchived ? "chevronDown" : "chevronRight"} size={14} />
              Archived ({archived.length})
            </div>
            {showArchived && <div className="sessions">{archived.filter(matches).map(sessionRow)}</div>}
          </div>
        )}
      </div>
    );
  };

  // Grouped layout: one bounded card per persona (sessions grouped by their `agent`), each with a
  // gear → PersonaView and an inline "New session". A flat alternative to the accordion surfaces.
  const groupedList = () => (
    <>
      {visibleSurfaces.map((s) => {
        const list = props.sessions
          .filter((x) => x.agent === s.key && !x.session_id.startsWith("__") && !x.archived)
          .filter(matches);
        return (
          <div className="persona-group" key={s.key}>
            <div className="persona-group-head">
              <span className={"persona-group-ico " + s.cls}>
                <Icon name={s.icon} size={12} />
              </span>
              <span className="persona-group-name">{s.label}</span>
              <span className="persona-group-count">{list.length}</span>
              <LiveDot state={liveByPersona.get(s.key)} />
              <AttnBadge n={attnByPersona.get(s.key) || 0} />
              <button
                className="persona-gear"
                title={`About the ${s.label} persona`}
                aria-label={`About the ${s.label} persona`}
                onClick={() => props.onOpenPersona(s.key)}
              >
                <Icon name="sliders" size={13} />
              </button>
            </div>
            <div className="persona-group-body">
              <div className="surf-action" onClick={() => props.onNewSession(s.key)}>
                <Icon name="plus" size={14} className="ico" /> New {s.key === "chat" ? "chat" : "session"}
              </div>
              {list.length === 0 ? (
                <div className="session-empty">
                  {normalizedQuery ? "No matching conversations." : "No conversations yet."}
                </div>
              ) : (
                list.map(sessionRow)
              )}
            </div>
          </div>
        );
      })}
    </>
  );

  return (
    <div className="sidebar">
      <div className="brand">
        {/* Multi-color mark: a green → purple gradient tying the surface palette together. */}
        <svg className="mark" width={17} height={17} viewBox="0 0 24 24" aria-hidden="true">
          <defs>
            <linearGradient id="ow-mark" x1="3" y1="3" x2="21" y2="21" gradientUnits="userSpaceOnUse">
              <stop offset="0" stopColor="#16a34a" />
              <stop offset="1" stopColor="#7c3aed" />
            </linearGradient>
          </defs>
          <path
            d="M12 2.4c.5 4.7 2.5 6.7 7.2 7.2-4.7.5-6.7 2.5-7.2 7.2-.5-4.7-2.5-6.7-7.2-7.2 4.7-.5 6.7-2.5 7.2-7.2z"
            fill="url(#ow-mark)"
          />
        </svg>
        <span className="name">OpenCoworker</span>
        {/* Layout toggle by the wordmark: flat (persona accordions) ↔ grouped (per-persona cards). */}
        <button
          className="layout-toggle"
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

      {/* Shared zone: capabilities common to ALL coworkers, above the persona accordions. */}
      <div className="shared-nav">
        {searchOpen ? (
          <div className="surf-search">
            <Icon name="search" size={15} className="ico" />
            <input
              className="surf-search-input"
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
              className="surf-search-x"
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
          <div className="shared-link" onClick={() => setSearchOpen(true)}>
            <Icon name="search" size={16} className="ico" /> Search
          </div>
        )}
        <div
          className={"shared-link" + (props.integrationsActive ? " active" : "")}
          onClick={props.onOpenIntegrations}
        >
          <Icon name="plug" size={16} className="ico" /> Integrations
        </div>
        <div
          className={"shared-link" + (props.scheduledActive ? " active" : "")}
          onClick={props.onOpenScheduled}
        >
          <Icon name="clock" size={16} className="ico" /> Automations
        </div>
      </div>

      {layout === "flat" && pinnedSessions.length > 0 && (
        <div className="pinned-band">
          <div className="pinned-label">Pinned</div>
          {pinnedSessions.map((s) => {
            const pi = personaIcon(s.agent);
            return (
              <div
                key={s.session_id}
                className={"pinned-row" + (s.session_id === props.activeSession ? " active" : "")}
                title={s.title || s.session_id}
                onClick={() => props.onSelectSession(s.session_id, s.workspace, s.agent)}
              >
                <span className={"pinned-ico " + pi.cls}>
                  <Icon name={pi.icon} size={11} />
                </span>
                <span className="pinned-title">{s.title || s.session_id}</span>
                <button
                  className="pinned-unpin"
                  title="Unpin"
                  onClick={(e) => {
                    e.stopPropagation();
                    props.onTogglePin(s.session_id, false);
                  }}
                >
                  <Icon name="pin" size={11} />
                </button>
              </div>
            );
          })}
        </div>
      )}

      {layout === "grouped" ? (
        <div className="surfaces persona-groups">{groupedList()}</div>
      ) : (
      <div className="surfaces">
        {visibleSurfaces.map((s) => {
          const expanded = isExpanded(s.key);
          return (
            <div className={"surf" + (expanded ? " open" : "")} key={s.key}>
              <div
                className={"surf-head" + (isCurrent(s.key) ? " active" : "")}
                onClick={() => onHeaderClick(s.key)}
              >
                <span className={"surf-ico " + s.cls}>
                  <Icon name={s.icon} size={13} />
                </span>
                <span className="surf-label">{s.label}</span>
                <LiveDot state={liveByPersona.get(s.key)} />
                <AttnBadge n={attnByPersona.get(s.key) || 0} />
                <Icon name={expanded ? "chevronDown" : "chevronRight"} size={16} className="surf-chev" />
              </div>
              {expanded && surfaceBody()}
            </div>
          );
        })}
      </div>
      )}

      <div className="sidebar-foot">
        <div
          className={"manage-link" + (props.inboxActive ? " active" : "")}
          onClick={props.onOpenInbox}
        >
          <Icon name="chat" size={15} className="ico" /> Inbox
          <AttnBadge n={totalAttention} />
        </div>
        <div
          className={"manage-link" + (props.auditActive ? " active" : "")}
          onClick={props.onOpenAudit}
        >
          <Icon name="audit" size={15} className="ico" /> Audit
        </div>
        <div className="manage-link" onClick={props.onManage}>
          <Icon name="sliders" size={15} className="ico" /> Manage
        </div>
        {workspaceSurface && (
          <div className="ws" title={props.workspace}>
            {props.workspace || "—"}
          </div>
        )}
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
    <div className="newsplit">
      <div className="newsplit-row">
        <button className="newsplit-primary" onClick={() => onNew(current)}>
          <Icon name="plus" size={15} /> New session
        </button>
        <button
          className="newsplit-caret"
          title="Start with a specific persona"
          aria-label="Choose a persona"
          onClick={() => setOpen((v) => !v)}
        >
          <Icon name="chevronDown" size={13} />
        </button>
      </div>
      {open && (
        <>
          <div className="newsplit-backdrop" onClick={() => setOpen(false)} />
          <div className="newsplit-menu">
            <div className="newsplit-menu-label">Start a session as</div>
            {enabled.map((p) => (
              <button
                key={p.id}
                className="newsplit-item"
                onClick={() => {
                  setOpen(false);
                  onNew(p.id);
                }}
              >
                <span className={"newsplit-item-ico ico-" + (p.icon || "cowork")}>
                  <Icon name={ICON_FOR[p.icon] ?? "diamond"} size={12} />
                </span>
                <span className="newsplit-item-text">
                  <span className="newsplit-item-name">{p.id === "cowork" ? "OpenCoworker" : p.name}</span>
                  {p.tagline && <span className="newsplit-item-tag">{p.tagline}</span>}
                </span>
              </button>
            ))}
            <div className="newsplit-menu-sep">
              <button
                className="newsplit-manage"
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
