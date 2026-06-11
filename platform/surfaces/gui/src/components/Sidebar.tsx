import { useEffect, useMemo, useState } from "react";
import type { RecentWorkspace, SurfaceVisibility } from "../api";
import type { SessionInfo } from "../types";
import { Icon } from "./Icon";

// Session surfaces shown as accordions, in display order. Cowork is always visible; Chat/Code
// are toggled via Settings. Each has a colored icon square.
const SURFACES: { key: string; label: string; icon: "diamond" | "chat" | "code"; cls: string }[] = [
  { key: "cowork", label: "OpenCoworker", icon: "diamond", cls: "ico-cowork" },
  { key: "chat", label: "Chat", icon: "chat", cls: "ico-chat" },
  { key: "code", label: "Code", icon: "code", cls: "ico-code" },
];

interface Props {
  agent: string;
  workspace: string;
  surfaces: SurfaceVisibility;
  sessions: SessionInfo[];
  projects: RecentWorkspace[];
  activeSession: string;
  onSwitchAgent: (agent: string) => void;
  onNewSession: () => void;
  onSelectSession: (id: string, workspace: string, agent: string) => void;
  onNewProject: () => void;
  onRenameSession: (id: string, title: string) => void;
  onDeleteSession: (id: string) => void;
  onManage: () => void;
  onOpenSuperagent: () => void;
  onOpenScheduled: () => void;
  onOpenIntegrations: () => void;
  onOpenAudit: () => void;
  superagentActive: boolean;
  scheduledActive: boolean;
  integrationsActive: boolean;
  auditActive: boolean;
  helperName?: string;
}

const baseName = (p: string) => p.split("/").filter(Boolean).pop() || p;

export function Sidebar(props: Props) {
  const [query, setQuery] = useState("");
  const [searchOpen, setSearchOpen] = useState(false);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editValue, setEditValue] = useState("");
  const [showArchived, setShowArchived] = useState(false);
  const all = props.sessions.filter((s) => s.agent === props.agent && !s.session_id.startsWith("__"));
  const mine = all.filter((s) => !s.archived);
  const archived = all.filter((s) => s.archived);
  // Only Code groups sessions by project folder. Cowork conversations are orphan (each has its
  // own per-conversation scratch dir), so they list flat like Chat.
  const workspaceSurface = props.agent === "code";

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
            <span className="session-actions" onClick={(e) => e.stopPropagation()}>
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
  if (props.workspace) {
    projectOrder.push(props.workspace);
    seen.add(props.workspace);
  }
  for (const s of mine) {
    if (s.workspace && !seen.has(s.workspace)) {
      seen.add(s.workspace);
      projectOrder.push(s.workspace);
    }
  }

  const visibleSurfaces = SURFACES.filter(
    (s) => s.key === "cowork" || props.surfaces[s.key as keyof SurfaceVisibility],
  );

  // Which accordion body is expanded. Follows the active surface, but can be set to null so ALL
  // accordions collapse (clicking the active header toggles it shut without leaving the surface).
  // Decoupled from the main view: opening Integrations/Automations keeps the accordion open.
  const [openKey, setOpenKey] = useState<string | null>(props.agent);
  useEffect(() => setOpenKey(props.agent), [props.agent]);

  const isCurrent = (key: string) => props.agent === key; // the surface you're in
  const isExpanded = (key: string) => openKey === key; // its body is open
  const onHeaderClick = (key: string) => {
    if (isCurrent(key)) setOpenKey((k) => (k === key ? null : key)); // collapse/expand in place
    else props.onSwitchAgent(key); // switch surface (effect re-opens it)
  };

  // The expanded body for the active surface: action rows (New / Search / Integrations /
  // Automations) then the project-grouped (or flat) session list.
  const surfaceBody = () => {
    const isCowork = props.agent === "cowork";
    return (
      <div className="surf-body">
        <div className="surf-actions">
          <div className="surf-action" onClick={props.onNewSession}>
            <Icon name="plus" size={16} className="ico" /> New {isCowork ? "session" : props.agent === "chat" ? "chat" : "session"}
          </div>
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
            <div className="surf-action" onClick={() => setSearchOpen(true)}>
              <Icon name="search" size={16} className="ico" /> Search
            </div>
          )}
          {isCowork && (
            <>
              <div
                className={"surf-action" + (props.integrationsActive ? " active" : "")}
                onClick={props.onOpenIntegrations}
              >
                <Icon name="plug" size={16} className="ico" /> Integrations
              </div>
              <div
                className={"surf-action" + (props.scheduledActive ? " active" : "")}
                onClick={props.onOpenScheduled}
              >
                <Icon name="clock" size={16} className="ico" /> Automations
              </div>
            </>
          )}
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
      </div>

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
                <Icon name={expanded ? "chevronDown" : "chevronRight"} size={16} className="surf-chev" />
              </div>
              {expanded && surfaceBody()}
            </div>
          );
        })}
      </div>

      <div className="sidebar-foot">
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
