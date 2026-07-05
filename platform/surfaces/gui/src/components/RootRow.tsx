import type { RootInfo } from "../api";
import { Icon } from "./Icon";
import { baseName } from "../paths";

// One directory row, shared by the composer popover and the session start panel. The primary is the
// session's bound workspace — the repo/folder for Code/Ops (shown by name), or a throwaway scratch
// for Cowork (shown as "Temporary space"). It's always read-write and can't be removed.
export function RootRow({
  root,
  busy,
  scratchPrimary,
  onToggle,
  onRemove,
}: {
  root: RootInfo;
  busy?: boolean;
  scratchPrimary?: boolean;
  onToggle: (r: RootInfo) => void;
  onRemove: (path: string) => void;
}) {
  const label = root.primary
    ? scratchPrimary
      ? "Temporary space"
      : baseName(root.path)
    : root.label;
  return (
    <div className={"root-row" + (root.exists ? "" : " missing")}>
      <Icon name="folder" size={14} className="root-ico" />
      <span className="root-text" title={root.path}>
        <span className="root-label">
          {label}
          {root.primary && !scratchPrimary && <span className="root-tag"> main</span>}
        </span>
        <span className="root-path">{root.path}</span>
      </span>
      {!root.exists && <span className="root-tag warn">missing</span>}
      <button
        className={"root-access" + (root.writable ? " rw" : " ro")}
        onClick={() => onToggle(root)}
        disabled={busy || root.primary}
        title={root.primary ? "The main workspace is always read-write" : "Toggle read-only / read-write"}
      >
        {root.writable ? "Read-write" : "Read-only"}
      </button>
      {!root.primary && (
        <button className="root-x" onClick={() => onRemove(root.path)} disabled={busy} title="Remove">
          ×
        </button>
      )}
    </div>
  );
}
