import type { RootInfo } from "../api";
import { Icon } from "./Icon";

// One directory row, shared by the composer popover and the session start panel. The primary
// scratch dir is shown as "Temporary space" (always read-write, can't be removed).
export function RootRow({
  root,
  busy,
  onToggle,
  onRemove,
}: {
  root: RootInfo;
  busy?: boolean;
  onToggle: (r: RootInfo) => void;
  onRemove: (path: string) => void;
}) {
  const label = root.primary ? "Temporary space" : root.label;
  return (
    <div className={"root-row" + (root.exists ? "" : " missing")}>
      <Icon name="folder" size={14} className="root-ico" />
      <span className="root-text" title={root.path}>
        <span className="root-label">{label}</span>
        <span className="root-path">{root.path}</span>
      </span>
      {!root.exists && <span className="root-tag warn">missing</span>}
      <button
        className={"root-access" + (root.writable ? " rw" : " ro")}
        onClick={() => onToggle(root)}
        disabled={busy || root.primary}
        title={root.primary ? "The temporary space is always read-write" : "Toggle read-only / read-write"}
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
