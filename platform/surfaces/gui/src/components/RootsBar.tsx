import { useEffect, useRef, useState } from "react";
import { useRoots } from "../useRoots";
import { AddFolderForm } from "./AddFolderForm";
import { Icon } from "./Icon";
import { RootRow } from "./RootRow";

// Cowork's per-conversation directory control, shown in the composer head. A chip summarizing how
// many directories the agent can touch; clicking opens a popover to add folders (read-only or
// read-write), flip a folder's access, or remove it. The primary scratch is fixed and read-write.
export function RootsBar({ sessionId }: { sessionId: string }) {
  const [open, setOpen] = useState(false);
  const [reloadKey, setReloadKey] = useState(0);
  const { roots, busy, error, addRoot, toggleAccess, removeRoot } = useRoots(sessionId, reloadKey);
  const wrap = useRef<HTMLDivElement | null>(null);

  // Refetch when the popover opens, so it reflects folders added since mount (e.g. via the agent).
  useEffect(() => {
    if (open) setReloadKey((k) => k + 1);
  }, [open]);

  // Close the popover on an outside click.
  useEffect(() => {
    if (!open) return;
    const onDoc = (e: MouseEvent) => {
      if (wrap.current && !wrap.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", onDoc);
    return () => document.removeEventListener("mousedown", onDoc);
  }, [open]);

  const count = roots.length;
  return (
    <div className="rootsbar" ref={wrap}>
      <button className="wschip" onClick={() => setOpen((o) => !o)} title="Directories the agent can use">
        <Icon name="folder" size={14} />
        <span className="wsname">
          {count} {count === 1 ? "directory" : "directories"}
        </span>
        <Icon name="chevronDown" size={12} className="edit" />
      </button>

      {open && (
        <div className="roots-pop">
          <div className="roots-head">Directories the agent can use</div>
          {roots.map((r) => (
            <RootRow key={r.path} root={r} busy={busy} onToggle={toggleAccess} onRemove={removeRoot} />
          ))}
          <div className="roots-add">
            <AddFolderForm onAdd={addRoot} busy={busy} />
          </div>
          {error && <div className="roots-err">{error}</div>}
        </div>
      )}
    </div>
  );
}
