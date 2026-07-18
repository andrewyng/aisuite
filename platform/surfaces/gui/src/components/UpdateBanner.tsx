import { useEffect, useRef, useState } from "react";
import { checkForUpdate, installUpdate, isTauri, type UpdateInfo } from "../tauri";

// Auto-update prompt (desktop shell only — the browser build never renders this).
// Deliberately a PROMPT, not a silent background install: swapping the app under a
// user mid-session would kill their running coworker turn, and quiet self-mutation
// sits badly with the local-first trust posture. "Later" dismisses until next launch.
//
// The check runs once, shortly after boot settles (the splash and session restore own
// the first seconds). Update integrity is enforced below this layer: the shell verifies
// the manifest's minisign signature against the pubkey compiled into tauri.conf.json.

export function UpdateBanner() {
  const [update, setUpdate] = useState<UpdateInfo | null>(null);
  const [phase, setPhase] = useState<"idle" | "installing" | "error">("idle");
  const checked = useRef(false);

  useEffect(() => {
    if (!isTauri() || checked.current) return;
    checked.current = true;
    const t = setTimeout(() => {
      checkForUpdate().then((u) => u && setUpdate(u)).catch(() => {});
    }, 15_000);
    return () => clearTimeout(t);
  }, []);

  if (!update) return null;

  const install = async () => {
    setPhase("installing");
    try {
      await installUpdate(); // success restarts the app — nothing to do after
    } catch {
      setPhase("error");
    }
  };

  return (
    <div
      className="fixed bottom-4 right-4 z-50 w-[300px] rounded-xl border border-line bg-panel shadow-2xl px-4 py-3.5"
      role="status"
      data-testid="update-banner"
    >
      <div className="text-[13px] font-semibold">Update available</div>
      <div className="text-[12px] text-muted mt-0.5">
        OpenWorker v{update.version} is ready to install.
      </div>
      {phase === "error" && (
        <div className="text-[11.5px] text-warnInk mt-1.5">
          The update couldn't be installed — it will be offered again next launch.
        </div>
      )}
      <div className="flex items-center gap-2 mt-2.5">
        <button
          className="px-3 py-1.5 rounded-full bg-accent text-white text-[12.5px] disabled:opacity-50"
          onClick={install}
          disabled={phase === "installing"}
          data-testid="update-install"
        >
          {phase === "installing" ? "Downloading…" : "Restart to update"}
        </button>
        <button
          className="px-2 py-1.5 text-[12.5px] text-faint hover:text-muted"
          onClick={() => setUpdate(null)}
          disabled={phase === "installing"}
          data-testid="update-later"
        >
          Later
        </button>
      </div>
    </div>
  );
}
