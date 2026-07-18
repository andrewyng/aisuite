import { useState } from "react";
import { cloudLogin } from "../../api";

// The signed-out state of every one-click pane: a REAL sign-in button, not a
// hint pointing at another page. Sign-in completes in the system browser; the
// Connectors section's 5s poll picks it up and the pane re-renders signed in.
export function CloudSignInInline({ blurb }: { blurb?: string }) {
  const [waiting, setWaiting] = useState(false);
  return (
    <div className="space-y-1.5">
      <button
        className="w-full px-3 py-2 rounded-lg border border-accent text-accent text-[13px] font-medium hover:bg-accentSoft/40"
        data-testid="inline-cloud-sign-in"
        onClick={async () => {
          setWaiting(true);
          await cloudLogin();
          setTimeout(() => setWaiting(false), 4000);
        }}
      >
        {waiting ? "Check your browser…" : "Sign in to OpenWorker Cloud"}
      </button>
      <div className="text-[11.5px] text-faint">
        {blurb || "Sign-in unlocks one-click connects — or switch to Manual, which works without it."}
      </div>
    </div>
  );
}
