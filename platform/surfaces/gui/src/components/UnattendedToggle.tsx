import { useEffect, useState } from "react";
import { getUnattended, setUnattended } from "../api";
import { Toggle } from "./Toggle";

// Per-session Unattended toggle: a labeled switch (only the switch carries the accent), not a filled
// pill. Turning it on is a one-tap confirm (handing over control): from then on the agent's
// approvals/questions route to the Inbox instead of prompting inline. Turning it off is immediate.
export function UnattendedToggle({
  sessionId,
  onChange,
}: {
  sessionId: string;
  onChange?: (on: boolean) => void;
}) {
  const [on, setOn] = useState(false);
  const [confirming, setConfirming] = useState(false);
  const report = (v: boolean) => {
    setOn(v);
    onChange?.(v);
  };

  // Reflect the session's actual state (survives reloads / session switches).
  useEffect(() => {
    let alive = true;
    setConfirming(false);
    getUnattended(sessionId)
      .then((v) => alive && report(v))
      .catch(() => {});
    return () => {
      alive = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionId]);

  const enable = async () => {
    await setUnattended(sessionId, true);
    report(true);
    setConfirming(false);
  };
  const disable = async () => {
    await setUnattended(sessionId, false);
    report(false);
  };

  if (confirming) {
    return (
      <span className="inline-flex items-center gap-2 text-[12px] text-muted">
        Run unattended? Approvals route to the Inbox.
        <button className="text-accent font-medium hover:underline" onClick={enable}>
          Confirm
        </button>
        <button className="text-faint hover:text-ink" onClick={() => setConfirming(false)}>
          Cancel
        </button>
      </span>
    );
  }
  return (
    <span
      className="inline-flex items-center gap-1.5"
      title={
        on
          ? "Approvals and questions are going to the Inbox — tap to take back control"
          : "Step away — route approvals/questions to the Inbox"
      }
    >
      <Toggle
        checked={on}
        onChange={(next) => (next ? setConfirming(true) : disable())}
        title="Unattended"
      />
      <span className="text-[12px] text-muted">Unattended</span>
    </span>
  );
}
