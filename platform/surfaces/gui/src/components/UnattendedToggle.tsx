import { useEffect, useState } from "react";
import { getUnattended, setUnattended } from "../api";

// Per-session Unattended toggle. Turning it on is a one-tap confirm (handing over control):
// from then on the agent's approvals/questions route to the Inbox instead of prompting inline.
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

  if (on) {
    return (
      <button
        className="wschip unattended-on"
        title="Approvals and questions are going to the Inbox — tap to take back control"
        onClick={disable}
      >
        ● Unattended
      </button>
    );
  }
  if (confirming) {
    return (
      <span className="unattended-confirm">
        Run unattended? Approvals go to the Inbox.
        <button className="btn-primary sm" onClick={enable}>
          Confirm
        </button>
        <button className="btn sm" onClick={() => setConfirming(false)}>
          Cancel
        </button>
      </span>
    );
  }
  return (
    <button
      className="wschip"
      title="Step away — route approvals/questions to the Inbox"
      onClick={() => setConfirming(true)}
    >
      Unattended
    </button>
  );
}
