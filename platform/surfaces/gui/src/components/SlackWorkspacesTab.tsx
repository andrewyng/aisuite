import { useEffect, useState } from "react";
import {
  connectManaged,
  disconnectSlackWorkspace,
  getCloudStatus,
  getConnectors,
  type CloudStatus,
  type Connector,
  type SlackWorkspace,
} from "../api";
import { AllowlistBlock, ListeningSessionsBlock, UnauthorizedBlock } from "./ManageTabs";

// The dedicated Slack page (M3.5): the managed relay is multi-workspace, and Slack ids are
// workspace-scoped, so everything the crowded connector card used to hold lives here PER
// WORKSPACE — allow-list, recent senders, parked messages — plus Add workspace / Disconnect.
// Manual Socket Mode (paste tokens) still works: it shows as one card with the flat allow-list.
const CARD = "rounded-xl2 border border-line bg-panel";
const BTN_ACCENT =
  "text-[12.5px] px-3 py-1.5 rounded-lg bg-accent text-white shrink-0 disabled:opacity-50";
const BTN_DANGER = "text-[12.5px] text-danger/80 hover:text-danger shrink-0";

export function SlackWorkspacesTab() {
  const [slack, setSlack] = useState<Connector | null>(null);
  const [cloud, setCloud] = useState<CloudStatus | null>(null);
  const [waiting, setWaiting] = useState(false); // Add-workspace browser flow in flight
  const [error, setError] = useState<string | null>(null);

  const refresh = () => {
    getConnectors()
      .then((cs) => setSlack(cs.find((c) => c.name === "slack") ?? null))
      .catch(() => setSlack(null));
    getCloudStatus().then(setCloud).catch(() => setCloud(null));
  };
  useEffect(() => {
    refresh();
    // Managed connect completes in the system browser; recent senders and parked
    // messages arrive over time — poll like the Connectors tab does.
    const t = setInterval(refresh, 5000);
    return () => clearInterval(t);
  }, []);

  const addWorkspace = async () => {
    setError(null);
    const res = await connectManaged("slack");
    if (res.ok) {
      setWaiting(true);
      refresh();
    } else setError(res.error || "could not start the Slack install");
  };

  // The install completes in the system browser; when the poll shows a new workspace,
  // stop the "Check your browser…" state.
  const wsCount = slack?.workspaces?.length ?? 0;
  useEffect(() => {
    if (waiting) setWaiting(false);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [wsCount]);

  if (!slack) return <div className="text-[13px] text-muted">Loading…</div>;

  const relay = slack.mode === "relay";
  const workspaces = slack.workspaces ?? [];
  const manualConnected = slack.connected && !relay;

  return (
    <div className="space-y-3" data-testid="slack-workspaces">
      {/* Mode line: which of the two connect paths is live. One at a time by design. */}
      <div className={CARD + " p-3.5 flex items-center gap-3"}>
        <div className="min-w-0">
          <div className="font-semibold text-[14px] flex items-center gap-2">
            Slack
            {slack.connected && (
              <span
                className="text-[10.5px] px-1.5 py-0.5 rounded border border-line text-muted"
                data-testid="slack-mode-badge"
              >
                {relay ? "managed relay" : "Socket Mode (manual tokens)"}
              </span>
            )}
          </div>
          <div className="text-[12px] text-muted">
            {relay
              ? "One @ocw app, installed per workspace. Each workspace has its own allow-list."
              : manualConnected
                ? "Connected with pasted tokens (single workspace). The managed relay is the multi-workspace path."
                : "Not connected. Add a workspace with one click (needs cloud sign-in), or paste tokens on the Connectors tab."}
          </div>
        </div>
        <div className="ml-auto">
          {cloud?.signed_in ? (
            !manualConnected && (
              <button
                className={BTN_ACCENT}
                data-testid="add-workspace"
                onClick={addWorkspace}
                disabled={waiting}
              >
                {waiting ? "Check your browser…" : "+ Add workspace"}
              </button>
            )
          ) : (
            <span className="text-[12px] text-muted">
              Sign in to OpenCoworker Cloud (Connectors tab) to add workspaces.
            </span>
          )}
        </div>
      </div>
      {error && <div className="text-[12.5px] text-danger">{error}</div>}

      {relay &&
        workspaces.map((w) => (
          <WorkspaceCard key={w.team_id} c={slack} w={w} onChanged={refresh} />
        ))}
      {relay && workspaces.length === 0 && (
        <div className={CARD + " p-4 text-[13px] text-muted"}>
          No workspaces yet — Add workspace opens Slack's consent page in your browser.
        </div>
      )}

      {/* Manual Socket Mode: single workspace, the flat allow-list (unchanged semantics). */}
      {manualConnected && (
        <div className={CARD} data-testid="slack-manual-card">
          <div className="px-3.5 py-3 flex items-center gap-2">
            <span className="font-semibold text-[13.5px]">{slack.account || "workspace"}</span>
            <span className="text-[11.5px] text-faint">manual tokens</span>
          </div>
          <AllowlistBlock c={slack} onChanged={refresh} />
          <UnauthorizedBlock c={slack} onChanged={refresh} />
        </div>
      )}

      {slack.connected && <div className={CARD}><ListeningSessionsBlock c={slack} /></div>}
    </div>
  );
}

function WorkspaceCard({
  c,
  w,
  onChanged,
}: {
  c: Connector;
  w: SlackWorkspace;
  onChanged: () => void;
}) {
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const disconnect = async () => {
    setBusy(true);
    setErr(null);
    const res = await disconnectSlackWorkspace(w.team_id);
    setBusy(false);
    if (res.ok) onChanged();
    else setErr(res.error || "could not disconnect");
  };
  return (
    <div className={CARD} data-testid={`slack-workspace-${w.team_id}`}>
      <div className="px-3.5 py-3 flex items-center gap-2">
        <span className="font-semibold text-[13.5px]">{w.account || w.team_id}</span>
        <span className="text-[11.5px] text-faint">{w.team_id}</span>
        <button
          className={"ml-auto " + BTN_DANGER}
          data-testid={`disconnect-workspace-${w.team_id}`}
          title="Stop relaying this workspace to this computer. The @ocw app stays installed in Slack."
          onClick={disconnect}
          disabled={busy}
        >
          {busy ? "Disconnecting…" : "Disconnect"}
        </button>
      </div>
      {err && <div className="px-3.5 pb-2 text-[12.5px] text-danger">{err}</div>}
      <AllowlistBlock
        c={c}
        onChanged={onChanged}
        teamId={w.team_id}
        allowed={w.allowed_users}
        allowedNames={w.allowed_user_names}
      />
      <UnauthorizedBlock c={c} onChanged={onChanged} teamId={w.team_id} />
    </div>
  );
}
