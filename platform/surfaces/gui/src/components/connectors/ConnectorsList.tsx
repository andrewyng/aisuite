import { useState } from "react";
import {
  cloudLogin,
  cloudLogout,
  setCloudTelemetry,
  type CloudStatus,
  type Connector,
} from "../../api";
import { ConnectorBadge } from "../../connectors/ConnectorIcon";
import { AddConnectionModal } from "./AddConnectionModal";
import { CHIP_OK, GRP, GRP_H, FOOT, PILL_QUIET, ROW } from "./ui";

// The Connectors LIST (UX-DECISIONS §21): connected first in their own inset group —
// rows navigate to the connector's detail subpage; problems surface as a chip in the
// list, never one click deep. Available connectors below with a Connect pill.

const AVAILABLE_FOLD = 8; // rows shown before "show all"

export function ConnectorsList({
  connectors,
  cloud,
  onOpen,
  onChanged,
}: {
  connectors: Connector[];
  cloud: CloudStatus | null;
  onOpen: (name: string) => void;
  onChanged: () => void;
}) {
  const [filter, setFilter] = useState("");
  const [showAll, setShowAll] = useState(false);
  const [connecting, setConnecting] = useState<string | null>(null);

  const q = filter.trim().toLowerCase();
  const match = (c: Connector) => !q || c.title.toLowerCase().includes(q) || c.name.includes(q);
  const connected = connectors.filter((c) => c.connected && match(c));
  const available = connectors.filter((c) => !c.connected && c.available && match(c));
  const shown = showAll || q ? available : available.slice(0, AVAILABLE_FOLD);
  const connectingC = connecting ? connectors.find((c) => c.name === connecting) : null;

  return (
    <div>
      <div className="flex items-center justify-end mb-4">
        <input
          placeholder="Search"
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          className="w-44 px-3.5 py-1.5 rounded-full border border-line bg-panel text-[13px] outline-none focus:border-accent"
        />
      </div>

      {connected.length > 0 && (
        <>
          <div className={GRP_H + " !mt-0"}>Connected · {connected.length}</div>
          <div className={GRP}>
            {connected.map((c) => (
              <button
                key={c.name}
                data-testid={`connector-${c.name}`}
                className={ROW + " w-full text-left hover:bg-paper/60"}
                onClick={() => onOpen(c.name)}
              >
                <ConnectorBadge connector={c} size={34} title={c.title} />
                <span className="min-w-0 flex-1">
                  <span className="font-medium text-[13.5px]">{c.title}</span>
                  <span className="block text-[12px] text-muted">{statusLine(c)}</span>
                </span>
                {healthChip(c)}
                <span className="text-faint text-[15px] shrink-0">›</span>
              </button>
            ))}
          </div>
        </>
      )}

      <div className={GRP_H}>Available</div>
      <div className={GRP}>
        {shown.map((c) => (
          <div key={c.name} data-testid={`connector-${c.name}`} className={ROW}>
            <ConnectorBadge connector={c} size={34} title={c.title} />
            <span className="min-w-0 flex-1">
              <span className="font-medium text-[13.5px]">{c.title}</span>
              <span className="block text-[12px] text-muted truncate">{c.blurb}</span>
            </span>
            <button className={PILL_QUIET} onClick={() => setConnecting(c.name)}>
              Connect
            </button>
          </div>
        ))}
        {shown.length === 0 && (
          <div className={ROW + " text-[12.5px] text-muted"}>Nothing matches.</div>
        )}
      </div>
      {!showAll && !q && available.length > AVAILABLE_FOLD && (
        <div className={FOOT}>
          {available.length - AVAILABLE_FOLD} more ·{" "}
          <button className="text-muted hover:text-ink" onClick={() => setShowAll(true)}>
            show all
          </button>
        </div>
      )}

      <CloudStrip cloud={cloud} onChanged={onChanged} />

      {connectingC && (
        <AddConnectionModal
          c={connectingC}
          cloud={cloud}
          onClose={() => setConnecting(null)}
          onChanged={onChanged}
        />
      )}
    </div>
  );
}

function statusLine(c: Connector): string {
  if (c.name === "slack" && c.mode === "relay") {
    const n = c.workspaces?.length ?? 0;
    return `${n} workspace${n === 1 ? "" : "s"} · relay`;
  }
  if (c.auth === "none") return "Built in";
  return c.account || "Connected";
}

function healthChip(c: Connector) {
  // Static until the per-connector status endpoints land (M3.6 Step 2): connected
  // two-way relay reads Live; everything else connected reads Ready.
  if (c.name === "slack" && c.mode === "relay")
    return <span className={CHIP_OK}>● Live</span>;
  if (c.two_way && c.connected) return <span className={CHIP_OK}>● Live</span>;
  return <span className={CHIP_OK}>● Ready</span>;
}

// The cloud account, shrunk to a slim strip (was a full card). Keeps the e2e-pinned
// affordances: sign-in/out, account text, and the telemetry opt-out when signed in.
function CloudStrip({ cloud, onChanged }: { cloud: CloudStatus | null; onChanged: () => void }) {
  const [busy, setBusy] = useState(false);
  const [telemetry, setTelemetry] = useState<boolean | null>(null);

  const signIn = async () => {
    setBusy(true);
    await cloudLogin(); // opens the system browser; the section poll picks up completion
    setTimeout(() => {
      setBusy(false);
      onChanged();
    }, 2500);
  };

  return (
    <div className={GRP + " mt-8"} data-testid="cloud-account">
      <div className={ROW}>
        <span
          className={
            "w-2 h-2 rounded-full shrink-0 " + (cloud?.signed_in ? "bg-ok" : "bg-faint/60")
          }
        />
        {cloud?.signed_in ? (
          <span className="min-w-0 flex-1 text-[12.5px] text-muted truncate">
            Signed in as <span className="text-ink font-medium">{cloud.account || "cloud"}</span>
            <span className="text-faint"> — one-click connects enabled</span>
          </span>
        ) : (
          <span className="min-w-0 flex-1 text-[12.5px] text-muted truncate">
            OpenCoworker Cloud: sign in for one-click connects. Manual token setup always works.
          </span>
        )}
        {cloud?.signed_in ? (
          <button
            className="text-[12.5px] text-muted hover:text-ink shrink-0"
            onClick={async () => {
              await cloudLogout();
              onChanged();
            }}
          >
            Sign out
          </button>
        ) : (
          <button className="text-[12.5px] text-accent font-medium shrink-0" onClick={signIn} disabled={busy}>
            {busy ? "Check your browser…" : "Sign in"}
          </button>
        )}
      </div>
      {cloud?.signed_in && (
        <label className={ROW + " select-none"}>
          <input
            type="checkbox"
            checked={telemetry ?? cloud.telemetry_enabled !== false}
            data-testid="telemetry-toggle"
            onChange={async (e) => {
              setTelemetry(e.target.checked);
              await setCloudTelemetry(e.target.checked);
              onChanged();
            }}
          />
          <span className="text-[12px] text-muted">
            Help improve OpenCoworker — which coworker type was started and when; never your
            prompts, files, or connector data.
          </span>
        </label>
      )}
    </div>
  );
}
