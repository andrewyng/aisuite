// SourcesDrawer — the per-session connections panel (§6, mock parity). Opened from the SourcesBar.
// Shows (when a persona is known) a why-connect blurb + "X of Y recommended connected" progress; a
// Connected list whose toggles set per-session overrides (POST /v1/sessions/{id}/connections — mute/
// unmute for THIS session only); a Recommended-connectors list (Connect → deep-link to Integrations);
// and a footer link to the global Integrations surface.
//
// The connection data (`conns`) and the connector index (`byName`, for real brand colors) are owned by
// the SourcesBar and passed down so toggling re-reads via `onReload`, keeping the bar's avatar stack +
// ⚠ count in sync. The persona blurb is fetched here, only when a `personaId` is provided.

import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import {
  getCloudStatus,
  getConnectors,
  getPersonaDetail,
  getRecentChannels,
  getSubscriptions,
  setSessionConnection,
  subscribeChannel,
  unsubscribeChannel,
  type CloudStatus,
  type Connector,
  type PersonaDetail,
  type RecentChannel,
  type SessionConnections,
  type Subscription,
} from "../api";
import { ConnectorBadge } from "../connectors/ConnectorIcon";
import { shortPersonaName } from "../personaScope";
import { Icon } from "./Icon";
import { ConnectSetup } from "./ManageTabs";
import { PersonaGlyph } from "./personaIcon";
import { ChannelPicker } from "./SubscriptionsChip";
import { Toggle } from "./Toggle";
import { labelFor, visualFor, type ConnectorMap } from "../connectors/visuals";

// A channel address's platform: "slack:C0123" → "slack"; a bare id or "#mention" defaults to slack
// (the backend's own default when no platform prefix is given).
const platformOf = (channel: string) => (channel.includes(":") ? channel.split(":")[0] : "slack");

const SEC_H = "text-[11px] uppercase tracking-[0.05em] text-faint font-semibold";
const TAG_CORE =
  "text-[10px] px-1.5 py-0.5 rounded-full bg-warnSoft/70 text-warnInk border border-warnInk/15";
const BTN_ACCENT = "text-[12px] px-2.5 py-1.5 rounded-lg bg-accent text-white shrink-0";
const BTN_BORDERED =
  "text-[12px] px-2.5 py-1.5 rounded-lg border border-line bg-paper hover:border-lineStrong shrink-0";

export function SourcesDrawer({
  sessionId,
  personaId,
  conns,
  byName,
  onReload,
  onClose,
  onOpenIntegrations,
  onOpenPersona,
}: {
  sessionId: string;
  personaId?: string;
  conns: SessionConnections;
  byName: ConnectorMap;
  onReload: () => void;
  onClose: () => void;
  onOpenIntegrations?: () => void;
  onOpenPersona?: (id: string) => void;
}) {
  const [persona, setPersona] = useState<PersonaDetail | null>(null);
  // The connector whose channel list is open as a child panel (null = main Sources view).
  const [channelsFor, setChannelsFor] = useState<string | null>(null);
  // The recommended connector being connected IN CONTEXT (child panel — no detour to the
  // global Connectors page; owner ask, 2026-07-03). Cloud status gates the one-click path.
  const [connectFor, setConnectFor] = useState<Connector | null>(null);
  const [cloud, setCloud] = useState<CloudStatus | null>(null);
  useEffect(() => {
    getCloudStatus().then(setCloud).catch(() => setCloud(null));
  }, []);
  const [subs, setSubs] = useState<Subscription[]>([]);
  const [recent, setRecent] = useState<RecentChannel[]>([]);
  const [draft, setDraft] = useState("");
  // Server rejection of an add (e.g. a bare #name, which can't be looked up) — shown inline.
  const [addErr, setAddErr] = useState<string | null>(null);

  useEffect(() => {
    if (!personaId) return;
    let live = true;
    getPersonaDetail(personaId)
      .then((d) => live && setPersona(d))
      .catch(() => {});
    return () => {
      live = false;
    };
  }, [personaId]);

  // Channel subscriptions (for the two-way-connector drill-down) + the recent-channels datalist.
  const loadSubs = () => getSubscriptions().then(setSubs).catch(() => setSubs([]));
  useEffect(() => {
    loadSubs();
    getRecentChannels().then(setRecent).catch(() => setRecent([]));
  }, []);

  const toggleSession = async (connector: string, next: boolean) => {
    await setSessionConnection(sessionId, connector, next);
    onReload();
  };

  // Channels this session listens to on a given connector (Slack/Telegram/…).
  const channelsOf = (connector: string) =>
    subs.filter((s) => s.session_id === sessionId && platformOf(s.channel) === connector);

  const addChannel = async () => {
    const raw = draft.trim();
    if (!raw || !channelsFor) return;
    // In a connector's panel, a bare id is scoped to that connector; explicit "platform:", a
    // URL, or a "#mention" are passed through as typed (the server parses link/mention forms).
    const channel = raw.includes(":") || raw.startsWith("#") ? raw : `${channelsFor}:${raw}`;
    const r = await subscribeChannel(sessionId, channel);
    if (!r.ok) {
      setAddErr(r.error || "Couldn't add that channel.");
      return;
    }
    setAddErr(null);
    setDraft("");
    loadSubs();
  };
  const removeChannel = async (channel: string) => {
    await unsubscribeChannel(sessionId, channel);
    loadSubs();
  };

  const { connected, recommended } = conns;
  const total = persona?.recommends.length ?? 0;
  const got = persona?.recommends.filter((r) => r.connected).length ?? 0;
  const pct = total > 0 ? Math.round((got / total) * 100) : 0;

  // Portaled to <body>: the opener row is docked inside the glass topbar (UX-008), whose
  // backdrop-filter creates a containing block — `fixed` would resolve against the 48px bar,
  // not the viewport, clipping the drawer. The portal makes the overlay ancestor-proof.
  return createPortal(
    <div className="fixed inset-0 z-40">
      <div className="absolute inset-0 bg-black/30 backdrop-blur-[1px]" onClick={onClose} />
      <aside
        className="absolute right-0 top-0 h-full w-[420px] max-w-[92vw] bg-panel border-l border-line shadow-2xl flex flex-col"
        role="dialog"
        aria-label="Session connections"
      >
        {connectFor ? (
          <ConnectPanel
            c={connectFor}
            cloud={cloud}
            onDone={() => {
              setConnectFor(null);
              onReload();
            }}
            onBack={() => setConnectFor(null)}
            onClose={onClose}
          />
        ) : channelsFor ? (
          <ChannelsPanel
            connector={channelsFor}
            label={labelFor(channelsFor, byName)}
            channels={channelsOf(channelsFor)}
            recent={recent}
            draft={draft}
            onDraft={(v) => {
              setDraft(v);
              setAddErr(null); // typing again clears the last rejection
            }}
            onAdd={addChannel}
            error={addErr}
            onRemove={removeChannel}
            onBack={() => setChannelsFor(null)}
            onClose={onClose}
          />
        ) : (
        <>
        <header className="px-4 h-12 shrink-0 flex items-center gap-2 border-b border-line">
          <span className="w-6 h-6 rounded-md bg-panel border border-line grid place-items-center text-[13px]">
            <PersonaGlyph icon={persona?.icon} size={13} />
          </span>
          <div className="min-w-0">
            <div className="text-[13px] font-semibold leading-tight truncate">
              {persona ? shortPersonaName(persona.name, personaId) : "Sources"}
            </div>
            <div className="text-[11px] text-faint leading-tight">Session connections</div>
          </div>
          <button
            className="ml-auto w-7 h-7 grid place-items-center rounded-md text-faint hover:text-ink hover:bg-paper"
            onClick={onClose}
            aria-label="Close"
          >
            ✕
          </button>
        </header>

        <div className="flex-1 overflow-y-auto hairline-scroll px-4 py-4 space-y-5">
          {/* persona why-connect blurb (when a persona is known) */}
          {persona && (
            <div className="rounded-xl2 border border-accent/25 bg-accentSoft/50 p-3.5">
              <div className="flex items-center gap-2 text-[13px] font-semibold">
                <span>⚡</span> Get the most out of {shortPersonaName(persona.name, personaId)}
              </div>
              {persona.description && (
                <p className="text-[12.5px] text-muted mt-1.5 leading-relaxed">{persona.description}</p>
              )}
              {total > 0 && (
                <>
                  <div className="mt-2 h-1.5 rounded-full bg-panel overflow-hidden">
                    <div className="h-full bg-accent" style={{ width: `${pct}%` }} />
                  </div>
                  <div className="mt-1 text-[11px] text-faint">
                    {got} of {total} recommended connected
                  </div>
                </>
              )}
              {onOpenPersona && (
                <button
                  className="mt-2 inline-block text-[12px] font-medium text-accent hover:underline"
                  onClick={() => onOpenPersona(persona.id)}
                >
                  About this persona →
                </button>
              )}
            </div>
          )}

          {/* connected — each toggle is a per-session override (mute for this session only) */}
          <section>
            <div className="flex items-center justify-between mb-2">
              <div className={SEC_H}>Connected · {connected.length}</div>
              <div className="text-[10.5px] text-faint">enabled for this session</div>
            </div>
            <div className="space-y-1.5">
              {connected.length === 0 && (
                <div className="text-[12.5px] text-faint px-0.5 py-1">
                  No connectors enabled for this session.
                </div>
              )}
              {connected.map((c) => (
                <div
                  className="flex items-center gap-3 p-2.5 rounded-xl2 border border-line bg-paper"
                  key={c.connector}
                >
                  <ConnectorBadge connector={visualFor(c.connector, "connector", byName)} size={32} />
                  <div className="min-w-0 flex-1">
                    <div className="text-[13px] font-medium">{labelFor(c.connector, byName)}</div>
                    {c.detail && <div className="text-[12px] text-muted truncate">{c.detail}</div>}
                    {/* Two-way messaging connectors (Slack/Telegram): drill into the channels this
                        session listens to. */}
                    {byName[c.connector]?.two_way && (
                      <button
                        className="mt-1 inline-flex items-center gap-0.5 text-[11.5px] text-accent hover:underline"
                        onClick={() => {
                          setDraft("");
                          setChannelsFor(c.connector);
                        }}
                      >
                        Channels · {channelsOf(c.connector).length}
                        <Icon name="chevronRight" size={11} />
                      </button>
                    )}
                  </div>
                  <Toggle
                    checked={c.enabled}
                    onChange={(next) => toggleSession(c.connector, next)}
                    title="Enabled for this session — tap to mute here"
                  />
                </div>
              ))}
            </div>
            {connected.length > 0 && (
              <p className="text-[11px] text-faint mt-1.5">
                Turning one off mutes it for <b>this session only</b> — the connector stays connected
                and enabled for the persona.
              </p>
            )}
          </section>

          {/* recommended connectors (not yet connected) */}
          {recommended.length > 0 && (
            <section>
              <div className={`${SEC_H} mb-2`}>Recommended connectors</div>
              <div className="space-y-1.5">
                {recommended.map((r) => (
                  <div
                    className="flex items-start gap-3 p-2.5 rounded-xl2 border border-line"
                    key={r.connector}
                  >
                    <ConnectorBadge connector={visualFor(r.connector, "connector", byName)} size={32} />
                    <div className="min-w-0 flex-1">
                      <div className="flex items-center gap-2">
                        <span className="text-[13px] font-medium">{labelFor(r.connector, byName)}</span>
                        {r.tier === "core" && <span className={TAG_CORE}>core</span>}
                      </div>
                      <div className="text-[12px] text-muted">{r.reason}</div>
                    </div>
                    <button
                      className={r.tier === "core" ? BTN_ACCENT : BTN_BORDERED}
                      onClick={() => {
                        // Connect IN CONTEXT when we ship this connector; unknown refs
                        // (no descriptor) still fall back to the global page.
                        const desc = byName[r.connector];
                        if (desc) setConnectFor(desc);
                        else onOpenIntegrations?.();
                      }}
                    >
                      Connect
                    </button>
                  </div>
                ))}
              </div>
            </section>
          )}
        </div>

        <footer className="px-4 py-3 border-t border-line">
          <button
            className="w-full text-[12.5px] text-accent font-medium hover:underline text-left"
            onClick={() => {
              onClose();
              onOpenIntegrations?.();
            }}
          >
            Manage all connectors (global) →
          </button>
        </footer>
        </>
        )}
      </aside>
    </div>,
    document.body,
  );
}

// Connect-in-context (§ Sources child panel): the same ConnectSetup the global Connectors page
// uses (one-click managed when signed in + manual fields always), hosted inside the drawer so
// connecting a recommended connector never navigates away from the session. Managed connects
// complete out-of-band (browser → broker → sidecar), so poll until the connector flips.
function ConnectPanel({
  c,
  cloud,
  onDone,
  onBack,
  onClose,
}: {
  c: Connector;
  cloud: CloudStatus | null;
  onDone: () => void;
  onBack: () => void;
  onClose: () => void;
}) {
  useEffect(() => {
    const t = setInterval(async () => {
      try {
        const list = await getConnectors();
        if (list.find((x) => x.name === c.name)?.connected) onDone();
      } catch {
        /* poll again */
      }
    }, 2500);
    return () => clearInterval(t);
  }, [c.name, onDone]);

  return (
    <>
      <header className="px-3 h-12 shrink-0 flex items-center gap-1.5 border-b border-line">
        <button
          className="w-7 h-7 grid place-items-center rounded-md text-faint hover:text-ink hover:bg-paper shrink-0"
          onClick={onBack}
          aria-label="Back to sources"
        >
          <Icon name="arrowLeft" size={16} />
        </button>
        <div className="min-w-0">
          <div className="text-[13px] font-semibold leading-tight truncate">Connect {c.title}</div>
          <div className="text-[11px] text-faint leading-tight">Stays in this session</div>
        </div>
        <button
          className="ml-auto w-7 h-7 grid place-items-center rounded-md text-faint hover:text-ink hover:bg-paper"
          onClick={onClose}
          aria-label="Close"
        >
          ✕
        </button>
      </header>
      <div className="flex-1 overflow-y-auto hairline-scroll px-4 py-3">
        {c.blurb && <p className="text-[12.5px] text-muted mb-1 leading-relaxed">{c.blurb}</p>}
        <div className="-mx-3.5">
          <ConnectSetup c={c} cloud={cloud} onConnected={onDone} />
        </div>
      </div>
    </>
  );
}

// The per-connector channels drill-down (§ Sources child panel): which channels THIS session listens
// to on a two-way messaging connector (Slack/Telegram). Reached from the connector's "Channels · N"
// row; ‹ back returns to the Sources list.
function ChannelsPanel({
  label,
  channels,
  recent,
  draft,
  onDraft,
  onAdd,
  error,
  onRemove,
  onBack,
  onClose,
}: {
  connector: string;
  label: string;
  channels: Subscription[];
  recent: RecentChannel[];
  draft: string;
  onDraft: (v: string) => void;
  onAdd: () => void;
  error?: string | null;
  onRemove: (channel: string) => void;
  onBack: () => void;
  onClose: () => void;
}) {
  return (
    <>
      <header className="px-3 h-12 shrink-0 flex items-center gap-1.5 border-b border-line">
        <button
          className="w-7 h-7 grid place-items-center rounded-md text-faint hover:text-ink hover:bg-paper shrink-0"
          onClick={onBack}
          aria-label="Back to sources"
        >
          <Icon name="arrowLeft" size={16} />
        </button>
        <div className="min-w-0">
          <div className="text-[13px] font-semibold leading-tight truncate">{label} channels</div>
          <div className="text-[11px] text-faint leading-tight">This session listens to</div>
        </div>
        <button
          className="ml-auto w-7 h-7 grid place-items-center rounded-md text-faint hover:text-ink hover:bg-paper"
          onClick={onClose}
          aria-label="Close"
        >
          ✕
        </button>
      </header>

      <div className="flex-1 overflow-y-auto hairline-scroll px-4 py-4 space-y-5">
        <section>
          <div className={`${SEC_H} mb-2`}>Subscribed channels · {channels.length}</div>
          {channels.length === 0 ? (
            <div className="text-[12.5px] text-faint px-0.5 py-1">
              Not listening to any {label} channel yet.
            </div>
          ) : (
            <div className="space-y-1.5">
              {channels.map((s) => (
                <div
                  className="flex items-center gap-2 p-2.5 rounded-xl2 border border-line bg-paper"
                  key={s.channel}
                >
                  <Icon name="plug" size={14} className="text-muted shrink-0" />
                  <span className="min-w-0 flex-1 text-[13px] truncate" title={s.channel}>
                    {s.channel_name ? `#${s.channel_name}` : s.channel}
                    {s.channel_name && (
                      <span className="ml-1.5 text-[11px] text-faint">{s.channel}</span>
                    )}
                  </span>
                  {s.collision && (
                    <span
                      className="text-[10.5px] text-warnInk bg-warnSoft/70 border border-warnInk/15 rounded px-1.5 py-0.5 shrink-0"
                      title="This channel is also this session's Inbox-routing target — inbound and outbound collide."
                    >
                      ⚠
                    </span>
                  )}
                  <button
                    className="w-6 h-6 grid place-items-center text-faint hover:text-danger shrink-0"
                    title="Stop listening"
                    onClick={() => onRemove(s.channel)}
                  >
                    ×
                  </button>
                </div>
              ))}
            </div>
          )}
        </section>

        <section>
          <div className={`${SEC_H} mb-2`}>Add a channel</div>
          <div className="flex items-center gap-2">
            <ChannelPicker value={draft} onChange={onDraft} recent={recent} onSubmit={onAdd} />
            <button className={BTN_ACCENT} disabled={!draft.trim()} onClick={onAdd}>
              Add
            </button>
          </div>
          {error && (
            <p
              className="text-[11.5px] text-warnInk mt-2 leading-relaxed"
              data-testid="channel-add-error"
            >
              {error}
            </p>
          )}
          <p className="text-[11px] text-faint mt-2 leading-relaxed">
            The agent receives messages posted to these channels. Removing one stops this session from
            listening — the connector stays connected.
          </p>
        </section>
      </div>
    </>
  );
}
