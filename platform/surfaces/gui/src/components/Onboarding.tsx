import { useEffect, useRef, useState } from "react";
import {
  cloudLogin,
  connectManaged,
  createAutomation,
  getCloudStatus,
  getConnectors,
  getProviders,
  getRecentChannels,
  setOnboarded,
  setProvider,
  verifyProvider,
  type CloudStatus,
  type Connector,
  type ProviderInfo,
  type RecentChannel,
} from "../api";
import { openExternal } from "../tauri";
import { ConnectorBadge } from "../connectors/ConnectorIcon";
import { ChannelPicker } from "./SubscriptionsChip";
import { Icon } from "./Icon";

// First-run onboarding (UX-DECISIONS §24): model → recipe → tips. Only step 1 gates;
// steps 2–3 are skippable. Replayable anytime from Settings ▸ Appearance ▸ "Run setup again".

// Where a non-developer gets an API key — deep link + one line of instructions.
const KEY_HELP: Record<string, { url: string; label: string }> = {
  anthropic: { url: "https://console.anthropic.com/settings/keys", label: "console.anthropic.com" },
  openai: { url: "https://platform.openai.com/api-keys", label: "platform.openai.com" },
  gemini: { url: "https://aistudio.google.com/apikey", label: "aistudio.google.com" },
  fireworks: { url: "https://fireworks.ai/account/api-keys", label: "fireworks.ai" },
  together: { url: "https://api.together.xyz/settings/api-keys", label: "together.xyz" },
};

// The role tabs (§24 step 2): each names its recipe, the two connectors it needs, and how the
// recipe card assembles. Cron strings are explicit — the user picks a phrase, we map it.
const CADENCES: Record<string, { label: string; cron: string }> = {
  mon9: { label: "Mondays, 9:00", cron: "0 9 * * 1" },
  daily9: { label: "Daily, 9:00", cron: "0 9 * * *" },
  fri17: { label: "Fridays, 17:00", cron: "0 17 * * 5" },
  mon830: { label: "Mondays, 8:30", cron: "30 8 * * 1" },
};

type TabKey = "eng" | "sales" | "everyday";

const TABS: Record<
  TabKey,
  { label: string; line: string; conns: { name: string; why: string }[] }
> = {
  eng: {
    label: "Engineering",
    line: "Every Monday: a digest of merged PRs and commits, posted to your team's Slack.",
    conns: [
      { name: "slack", why: "Where the digest posts" },
      { name: "github", why: "What the digest summarizes" },
    ],
  },
  sales: {
    label: "Sales",
    line: "Every Monday: deals that moved — and deals going quiet — posted to Slack.",
    conns: [
      { name: "slack", why: "Where the digest posts" },
      { name: "hubspot", why: "Pipeline and deal activity" },
    ],
  },
  everyday: {
    label: "Everyday",
    line: "Each morning: your calendar and unread email, summarized before your day starts.",
    conns: [
      { name: "google_calendar", why: "Today's meetings and gaps" },
      { name: "gmail", why: "What arrived overnight" },
    ],
  },
};

type Verify = { state: "idle" | "testing" | "ok" | "error"; msg?: string };

export function Onboarding({ onDone }: { onDone: (next?: "work" | "gallery") => void }) {
  const [step, setStep] = useState(0);

  // -- step 1: model ------------------------------------------------------------
  const [providers, setProviders] = useState<ProviderInfo[]>([]);
  const [prov, setProv] = useState("anthropic");
  const [fields, setFields] = useState<Record<string, string>>({});
  const [model, setModel] = useState("");
  const [verify, setVerify] = useState<Verify>({ state: "idle" });
  const [skipConfirm, setSkipConfirm] = useState(false);

  useEffect(() => {
    getProviders()
      .then((ps) => {
        setProviders(ps);
        // Prefer a provider with a REAL credential; keyless ones (Ollama) report configured
        // without proving anything is running — they must pass Test instead.
        const preferred =
          ps.find((p) => p.configured && p.needs_key) ||
          ps.find((p) => p.name === "anthropic") ||
          ps[0];
        if (preferred) pickProvider(preferred.name, ps);
      })
      .catch(() => {});
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const info = providers.find((p) => p.name === prov);

  const pickProvider = (name: string, list?: ProviderInfo[]) => {
    const p = (list || providers).find((x) => x.name === name);
    setProv(name);
    setVerify({ state: "idle" });
    const next: Record<string, string> = {};
    for (const f of p?.fields || []) next[f.key] = p?.values?.[f.key] || f.default || "";
    setFields(next);
    setModel(p?.recommended_model || p?.suggested_models?.[0] || "");
  };

  const runTest = async () => {
    setVerify({ state: "testing" });
    const res = await verifyProvider(prov, fields).catch(() => ({ ok: false, error: "unreachable" }));
    setVerify(res.ok ? { state: "ok" } : { state: "error", msg: res.error || "couldn't verify" });
  };

  const credentialed = !!info?.configured && !!info?.needs_key;
  const ready = credentialed || verify.state === "ok";

  const saveAndContinue = async () => {
    if (!info?.configured || Object.values(fields).some(Boolean)) {
      await setProvider(prov, fields).catch(() => {});
    }
    setStep(1);
  };

  // -- step 2: recipe -------------------------------------------------------------
  const [tab, setTab] = useState<TabKey>("eng");
  const [connectors, setConnectors] = useState<Connector[]>([]);
  const [cloud, setCloud] = useState<CloudStatus | null>(null);
  const [pendingConn, setPendingConn] = useState<string | null>(null);
  const [recent, setRecent] = useState<RecentChannel[]>([]);
  const [repo, setRepo] = useState("");
  const [channel, setChannel] = useState("");
  const [cadence, setCadence] = useState("mon9");
  const [briefTime, setBriefTime] = useState("08:00");
  const [deliver, setDeliver] = useState<"app" | "slack">("app");
  const [consent, setConsent] = useState(true);
  const [creating, setCreating] = useState(false);
  const [recap, setRecap] = useState<string | null>(null);

  const refreshStep2 = () => {
    getConnectors().then(setConnectors).catch(() => {});
    getCloudStatus().then(setCloud).catch(() => {});
  };
  // Poll while on the recipe step: connects and the cloud sign-in land out-of-band.
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  useEffect(() => {
    if (step !== 1) return;
    refreshStep2();
    getRecentChannels().then(setRecent).catch(() => {});
    pollRef.current = setInterval(refreshStep2, 3000);
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, [step]);

  const connState = (name: string) => connectors.find((c) => c.name === name);
  const tabConns = TABS[tab].conns;
  const allConnected = tabConns.every((c) => connState(c.name)?.connected);

  const startConnect = async (name: string) => {
    if (!cloud?.signed_in) {
      setPendingConn(name); // the pane appears; sign-in completes it
      return;
    }
    await connectManaged(name).catch(() => {});
    refreshStep2();
  };

  const signInThenConnect = async () => {
    await cloudLogin().catch(() => {});
    // Poll until the browser flow lands, then finish the pending connect (bounded).
    let polls = 0;
    const t = setInterval(async () => {
      polls += 1;
      const s = await getCloudStatus().catch(() => null);
      if (s?.signed_in) {
        clearInterval(t);
        setCloud(s);
        if (pendingConn) {
          await connectManaged(pendingConn).catch(() => {});
          setPendingConn(null);
          refreshStep2();
        }
      } else if (polls > 90) clearInterval(t);
    }, 2000);
  };

  const create = async () => {
    setCreating(true);
    let payload: Parameters<typeof createAutomation>[0] & { permissions?: unknown[] };
    if (tab === "eng") {
      payload = {
        title: "Weekly GitHub digest",
        instructions:
          `Summarize the past week in the GitHub repository ${repo || "(the connected repository)"}: ` +
          `merged pull requests, notable commits, and anything needing attention. ` +
          `Post the digest to the Slack channel ${channel} using send_message.`,
        cron: CADENCES[cadence].cron,
        permissions: consent && channel ? [{ tool: "send_message", target: channel, access: "write" }] : [],
      };
    } else if (tab === "sales") {
      payload = {
        title: "Weekly pipeline digest",
        instructions:
          `Review HubSpot for the past week: deals that changed stage, deals going quiet, and ` +
          `deals past their close date. Post a short pipeline digest to the Slack channel ` +
          `${channel} using send_message.`,
        cron: CADENCES[cadence].cron,
        permissions: consent && channel ? [{ tool: "send_message", target: channel, access: "write" }] : [],
      };
    } else {
      const [h, m] = briefTime.split(":");
      payload = {
        title: "Morning brief",
        instructions:
          `Prepare a short morning brief: today's calendar events and gaps, plus email that ` +
          `arrived since yesterday evening. ` +
          (deliver === "app" ? "Save it as the session deliverable." : "Send it to me as a Slack DM."),
        cron: `${Number(m) || 0} ${Number(h) || 8} * * *`,
        permissions: [], // reads don't gate; the consent line here is disclosure
      };
    }
    const res = await createAutomation(payload as any).catch(() => ({ ok: false }) as any);
    setCreating(false);
    if (res.ok) {
      const when = tab === "everyday" ? `every day ${briefTime}` : CADENCES[cadence].label;
      setRecap(`${payload.title} · ${when} · manage under Automations`);
      setStep(2);
    }
  };

  const finish = async (next?: "work" | "gallery") => {
    await setOnboarded(true).catch(() => {});
    onDone(next);
  };

  // -- shared bits ----------------------------------------------------------------
  const seg = (on: boolean) =>
    "px-4 py-1.5 rounded-full text-[12.5px] " + (on ? "bg-ink text-panel" : "text-muted hover:text-ink");
  const label = "block text-[12px] text-muted mt-3 mb-1";
  const input =
    "w-full px-3 py-2 rounded-lg border border-line bg-panel text-[13.5px] outline-none focus:border-accent";
  const dots = (
    <div className="flex justify-center gap-2 mb-6">
      {[0, 1, 2].map((i) => (
        <span key={i} className={"w-1.5 h-1.5 rounded-full " + (i <= step ? "bg-accent" : "bg-line")} />
      ))}
    </div>
  );

  return (
    <div className="fixed inset-0 z-50 bg-ink/30 grid place-items-center" data-testid="onboarding">
      <div className="w-[600px] max-w-[92vw] max-h-[88vh] overflow-y-auto rounded-2xl border border-line bg-panel shadow-2xl p-8">
        {dots}

        {step === 0 && (
          <section data-testid="ob-step-model">
            <h1 className="text-[19px] font-semibold">Connect a model</h1>
            <p className="text-[13px] text-muted mt-0.5 mb-5">
              OpenCoworker runs on your own API key — your key and your data stay on this Mac.
            </p>
            <label className={label}>Provider</label>
            <select className={input} value={prov} onChange={(e) => pickProvider(e.target.value)}>
              {providers.map((p) => (
                <option key={p.name} value={p.name}>
                  {p.title}
                  {p.configured && p.needs_key ? " — connected" : ""}
                </option>
              ))}
            </select>
            {info?.blurb && <p className="text-[11.5px] text-faint mt-1">{info.blurb}</p>}

            {(info?.fields || []).map((f) => (
              <div key={f.key}>
                <label className={label}>{f.label}</label>
                <div className="flex gap-2">
                  <input
                    className={input}
                    type={f.secret ? "password" : "text"}
                    placeholder={f.placeholder}
                    value={fields[f.key] || ""}
                    data-testid={`ob-field-${f.key}`}
                    onChange={(e) => {
                      setFields((cur) => ({ ...cur, [f.key]: e.target.value }));
                      setVerify({ state: "idle" });
                    }}
                  />
                </div>
                {f.help && <p className="text-[11.5px] text-faint mt-1">{f.help}</p>}
              </div>
            ))}

            {KEY_HELP[prov] && (
              <p className="text-[11.5px] text-faint mt-2">
                No key yet?{" "}
                <button
                  className="text-accent hover:underline"
                  onClick={() => openExternal(KEY_HELP[prov].url)}
                >
                  Create one at {KEY_HELP[prov].label} ↗
                </button>{" "}
                — takes about a minute.
              </p>
            )}

            <label className={label}>Default model</label>
            <select className={input} value={model} onChange={(e) => setModel(e.target.value)} data-testid="ob-model">
              {Array.from(new Set([info?.recommended_model, ...(info?.suggested_models || []), model].filter(Boolean))).map(
                (m) => (
                  <option key={m as string} value={m as string}>
                    {m}
                    {m === info?.recommended_model ? " — recommended" : ""}
                  </option>
                ),
              )}
            </select>
            <p className="text-[11.5px] text-faint mt-1">Add or hide models later under Settings ▸ Models.</p>

            <div className="flex items-center gap-3 mt-4">
              <button
                className="px-4 py-1.5 rounded-full border border-line text-[13px] hover:bg-paper"
                onClick={runTest}
                disabled={verify.state === "testing"}
                data-testid="ob-test"
              >
                {verify.state === "testing" ? "Testing…" : "Test"}
              </button>
              {verify.state === "ok" && <span className="text-[12.5px] text-ok">✓ Connected</span>}
              {credentialed && verify.state === "idle" && (
                <span className="text-[12.5px] text-ok">✓ Already connected</span>
              )}
              {verify.state === "error" && (
                <span className="text-[12.5px] text-warnInk">{verify.msg}</span>
              )}
            </div>

            <div className="flex items-center mt-7">
              {!skipConfirm ? (
                <button className="text-[12.5px] text-faint hover:text-muted" onClick={() => setSkipConfirm(true)}>
                  Skip setup
                </button>
              ) : (
                <span className="text-[12.5px] text-muted">
                  Nothing works without a model —{" "}
                  <button className="text-accent" onClick={() => finish()}>
                    skip anyway
                  </button>
                </span>
              )}
              <button
                className="ml-auto px-5 py-2 rounded-full bg-ink text-panel text-[13px] disabled:opacity-40"
                disabled={!ready}
                onClick={saveAndContinue}
                data-testid="ob-continue"
              >
                Continue
              </button>
            </div>
          </section>
        )}

        {step === 1 && (
          <section data-testid="ob-step-recipe">
            <h1 className="text-[19px] font-semibold">Get your first automation running</h1>
            <p className="text-[13px] text-muted mt-0.5 mb-4">Pick what sounds most like your week:</p>

            <div className="inline-flex border border-line rounded-full p-0.5 mb-3">
              {(Object.keys(TABS) as TabKey[]).map((k) => (
                <button key={k} className={seg(tab === k)} onClick={() => setTab(k)} data-testid={`ob-tab-${k}`}>
                  {TABS[k].label}
                </button>
              ))}
            </div>
            <p className="text-[13px] bg-paper rounded-lg px-3.5 py-2.5 mb-1">{TABS[tab].line}</p>

            {tabConns.map(({ name, why }) => {
              const c = connState(name);
              return (
                <div key={name} className="flex items-center gap-3 py-2.5 border-b border-line last:border-b-0">
                  {c && <ConnectorBadge connector={c} size={26} title={c.title} />}
                  <span className="min-w-0 flex-1">
                    <span className="block text-[13.5px] font-medium">{c?.title || name}</span>
                    <span className="block text-[11.5px] text-faint">{why}</span>
                  </span>
                  {c?.connected ? (
                    <span className="text-[12.5px] text-ok">✓ Connected</span>
                  ) : (
                    <button
                      className="px-3.5 py-1 rounded-full border border-line text-[12.5px] hover:bg-paper"
                      onClick={() => startConnect(name)}
                      data-testid={`ob-connect-${name}`}
                    >
                      Connect
                    </button>
                  )}
                </div>
              );
            })}

            {pendingConn && !cloud?.signed_in && (
              <div className="bg-accentSoft/50 rounded-xl px-4 py-3 mt-3 text-[12.5px] text-muted" data-testid="ob-cloudpane">
                <span className="block text-[13px] text-ink font-medium">
                  One sign-in unlocks every one-click connection
                </span>
                Connections are brokered by OpenCoworker Cloud — your tokens stay on this Mac.
                <div>
                  <button
                    className="mt-2 px-3.5 py-1 rounded-full border border-line text-[12.5px] text-accent hover:bg-panel"
                    onClick={signInThenConnect}
                    data-testid="ob-cloud-signin"
                  >
                    Sign in to OpenCoworker Cloud
                  </button>
                </div>
              </div>
            )}

            {allConnected && (
              <div className="bg-paper rounded-xl px-4 py-3.5 mt-4" data-testid="ob-recipe">
                {tab !== "everyday" ? (
                  <>
                    {tab === "eng" && (
                      <>
                        <label className={label}>Repository</label>
                        <input
                          className={input}
                          placeholder="owner/repo"
                          value={repo}
                          onChange={(e) => setRepo(e.target.value)}
                          data-testid="ob-repo"
                        />
                      </>
                    )}
                    <label className={label}>Post to channel</label>
                    <div data-testid="ob-channel">
                      <ChannelPicker value={channel} onChange={setChannel} recent={recent} />
                    </div>
                    <p className="text-[11px] text-warnInk mt-1">
                      The bot must be a member of the channel — invite @ocw in Slack if it isn't.
                    </p>
                    <label className={label}>When</label>
                    <select className={input} value={cadence} onChange={(e) => setCadence(e.target.value)}>
                      {Object.entries(CADENCES).map(([k, v]) => (
                        <option key={k} value={k}>
                          {v.label}
                        </option>
                      ))}
                    </select>
                    <label className="flex items-start gap-2.5 mt-3.5 text-[12.5px] text-muted select-none">
                      <input
                        type="checkbox"
                        className="mt-0.5"
                        checked={consent}
                        onChange={(e) => setConsent(e.target.checked)}
                        data-testid="ob-consent"
                      />
                      <span>
                        Allow this automation to post its digest to{" "}
                        <b className="text-ink">{channel || "the channel"}</b> without asking each
                        time. Anything else still asks first.
                      </span>
                    </label>
                  </>
                ) : (
                  <>
                    <label className={label}>When</label>
                    <input
                      className={input}
                      type="time"
                      value={briefTime}
                      onChange={(e) => setBriefTime(e.target.value)}
                    />
                    <label className={label}>Deliver to</label>
                    <select className={input} value={deliver} onChange={(e) => setDeliver(e.target.value as any)}>
                      <option value="app">In the app</option>
                      <option value="slack">Slack DM (connect Slack later)</option>
                    </select>
                    <p className="text-[12.5px] text-muted mt-3">
                      This brief only <b className="text-ink">reads</b> Calendar and Gmail on
                      schedule — reading never needs approval.
                    </p>
                  </>
                )}
              </div>
            )}

            <div className="flex items-center mt-6">
              <button className="text-[12.5px] text-faint hover:text-muted" onClick={() => setStep(2)}>
                Skip for now
              </button>
              <button
                className="ml-auto px-5 py-2 rounded-full bg-ink text-panel text-[13px] disabled:opacity-40"
                disabled={!allConnected || creating || (tab !== "everyday" && !channel)}
                onClick={create}
                data-testid="ob-create"
              >
                {creating ? "Creating…" : "Create automation"}
              </button>
            </div>
          </section>
        )}

        {step === 2 && (
          <section data-testid="ob-step-done">
            <div className="text-center">
              <div className="w-12 h-12 rounded-full bg-okSoft text-ok grid place-items-center mx-auto mb-3 text-[22px]">
                ✓
              </div>
              <h1 className="text-[19px] font-semibold mb-4">You're set up</h1>
            </div>

            {recap && (
              <div className="flex items-start gap-3 border border-ok/50 bg-okSoft/40 rounded-xl px-4 py-3 mb-3" data-testid="ob-recap">
                <Icon name="clock" size={16} className="text-ok mt-0.5 shrink-0" />
                <span className="text-[12.5px]">
                  <b className="block text-[13px]">Your automation is scheduled</b>
                  <span className="text-muted">{recap}</span>
                </span>
              </div>
            )}

            <div className="flex items-start gap-3 border border-line rounded-xl px-4 py-3">
              <Icon name="sparkle" size={16} className="text-accent mt-0.5 shrink-0" />
              <span className="text-[12.5px] flex-1">
                <b className="block text-[13px]">Specialist coworkers</b>
                <span className="text-muted">
                  Pre-scoped coworkers for sales, ops, code review and more — browse the gallery
                  and enable the ones you want.
                </span>
              </span>
              <button
                className="px-3.5 py-1 rounded-full border border-line text-[12.5px] hover:bg-paper shrink-0"
                onClick={() => finish("gallery")}
                data-testid="ob-gallery"
              >
                Show me
              </button>
            </div>

            <p className="text-[12px] text-muted text-center mt-5">
              Every session shows what it can touch — sources and folders — and you can switch any
              of them off, just for that session.
            </p>

            <div className="text-center mt-5">
              <button
                className="px-5 py-2 rounded-full bg-ink text-panel text-[13px]"
                onClick={() => finish("work")}
                data-testid="ob-start"
              >
                Start working
              </button>
              <p className="text-[11px] text-faint mt-3">
                Replay this setup anytime: Settings ▸ Appearance ▸ Run setup again.
              </p>
            </div>
          </section>
        )}
      </div>
    </div>
  );
}
