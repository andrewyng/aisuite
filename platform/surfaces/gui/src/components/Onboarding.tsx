import { useEffect, useState } from "react";
import {
  cloudLogin,
  getCloudStatus,
  getConnectors,
  getProviders,
  setOnboarded,
  setProvider,
  verifyProvider,
  type CloudStatus,
  type Connector,
  type ProviderInfo,
} from "../api";
import { openExternal } from "../tauri";
import { ConnectorBadge } from "../connectors/ConnectorIcon";
import { SelectMenu } from "./SelectMenu";
import { Spinner } from "./AutomationQuickstart";

// First-run onboarding (UX-DECISIONS §24, restructured by §29): model → your tools → go.
// Only the model step gates; the recipe machinery moved to the Automations quickstart
// (AutomationQuickstart.tsx). Replayable from Settings ▸ Appearance ▸ "Run setup again".

// Where a non-developer gets an API key — deep link + one line of instructions.
const KEY_HELP: Record<string, { url: string; label: string }> = {
  anthropic: { url: "https://console.anthropic.com/settings/keys", label: "console.anthropic.com" },
  openai: { url: "https://platform.openai.com/api-keys", label: "platform.openai.com" },
  gemini: { url: "https://aistudio.google.com/apikey", label: "aistudio.google.com" },
  fireworks: { url: "https://fireworks.ai/account/api-keys", label: "fireworks.ai" },
  together: { url: "https://api.together.xyz/settings/api-keys", label: "together.xyz" },
};

type Verify = { state: "idle" | "testing" | "ok" | "error"; msg?: string };

export function Onboarding({ onDone }: { onDone: (next?: "work" | "gallery" | "automations") => void }) {
  const [step, setStep] = useState(0);

  // -- step 1: model ------------------------------------------------------------
  const [providers, setProviders] = useState<ProviderInfo[]>([]);
  const [prov, setProv] = useState("anthropic");
  const [fields, setFields] = useState<Record<string, string>>({});
  // Optional endpoint (a base_url WITH a default, on a keyed provider) collapses behind a
  // "Configure custom endpoint" link (owner call 2026-07-11) — most users never touch it.
  const [showEndpoint, setShowEndpoint] = useState(false);
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
    setShowEndpoint(false);
    const next: Record<string, string> = {};
    for (const f of p?.fields || []) next[f.key] = p?.values?.[f.key] || f.default || "";
    setFields(next);
  };

  const runTest = async (): Promise<boolean> => {
    setVerify({ state: "testing" });
    const res = await verifyProvider(prov, fields).catch(() => ({ ok: false, error: "unreachable" }));
    setVerify(res.ok ? { state: "ok" } : { state: "error", msg: res.error || "couldn't verify" });
    return res.ok;
  };

  const credentialed = !!info?.configured && !!info?.needs_key;
  // What the Test button shows: a fresh pass OR already-stored credentials read "✓ Connected".
  const verified = verify.state === "ok" || (credentialed && verify.state === "idle");
  // Continue is clickable as soon as the required (secret) fields are filled — the verify runs
  // AUTOMATICALLY on Continue (tester catch 2026-07-12: "did I have to click Test first?" — the
  // manual Test-then-Continue two-step read as a puzzle). Test stays as an optional explicit check.
  const requiredFilled = (info?.fields || []).every(
    (f) => !f.secret || (fields[f.key] || "").trim(),
  );
  const canContinue = credentialed || requiredFilled;

  const saveAndContinue = async () => {
    if (!credentialed && verify.state !== "ok" && !(await runTest())) return;
    if (!info?.configured || Object.values(fields).some(Boolean)) {
      await setProvider(prov, fields).catch(() => {});
    }
    setStep(1);
  };

  // -- step 2: Connect your tools (§29 — value-framed cloud sign-in) ---------------
  const [connectors, setConnectors] = useState<Connector[]>([]);
  const [cloud, setCloud] = useState<CloudStatus | null>(null);
  // §30: narrate the sign-in — "opening" while the POST is in flight, "waiting" once the
  // browser is off (the 3 s poll below flips to the ✓ state when the flow lands).
  const [signinPhase, setSigninPhase] = useState<"opening" | "waiting" | null>(null);
  // Poll while on the tools page: the browser sign-in flow lands out-of-band.
  useEffect(() => {
    if (step !== 1) return;
    const load = () => {
      getConnectors().then(setConnectors).catch(() => {});
      getCloudStatus().then(setCloud).catch(() => {});
    };
    load();
    const t = setInterval(load, 3000);
    return () => clearInterval(t);
  }, [step]);

  // The logo row: the headline managed connectors, REAL brand icons (owner call 2026-07-12 —
  // the mock's letter badges were stand-ins).
  const TOOL_ROW = ["slack", "github", "hubspot", "gmail", "google_calendar"];

  const finish = async (next?: "work" | "gallery" | "automations") => {
    await setOnboarded(true).catch(() => {});
    onDone(next);
  };

  // -- shared bits ----------------------------------------------------------------
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
      {/* FIXED height across all three steps (owner call 2026-07-12: the modal resizing per
          step felt unsettled). Steps are flex columns; each action row `mt-auto`-pins to the
          bottom; taller content scrolls inside the step. */}
      {/* 700px fits the tallest step (the recipe, ~593px content) with headroom; shorter
          windows cap at 88vh and the step scrolls inside. */}
      <div className="w-[600px] max-w-[92vw] h-[700px] max-h-[88vh] rounded-2xl border border-line bg-panel shadow-2xl p-8 flex flex-col">
        {dots}

        {step === 0 && (
          <section data-testid="ob-step-model" className="flex-1 min-h-0 flex flex-col overflow-y-auto">
            <h1 className="text-[19px] font-semibold">Welcome to OpenCoworker</h1>
            <p className="text-[13px] text-muted mt-0.5 mb-5">
              Connect a model to get started — OpenCoworker runs on your own API key, and your
              key and your data stay on this Mac.
            </p>
            <label className={label}>Provider</label>
            {/* The same SelectMenu Settings ▸ Models uses (owner call 2026-07-11: the native
                <select> read as a raw OS control) — sections + green key-set dots included. */}
            <SelectMenu
              ariaLabel="Provider"
              value={prov}
              options={[...providers]
                .sort(
                  (a, b) =>
                    Number(b.configured && b.needs_key) - Number(a.configured && a.needs_key),
                )
                .map((p) => {
                  const ready = p.configured && p.needs_key;
                  return {
                    value: p.name,
                    label: p.title,
                    group: ready ? "Ready to use" : "Needs setup",
                    dot: ready,
                  };
                })}
              onChange={(name) => pickProvider(name)}
            />
            {info?.blurb && <p className="text-[11.5px] text-faint mt-1">{info.blurb}</p>}

            {(info?.fields || []).map((f) => {
              // ANY base_url on a keyed provider is an expert option: collapsed behind a link
              // (owner catch 2026-07-12: OpenAI's endpoint has no default, so the earlier
              // default-only condition left it visible on the first pass). Keyless providers
              // (Ollama) keep it visible — the endpoint IS the connection there.
              const keyed = (info?.fields || []).some((x) => x.secret);
              if (f.key === "base_url" && keyed && !showEndpoint) {
                return (
                  <button
                    key={f.key}
                    className="block self-start text-[12px] text-accent hover:underline mt-3"
                    onClick={() => setShowEndpoint(true)}
                    data-testid="ob-endpoint-link"
                  >
                    Configure custom endpoint ›
                  </button>
                );
              }
              return (
                <div key={f.key}>
                  <label className={label}>{f.label}</label>
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
                  {f.help && <p className="text-[11.5px] text-faint mt-1">{f.help}</p>}
                </div>
              );
            })}

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

            {/* No model picker here (owner call 2026-07-11): the model is chosen per session,
                and the old select never persisted anything anyway. One pointer instead. */}
            <p className="text-[11.5px] text-faint mt-3">
              Models can be enabled or hidden anytime in Settings ▸ Models.
            </p>

            {/* Error line: fixed height so verify failures never reflow the form. Success is
                NOT a line — the Test button itself flips to "✓ Connected" (owner call
                2026-07-12: the green status text was louder than the moment deserved). */}
            <div className="mt-3 min-h-[19px] text-[12.5px]">
              {verify.state === "error" && <span className="text-warnInk">{verify.msg}</span>}
            </div>

            <div className="flex items-center gap-3 mt-auto pt-5">
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
                className={
                  "ml-auto px-4 py-2 rounded-full border text-[13px] disabled:opacity-40 " +
                  (verified ? "border-ok/40 text-ok" : "border-line hover:bg-paper")
                }
                onClick={runTest}
                disabled={verify.state === "testing"}
                data-testid="ob-test"
              >
                {verify.state === "testing" ? "Testing…" : verified ? "✓ Connected" : "Test"}
              </button>
              <button
                className="px-5 py-2 rounded-full bg-ink text-panel text-[13px] disabled:opacity-40"
                disabled={!canContinue || verify.state === "testing"}
                onClick={saveAndContinue}
                data-testid="ob-continue"
              >
                {verify.state === "testing" ? "Checking…" : "Continue"}
              </button>
            </div>
          </section>
        )}

        {step === 1 && (
          <section data-testid="ob-step-tools" className="flex-1 min-h-0 flex flex-col overflow-y-auto">
            <h1 className="text-[19px] font-semibold">Connect your tools</h1>
            <p className="text-[13px] text-muted mt-0.5 mb-5">
              OpenCoworker works with the apps you already use.
            </p>

            <div className="flex items-center gap-2.5 mb-5">
              {TOOL_ROW.map((name) => {
                const c = connectors.find((x) => x.name === name);
                return c ? <ConnectorBadge key={name} connector={c} size={38} title={c.title} /> : null;
              })}
            </div>

            <div className="rounded-xl2 border border-line bg-paper px-4 py-3.5 text-[12.5px] text-muted">
              <span className="block text-[13px] text-ink font-medium mb-0.5">
                One sign-in unlocks every one-click connection
              </span>
              Connections are brokered by OpenCoworker Cloud — your tokens stay on this Mac.
              Connect Slack, GitHub, HubSpot, Gmail and Calendar later with a single click each.
            </div>
            {/* Sign-in is the one-click path, never the ONLY path (owner call 2026-07-12). */}
            <p className="text-[11.5px] text-faint mt-2.5">
              Prefer manual setup? Every tool can also be connected with your own API keys from
              the Connectors page — signing in just makes it one click.
            </p>

            <div className="mt-4">
              {cloud?.signed_in ? (
                <span
                  className="inline-flex items-center gap-2 text-[13px] text-ok bg-okSoft/60 rounded-lg px-3 py-2"
                  data-testid="ob-tools-signedin"
                >
                  ✓ Signed in{cloud.account ? ` as ${cloud.account}` : ""}
                </span>
              ) : signinPhase ? (
                <span className="inline-flex items-center gap-2 text-[13px] text-muted">
                  <Spinner />
                  {signinPhase === "opening" ? "Opening browser…" : "Waiting for sign-in…"}
                  {signinPhase === "waiting" && (
                    <span className="text-[11.5px] text-faint">
                      finish in your browser — this page updates by itself ·{" "}
                      <button
                        className="underline hover:text-muted"
                        onClick={() => setSigninPhase(null)}
                        data-testid="ob-signin-cancel"
                      >
                        Cancel
                      </button>
                    </span>
                  )}
                </span>
              ) : (
                <button
                  className="px-4 py-2 rounded-full bg-accent text-white text-[13px]"
                  onClick={async () => {
                    setSigninPhase("opening");
                    await cloudLogin().catch(() => {});
                    setSigninPhase("waiting");
                  }}
                  data-testid="ob-cloud-signin"
                >
                  Sign in to OpenCoworker Cloud
                </button>
              )}
            </div>

            <div className="flex items-center gap-3 mt-auto pt-5">
              <button
                className="text-[12.5px] text-faint hover:text-muted text-left"
                onClick={() => setStep(2)}
                data-testid="ob-tools-skip"
              >
                Skip — you can sign in whenever you first connect something
              </button>
              <button
                className="ml-auto px-5 py-2 rounded-full bg-ink text-panel text-[13px] shrink-0"
                onClick={() => setStep(2)}
                data-testid="ob-continue-tools"
              >
                Continue
              </button>
            </div>
          </section>
        )}

        {step === 2 && (
          <section data-testid="ob-step-done" className="flex-1 min-h-0 flex flex-col overflow-y-auto">
            <div className="text-center">
              <div className="w-12 h-12 rounded-full bg-okSoft text-ok grid place-items-center mx-auto mb-3 text-[22px]">
                ✓
              </div>
              <h1 className="text-[19px] font-semibold mb-1">You're set up</h1>
              <p className="text-[13px] text-muted mb-5">Two good ways to start:</p>
            </div>

            <button
              className="w-full flex items-start gap-3 rounded-xl2 border border-line hover:border-accent bg-panel px-4 py-3.5"
              onClick={() => finish("automations")}
              data-testid="ob-cta-automation"
            >
              <span className="w-9 h-9 rounded-lg bg-accentSoft text-accent grid place-items-center text-[15px] shrink-0">
                ◷
              </span>
              <span className="flex-1 min-w-0 text-left">
                <b className="block text-[13.5px]">Create your first automation</b>
                <span className="text-[12px] text-muted">
                  A weekly digest, a morning brief — pick a template, running in two minutes.
                </span>
              </span>
              <span className="text-faint self-center">›</span>
            </button>
            <button
              className="w-full flex items-start gap-3 rounded-xl2 border border-line hover:border-accent bg-panel px-4 py-3.5 mt-2.5"
              onClick={() => finish("work")}
              data-testid="ob-start"
            >
              <span className="w-9 h-9 rounded-lg bg-accentSoft text-accent grid place-items-center text-[15px] shrink-0">
                ✦
              </span>
              <span className="flex-1 min-w-0 text-left">
                <b className="block text-[13.5px]">Start working with Coworker</b>
                <span className="text-[12px] text-muted">
                  Open a session and just ask — analyze files, draft, research, build.
                </span>
              </span>
              <span className="text-faint self-center">›</span>
            </button>

            {/* The Specialist-coworkers gallery card and the per-session-scope line stay HIDDEN
                (owner call 2026-07-12); the finish("gallery") plumbing remains for their return. */}

            <p className="text-[11px] text-faint text-center mt-auto pt-5">
              Replay this setup anytime: Settings ▸ Appearance ▸ Run setup again.
            </p>
          </section>
        )}
      </div>
    </div>
  );
}
