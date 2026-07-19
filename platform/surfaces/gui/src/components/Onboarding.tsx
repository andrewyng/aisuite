import { useEffect, useRef, useState } from "react";
import {
  cloudLogin,
  connectManaged,
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
import { PROVIDER_LOGOS, providerRank } from "../providers/logos";
import { Spinner } from "./AutomationQuickstart";

// First-run onboarding (UX-DECISIONS §24 → §29 → §39): model → your tools → go.
// §39 (owner design, 2026-07-18): step 1 is a PROVIDER GALLERY — 13 real brand
// marks, two per row, each card wearing its own state — and step 2 is a
// two-state tools page whose post-sign-in body is a mini connector gallery with
// live one-click connects. Both steps share one frame rule: the header and
// footer never move; only the middle region swaps, at a fixed height.
// Replayable from Settings ▸ Appearance ▸ "Run setup again".

// Where a non-developer gets an API key — deep link + one line of instructions.
const KEY_HELP: Record<string, { url: string; label: string }> = {
  anthropic: { url: "https://console.anthropic.com/settings/keys", label: "console.anthropic.com" },
  openai: { url: "https://platform.openai.com/api-keys", label: "platform.openai.com" },
  gemini: { url: "https://aistudio.google.com/apikey", label: "aistudio.google.com" },
  fireworks: { url: "https://fireworks.ai/account/api-keys", label: "fireworks.ai" },
  together: { url: "https://api.together.xyz/settings/api-keys", label: "together.xyz" },
  zai: { url: "https://z.ai/manage-apikey/apikey-list", label: "z.ai" },
  kimi: { url: "https://platform.moonshot.ai/console/api-keys", label: "platform.moonshot.ai" },
  deepseek: { url: "https://platform.deepseek.com/api_keys", label: "platform.deepseek.com" },
  mistral: { url: "https://console.mistral.ai/api-keys", label: "console.mistral.ai" },
  qwen: { url: "https://modelstudio.console.alibabacloud.com", label: "alibabacloud.com" },
  minimax: { url: "https://platform.minimax.io", label: "platform.minimax.io" },
  xai: { url: "https://console.x.ai", label: "console.x.ai" },
};

// Step 2's mini gallery (§39): managed connectors with LIVE prod OAuth apps only.
// gmail + google_calendar ship grayed "Coming soon" — both ride the same Google
// app, gated on Google verification/CASA; flip them into ACTIVE when it lands.
const TOOLS_ACTIVE = ["outlook", "slack", "github", "notion", "hubspot"];
const TOOLS_SOON = ["gmail", "google_calendar"];

type Verify = { state: "idle" | "testing" | "ok" | "error"; msg?: string };

/** Brand chip: always a light plate so multicolor marks read on any theme. */
function ProviderMark({ name, title, size = 32 }: { name: string; title: string; size?: number }) {
  const url = PROVIDER_LOGOS[name];
  return (
    <span
      className="rounded-lg border border-line grid place-items-center shrink-0"
      style={{ width: size, height: size, background: "#f6f7f8" }}
    >
      {url ? (
        <img src={url} alt="" style={{ width: size * 0.6, height: size * 0.6 }} />
      ) : (
        <span className="text-[13px] font-semibold text-muted">{title[0]}</span>
      )}
    </span>
  );
}

export function Onboarding({ onDone }: { onDone: (next?: "work" | "gallery" | "automations") => void }) {
  const [step, setStep] = useState(0);

  // -- step 1: model (provider gallery ⇄ key form) -------------------------------
  const [providers, setProviders] = useState<ProviderInfo[]>([]);
  // null = the gallery; a provider name = that provider's key form.
  const [sel, setSel] = useState<string | null>(null);
  const [fields, setFields] = useState<Record<string, string>>({});
  const [dirty, setDirty] = useState(false);
  const [showEndpoint, setShowEndpoint] = useState(false);
  const [verify, setVerify] = useState<Verify>({ state: "idle" });
  const [skipConfirm, setSkipConfirm] = useState(false);
  // Keyless providers (Ollama) report configured without proving anything runs —
  // a passing Detect this session is what arms Next for them.
  const [keylessOk, setKeylessOk] = useState<Set<string>>(new Set());
  const backTimer = useRef<number | null>(null);

  const refreshProviders = () =>
    getProviders()
      .then(setProviders)
      .catch(() => {});
  useEffect(() => {
    refreshProviders();
    return () => {
      if (backTimer.current) window.clearTimeout(backTimer.current);
    };
  }, []);

  const info = providers.find((p) => p.name === sel);
  const credentialed = !!info?.configured && !!info?.needs_key;

  // Unsaved per-provider input survives switching cards (owner complaint 2026-07-16).
  const [drafts, setDrafts] = useState<Record<string, Record<string, string>>>({});

  const openProvider = (name: string) => {
    const p = providers.find((x) => x.name === name);
    if (sel) setDrafts((d) => ({ ...d, [sel]: fields }));
    const draft = drafts[name];
    const next: Record<string, string> = {};
    for (const f of p?.fields || []) next[f.key] = draft?.[f.key] || p?.values?.[f.key] || f.default || "";
    setSel(name);
    setFields(next);
    setDirty(!!draft && Object.values(draft).some(Boolean));
    setVerify({ state: "idle" });
    setShowEndpoint(false);
  };

  const backToGallery = () => {
    if (sel) setDrafts((d) => ({ ...d, [sel]: fields }));
    setSel(null);
    setVerify({ state: "idle" });
  };

  // Test = verify AND save AND return (§39: a passing Test auto-saves and takes
  // you back to the gallery, where the card now wears its ✓ — no extra clicks).
  const runTestAndSave = async (): Promise<boolean> => {
    if (!sel) return false;
    setVerify({ state: "testing" });
    const res = await verifyProvider(sel, fields).catch(() => ({ ok: false, error: "unreachable" }));
    if (!res.ok) {
      setVerify({ state: "error", msg: res.error || "couldn't verify" });
      return false;
    }
    if (dirty || !info?.configured) await setProvider(sel, fields).catch(() => {});
    if (!info?.needs_key) setKeylessOk((s) => new Set(s).add(sel));
    setVerify({ state: "ok" });
    setDirty(false);
    setDrafts((d) => ({ ...d, [sel]: {} }));
    await refreshProviders();
    // Let the in-field "✓ Tested & saved" register, then slide home.
    backTimer.current = window.setTimeout(backToGallery, 900);
    return true;
  };

  const anyReady =
    providers.some((p) => p.configured && p.needs_key) || keylessOk.size > 0;
  // In the form with typed-but-untested input, Next verifies+saves first (tester
  // catch 2026-07-12: a manual Test-then-Continue two-step reads as a puzzle).
  const secretFilled = (info?.fields || []).every((f) => !f.secret || (fields[f.key] || "").trim());
  const nextFromForm = !!sel && dirty && secretFilled;
  const canNext = anyReady || nextFromForm;

  const advance = async () => {
    if (nextFromForm && !credentialed) {
      if (backTimer.current) window.clearTimeout(backTimer.current);
      if (!(await runTestAndSave())) return;
    }
    setStep(1);
  };

  const providerStatus = (p: ProviderInfo) =>
    p.configured && p.needs_key ? (
      <span className="block text-[11.5px] text-ok font-medium">✓ Connected</span>
    ) : !p.needs_key ? (
      <span className="block text-[11.5px] text-faint">
        {keylessOk.has(p.name) ? <span className="text-ok font-medium">✓ Running</span> : "No key needed"}
      </span>
    ) : (
      <span className="block text-[11.5px] text-faint">Not set up</span>
    );

  // -- step 2: connect your everyday tools (§39 two-state page) -------------------
  const [connectors, setConnectors] = useState<Connector[]>([]);
  const [cloud, setCloud] = useState<CloudStatus | null>(null);
  const [signinPhase, setSigninPhase] = useState<"opening" | "waiting" | null>(null);
  // One in-flight connect at a time; clicking another card quietly resets the first.
  const [pendingTool, setPendingTool] = useState<string | null>(null);

  // Poll while on the tools page: sign-in AND vendor consents land out-of-band in
  // the system browser. Tighten while either is actually in flight.
  useEffect(() => {
    if (step !== 1) return;
    const load = () => {
      getConnectors().then(setConnectors).catch(() => {});
      getCloudStatus().then(setCloud).catch(() => {});
    };
    load();
    const fast = signinPhase === "waiting" || pendingTool !== null;
    const t = setInterval(load, fast ? 750 : 3000);
    return () => clearInterval(t);
  }, [step, signinPhase, pendingTool]);

  // The poll flips the card to ✓ when the consent lands.
  useEffect(() => {
    if (pendingTool && connectors.find((c) => c.name === pendingTool)?.connected)
      setPendingTool(null);
  }, [connectors, pendingTool]);

  const startTool = async (name: string) => {
    setPendingTool(name); // replaces any previous pending connect
    const res = await connectManaged(
      name,
      name === "hubspot" ? { access: "read" } : undefined, // least privilege in onboarding
    ).catch(() => ({ ok: false }));
    if (!res.ok) setPendingTool((cur) => (cur === name ? null : cur)); // silent reset — no error walls here
  };

  const finish = async (next?: "work" | "gallery" | "automations") => {
    await setOnboarded(true).catch(() => {});
    onDone(next);
  };

  // -- shared bits ----------------------------------------------------------------
  const label = "block text-[12px] text-muted mt-3 mb-1";
  const input =
    "w-full px-3 py-2 rounded-lg border bg-panel text-[13.5px] outline-none focus:border-accent";
  const card =
    "flex items-center gap-2.5 rounded-xl border border-line bg-panel px-3 py-2.5 text-left hover:border-lineStrong transition-colors";
  const dots = (
    <div className="flex justify-center gap-2 mb-6">
      {[0, 1, 2].map((i) => (
        <span key={i} className={"w-1.5 h-1.5 rounded-full " + (i <= step ? "bg-accent" : "bg-line")} />
      ))}
    </div>
  );

  const ordered = [...providers].sort((a, b) => providerRank(a.name) - providerRank(b.name));
  // The in-field saved state (§39): green border + pill INSIDE the key box — shown
  // for stored credentials and fresh test-passes alike; typing clears it.
  const savedState = (credentialed && !dirty) || verify.state === "ok";

  return (
    <div className="fixed inset-0 z-50 bg-ink/30 grid place-items-center" data-testid="onboarding">
      {/* FIXED height across all three steps (owner call 2026-07-12, reaffirmed §39: the
          modal must never resize — the gallery⇄form swap happens inside this box). */}
      <div className="w-[600px] max-w-[92vw] h-[700px] max-h-[88vh] rounded-2xl border border-line bg-panel shadow-2xl p-8 flex flex-col">
        {dots}

        {step === 0 && (
          <section data-testid="ob-step-model" className="flex-1 min-h-0 flex flex-col">
            {/* Persistent header — stays put while the region below swaps (§39). */}
            <h1 className="text-[19px] font-semibold">Welcome to OpenWorker</h1>
            <p className="text-[13px] text-muted mt-0.5 mb-4">
              Pick a model provider to get started — OpenWorker runs on your own key, and your
              key and your data stay on this Mac.
            </p>

            {!sel ? (
              /* ---- the provider GALLERY ---- */
              <div className="flex-1 min-h-0 overflow-y-auto pr-1" data-testid="ob-provider-gallery">
                <div className="grid grid-cols-2 gap-2.5">
                  {ordered.map((p) => (
                    <button
                      key={p.name}
                      className={card}
                      data-testid={`ob-provider-${p.name}`}
                      onClick={() => openProvider(p.name)}
                    >
                      <ProviderMark name={p.name} title={p.title} />
                      <span className="min-w-0 flex-1">
                        <span className="block text-[13px] font-semibold leading-tight truncate">
                          {p.title}
                        </span>
                        {providerStatus(p)}
                      </span>
                      <span className="text-faint text-[14px]">›</span>
                    </button>
                  ))}
                </div>
              </div>
            ) : (
              /* ---- one provider's key form, same box ---- */
              <div className="flex-1 min-h-0 overflow-y-auto pr-1">
                <button
                  className="text-[12.5px] text-muted hover:text-ink"
                  onClick={backToGallery}
                  data-testid="ob-back"
                >
                  ‹ All providers
                </button>
                <div className="flex items-center gap-3 mt-3 mb-1">
                  <ProviderMark name={info?.name || ""} title={info?.title || ""} size={36} />
                  <span className="min-w-0">
                    <span className="block text-[15px] font-semibold leading-tight">{info?.title}</span>
                    {info ? providerStatus(info) : null}
                  </span>
                </div>
                {info?.blurb && <p className="text-[11.5px] text-faint mt-1">{info.blurb}</p>}

                {(info?.fields || []).map((f) => {
                  const keyed = (info?.fields || []).some((x) => x.secret);
                  // ANY base_url on a keyed provider is an expert option: a quiet disclosure
                  // (owner call 2026-07-18: no explainer copy — its users know what it's for).
                  if (f.key === "base_url" && keyed && !showEndpoint) {
                    return (
                      <button
                        key={f.key}
                        className="block self-start text-[12.5px] text-muted hover:text-ink mt-4"
                        onClick={() => setShowEndpoint(true)}
                        data-testid="ob-endpoint-link"
                      >
                        Custom endpoint ⌄
                      </button>
                    );
                  }
                  const testable =
                    (f.secret && f.key === (info?.fields || []).find((x) => x.secret)?.key) ||
                    (!keyed && f.key === (info?.fields || [])[0]?.key);
                  return (
                    <div key={f.key}>
                      <label className={label}>{f.label}</label>
                      <div className="flex gap-2">
                        <div className="relative flex-1 min-w-0">
                          <input
                            className={
                              input +
                              (savedState && f.secret ? " border-ok pr-32" : " border-line")
                            }
                            type={f.secret ? "password" : "text"}
                            placeholder={f.secret && credentialed && !dirty ? "••••••••" : f.placeholder}
                            value={fields[f.key] || ""}
                            data-testid={`ob-field-${f.key}`}
                            onChange={(e) => {
                              setFields((cur) => ({ ...cur, [f.key]: e.target.value }));
                              setDirty(true);
                              setVerify({ state: "idle" });
                            }}
                          />
                          {/* §39: state lives IN the field — no status lines below. */}
                          {savedState && f.secret && (
                            <span
                              className="absolute right-2 top-1/2 -translate-y-1/2 text-[11px] font-medium text-ok bg-okSoft rounded-full px-2 py-0.5 pointer-events-none"
                              data-testid="ob-saved-pill"
                            >
                              ✓ Tested &amp; saved
                            </span>
                          )}
                          {savedState && !f.secret && testable && (
                            <span
                              className="absolute right-2 top-1/2 -translate-y-1/2 text-[11px] font-medium text-ok bg-okSoft rounded-full px-2 py-0.5 pointer-events-none"
                              data-testid="ob-saved-pill"
                            >
                              ✓ Detected
                            </span>
                          )}
                        </div>
                        {testable && (
                          <button
                            className="px-4 rounded-lg border border-line text-[13px] font-medium text-ink hover:border-lineStrong shrink-0 disabled:opacity-40"
                            onClick={() => runTestAndSave()}
                            disabled={
                              verify.state === "testing" || (f.secret && !secretFilled && !credentialed)
                            }
                            data-testid="ob-test"
                          >
                            {verify.state === "testing" ? "…" : info?.needs_key ? "Test" : "Detect"}
                          </button>
                        )}
                      </div>
                      {f.help && !f.secret && <p className="text-[11.5px] text-faint mt-1">{f.help}</p>}
                    </div>
                  );
                })}

                {info?.needs_key && KEY_HELP[sel] && (
                  <p className="text-[11.5px] text-faint mt-2">
                    No key yet?{" "}
                    <button
                      className="text-muted underline decoration-line underline-offset-2 hover:text-ink"
                      onClick={() => openExternal(KEY_HELP[sel].url)}
                    >
                      Create one at {KEY_HELP[sel].label} ↗
                    </button>{" "}
                    — takes about a minute.
                  </p>
                )}
                {info && !info.needs_key && (
                  <p className="text-[11.5px] text-faint mt-2">
                    No API key needed — Ollama runs models on this Mac.{" "}
                    <button
                      className="text-muted underline decoration-line underline-offset-2 hover:text-ink"
                      onClick={() => openExternal("https://ollama.com/download")}
                    >
                      Install Ollama ↗
                    </button>
                  </p>
                )}

                {/* Error line: fixed height so failures never reflow the form. */}
                <div className="mt-3 min-h-[19px] text-[12.5px]">
                  {verify.state === "error" && <span className="text-warnInk">{verify.msg}</span>}
                </div>
              </div>
            )}

            {/* Persistent footer (§39). */}
            <div className="flex items-center gap-3 pt-5">
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
                className="ml-auto px-6 py-2 rounded-full bg-ink text-panel text-[13px] disabled:opacity-40"
                disabled={!canNext || verify.state === "testing"}
                onClick={advance}
                data-testid="ob-continue"
              >
                {verify.state === "testing" ? "Checking…" : "Next"}
              </button>
            </div>
            <p className="text-[11px] text-faint mt-3">
              Models can be enabled or hidden anytime in Settings ▸ Models.
            </p>
          </section>
        )}

        {step === 1 && (
          /* §39 (owner design, 2026-07-18): a two-state page. Pre-sign-in it makes the case —
             tools are what turn a chatbot into a coworker — and asks for the sign-in that
             makes connecting one click. Post-sign-in the SAME region becomes a mini connector
             gallery with live one-click connects, so the headline's promise is kept on this
             page, resolving §29's promise-with-no-action concern. */
          <section data-testid="ob-step-tools" className="flex-1 min-h-0 flex flex-col">
            <h1 className="text-[19px] font-semibold">Connect your everyday tools</h1>
            <p className="text-[13px] text-muted mt-0.5 mb-4">
              A coworker that can only chat can only advise. Connected to your email, calendar,
              CRM, and code, it can do the actual work — triage the inbox, prep the meeting,
              update the pipeline, review the PR.
            </p>

            {!cloud?.signed_in ? (
              <div className="flex-1 min-h-0 overflow-y-auto pr-1">
                <div className="flex items-center gap-2.5 mb-4">
                  {TOOLS_ACTIVE.map((name) => {
                    const c = connectors.find((x) => x.name === name);
                    return c ? <ConnectorBadge key={name} connector={c} size={38} title={c.title} /> : null;
                  })}
                </div>
                <div className="rounded-xl2 border border-line bg-paper px-4 py-3.5 text-[12.5px] text-muted">
                  <span className="block text-[13px] text-ink font-medium mb-0.5">
                    One click, keys handled
                  </span>
                  Sign in to OpenWorker and connecting is one click — OAuth and tokens for 20+
                  integrations are handled for you, and tokens stay local on this Mac. No
                  digging through dev consoles for API keys.
                </div>
                <div className="mt-4 min-h-[36px]">
                  {signinPhase && (
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
                  )}
                </div>
              </div>
            ) : (
              /* ---- signed in: the mini connector gallery (§39) ---- */
              <div className="flex-1 min-h-0 overflow-y-auto pr-1">
                <p className="text-[12.5px] text-ok mb-3" data-testid="ob-tools-signedin">
                  ✓ Signed in{cloud.account ? ` as ${cloud.account}` : ""}
                </p>
                <div className="grid grid-cols-2 gap-2.5" data-testid="ob-tool-gallery">
                  {[...TOOLS_ACTIVE, ...TOOLS_SOON].map((name) => {
                    const c = connectors.find((x) => x.name === name);
                    if (!c) return null;
                    const soon = TOOLS_SOON.includes(name);
                    if (soon)
                      return (
                        /* Grayed "Coming soon" (§39): the whole Google trio is gated on
                           Google verification — present-but-gray says "coming", not "missing". */
                        <div
                          key={name}
                          className="group relative flex items-center gap-2.5 rounded-xl border border-line bg-panel px-3 py-2.5"
                          data-testid={`ob-tool-${name}`}
                        >
                          <span className="opacity-40 grayscale">
                            <ConnectorBadge connector={c} size={34} title={c.title} />
                          </span>
                          <span className="block text-[13px] font-semibold text-faint">{c.title}</span>
                          <span className="absolute -top-2 right-2.5 rounded-full bg-ink text-panel text-[10.5px] px-2 py-0.5 opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none">
                            Coming soon
                          </span>
                        </div>
                      );
                    return (
                      <button
                        key={name}
                        className={card}
                        data-testid={`ob-tool-${name}`}
                        onClick={() => !c.connected && startTool(name)}
                      >
                        <ConnectorBadge connector={c} size={34} title={c.title} />
                        <span className="min-w-0 flex-1">
                          <span className="block text-[13px] font-semibold leading-tight">{c.title}</span>
                          {c.connected ? (
                            <span className="block text-[11.5px] text-ok font-medium">✓ Connected</span>
                          ) : pendingTool === name ? (
                            <span className="block text-[11.5px] text-muted">Check your browser…</span>
                          ) : (
                            <span className="block text-[11.5px] text-faint">One click</span>
                          )}
                        </span>
                      </button>
                    );
                  })}
                </div>
              </div>
            )}

            <div className="flex items-center gap-3 pt-5">
              <button
                className="text-[12.5px] text-faint hover:text-muted text-left"
                onClick={() => setStep(2)}
                data-testid="ob-tools-skip"
              >
                {cloud?.signed_in ? "Set up later" : "Skip — I’ll use my own API tokens"}
              </button>
              {cloud?.signed_in ? (
                <button
                  className="ml-auto px-6 py-2 rounded-full bg-ink text-panel text-[13px] shrink-0"
                  onClick={() => setStep(2)}
                  data-testid="ob-continue-tools"
                >
                  Next
                </button>
              ) : (
                <button
                  className="ml-auto px-6 py-2 rounded-full bg-accent text-white text-[13px] shrink-0 disabled:opacity-40"
                  disabled={signinPhase === "opening"}
                  onClick={async () => {
                    setSigninPhase("opening");
                    await cloudLogin().catch(() => {});
                    setSigninPhase("waiting");
                  }}
                  data-testid="ob-cloud-signin"
                >
                  Sign in
                </button>
              )}
            </div>
            <p className="text-[11px] text-faint mt-3">
              {cloud?.signed_in
                ? "30+ more tools on the Connectors page — add or remove anytime."
                : "Even signed in, your data flows vendor → this Mac — the cloud only brokers the connection."}
            </p>
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
