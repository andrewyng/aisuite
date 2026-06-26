import { useEffect, useState } from "react";
import {
  detectProvider,
  getProviders,
  getSettings,
  setOnboarded,
  setProvider,
  setScratchBase,
  verifyProvider,
  type ProviderInfo,
} from "../api";
import {
  getAutostart,
  getKeepAwake,
  isTauri,
  openExternal,
  pickFolder,
  setAutostart,
  setKeepAwake,
} from "../tauri";
import { ModelChecklist } from "./ModelChecklist";

const STEPS = ["Welcome", "Files", "Model", "Always-on"];

// Where a non-developer gets an API key for each provider — shown in the "Don't have a key?"
// helper on the model step. Rendered as a copyable link so it works even in the desktop webview.
const KEY_HELP: Record<string, { url: string; steps: string }> = {
  openai: {
    url: "https://platform.openai.com/api-keys",
    steps: "Sign in, click “Create new secret key”, then copy it here.",
  },
  anthropic: {
    url: "https://console.anthropic.com/settings/keys",
    steps: "Sign in, click “Create Key”, then copy it here.",
  },
  gemini: {
    url: "https://aistudio.google.com/apikey",
    steps: "Sign in, click “Create API key”, then copy it here.",
  },
};

type Verify = { state: "idle" | "testing" | "ok" | "error"; msg?: string };

/**
 * First-run setup wizard (desktop). Walks through where files are saved, connecting a model
 * (API key or local Ollama), and the always-on toggles. Each field saves as you go; "Finish"
 * records completion unless you unticked "Show this on next startup".
 *
 * NOTE: MyHelper's working-folder step is hidden for now — the always-on helper isn't shipping
 * in this beta. Restore it from git history when MyHelper lands in a future version.
 */
export function Onboarding({ onDone }: { onDone: () => void }) {
  const [step, setStep] = useState(0);

  // Cowork scratch location (where each conversation's per-conversation folder is created)
  const [scratch, setScratch] = useState("");
  const [scratchMsg, setScratchMsg] = useState<string | null>(null);

  // model + key
  const [models, setModels] = useState<string[]>([]);
  const [model, setModel] = useState("");
  const [keyDraft, setKeyDraft] = useState("");
  const [keyMsg, setKeyMsg] = useState<string | null>(null);
  const [secretsPath, setSecretsPath] = useState("");

  // provider choice (API pane) + local models (Ollama)
  const [conn, setConn] = useState<"api" | "local">("api");
  const [apiProv, setApiProv] = useState("openai");
  // True once the user manually picks a provider — stops key auto-detect from overriding them.
  const [manualProv, setManualProv] = useState(false);
  const [detected, setDetected] = useState<string | null>(null);
  const [verify, setVerify] = useState<Verify>({ state: "idle" });
  const [keyHelpOpen, setKeyHelpOpen] = useState(false);
  const [endpoint, setEndpoint] = useState(""); // OpenAI custom endpoint (Azure, OpenRouter, …)
  const [providers, setProviders] = useState<ProviderInfo[]>([]);
  const [ollamaUrl, setOllamaUrl] = useState("");
  const [ollamaMsg, setOllamaMsg] = useState<string | null>(null);
  // First Skip click with no model connected asks for confirmation rather than leaving silently.
  const [skipConfirm, setSkipConfirm] = useState(false);

  // always-on
  const [autostart, setAuto] = useState(false);
  const [keepAwake, setKeep] = useState(false);
  const [showAgain, setShowAgain] = useState(false); // inverse of "don't show again"; default = don't show

  const refreshSettings = () =>
    getSettings()
      .then((s) => {
        setModels(s.models || []);
        setModel(s.model);
        setScratch((cur) => cur || s.scratch_base || "");
        setSecretsPath(s.secrets_path || "");
      })
      .catch(() => {});
  const refreshProviders = () =>
    getProviders()
      .then((ps) => {
        setProviders(ps);
        const oll = ps.find((p) => p.name === "ollama");
        if (oll?.values?.base_url) setOllamaUrl((cur) => cur || oll.values.base_url);
        const oai = ps.find((p) => p.name === "openai");
        if (oai?.values?.base_url) setEndpoint((cur) => cur || oai.values.base_url);
      })
      .catch(() => {});

  useEffect(() => {
    refreshSettings();
    refreshProviders();
    if (isTauri()) {
      getAutostart().then((v) => setAuto(!!v));
      getKeepAwake().then((v) => setKeep(!!v));
    }
  }, []);

  const browseScratch = async () => {
    const p = await pickFolder();
    if (p) saveScratch(p);
  };
  const saveScratch = async (p: string) => {
    setScratch(p);
    setScratchMsg(null);
    const res = await setScratchBase(p.trim());
    setScratchMsg(res.ok ? "Saved." : res.error || "Couldn't use that folder.");
  };

  // Build the {api_key, base_url?} payload for the currently selected API provider.
  const keyFields = (): Record<string, string> => {
    const fields: Record<string, string> = { api_key: keyDraft.trim() };
    if (apiProv === "openai") fields.base_url = endpoint.trim();
    return fields;
  };

  // Guess the provider from the key as the user types/pastes it, and switch the dropdown to match
  // (until they pick one by hand). Mirrors the "OpenAI key detected automatically" affordance.
  const onKeyChange = (v: string) => {
    setKeyDraft(v);
    setKeyMsg(null);
    setVerify({ state: "idle" });
    const det = detectProvider(v);
    setDetected(det);
    if (det && !manualProv && det !== apiProv && providers.some((p) => p.name === det)) {
      setApiProv(det);
    }
  };

  const testKey = async () => {
    if (!keyDraft.trim() && !selProv?.configured) return;
    setVerify({ state: "testing" });
    setKeyMsg(null);
    const res = await verifyProvider(apiProv, keyFields());
    setVerify(res.ok ? { state: "ok" } : { state: "error", msg: res.error || "Couldn't verify." });
  };

  const saveKey = async () => {
    if (!keyDraft.trim()) return;
    setKeyMsg(null);
    const res = await setProvider(apiProv, keyFields());
    if (res.ok) {
      setKeyDraft("");
      setVerify({ state: "idle" });
      setKeyMsg("Saved locally.");
      refreshProviders();
      refreshSettings(); // the provider's recommended model may have been added to the list
    } else {
      setKeyMsg(res.error || "Couldn't save key.");
    }
  };

  const ollama = providers.find((p) => p.name === "ollama");
  const saveOllama = async () => {
    setOllamaMsg(null);
    const res = await setProvider("ollama", { base_url: ollamaUrl.trim() });
    if (res.ok) {
      const rec = res.recommended_model;
      setOllamaMsg(
        rec
          ? `Saved. ${rec} is the recommended model — pick it below (pull it first with: ollama pull ${rec}).`
          : "Saved.",
      );
      refreshSettings(); // the recommended model may have been added to the list
    } else {
      setOllamaMsg(res.error || "Couldn't save the Ollama URL.");
    }
  };

  // The provider the model step is currently configuring. Its models render as a checklist
  // (tick = in the composer picker, black badge = default) once the provider is usable.
  const apiProviders = providers.filter((p) => p.name !== "ollama");
  const provName = conn === "local" ? "ollama" : apiProv;
  const selProv = providers.find((p) => p.name === provName);
  const knownNames = providers.map((p) => p.name);

  const toggleAuto = async (v: boolean) => setAuto(!!(await setAutostart(v)));
  const toggleKeep = async (v: boolean) => setKeep(!!(await setKeepAwake(v)));

  // The provider the default model routes to (prefix before `:`, else OpenAI). A model is "ready"
  // only if that provider is configured — used to warn when Skipping with nothing connected.
  const modelProviderName = (m: string): string => {
    const i = (m || "").indexOf(":");
    if (i > 0) {
      const p = m.slice(0, i);
      if (providers.some((x) => x.name === p)) return p;
    }
    return "openai";
  };
  const modelReady = providers.some(
    (p) => p.name === modelProviderName(model) && p.configured,
  );

  const finish = async () => {
    await setOnboarded(!showAgain); // ticked "show again" → keep showing → onboarded=false
    onDone();
  };
  // Skip warns once if no model is connected (chat would stay paused), then leaves on confirm.
  const requestSkip = () => {
    if (!modelReady && !skipConfirm) {
      setSkipConfirm(true);
      return;
    }
    finish();
  };

  const last = step === STEPS.length - 1;

  return (
    <div className="ob-overlay">
      <div className="ob">
        <div className="ob-rail">
          {STEPS.map((s, i) => (
            <div key={s} className={"ob-rail-item" + (i === step ? " active" : i < step ? " done" : "")}>
              <span className="ob-dot">{i < step ? "✓" : i + 1}</span>
              {s}
            </div>
          ))}
        </div>

        <div className="ob-body">
          {step === 0 && (
            <div className="ob-step">
              <div className="ob-mark">✳</div>
              <h2>Welcome to OpenCoworker</h2>
              <p className="ob-sub">
                An open-source desktop agent that does real work on your machine — research, code,
                and documents. Your files and your keys stay on this computer; nothing leaves
                unless you say so.
              </p>
              <ul className="ob-valueprops">
                <li><strong>Private by default</strong> — runs locally, bring your own API key (or use a free local model).</li>
                <li><strong>Real deliverables</strong> — it writes files, reports, and code, not just chat.</li>
                <li><strong>Always reachable</strong> — schedule automations and connect your tools.</li>
              </ul>
              <p className="ob-sub dim">A quick setup takes about a minute.</p>
            </div>
          )}

          {step === 1 && (
            <div className="ob-step">
              <h2>Where files go</h2>
              <p className="ob-sub">
                Each conversation gets its own folder under the location below — that's where the
                agent saves the files it produces. You can grant access to more folders any time.
              </p>

              <label className="ob-label">Save files under</label>
              <div className="ob-row">
                <input
                  placeholder="~/OpenCoworker"
                  value={scratch}
                  onChange={(e) => setScratch(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && saveScratch(scratch)}
                />
                {isTauri() && (
                  <button className="btn" onClick={browseScratch}>
                    Browse…
                  </button>
                )}
                <button className="btn primary" onClick={() => saveScratch(scratch)} disabled={!scratch.trim()}>
                  Set
                </button>
              </div>
              {scratchMsg && <div className="ob-note">{scratchMsg}</div>}

              {/* MyHelper's working folder lived here. Hidden for this beta (MyHelper isn't
                  shipping yet) — bring it back in a future version. */}
            </div>
          )}

          {step === 2 && (
            <div className="ob-step">
              <h2>Connect a model</h2>
              <p className="ob-sub">
                Connect a model provider with an API key — or run models locally with Ollama
                (free, runs on your Mac) — then pick the default model for new sessions.
              </p>

              <div className="subtabs ob-subtabs">
                <div className="manage-tabs">
                  <div className={"mtab" + (conn === "api" ? " active" : "")} onClick={() => setConn("api")}>
                    API key
                  </div>
                  <div className={"mtab" + (conn === "local" ? " active" : "")} onClick={() => setConn("local")}>
                    Local (Ollama)
                  </div>
                </div>
              </div>
              {conn === "api" ? (
                <>
                  <label className="ob-label">Provider</label>
                  <select
                    className="ob-select"
                    value={apiProv}
                    onChange={(e) => {
                      setManualProv(true);
                      setApiProv(e.target.value);
                      setKeyDraft("");
                      setKeyMsg(null);
                      setVerify({ state: "idle" });
                      setDetected(null);
                    }}
                  >
                    {apiProviders.map((p) => (
                      <option key={p.name} value={p.name}>
                        {p.title}
                      </option>
                    ))}
                  </select>

                  {apiProv === "openai" && (
                    <>
                      <label className="ob-label">Custom endpoint (optional)</label>
                      <input
                        className="ob-input"
                        placeholder="https://…/openai/v1 — for Azure OpenAI or any OpenAI-compliant server"
                        value={endpoint}
                        autoComplete="off"
                        spellCheck={false}
                        onChange={(e) => setEndpoint(e.target.value)}
                      />
                    </>
                  )}

                  <label className="ob-label">
                    {selProv?.fields.find((f) => f.key === "api_key")?.label || "API key"}{" "}
                    {selProv?.configured && <span className="ob-ok">· configured</span>}
                  </label>
                  <div className="ob-row">
                    <input
                      type="password"
                      placeholder={
                        selProv?.configured
                          ? "•••••••• (saved) — enter to replace"
                          : selProv?.fields.find((f) => f.key === "api_key")?.placeholder || "sk-…"
                      }
                      value={keyDraft}
                      autoComplete="off"
                      spellCheck={false}
                      onChange={(e) => onKeyChange(e.target.value)}
                      onKeyDown={(e) => e.key === "Enter" && testKey()}
                    />
                    <button
                      className="btn"
                      onClick={testKey}
                      disabled={verify.state === "testing" || (!keyDraft.trim() && !selProv?.configured)}
                      title="Check the key works — without saving it"
                    >
                      {verify.state === "testing" ? "Testing…" : "Test"}
                    </button>
                    <button
                      className="btn primary"
                      onClick={saveKey}
                      disabled={!keyDraft.trim() && !(apiProv === "openai" && endpoint.trim())}
                    >
                      Save
                    </button>
                  </div>

                  {/* One status line at a time: verified (strongest) > error > auto-detected. */}
                  {verify.state === "ok" ? (
                    <div className="ob-status ob-ok">✓ Key verified — you're good to go.</div>
                  ) : verify.state === "error" ? (
                    <div className="ob-status ob-err">{verify.msg}</div>
                  ) : detected && !manualProv ? (
                    <div className="ob-status ob-detected">
                      ✓ {providers.find((p) => p.name === detected)?.title || detected} key detected
                      automatically. <span className="dim">Not right? Pick a provider above.</span>
                    </div>
                  ) : null}

                  <div className="ob-keyhelp">
                    <button className="ob-link" onClick={() => setKeyHelpOpen((o) => !o)}>
                      {keyHelpOpen ? "▾" : "▸"} Don't have an API key? Get one in about 2 minutes
                    </button>
                    {keyHelpOpen && KEY_HELP[apiProv] && (
                      <div className="ob-keyhelp-body">
                        <div>{KEY_HELP[apiProv].steps}</div>
                        <div className="ob-row" style={{ marginTop: 6 }}>
                          <button className="btn" onClick={() => openExternal(KEY_HELP[apiProv].url)}>
                            Open {selProv?.title || "provider"} ↗
                          </button>
                          <code className="ob-url">{KEY_HELP[apiProv].url}</code>
                        </div>
                      </div>
                    )}
                  </div>

                  <div className="ob-note dim" style={{ marginTop: 18 }}>
                    Stored locally{secretsPath ? ` at ${secretsPath}` : ""}, readable only by your account. Never sent to the model.
                  </div>
                  {keyMsg && <div className="ob-note">{keyMsg}</div>}

                  {selProv?.configured && (
                    <>
                      <label className="ob-label">Models</label>
                      <div className="ob-note dim" style={{ margin: "0 0 4px" }}>
                        Ticked models show in the composer's picker; the default is what new
                        sessions start with.
                      </div>
                      <ModelChecklist
                        provider={provName}
                        knownProviders={knownNames}
                        suggested={selProv.suggested_models}
                        curated={models}
                        defaultModel={model}
                        onChanged={(next) => {
                          setModels(next.models);
                          setModel(next.model);
                        }}
                      />
                    </>
                  )}
                </>
              ) : (
                <>
                  <label className="ob-label">Ollama server URL</label>
                  <div className="ob-row">
                    <input
                      placeholder="http://localhost:11434"
                      value={ollamaUrl}
                      autoComplete="off"
                      spellCheck={false}
                      onChange={(e) => setOllamaUrl(e.target.value)}
                      onKeyDown={(e) => e.key === "Enter" && saveOllama()}
                    />
                    <button className="btn primary" onClick={saveOllama}>
                      Save
                    </button>
                  </div>
                  <div className="ob-note dim">
                    Needs <code>ollama serve</code> running
                    {ollama?.recommended_model ? (
                      <> and a tool-capable model pulled, e.g. <code>ollama pull {ollama.recommended_model}</code></>
                    ) : null}
                    . No API key needed; you can fine-tune models later in Manage.
                  </div>
                  {ollamaMsg && <div className="ob-note">{ollamaMsg}</div>}

                  <label className="ob-label">Models</label>
                  <div className="ob-note dim" style={{ margin: "0 0 4px" }}>
                    Your pulled models. Ticked ones show in the composer's picker; the default is
                    what new sessions start with.
                  </div>
                  <ModelChecklist
                    provider="ollama"
                    knownProviders={knownNames}
                    suggested={ollama?.suggested_models || []}
                    curated={models}
                    defaultModel={model}
                    onChanged={(next) => {
                      setModels(next.models);
                      setModel(next.model);
                    }}
                  />
                </>
              )}
            </div>
          )}

          {step === 3 && (
            <div className="ob-step">
              <h2>Staying on</h2>
              <p className="ob-sub">
                Scheduled automations only run while OpenCoworker is running.
                {!isTauri() && " (Desktop app only.)"}
              </p>
              <label className={"ob-toggle" + (isTauri() ? "" : " disabled")}>
                <input type="checkbox" checked={autostart} disabled={!isTauri()} onChange={(e) => toggleAuto(e.target.checked)} />
                <span>
                  <strong>Open at login</strong>
                  <small>Launch OpenCoworker automatically when you sign in.</small>
                </span>
              </label>
              <label className={"ob-toggle" + (isTauri() ? "" : " disabled")}>
                <input type="checkbox" checked={keepAwake} disabled={!isTauri()} onChange={(e) => toggleKeep(e.target.checked)} />
                <span>
                  <strong>Keep this system awake</strong>
                  <small>Prevent idle sleep so scheduled tasks fire on time.</small>
                </span>
              </label>

              <label className="ob-check">
                <input type="checkbox" checked={showAgain} onChange={(e) => setShowAgain(e.target.checked)} />
                Show this setup again on next startup
              </label>
            </div>
          )}
        </div>

        {skipConfirm && (
          <div className="ob-skipwarn">
            <span>
              No model is connected yet — chat stays paused until you add one. You can still browse,
              and connect a model later from Settings.
            </span>
            <div className="ob-skipwarn-actions">
              <button className="btn" onClick={() => { setSkipConfirm(false); setStep(2); setConn("api"); }}>
                Connect a model
              </button>
              <button className="btn ghost" onClick={finish}>
                Skip anyway
              </button>
            </div>
          </div>
        )}
        <div className="ob-foot">
          <button className="btn ghost" onClick={requestSkip}>
            Skip
          </button>
          <div className="ob-foot-right">
            {step > 0 && (
              <button className="btn" onClick={() => setStep(step - 1)}>
                Back
              </button>
            )}
            {last ? (
              <button className="btn primary" onClick={finish}>
                Finish
              </button>
            ) : (
              <button className="btn primary" onClick={() => setStep(step + 1)}>
                Next
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
