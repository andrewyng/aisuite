import { useEffect, useState } from "react";
import {
  getProviders,
  getSettings,
  setOnboarded,
  setProvider,
  setScratchBase,
  type ProviderInfo,
} from "../api";
import {
  getAutostart,
  getKeepAwake,
  isTauri,
  pickFolder,
  setAutostart,
  setKeepAwake,
} from "../tauri";
import { ModelChecklist } from "./ModelChecklist";

const STEPS = ["Welcome", "Files", "Model", "Always-on"];

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
  const [endpoint, setEndpoint] = useState(""); // OpenAI custom endpoint (Azure, OpenRouter, …)
  const [providers, setProviders] = useState<ProviderInfo[]>([]);
  const [ollamaUrl, setOllamaUrl] = useState("");
  const [ollamaMsg, setOllamaMsg] = useState<string | null>(null);

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

  const saveKey = async () => {
    if (!keyDraft.trim()) return;
    setKeyMsg(null);
    const fields: Record<string, string> = { api_key: keyDraft.trim() };
    if (apiProv === "openai") fields.base_url = endpoint.trim();
    const res = await setProvider(apiProv, fields);
    if (res.ok) {
      setKeyDraft("");
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

  const finish = async () => {
    await setOnboarded(!showAgain); // ticked "show again" → keep showing → onboarded=false
    onDone();
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
                A quick setup: choose where your files are saved, then connect a model — an API
                key or a local Ollama model. Takes about a minute.
              </p>
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
                      setApiProv(e.target.value);
                      setKeyDraft("");
                      setKeyMsg(null);
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
                      onChange={(e) => setKeyDraft(e.target.value)}
                      onKeyDown={(e) => e.key === "Enter" && saveKey()}
                    />
                    <button
                      className="btn primary"
                      onClick={saveKey}
                      disabled={!keyDraft.trim() && !(apiProv === "openai" && endpoint.trim())}
                    >
                      Save
                    </button>
                  </div>
                  <div className="ob-note dim">
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

        <div className="ob-foot">
          <button className="btn ghost" onClick={finish}>
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
