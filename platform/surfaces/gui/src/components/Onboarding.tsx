import { useEffect, useState } from "react";
import {
  getSettings,
  getSuperagent,
  setDefaultModel,
  setModelKey,
  setOnboarded,
  setSuperagentWorkspace,
} from "../api";
import {
  getAutostart,
  getKeepAwake,
  isTauri,
  pickFolder,
  setAutostart,
  setKeepAwake,
} from "../tauri";

const STEPS = ["Welcome", "Workspace", "Model & key", "Always-on"];

/**
 * First-run setup wizard (desktop). Walks through MyHelper's working folder, the model + API
 * key, and the always-on toggles. Each field saves as you go; "Finish" records completion
 * unless you unticked "Show this on next startup".
 */
export function Onboarding({ onDone }: { onDone: () => void }) {
  const [step, setStep] = useState(0);

  // workspace
  const [workspace, setWorkspace] = useState("");
  const [wsMsg, setWsMsg] = useState<string | null>(null);

  // model + key
  const [models, setModels] = useState<string[]>([]);
  const [model, setModel] = useState("");
  const [hasKey, setHasKey] = useState(false);
  const [keyDraft, setKeyDraft] = useState("");
  const [keyMsg, setKeyMsg] = useState<string | null>(null);

  // always-on
  const [autostart, setAuto] = useState(false);
  const [keepAwake, setKeep] = useState(false);
  const [showAgain, setShowAgain] = useState(false); // inverse of "don't show again"; default = don't show

  useEffect(() => {
    getSettings()
      .then((s) => {
        setModels(s.models || []);
        setModel(s.model);
        setHasKey(s.has_key);
      })
      .catch(() => {});
    getSuperagent()
      .then((s) => s?.workspace && setWorkspace(s.workspace))
      .catch(() => {});
    if (isTauri()) {
      getAutostart().then((v) => setAuto(!!v));
      getKeepAwake().then((v) => setKeep(!!v));
    }
  }, []);

  const browse = async () => {
    const p = await pickFolder();
    if (p) saveWorkspace(p);
  };
  const saveWorkspace = async (p: string) => {
    setWorkspace(p);
    setWsMsg(null);
    const res = await setSuperagentWorkspace(p.trim());
    setWsMsg(res.ok ? "Saved." : res.error || "Couldn't set that folder.");
  };

  const chooseModel = async (m: string) => {
    setModel(m);
    await setDefaultModel(m);
  };
  const saveKey = async () => {
    if (!keyDraft.trim()) return;
    const res = await setModelKey(keyDraft.trim());
    if (res.ok) {
      setHasKey(true);
      setKeyDraft("");
      setKeyMsg("Saved locally.");
    } else {
      setKeyMsg(res.error || "Couldn't save key.");
    }
  };

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
              <h2>Welcome to Coworker</h2>
              <p className="ob-sub">
                A quick setup: pick a working folder for your always-on helper, set your model and
                API key, and choose how it stays running. Takes about a minute.
              </p>
            </div>
          )}

          {step === 1 && (
            <div className="ob-step">
              <h2>MyHelper's working folder</h2>
              <p className="ob-sub">
                Your always-on helper reads, writes, and saves deliverables here. You can change it
                later in Manage → Super-agent.
              </p>
              <div className="ob-row">
                <input
                  placeholder="/Users/you/coworker"
                  value={workspace}
                  onChange={(e) => setWorkspace(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && saveWorkspace(workspace)}
                />
                {isTauri() && (
                  <button className="btn" onClick={browse}>
                    Browse…
                  </button>
                )}
                <button className="btn primary" onClick={() => saveWorkspace(workspace)} disabled={!workspace.trim()}>
                  Set
                </button>
              </div>
              {wsMsg && <div className="ob-note">{wsMsg}</div>}
            </div>
          )}

          {step === 2 && (
            <div className="ob-step">
              <h2>Model & API key</h2>
              <p className="ob-sub">The default model for new sessions, and your OpenAI key.</p>
              <label className="ob-label">Default model</label>
              <select className="ob-select" value={model} onChange={(e) => chooseModel(e.target.value)}>
                {models.map((m) => (
                  <option key={m} value={m}>
                    {m}
                  </option>
                ))}
              </select>

              <label className="ob-label">
                OpenAI API key {hasKey && <span className="ob-ok">· configured</span>}
              </label>
              <div className="ob-row">
                <input
                  type="password"
                  placeholder={hasKey ? "•••••••• (saved) — enter to replace" : "sk-…"}
                  value={keyDraft}
                  autoComplete="off"
                  spellCheck={false}
                  onChange={(e) => setKeyDraft(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && saveKey()}
                />
                <button className="btn primary" onClick={saveKey} disabled={!keyDraft.trim()}>
                  Save
                </button>
              </div>
              <div className="ob-note dim">
                Stored locally at ~/.config/coworker/secrets.json (0600). Never sent to the model.
              </div>
              {keyMsg && <div className="ob-note">{keyMsg}</div>}
            </div>
          )}

          {step === 3 && (
            <div className="ob-step">
              <h2>Staying on</h2>
              <p className="ob-sub">
                The scheduler and MyHelper only run while Coworker is running.
                {!isTauri() && " (Desktop app only.)"}
              </p>
              <label className={"ob-toggle" + (isTauri() ? "" : " disabled")}>
                <input type="checkbox" checked={autostart} disabled={!isTauri()} onChange={(e) => toggleAuto(e.target.checked)} />
                <span>
                  <strong>Open at login</strong>
                  <small>Launch Coworker automatically when you sign in.</small>
                </span>
              </label>
              <label className={"ob-toggle" + (isTauri() ? "" : " disabled")}>
                <input type="checkbox" checked={keepAwake} disabled={!isTauri()} onChange={(e) => toggleKeep(e.target.checked)} />
                <span>
                  <strong>Keep the Mac awake</strong>
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
