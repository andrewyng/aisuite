import { useEffect, useState } from "react";
import {
  getPersonas,
  installPersona,
  updatePersona,
  type Persona,
  type PersonaConsent,
} from "../api";

// Personas management: enable a persona, choose whether it shows in the new-session picker,
// set the default, and install more from a local directory or a GitHub repo (snapshotted).
export function PersonasTab() {
  const [personas, setPersonas] = useState<Persona[]>([]);
  const [mode, setMode] = useState<"git" | "dir">("git");
  const [src, setSrc] = useState("");
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);
  const [consent, setConsent] = useState<PersonaConsent[] | null>(null);

  const reload = () => getPersonas().then(setPersonas).catch(() => {});
  useEffect(() => {
    reload();
  }, []);

  const toggle = async (
    id: string,
    body: { enabled?: boolean; surfaced?: boolean; default?: boolean },
  ) => {
    const r = await updatePersona(id, body);
    if (r.personas) setPersonas(r.personas);
    else reload();
  };

  const install = async () => {
    if (!src.trim()) return;
    setBusy(true);
    setMsg(null);
    setConsent(null);
    const r = await installPersona(
      mode === "git" ? { git_url: src.trim() } : { dir: src.trim() },
    );
    setBusy(false);
    if (!r.ok) {
      setMsg(r.error || "install failed");
      return;
    }
    setConsent(r.consent || []);
    if (r.personas) setPersonas(r.personas);
    setMsg(`Installed ${(r.consent || []).length} persona(s) — review and enable below.`);
    setSrc("");
  };

  return (
    <div className="conn-tab">
      <div className="sa-sub" style={{ marginTop: 4 }}>
        Personas
      </div>
      <div className="conn-note dim" style={{ marginBottom: 10 }}>
        Specialized coworkers. Enable one, then choose whether it appears in the new-session
        picker. The starred persona is the default for new sessions.
      </div>

      {personas.map((p) => (
        <div key={p.id} className="persona-row">
          <div className="persona-row-main">
            <div className="persona-row-name">
              {p.name}
              {p.default ? " ★" : ""}
              {p.builtin ? <span className="dim"> · built-in</span> : null}
            </div>
            <div className="dim persona-row-tag">{p.tagline}</div>
          </div>
          <label className="persona-row-toggle">
            <input
              type="checkbox"
              checked={p.enabled}
              onChange={(e) => toggle(p.id, { enabled: e.target.checked })}
            />{" "}
            Enabled
          </label>
          <label className="persona-row-toggle">
            <input
              type="checkbox"
              checked={p.surfaced}
              disabled={!p.enabled}
              onChange={(e) => toggle(p.id, { surfaced: e.target.checked })}
            />{" "}
            In picker
          </label>
          <button
            className="btn sm"
            disabled={p.default || !p.enabled}
            onClick={() => toggle(p.id, { default: true })}
          >
            Set default
          </button>
        </div>
      ))}

      <div className="sa-sub" style={{ marginTop: 22 }}>
        Add personas
      </div>
      <div className="conn-note dim" style={{ marginBottom: 8 }}>
        Load from a local directory or a public GitHub repo. Files are copied into a managed area
        (a snapshot), so the persona stays stable even if the source changes. No code runs — a
        persona only composes vetted tools.
      </div>
      <div className="persona-install">
        <select value={mode} onChange={(e) => setMode(e.target.value as "git" | "dir")}>
          <option value="git">GitHub URL</option>
          <option value="dir">Local directory</option>
        </select>
        <input
          placeholder={
            mode === "git" ? "https://github.com/acme/ops-persona" : "/path/to/personas"
          }
          value={src}
          onChange={(e) => setSrc(e.target.value)}
        />
        <button className="btn-primary sm" disabled={busy || !src.trim()} onClick={install}>
          {busy ? "Installing…" : "Install"}
        </button>
      </div>
      {msg ? (
        <div className="conn-note dim" style={{ marginTop: 8 }}>
          {msg}
        </div>
      ) : null}

      {consent && consent.length ? (
        <div style={{ marginTop: 10 }}>
          {consent.map((c) => (
            <div key={c.id} className="consent-card">
              <div className="persona-row-name">{c.name}</div>
              <div className="dim" style={{ fontSize: 12, marginBottom: 6 }}>
                {c.description}
              </div>
              <div style={{ fontSize: 12 }}>Tools: {c.tools.join(", ") || "—"}</div>
              <div style={{ fontSize: 12 }}>
                Risk: {c.risk.join(", ") || "read"}
                {c.connectors ? " · connectors" : ""}
                {c.messaging ? " · messaging" : ""}
                {c.mcp.length ? ` · mcp: ${c.mcp.join(", ")}` : ""}
              </div>
              <div className="dim" style={{ fontSize: 12, marginTop: 4 }}>
                Recommended mode: {c.recommended_mode}. Enable it above to use it.
              </div>
            </div>
          ))}
        </div>
      ) : null}
    </div>
  );
}
