import { useEffect, useState } from "react";
import {
  addMcpServer,
  allowUser,
  connectConnector,
  getAudit,
  deleteMcpServer,
  disallowUser,
  disconnectConnector,
  getConnectors,
  getMcpServers,
  getMcpTools,
  getProviders,
  getSettings,
  patchMcpServer,
  reloadMcp,
  setOnboarded,
  setProvider,
  setScratchBase,
  updateConnectorTools,
  verifyProvider,
  type AuditEvent,
  type Connector,
  type McpServer,
  type ModelSettings,
  type ProviderInfo,
} from "../api";
import { getAutostart, getKeepAwake, isTauri, pickFolder, setAutostart, setKeepAwake } from "../tauri";
import { useThemePref } from "../theme";
import { ModelChecklist } from "./ModelChecklist";
import { PersonasTab } from "./PersonasTab";
import { ConnectorBadge } from "../connectors/ConnectorIcon";
import { Toggle } from "./Toggle";

type Tab = "settings" | "models" | "personas" | "mcps" | "skills";

// Shared utility strings for the mock-ported Connectors / MCP surfaces (used by IntegrationsView).
const SEC_H = "text-[11px] uppercase tracking-[0.05em] text-faint font-semibold";
const CARD = "rounded-xl2 border border-line bg-panel";
const BTN_BORDERED =
  "text-[12.5px] px-3 py-1.5 rounded-lg border border-line bg-paper hover:border-lineStrong shrink-0";
const BTN_ACCENT = "text-[12.5px] px-3 py-1.5 rounded-lg bg-accent text-white shrink-0 disabled:opacity-50";
const BTN_DANGER = "text-[12.5px] text-danger/80 hover:text-danger shrink-0";

/** Two-letter initials for a chip/avatar (first+last word, else first two chars). */
function initials(name: string): string {
  const parts = name.trim().split(/\s+/).filter(Boolean);
  if (parts.length === 0) return "?";
  if (parts.length === 1) return parts[0].slice(0, 2).toUpperCase();
  return (parts[0][0] + parts[parts.length - 1][0]).toUpperCase();
}

const EXAMPLE = `{
  "filesystem": {
    "command": "npx",
    "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/dir"],
    "enabled": true
  }
}`;

const TABS: { key: Tab; label: string; disabled?: boolean }[] = [
  { key: "settings", label: "Settings" },
  { key: "models", label: "Configure Models" },
  { key: "personas", label: "Personas" },
  { key: "mcps", label: "MCPs" },
  { key: "skills", label: "Skills" },
];

export function ManageModal({
  onClose,
  initialTab,
}: {
  onClose: () => void;
  initialTab?: Tab;
}) {
  const tabs = TABS;
  const initial =
    initialTab && tabs.some((t) => t.key === initialTab && !t.disabled) ? initialTab : "settings";
  const [tab, setTab] = useState<Tab>(initial);

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal manage" onClick={(e) => e.stopPropagation()}>
        <div className="manage-head">
          <div className="manage-tabs">
            {tabs.map((t) => (
              <div
                key={t.key}
                className={"mtab" + (tab === t.key ? " active" : "") + (t.disabled ? " disabled" : "")}
                title={t.disabled ? "Coming soon" : undefined}
                onClick={t.disabled ? undefined : () => setTab(t.key)}
              >
                {t.label}
              </div>
            ))}
          </div>
          <div className="modal-close" onClick={onClose}>
            ✕
          </div>
        </div>
        <div className="manage-body">
          {tab === "settings" ? (
            <SettingsTab />
          ) : tab === "models" ? (
            <ModelsTab />
          ) : tab === "personas" ? (
            <PersonasTab />
          ) : tab === "mcps" ? (
            <McpTab />
          ) : (
            <div className="manage-empty">Skill management — coming soon.</div>
          )}
        </div>
        <div className="manage-foot">
          <span className="manage-foot-note dim">● Changes save automatically</span>
          <button className="btn-primary sm" onClick={onClose}>
            Done — back to chat
          </button>
        </div>
      </div>
    </div>
  );
}

// -- Configure Models tab (providers, model list, keys) -----------------------
function ModelsTab() {
  const [settings, setSettings] = useState<ModelSettings | null>(null);
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);
  const [providers, setProviders] = useState<ProviderInfo[]>([]);
  const [ollamaUrl, setOllamaUrl] = useState("");
  const [ollamaMsg, setOllamaMsg] = useState<string | null>(null);
  // Two panes: hosted API models (key-based, per provider) and local Ollama models.
  const [sub, setSub] = useState<"api" | "local">("api");
  const [apiProv, setApiProv] = useState("openai");
  const [endpoint, setEndpoint] = useState(""); // OpenAI custom endpoint (Azure, OpenRouter, …)
  const [verify, setVerify] = useState<{ state: "idle" | "testing" | "ok" | "error"; msg?: string }>({ state: "idle" });

  const refresh = () => getSettings().then(setSettings).catch(() => setSettings(null));
  const loadProviders = () =>
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
    refresh();
    loadProviders();
  }, []);

  const saveOllama = async () => {
    setOllamaMsg(null);
    const res = await setProvider("ollama", { base_url: ollamaUrl.trim() });
    if (res.ok) {
      const rec = res.recommended_model;
      setOllamaMsg(
        rec
          ? `Saved. ${rec} is the recommended model — added to your list if it's pulled (else: ollama pull ${rec}).`
          : "Saved. Tick or add an Ollama model under “Models” below to use it.",
      );
      loadProviders();
      refresh(); // pick up the auto-added recommended model in the curated list
    } else {
      setOllamaMsg(res.error || "Failed to save Ollama URL.");
    }
  };

  // The provider the pane is configuring (the ModelChecklist owns add/remove/default).
  const knownNames = providers.map((p) => p.name);
  const provName = sub === "local" ? "ollama" : apiProv;
  const selProv = providers.find((p) => p.name === provName);

  const keyFields = (): Record<string, string> => {
    const fields: Record<string, string> = {};
    if (draft.trim()) fields.api_key = draft.trim();
    if (apiProv === "openai") fields.base_url = endpoint.trim();
    return fields;
  };

  const testKey = async () => {
    if (!draft.trim() && !selProv?.configured) return;
    setVerify({ state: "testing" });
    const res = await verifyProvider(apiProv, keyFields());
    setVerify(res.ok ? { state: "ok" } : { state: "error", msg: res.error || "Couldn't verify." });
  };

  const save = async () => {
    if (!draft.trim() && !(apiProv === "openai" && endpoint.trim())) return;
    setBusy(true);
    setMsg(null);
    const res = await setProvider(apiProv, keyFields());
    setBusy(false);
    if (res.ok) {
      setDraft("");
      setVerify({ state: "idle" });
      // set_provider auto-adds the provider's model and makes it the default if the old default
      // wasn't usable — so the composer is ready immediately. Confirm that to the user.
      const updated = await getSettings().catch(() => null);
      if (updated) setSettings(updated);
      setMsg(
        updated
          ? `Saved. “${updated.model}” is ready in the composer — stored locally, never sent to the model.`
          : "Saved. The key is stored locally and never sent to the model.",
      );
      loadProviders();
    } else {
      setMsg(res.error || "Failed to save key.");
    }
  };

  if (!settings) return <div className="manage-empty">Loading…</div>;

  // The checklist shown once the pane's provider is usable (always, for keyless Ollama).
  const checklist = (selProv?.configured || sub === "local") && (
    <>
      <div className="sa-sub" style={{ marginTop: 22 }}>Models</div>
      <div className="conn-meta dim" style={{ marginBottom: 4 }}>
        {sub === "local" ? (
          <>Your pulled Ollama models. Ticked ones show in the composer's picker; the black badge marks the default for new sessions.</>
        ) : (
          <>Ticked models show in the composer's picker; the black badge marks the default for new sessions.</>
        )}
      </div>
      <ModelChecklist
        provider={provName}
        knownProviders={knownNames}
        suggested={selProv?.suggested_models || []}
        curated={settings.models}
        defaultModel={settings.model}
        onChanged={(next) => setSettings((s) => (s ? { ...s, models: next.models, model: next.model } : s))}
      />
    </>
  );

  return (
    <div className="conn-tab">
      <div className="subtabs" style={{ marginTop: 4 }}>
        <div className="manage-tabs">
          <div
            className={"mtab" + (sub === "api" ? " active" : "")}
            onClick={() => setSub("api")}
          >
            API models
          </div>
          <div
            className={"mtab" + (sub === "local" ? " active" : "")}
            onClick={() => setSub("local")}
          >
            Local models
          </div>
        </div>
      </div>

      {sub === "api" ? (
        <>
          <label className="conn-field">
            <span className="conn-field-label">Provider</span>
            <select
              value={apiProv}
              onChange={(e) => {
                setApiProv(e.target.value);
                setDraft("");
                setMsg(null);
                setVerify({ state: "idle" });
              }}
            >
              {providers
                .filter((p) => p.name !== "ollama")
                .map((p) => (
                  <option key={p.name} value={p.name}>
                    {p.title}
                  </option>
                ))}
            </select>
          </label>
          <div className="conn-meta" style={{ marginBottom: 12 }}>
            {selProv?.configured ? (
              <span className="ok">● Connected — key set{provName === "openai" && settings.source === "env" ? " (from environment)" : ""}</span>
            ) : (
              <span className="danger">● Not connected — add a key below to use this provider</span>
            )}
          </div>
        </>
      ) : (
        <div className="conn-note dim" style={{ marginBottom: 12 }}>
          Run models locally with <code>ollama serve</code>. OpenCoworker uses Ollama's
          OpenAI-compatible API, so tools work. No API key needed.
        </div>
      )}

      {sub === "api" ? (
        <>
          {provName === "openai" && settings.source === "env" ? (
            <div className="conn-note dim">
              A key is set via <code>OPENAI_API_KEY</code> in this server's environment. You can override
              it below; the stored key is used only when the environment variable is absent.
            </div>
          ) : null}
          {provName === "openai" && (
            <label className="conn-field">
              <span className="conn-field-label">Custom endpoint (optional)</span>
              <input
                type="text"
                placeholder="https://…/openai/v1"
                value={endpoint}
                spellCheck={false}
                autoComplete="off"
                onChange={(e) => setEndpoint(e.target.value)}
              />
              <span className="conn-field-help">
                For Azure OpenAI, OpenRouter, vLLM, or any OpenAI-compliant server. Leave blank for
                api.openai.com.
              </span>
            </label>
          )}
          <label className="conn-field">
            <span className="conn-field-label">
              {selProv?.fields.find((f) => f.key === "api_key")?.label || "API key"}
            </span>
            <input
              type="password"
              placeholder={selProv?.fields.find((f) => f.key === "api_key")?.placeholder || "sk-…"}
              value={draft}
              spellCheck={false}
              autoComplete="off"
              onChange={(e) => setDraft(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && save()}
            />
            <span className="conn-field-help">
              Stored locally at <code>~/.config/coworker/secrets.json</code> (0600). Required for the
              desktop app, where the server can't read your shell environment.
            </span>
          </label>
          <div className="conn-setup-actions">
            <button
              className="btn sm"
              onClick={testKey}
              disabled={verify.state === "testing" || (!draft.trim() && !selProv?.configured)}
              title="Check the key works — without saving it"
            >
              {verify.state === "testing" ? "Testing…" : "Test"}
            </button>
            <button
              className="btn-primary sm"
              onClick={save}
              disabled={busy || (!draft.trim() && !(apiProv === "openai" && endpoint.trim()))}
            >
              {busy ? "Saving…" : "Save"}
            </button>
          </div>
          {verify.state === "ok" && <div className="conn-meta ok" style={{ marginTop: 10 }}>✓ Key verified.</div>}
          {verify.state === "error" && <div className="conn-meta danger" style={{ marginTop: 10 }}>{verify.msg}</div>}
          {msg && <div className="conn-meta" style={{ marginTop: 10 }}>{msg}</div>}
          {checklist}
        </>
      ) : (
        <>
          <label className="conn-field">
            <span className="conn-field-label">Ollama server URL</span>
            <input
              type="text"
              placeholder="http://localhost:11434"
              value={ollamaUrl}
              spellCheck={false}
              autoComplete="off"
              onChange={(e) => setOllamaUrl(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && saveOllama()}
            />
            <span className="conn-field-help">
              The OpenAI-compatible <code>/v1</code> path is added automatically. Leave blank for the
              default. Your pulled models appear under <strong>Models</strong> below.
            </span>
          </label>
          <div className="conn-setup-actions">
            <button className="btn-primary sm" onClick={saveOllama}>
              Save Ollama URL
            </button>
          </div>
          {ollamaMsg && <div className="conn-meta" style={{ marginTop: 10 }}>{ollamaMsg}</div>}
          {checklist}
        </>
      )}
    </div>
  );
}

// -- Settings tab (files, surfaces, app behavior) ------------------------------
function SettingsTab() {
  const [settings, setSettings] = useState<ModelSettings | null>(null);
  const [theme, setTheme] = useThemePref();
  const [autostart, setAuto] = useState(false);
  const [keepAwake, setKeep] = useState(false);
  const [scratchDraft, setScratchDraft] = useState("");
  const [scratchMsg, setScratchMsg] = useState<string | null>(null);
  const desktop = isTauri();

  const refresh = () =>
    getSettings()
      .then((s) => {
        setSettings(s);
        setScratchDraft((d) => d || s.scratch_base || "");
      })
      .catch(() => setSettings(null));
  useEffect(() => {
    refresh();
    if (isTauri()) {
      getAutostart().then((v) => setAuto(!!v));
      getKeepAwake().then((v) => setKeep(!!v));
    }
  }, []);


  const saveScratch = async () => {
    setScratchMsg(null);
    const res = await setScratchBase(scratchDraft.trim());
    if (res.ok) {
      setScratchMsg("Saved. New conversations will use this location.");
      refresh();
    } else {
      setScratchMsg(res.error || "Could not use that location.");
    }
  };
  const browseScratch = async () => {
    const picked = await pickFolder();
    if (picked) setScratchDraft(picked);
  };

  const toggleAuto = async (v: boolean) => setAuto(!!(await setAutostart(v)));
  const toggleKeep = async (v: boolean) => setKeep(!!(await setKeepAwake(v)));
  const runSetupAgain = async () => {
    await setOnboarded(false);
    window.dispatchEvent(new CustomEvent("coworker:open-onboarding"));
  };

  if (!settings) return <div className="manage-empty">Loading…</div>;

  return (
    <div className="conn-tab">
      <div className="sa-sub">Appearance</div>
      <div className="seg" role="radiogroup" aria-label="Appearance" style={{ marginTop: 8 }}>
        {(["light", "dark", "auto"] as const).map((p) => (
          <button key={p} className={p === theme ? "active" : ""} onClick={() => setTheme(p)}>
            {p === "light" ? "Light" : p === "dark" ? "Dark" : "Auto"}
          </button>
        ))}
      </div>
      <div className="conn-meta dim" style={{ marginTop: 8 }}>
        Auto follows your Mac&rsquo;s appearance.
      </div>

      <div className="sa-sub" style={{ marginTop: 22 }}>OpenCoworker files</div>
      <div className="conn-meta dim" style={{ marginBottom: 10 }}>
        Each OpenCoworker conversation gets its own scratch folder under this location, where the agent
        saves files by default. You can grant access to more folders inside any conversation.
      </div>
      <label className="conn-field">
        <span className="conn-field-label">Scratch location</span>
        <div className="dirreq-pathrow">
          <input
            className="dirreq-path"
            type="text"
            placeholder="~/OpenCoworker"
            value={scratchDraft}
            spellCheck={false}
            autoComplete="off"
            onChange={(e) => setScratchDraft(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && saveScratch()}
          />
          {desktop && (
            <button className="btn" onClick={browseScratch} title="Pick a folder">
              Browse
            </button>
          )}
          <button className="btn-primary sm" onClick={saveScratch} disabled={!scratchDraft.trim()}>
            Save
          </button>
        </div>
        <span className="conn-field-help">
          New conversations use this; existing ones keep their current folder.
        </span>
      </label>
      {scratchMsg && <div className="conn-meta" style={{ marginTop: 8 }}>{scratchMsg}</div>}

      <div className="sa-sub" style={{ marginTop: 22 }}>Surfaces</div>
      <div className="conn-meta dim" style={{ marginBottom: 10 }}>
        Which coworkers appear in the left panel is now managed per persona in the{" "}
        <strong>Personas</strong> tab (Enabled + In picker).
      </div>

      {desktop && (
        <>
          <div className="sa-sub" style={{ marginTop: 22 }}>
            Always-on
          </div>
          <label className="ob-toggle">
            <input type="checkbox" checked={autostart} onChange={(e) => toggleAuto(e.target.checked)} />
            <span>
              <strong>Open at login</strong>
              <small>Launch OpenCoworker automatically when you sign in.</small>
            </span>
          </label>
          <label className="ob-toggle">
            <input type="checkbox" checked={keepAwake} onChange={(e) => toggleKeep(e.target.checked)} />
            <span>
              <strong>Keep this system awake</strong>
              <small>Prevent idle sleep so scheduled tasks fire on time.</small>
            </span>
          </label>
          <div className="conn-setup-actions" style={{ marginTop: 14 }}>
            <button className="btn sm" onClick={runSetupAgain}>
              Run setup again
            </button>
          </div>
        </>
      )}
    </div>
  );
}

export function McpTab() {
  const [servers, setServers] = useState<McpServer[]>([]);
  const [adding, setAdding] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refresh = () => getMcpServers().then(setServers).catch(() => setServers([]));
  useEffect(() => {
    refresh();
  }, []);

  const toggle = async (s: McpServer) => {
    await patchMcpServer(s.name, { enabled: !s.enabled });
    refresh();
  };
  const remove = async (s: McpServer) => {
    await deleteMcpServer(s.name);
    refresh();
  };

  return (
    <div className="space-y-3">
      <p className="text-[12.5px] text-muted leading-relaxed">
        External tool servers (stdio or HTTP), shared across all agents. Enabled servers' tools are
        permission-gated. Changes apply to new sessions —{" "}
        <button
          className="text-accent font-medium hover:underline"
          onClick={() => reloadMcp().then(refresh)}
        >
          reload now
        </button>
        .
      </p>

      {servers.length === 0 && !adding ? (
        <div className={CARD + " p-4 text-[13px] text-muted"}>
          No MCP servers configured.{" "}
          <button className="text-accent font-medium" onClick={() => setAdding(true)}>
            Add a server
          </button>
        </div>
      ) : (
        <div className="space-y-2">
          {servers.map((s) => (
            <McpRow key={s.name} server={s} onToggle={() => toggle(s)} onRemove={() => remove(s)} />
          ))}
        </div>
      )}

      {adding ? (
        <AddForm
          onCancel={() => {
            setAdding(false);
            setError(null);
          }}
          onError={setError}
          onAdded={() => {
            setAdding(false);
            setError(null);
            refresh();
          }}
        />
      ) : servers.length > 0 ? (
        <button className={BTN_ACCENT} onClick={() => setAdding(true)}>
          + Add server
        </button>
      ) : null}
      {error && <div className="text-[12.5px] text-danger">{error}</div>}
    </div>
  );
}

function McpRow({
  server,
  onToggle,
  onRemove,
}: {
  server: McpServer;
  onToggle: () => void;
  onRemove: () => void;
}) {
  const [tools, setTools] = useState<{ name: string; description: string }[] | null>(null);
  const [busy, setBusy] = useState(false);
  const [toolErr, setToolErr] = useState<string | null>(null);

  const loadTools = async () => {
    if (tools) {
      setTools(null);
      return;
    }
    setBusy(true);
    setToolErr(null);
    const res = await getMcpTools(server.name);
    setBusy(false);
    if (res.ok) setTools(res.tools);
    else setToolErr(res.error || "failed to connect");
  };

  return (
    <div className={CARD + " p-3.5"}>
      <div className="flex items-center gap-3">
        <Toggle checked={server.enabled} onChange={onToggle} title="Enable this server" />
        <div className="flex-1 min-w-0">
          <div className="text-[14px] font-medium">{server.name}</div>
          <div className="text-[11.5px] text-faint">
            {server.transport} · {server.status}
            {server.tool_count != null ? ` · ${server.tool_count} tools` : ""}
            {server.requires_approval ? " · asks" : ""}
          </div>
        </div>
        <button
          className="text-[12px] text-muted hover:text-ink shrink-0"
          onClick={loadTools}
          disabled={busy}
        >
          {busy ? "…" : tools ? "hide tools" : "tools"}
        </button>
        <button className={BTN_DANGER} onClick={onRemove}>
          remove
        </button>
      </div>
      {toolErr && <div className="text-[12.5px] text-danger mt-1.5">{toolErr}</div>}
      {tools && (
        <div className="mt-2.5 pt-2.5 border-t border-line flex flex-wrap gap-1.5">
          {tools.length === 0 && <div className="text-[12px] text-faint">No tools.</div>}
          {tools.map((t) => (
            <span
              key={t.name}
              title={t.description}
              className="font-mono text-[11.5px] px-1.5 py-0.5 rounded-md bg-paper border border-line"
            >
              {t.name}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

function AddForm({
  onCancel,
  onAdded,
  onError,
}: {
  onCancel: () => void;
  onAdded: () => void;
  onError: (e: string | null) => void;
}) {
  const [text, setText] = useState(EXAMPLE);

  const save = async () => {
    onError(null);
    let parsed: any;
    try {
      parsed = JSON.parse(text);
    } catch (e: any) {
      onError("Invalid JSON: " + e.message);
      return;
    }
    // Accept either {mcpServers:{...}}, {name:{...}}, or a single bare config.
    const map = parsed.mcpServers || parsed;
    const entries =
      map && typeof map === "object" && !map.command && !map.url
        ? Object.entries(map)
        : null;
    if (!entries || entries.length === 0) {
      onError('Paste a `{ "<name>": { … } }` object (or a full mcpServers block).');
      return;
    }
    for (const [name, config] of entries) {
      await addMcpServer(name, config as Record<string, any>);
    }
    onAdded();
  };

  return (
    <div className="space-y-2">
      <div className="text-[12.5px] text-muted">Paste server JSON (name → config):</div>
      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        spellCheck={false}
        rows={9}
        className="w-full font-mono text-[12px] px-3 py-2.5 rounded-lg border border-line bg-paper text-ink outline-none focus:border-accent resize-y"
      />
      <div className="flex items-center gap-3">
        <button className={BTN_ACCENT} onClick={save}>
          Add
        </button>
        <button className="text-[12.5px] text-muted hover:text-ink" onClick={onCancel}>
          cancel
        </button>
      </div>
    </div>
  );
}

// -- Connectors tab ----------------------------------------------------------
export function ConnectorsTab() {
  const [connectors, setConnectors] = useState<Connector[]>([]);
  const [openName, setOpenName] = useState<string | null>(null);

  const refresh = () => getConnectors().then(setConnectors).catch(() => setConnectors([]));
  useEffect(() => {
    refresh();
    // Recent senders arrive over time (someone DMs the bot) — poll so they surface to Allow.
    const t = setInterval(refresh, 5000);
    return () => clearInterval(t);
  }, []);

  return (
    <div className="grid grid-cols-2 gap-3">
      {connectors.map((c) => (
        <ConnectorRow
          key={c.name}
          c={c}
          open={openName === c.name}
          onToggleOpen={() => setOpenName(openName === c.name ? null : c.name)}
          onChanged={() => {
            setOpenName(null);
            refresh();
          }}
          onRefresh={refresh}
        />
      ))}
    </div>
  );
}

function ConnectorRow({
  c,
  open,
  onToggleOpen,
  onChanged,
  onRefresh,
}: {
  c: Connector;
  open: boolean;
  onToggleOpen: () => void;
  onChanged: () => void;
  onRefresh: () => void;
}) {
  // Connected (with its expandable settings) and any open card claim the full row width so the
  // expanded section has room; compact unconnected/soon cards stay one-per-column (mock grid).
  const wide = c.connected || open;

  return (
    <div className={CARD + (wide ? " col-span-2" : "")}>
      <div className="flex items-center gap-3 p-3.5">
        {/* Real brand color + logo from the API descriptor (Phase 1) via ConnectorBadge; unknown
            logo ids fall back to the neutral plug glyph. */}
        <ConnectorBadge connector={c} size={36} title={c.title} />
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            <span className="font-semibold text-[14px]">{c.title}</span>
            {c.two_way && (
              <span className="text-[10.5px] px-1.5 py-0.5 rounded border border-line text-muted">
                two-way
              </span>
            )}
          </div>
          {c.connected ? (
            <div className="text-[12px] text-muted flex items-center gap-1.5">
              <span className="w-1.5 h-1.5 rounded-full bg-ok shrink-0" />
              {c.account || "connected"}
              {c.allowed_users.length > 0 ? ` · ${c.allowed_users.length} allowed` : ""}
            </div>
          ) : (
            <div className="text-[12px] text-muted">{c.blurb || (c.available ? "Not connected" : "")}</div>
          )}
        </div>
        <div className="ml-auto flex items-center gap-2">
          {!c.available ? (
            <span className="text-[12px] text-faint">Soon</span>
          ) : c.connected && c.auth === "none" ? (
            <button className={BTN_BORDERED} onClick={onToggleOpen}>
              {open ? "Close" : "Settings"}
            </button>
          ) : c.connected ? (
            <>
              <button className={BTN_BORDERED} onClick={onToggleOpen}>
                {open ? "Close" : "Settings"}
              </button>
              <button
                className={BTN_DANGER}
                onClick={async () => {
                  await disconnectConnector(c.name);
                  onChanged();
                }}
              >
                Disconnect
              </button>
            </>
          ) : (
            <button className={BTN_BORDERED} onClick={onToggleOpen}>
              {open ? "Cancel" : "Connect"}
            </button>
          )}
        </div>
      </div>
      {open && !c.connected && <ConnectSetup c={c} onConnected={onChanged} />}
      {open && c.connected && <ConnectorTools c={c} onChanged={onChanged} />}
      {open && c.connected && c.two_way && <AllowlistBlock c={c} onChanged={onRefresh} />}
    </div>
  );
}

// Who may message this two-way bot. Recent senders surface here once they DM/mention the bot, so you
// can Allow them; allowed users are chips you can remove. (Was orphaned in the super-agent view.)
function AllowlistBlock({ c, onChanged }: { c: Connector; onChanged: () => void }) {
  const recent = c.recent ?? [];
  const unknownRecent = recent.filter((r) => !r.authorized);

  return (
    <div className="border-t border-line px-3.5 py-3 grid grid-cols-2 gap-5">
      <div>
        <div className={SEC_H + " mb-2"}>Allowed to message</div>
        <div className="flex flex-wrap gap-1.5">
          {c.allowed_users.length === 0 && (
            <span className="text-[12px] text-faint">nobody yet — Allow a recent sender →</span>
          )}
          {c.allowed_users.map((u) => (
            <span
              key={u}
              className="inline-flex items-center gap-1.5 pl-2 pr-1 py-1 rounded-full bg-paper border border-line text-[12px]"
            >
              <span className="w-4 h-4 rounded-full bg-accentSoft text-accent grid place-items-center text-[9px] font-bold">
                {initials(u)}
              </span>
              {u}
              <button
                className="w-4 h-4 grid place-items-center text-faint hover:text-danger"
                title="remove"
                onClick={async () => {
                  await disallowUser(c.name, u);
                  onChanged();
                }}
              >
                ×
              </button>
            </span>
          ))}
        </div>
      </div>
      <div>
        <div className={SEC_H + " mb-2"}>Recent senders</div>
        {unknownRecent.length === 0 ? (
          <div className="text-[12px] text-faint">None yet. Message the bot once and it'll show here.</div>
        ) : (
          <div className="space-y-1.5">
            {unknownRecent.map((r) => (
              <div className="flex items-center gap-2 text-[12.5px]" key={r.user_id}>
                <span className="w-5 h-5 rounded-full bg-paper border border-line grid place-items-center text-[9px] font-bold text-muted shrink-0">
                  {initials(r.user_name || "?")}
                </span>
                <span className="min-w-0 truncate" title={`id ${r.user_id}`}>
                  {r.user_name || "unknown"} <span className="text-faint">· {r.chat_type}</span>
                </span>
                <button
                  className="ml-auto text-[11.5px] px-2 py-0.5 rounded-md bg-accent text-white shrink-0"
                  onClick={async () => {
                    await allowUser(c.name, r.user_id);
                    onChanged();
                  }}
                >
                  Allow
                </button>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function ConnectorTools({ c, onChanged }: { c: Connector; onChanged: () => void }) {
  const toggle = async (toolName: string, enabled: boolean) => {
    await updateConnectorTools(c.name, { [toolName]: enabled });
    onChanged();
  };
  if (!c.tools?.length)
    return (
      <div className="border-t border-line px-3.5 py-3 text-[12.5px] text-muted">
        No tools for this connector yet.
      </div>
    );
  return (
    <div className="border-t border-line px-3.5 py-3">
      <div className={SEC_H + " mb-2"}>Tools exposed to OpenCoworker</div>
      <div className="space-y-1.5">
        {c.tools.map((tool) => (
          <label
            className="flex items-start gap-2.5 p-2 rounded-lg border border-line bg-paper"
            key={tool.name}
          >
            <input
              type="checkbox"
              className="mt-0.5 shrink-0"
              checked={tool.enabled}
              onChange={(e) => toggle(tool.name, e.target.checked)}
            />
            <span className="min-w-0">
              <span className="block text-[13px]">{tool.label}</span>
              <span className="block text-[11.5px] text-faint">
                {tool.name} · {tool.kind} · asks approval
              </span>
              <span className="block text-[11.5px] text-faint">{tool.description}</span>
            </span>
          </label>
        ))}
      </div>
    </div>
  );
}

function ConnectSetup({ c, onConnected }: { c: Connector; onConnected: () => void }) {
  const [values, setValues] = useState<Record<string, string>>({});
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const submit = async () => {
    setBusy(true);
    setError(null);
    const res = await connectConnector(c.name, values);
    setBusy(false);
    if (res.ok) onConnected();
    else setError(res.error || "could not connect");
  };

  return (
    <div className="border-t border-line px-3.5 py-3 space-y-3">
      {c.instructions.length > 0 && (
        <ol className="list-decimal pl-4 text-[12.5px] text-muted leading-relaxed space-y-1">
          {c.instructions.map((step, i) => (
            <li key={i}>{step}</li>
          ))}
        </ol>
      )}
      {c.fields.map((f) => (
        <label className="conn-field" key={f.key}>
          <span className="conn-field-label">
            {f.label}
            {!f.required && <em> (optional)</em>}
          </span>
          <input
            type={f.secret ? "password" : "text"}
            placeholder={f.placeholder}
            value={values[f.key] || ""}
            spellCheck={false}
            onChange={(e) => setValues({ ...values, [f.key]: e.target.value })}
          />
          {f.help && <span className="conn-field-help">{f.help}</span>}
        </label>
      ))}
      <div>
        <button className={BTN_ACCENT} onClick={submit} disabled={busy}>
          {busy ? "Validating…" : "Connect"}
        </button>
      </div>
      {error && <div className="text-[12.5px] text-danger">{error}</div>}
    </div>
  );
}

// -- Audit tab ---------------------------------------------------------------
export function AuditTab() {
  const [events, setEvents] = useState<AuditEvent[]>([]);
  const [sessionFilter, setSessionFilter] = useState("");
  const [connectorFilter, setConnectorFilter] = useState("");
  const [toolFilter, setToolFilter] = useState("");

  const refresh = () =>
    getAudit({
      limit: 150,
      session_id: sessionFilter.trim() || undefined,
      connector: connectorFilter.trim() || undefined,
      tool: toolFilter.trim() || undefined,
    })
      .then(setEvents)
      .catch(() => setEvents([]));

  useEffect(() => {
    refresh();
  }, []);

  return (
    <div className="conn-tab">
      <div className="mcp-intro">Recent connector and browser tool activity. Arguments are sanitized before storage.</div>
      <div className="audit-filters">
        <input placeholder="session id" value={sessionFilter} onChange={(e) => setSessionFilter(e.target.value)} />
        <input placeholder="connector" value={connectorFilter} onChange={(e) => setConnectorFilter(e.target.value)} />
        <input placeholder="tool" value={toolFilter} onChange={(e) => setToolFilter(e.target.value)} />
        <button className="btn-primary sm" onClick={refresh}>Filter</button>
      </div>
      <div className="audit-list">
        {events.length === 0 ? (
          <div className="manage-empty">No audit events yet.</div>
        ) : (
          events.map((ev) => <AuditRow ev={ev} key={ev.id} />)
        )}
      </div>
    </div>
  );
}

function AuditRow({ ev }: { ev: AuditEvent }) {
  return (
    <div className="audit-row">
      <div className="audit-head">
        <span className="audit-tool">{ev.tool}</span>
        <span className="conn-meta">
          {ev.connector || "tool"} · {ev.stage || ev.status || "event"} · {ev.timestamp}
        </span>
      </div>
      <div className="conn-meta">
        session {ev.session_id || "-"} {ev.approval ? `· ${ev.approval}` : ""} {ev.status ? `· ${ev.status}` : ""}
      </div>
      {ev.resource && <div className="conn-field-help">resource: {ev.resource}</div>}
      <div className="audit-args">{formatAuditArgs(ev.args)}</div>
      {(ev.reason || ev.result_preview) && <div className="conn-field-help">{ev.reason || ev.result_preview}</div>}
    </div>
  );
}

function formatAuditArgs(args: Record<string, any>) {
  if (!args || Object.keys(args).length === 0) return "";
  return Object.entries(args)
    .map(([k, v]) => `${k}=${typeof v === "string" ? v : JSON.stringify(v)}`)
    .join("  ");
}
