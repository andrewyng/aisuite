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
  getSuperagent,
  patchMcpServer,
  reloadMcp,
  setOnboarded,
  setProvider,
  setSurfaces,
  setScratchBase,
  setSuperagentName,
  setSuperagentWorkspace,
  updateConnectorTools,
  type AuditEvent,
  type Connector,
  type McpServer,
  type ModelSettings,
  type ProviderInfo,
  type SuperagentStatus,
} from "../api";
import { getAutostart, getKeepAwake, isTauri, pickFolder, setAutostart, setKeepAwake } from "../tauri";
import { useThemePref } from "../theme";
import { ModelChecklist } from "./ModelChecklist";

type Tab = "settings" | "models" | "superagent" | "mcps" | "skills";

const EXAMPLE = `{
  "filesystem": {
    "command": "npx",
    "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/dir"],
    "enabled": true
  }
}`;

// Super-agent isn't shipping in this release: kept visible (last) but disabled.
const TABS: { key: Tab; label: string; disabled?: boolean }[] = [
  { key: "settings", label: "Settings" },
  { key: "models", label: "Configure Models" },
  { key: "mcps", label: "MCPs" },
  { key: "skills", label: "Skills" },
  { key: "superagent", label: "Super-agent", disabled: true },
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
          ) : tab === "superagent" ? (
            <SuperagentTab />
          ) : tab === "mcps" ? (
            <McpTab />
          ) : (
            <div className="manage-empty">Skill management — coming soon.</div>
          )}
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

  const save = async () => {
    if (!draft.trim() && !(apiProv === "openai" && endpoint.trim())) return;
    setBusy(true);
    setMsg(null);
    const fields: Record<string, string> = {};
    if (draft.trim()) fields.api_key = draft.trim();
    if (apiProv === "openai") fields.base_url = endpoint.trim();
    const res = await setProvider(apiProv, fields);
    setBusy(false);
    if (res.ok) {
      setDraft("");
      setMsg("Saved. The key is stored locally and never sent to the model.");
      refresh();
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
              <span className="ok">key configured{provName === "openai" && settings.source === "env" ? " (from environment)" : ""}</span>
            ) : (
              <span className="danger">no API key — calls to this provider will fail</span>
            )}
          </div>
        </>
      ) : (
        <div className="conn-meta dim" style={{ marginBottom: 12 }}>
          Run models locally with <code>ollama serve</code>. OpenCoworker uses Ollama's
          OpenAI-compatible API, so tools work. No API key needed.
        </div>
      )}

      {sub === "api" ? (
        <>
          {provName === "openai" && settings.source === "env" ? (
            <div className="conn-meta dim">
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
              className="btn-primary sm"
              onClick={save}
              disabled={busy || (!draft.trim() && !(apiProv === "openai" && endpoint.trim()))}
            >
              {busy ? "Saving…" : "Save"}
            </button>
          </div>
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

  const toggleSurface = async (key: "chat" | "code", value: boolean) => {
    const res = await setSurfaces({ [key]: value });
    if (res.surfaces) setSettings((s) => (s ? { ...s, surfaces: res.surfaces } : s));
  };

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
        OpenCoworker is always shown. Turn on Chat and Code to add them to the left panel.
      </div>
      <label className="ob-toggle">
        <input
          type="checkbox"
          checked={!!settings.surfaces?.chat}
          onChange={(e) => toggleSurface("chat", e.target.checked)}
        />
        <span>
          <strong>Show Chat</strong>
          <small>Quick Q&amp;A and brainstorming — no workspace.</small>
        </span>
      </label>
      <label className="ob-toggle">
        <input
          type="checkbox"
          checked={!!settings.surfaces?.code}
          onChange={(e) => toggleSurface("code", e.target.checked)}
        />
        <span>
          <strong>Show Code</strong>
          <small>Work inside a codebase with git, diffs, and the shell.</small>
        </span>
      </label>

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
    <div className="mcp-tab">
      <div className="mcp-intro">
        External tool servers (stdio or HTTP), shared across all agents. Enabled servers'
        tools are permission-gated. Changes apply to new sessions —{" "}
        <button className="link" onClick={() => reloadMcp().then(refresh)}>
          reload now
        </button>
        .
      </div>

      {servers.length === 0 && !adding && (
        <div className="manage-empty">No MCP servers configured yet.</div>
      )}

      <div className="mcp-list">
        {servers.map((s) => (
          <McpRow key={s.name} server={s} onToggle={() => toggle(s)} onRemove={() => remove(s)} />
        ))}
      </div>

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
      ) : (
        <button className="btn-primary" onClick={() => setAdding(true)}>
          ＋ Add server
        </button>
      )}
      {error && <div className="mcp-error">{error}</div>}
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
    <div className="mcp-row">
      <div className="mcp-main">
        <label className="switch">
          <input type="checkbox" checked={server.enabled} onChange={onToggle} />
          <span className="slider" />
        </label>
        <div className="mcp-id">
          <div className="mcp-name">{server.name}</div>
          <div className="mcp-meta">
            {server.transport} · {server.status}
            {server.tool_count != null ? ` · ${server.tool_count} tools` : ""}
            {server.requires_approval ? " · asks" : ""}
          </div>
        </div>
        <button className="link" onClick={loadTools} disabled={busy}>
          {busy ? "…" : tools ? "hide tools" : "tools"}
        </button>
        <button className="link danger" onClick={onRemove}>
          remove
        </button>
      </div>
      {toolErr && <div className="mcp-error">{toolErr}</div>}
      {tools && (
        <div className="mcp-tools">
          {tools.length === 0 && <div className="dim">No tools.</div>}
          {tools.map((t) => (
            <div className="mcp-tool" key={t.name} title={t.description}>
              <code>{t.name}</code>
            </div>
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
    <div className="mcp-add">
      <div className="mcp-add-label">Paste server JSON (name → config):</div>
      <textarea value={text} onChange={(e) => setText(e.target.value)} spellCheck={false} rows={9} />
      <div className="mcp-add-actions">
        <button className="btn-primary" onClick={save}>
          Add
        </button>
        <button className="link" onClick={onCancel}>
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
  }, []);

  return (
    <div className="conn-tab">
      <div className="mcp-intro">
        Connect the accounts and tools OpenCoworker can use. You bring the token for this local
        build; managed one-click sign-in can replace this setup later.
      </div>
      <div className="conn-list">
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
          />
        ))}
      </div>
    </div>
  );
}

function ConnectorRow({
  c,
  open,
  onToggleOpen,
  onChanged,
}: {
  c: Connector;
  open: boolean;
  onToggleOpen: () => void;
  onChanged: () => void;
}) {
  const status = !c.available
    ? "soon"
    : c.connected
      ? c.account || "connected"
      : "not connected";

  return (
    <div className="conn-row">
      <div className="conn-main">
        <span className="conn-icon">{c.icon}</span>
        <div className="conn-id">
          <div className="conn-name">
            {c.title}
            {c.two_way && <span className="conn-tag">two-way</span>}
          </div>
          <div className="conn-meta">
            {c.connected ? <span className="conn-dot" /> : null}
            {status}
            {c.connected && c.allowed_users > 0 ? ` · ${c.allowed_users} allowed` : ""}
          </div>
        </div>
        {!c.available ? (
          <span className="conn-soon">soon</span>
        ) : c.connected && c.auth === "none" ? (
          <button className="btn-primary sm" onClick={onToggleOpen}>
            {open ? "close" : "Settings"}
          </button>
        ) : c.connected ? (
          <div className="conn-actions">
            <button className="btn-primary sm" onClick={onToggleOpen}>
              {open ? "close" : "Settings"}
            </button>
            <button
              className="link danger"
              onClick={async () => {
                await disconnectConnector(c.name);
                onChanged();
              }}
            >
              disconnect
            </button>
          </div>
        ) : (
          <button className="btn-primary sm" onClick={onToggleOpen}>
            {open ? "cancel" : "Connect"}
          </button>
        )}
      </div>
      {open && !c.connected && <ConnectSetup c={c} onConnected={onChanged} />}
      {open && c.connected && <ConnectorTools c={c} onChanged={onChanged} />}
    </div>
  );
}

function ConnectorTools({ c, onChanged }: { c: Connector; onChanged: () => void }) {
  const toggle = async (toolName: string, enabled: boolean) => {
    await updateConnectorTools(c.name, { [toolName]: enabled });
    onChanged();
  };
  if (!c.tools?.length) return <div className="conn-setup">No tools for this connector yet.</div>;
  return (
    <div className="conn-setup">
      <div className="sa-sub">Tools exposed to OpenCoworker</div>
      <div className="tool-settings">
        {c.tools.map((tool) => (
          <label className="tool-setting-row" key={tool.name}>
            <input
              type="checkbox"
              checked={tool.enabled}
              onChange={(e) => toggle(tool.name, e.target.checked)}
            />
            <span className="tool-setting-main">
              <span className="tool-setting-name">{tool.label}</span>
              <span className="conn-meta">
                {tool.name} · {tool.kind} · asks approval
              </span>
              <span className="conn-field-help">{tool.description}</span>
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
    <div className="conn-setup">
      {c.instructions.length > 0 && (
        <ol className="conn-steps">
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
      <div className="conn-setup-actions">
        <button className="btn-primary" onClick={submit} disabled={busy}>
          {busy ? "Validating…" : "Connect"}
        </button>
      </div>
      {error && <div className="mcp-error">{error}</div>}
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

// -- Super-agent tab ---------------------------------------------------------
function SuperagentTab() {
  const [status, setStatus] = useState<SuperagentStatus | null>(null);
  const [wsDraft, setWsDraft] = useState("");
  const [wsMsg, setWsMsg] = useState<string | null>(null);
  const [nameDraft, setNameDraft] = useState("");
  const [nameMsg, setNameMsg] = useState<string | null>(null);

  const refresh = () =>
    getSuperagent()
      .then((s) => {
        setStatus(s);
        setWsDraft((d) => (d === "" ? s.workspace : d));
        setNameDraft((d) => (d === "" ? s.name : d));
      })
      .catch(() => setStatus(null));
  useEffect(() => {
    refresh();
  }, []);

  if (!status) return <div className="manage-empty">Loading…</div>;

  const saveWs = async () => {
    setWsMsg(null);
    const res = await setSuperagentWorkspace(wsDraft.trim());
    if (res.ok) setWsMsg(res.restart_required ? "Saved — restart the server to apply." : "Saved.");
    else setWsMsg(res.error || "could not save");
    refresh();
  };

  const saveName = async () => {
    setNameMsg(null);
    const res = await setSuperagentName(nameDraft.trim());
    if (res.ok) setNameMsg("Saved — restart the server to apply.");
    else setNameMsg(res.error || "could not save");
    refresh();
  };

  return (
    <div className="sa-tab">
      <div className="mcp-intro">
        Your always-on personal helper. It runs on one continuous thread and replies in the app
        and over any connected chat. {status.running ? <span className="conn-dot" /> : null}
        {status.running ? " listening" : " not running"}.
      </div>

      <div className="sa-field">
        <span className="conn-field-label">Name (make it yours)</span>
        <div className="sa-ws-row">
          <input value={nameDraft} spellCheck={false} placeholder="MyHelper" onChange={(e) => setNameDraft(e.target.value)} />
          <button className="btn-primary sm" onClick={saveName}>
            Save
          </button>
        </div>
        {nameMsg && <span className="conn-field-help">{nameMsg}</span>}
      </div>

      <div className="sa-field">
        <span className="conn-field-label">Workspace (where it reads/writes files)</span>
        <div className="sa-ws-row">
          <input value={wsDraft} spellCheck={false} onChange={(e) => setWsDraft(e.target.value)} />
          <button className="btn-primary sm" onClick={saveWs}>
            Save
          </button>
        </div>
        {wsMsg && <span className="conn-field-help">{wsMsg}</span>}
      </div>

      {status.connectors.length === 0 ? (
        <div className="manage-empty">No two-way bot connected yet — add one in Connectors.</div>
      ) : (
        status.connectors.map((c) => (
          <SuperagentConnectorBlock key={c.name} c={c} onChanged={refresh} />
        ))
      )}
    </div>
  );
}

function SuperagentConnectorBlock({
  c,
  onChanged,
}: {
  c: import("../api").SuperagentConnector;
  onChanged: () => void;
}) {
  const unknownRecent = c.recent.filter((r) => !r.authorized);

  return (
    <div className="sa-conn">
      <div className="sa-conn-head">
        <span className="conn-name">{c.name}</span>
        <span className="conn-meta">
          {c.account || ""} {c.listening ? "· listening" : "· idle"}
        </span>
      </div>

      <div className="sa-sub">Allowed to message ({c.allowed_users.length})</div>
      <div className="sa-chips">
        {c.allowed_users.length === 0 && <span className="dim">nobody yet — add yourself below</span>}
        {c.allowed_users.map((u) => (
          <span className="sa-chip" key={u}>
            {u}
            <button
              className="sa-chip-x"
              title="remove"
              onClick={async () => {
                await disallowUser(c.name, u);
                onChanged();
              }}
            >
              ✕
            </button>
          </span>
        ))}
      </div>

      <div className="sa-sub">
        Recent senders{" "}
        <span className="dim">— DM the bot, then add yourself</span>
      </div>
      {unknownRecent.length === 0 ? (
        <div className="dim sa-recent-empty">None yet. Message the bot once and it'll show here.</div>
      ) : (
        <div className="sa-recent">
          {unknownRecent.map((r) => (
            <div className="sa-recent-row" key={r.user_id}>
              <span>
                {r.user_name || "unknown"} <span className="dim">· {r.chat_type} · id {r.user_id}</span>
              </span>
              <button
                className="btn-primary sm"
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
  );
}
