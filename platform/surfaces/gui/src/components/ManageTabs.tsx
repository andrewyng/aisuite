import { useEffect, useState } from "react";
import {
  addMcpServer,
  allowUser,
  cloudLogin,
  cloudLogout,
  setCloudTelemetry,
  connectConnector,
  connectManaged,
  deleteMcpServer,
  disallowUser,
  disconnectConnector,
  getCloudStatus,
  getConnectors,
  getMcpServers,
  getMcpTools,
  getProviders,
  getSettings,
  patchMcpServer,
  reloadMcp,
  setProvider,
  updateConnectorTools,
  verifyProvider,
  type CloudStatus,
  type Connector,
  type McpServer,
  type ModelSettings,
  type ProviderInfo,
} from "../api";
import { ModelChecklist } from "./ModelChecklist";
import { ConnectorBadge } from "../connectors/ConnectorIcon";
import { SelectMenu } from "./SelectMenu";
import { Toggle } from "./Toggle";

// "2h ago"-style label for the providers' Last-used line (null when never used).
const relTime = (epoch?: number | null): string | null => {
  if (!epoch) return null;
  const secs = Math.max(0, Math.floor(Date.now() / 1000 - epoch));
  if (secs < 90) return "just now";
  const mins = Math.floor(secs / 60);
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 48) return `${hrs}h ago`;
  return `${Math.floor(hrs / 24)}d ago`;
};

const fmtDate = (iso?: string | null): string | null => {
  if (!iso) return null;
  const d = new Date(iso + "T00:00:00");
  return Number.isNaN(d.getTime()) ? null : d.toLocaleDateString();
};

// Shared tab bodies for the Settings and Integrations pages (the old top-tab ManageModal was retired
// when Settings/Activity became full-page surfaces): ModelsTab → Settings ▸ Models; ConnectorsTab +
// McpTab → Integrations ▸ Connectors / MCP servers.
const SEC_H = "text-[11px] uppercase tracking-[0.05em] text-faint font-semibold";
const CARD = "rounded-xl2 border border-line bg-panel";
const BTN_BORDERED =
  "text-[12.5px] px-3 py-1.5 rounded-lg border border-line bg-paper hover:border-lineStrong shrink-0";
const BTN_ACCENT = "text-[12.5px] px-3 py-1.5 rounded-lg bg-accent text-white shrink-0 disabled:opacity-50";
const BTN_DANGER = "text-[12.5px] text-danger/80 hover:text-danger shrink-0";
const FIELD_LABEL = "block text-[12.5px] font-medium text-ink mb-1.5";
const FIELD_HELP = "block text-[12px] text-muted mt-1.5 leading-relaxed";
const INPUT_W =
  "w-full px-3 py-2 rounded-lg border border-line bg-paper text-[13px] text-ink outline-none focus:border-accent";

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

// -- Configure Models tab (providers, model list, keys) -----------------------
export function ModelsTab() {
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
        // Seed the endpoint for the selected provider: stored value, else the descriptor's
        // pre-filled default (OpenAI-compatible vendors ship their official endpoint).
        const sel = ps.find((p) => p.name === apiProv);
        const seeded =
          sel?.values?.base_url || sel?.fields.find((f) => f.key === "base_url")?.default || "";
        if (seeded) setEndpoint((cur) => cur || seeded);
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

  // The provider's endpoint field, when it has one (OpenAI's optional custom endpoint + the
  // OpenAI-compatible vendors' pre-filled one).
  const endpointField = selProv?.fields.find((f) => f.key === "base_url");

  const keyFields = (): Record<string, string> => {
    const fields: Record<string, string> = {};
    if (draft.trim()) fields.api_key = draft.trim();
    if (endpointField) fields.base_url = endpoint.trim();
    return fields;
  };

  const testKey = async () => {
    if (!draft.trim() && !selProv?.configured) return;
    setVerify({ state: "testing" });
    const res = await verifyProvider(apiProv, keyFields());
    setVerify(res.ok ? { state: "ok" } : { state: "error", msg: res.error || "Couldn't verify." });
  };

  const save = async () => {
    if (!draft.trim() && !(endpointField && endpoint.trim())) return;
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

  if (!settings) return <div className="text-[13px] text-muted">Loading…</div>;

  // The checklist shown once the pane's provider is usable (always, for keyless Ollama).
  const checklist = (selProv?.configured || sub === "local") && (
    <div className="mt-6">
      <div className={SEC_H + " mb-1.5"}>Models</div>
      <p className="text-[12px] text-muted mb-2.5 leading-relaxed">
        {sub === "local"
          ? "Your pulled Ollama models. Ticked ones show in the composer's picker; the black badge marks the default for new sessions."
          : "Ticked models show in the composer's picker; the black badge marks the default for new sessions."}
      </p>
      <ModelChecklist
        provider={provName}
        knownProviders={knownNames}
        suggested={selProv?.suggested_models || []}
        curated={settings.models}
        defaultModel={settings.model}
        labels={settings.model_labels}
        onChanged={(next) => setSettings((s) => (s ? { ...s, models: next.models, model: next.model } : s))}
      />
    </div>
  );

  return (
    <div>
      <div className="seg mb-4" role="tablist" aria-label="Model source">
        <button className={sub === "api" ? "active" : ""} onClick={() => setSub("api")}>
          API models
        </button>
        <button className={sub === "local" ? "active" : ""} onClick={() => setSub("local")}>
          Local models
        </button>
      </div>

      {sub === "api" ? (
        <div className={CARD + " p-4"}>
          <div className="mb-4">
            <span className={FIELD_LABEL}>Provider</span>
            {/* Custom select, sectioned "Ready to use" / "Needs a key" (configured providers
                float to the top). Rows carry a green "key set" dot + a Last-used sub-line, so
                which providers are configured (and still in use) is visible at a glance. */}
            <SelectMenu
              ariaLabel="Provider"
              value={apiProv}
              options={[...providers]
                .filter((p) => p.name !== "ollama")
                .sort((a, b) => Number(b.configured) - Number(a.configured)) // stable: keeps registry order within each section
                .map((p) => ({
                  value: p.name,
                  label: p.title,
                  group: p.configured ? "Ready to use" : "Needs a key",
                  dot: p.configured,
                  sub: p.configured
                    ? relTime(p.last_used_at)
                      ? `Last used ${relTime(p.last_used_at)}`
                      : "Not used yet"
                    : undefined,
                }))}
              onChange={(name) => {
                setApiProv(name);
                setDraft("");
                setMsg(null);
                setVerify({ state: "idle" });
                // Re-seed the endpoint for the newly selected provider (stored → default → blank).
                const p = providers.find((x) => x.name === name);
                setEndpoint(
                  p?.values?.base_url || p?.fields.find((f) => f.key === "base_url")?.default || "",
                );
              }}
            />
          </div>

          {selProv?.blurb && (
            <p className="text-[12px] text-muted -mt-2 mb-4 leading-relaxed">{selProv.blurb}</p>
          )}

          <div className="text-[12px] mb-4">
            {selProv?.configured ? (
              <span className="text-ok">
                ● Connected
                {fmtDate(selProv.key_set_at) ? ` — key added ${fmtDate(selProv.key_set_at)}` : " — key set"}
                {provName === "openai" && settings.source === "env" ? " (from environment)" : ""}
                <span className="text-muted">
                  {" · "}
                  {relTime(selProv.last_used_at) ? `last used ${relTime(selProv.last_used_at)}` : "not used yet"}
                </span>
              </span>
            ) : (
              <span className="text-danger">● Not connected — add a key below to use this provider</span>
            )}
          </div>

          {provName === "openai" && settings.source === "env" && (
            <p className="text-[12px] text-muted mb-4 leading-relaxed">
              A key is set via <code>OPENAI_API_KEY</code> in this server's environment. You can override
              it below; the stored key is used only when the environment variable is absent.
            </p>
          )}
          {endpointField && (
            <label className="block mb-4">
              <span className={FIELD_LABEL}>{endpointField.label}</span>
              <input
                className={INPUT_W}
                type="text"
                placeholder={endpointField.placeholder}
                value={endpoint}
                spellCheck={false}
                autoComplete="off"
                onChange={(e) => setEndpoint(e.target.value)}
              />
              <span className={FIELD_HELP}>{endpointField.help}</span>
            </label>
          )}
          <label className="block mb-4">
            <span className={FIELD_LABEL}>
              {selProv?.fields.find((f) => f.key === "api_key")?.label || "API key"}
            </span>
            <input
              className={INPUT_W}
              type="password"
              placeholder={selProv?.fields.find((f) => f.key === "api_key")?.placeholder || "sk-…"}
              value={draft}
              spellCheck={false}
              autoComplete="off"
              onChange={(e) => setDraft(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && save()}
            />
            <span className={FIELD_HELP}>
              Stored locally at <code>~/.config/coworker/secrets.json</code> (0600). Required for the
              desktop app, where the server can't read your shell environment.
            </span>
          </label>
          <div className="flex items-center gap-2">
            <button
              className={BTN_BORDERED}
              onClick={testKey}
              disabled={verify.state === "testing" || (!draft.trim() && !selProv?.configured)}
              title="Check the key works — without saving it"
            >
              {verify.state === "testing" ? "Testing…" : "Test"}
            </button>
            <button
              className={BTN_ACCENT}
              onClick={save}
              disabled={busy || (!draft.trim() && !(apiProv === "openai" && endpoint.trim()))}
            >
              {busy ? "Saving…" : "Save"}
            </button>
          </div>
          {verify.state === "ok" && <div className="text-[12px] text-ok mt-2.5">✓ Key verified.</div>}
          {verify.state === "error" && <div className="text-[12px] text-danger mt-2.5">{verify.msg}</div>}
          {msg && <div className="text-[12px] text-muted mt-2.5">{msg}</div>}
        </div>
      ) : (
        <div className={CARD + " p-4"}>
          <p className="text-[12.5px] text-muted mb-4 leading-relaxed">
            Run models locally with <code>ollama serve</code>. OpenCoworker uses Ollama's
            OpenAI-compatible API, so tools work. No API key needed.
          </p>
          <label className="block mb-4">
            <span className={FIELD_LABEL}>Ollama server URL</span>
            <input
              className={INPUT_W}
              type="text"
              placeholder="http://localhost:11434"
              value={ollamaUrl}
              spellCheck={false}
              autoComplete="off"
              onChange={(e) => setOllamaUrl(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && saveOllama()}
            />
            <span className={FIELD_HELP}>
              The OpenAI-compatible <code>/v1</code> path is added automatically. Leave blank for the
              default. Your pulled models appear under <strong>Models</strong> below.
            </span>
          </label>
          <button className={BTN_ACCENT} onClick={saveOllama}>
            Save Ollama URL
          </button>
          {ollamaMsg && <div className="text-[12px] text-muted mt-2.5">{ollamaMsg}</div>}
        </div>
      )}

      {checklist}
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
  const [cloud, setCloud] = useState<CloudStatus | null>(null);

  const refresh = () => {
    getConnectors().then(setConnectors).catch(() => setConnectors([]));
    getCloudStatus().then(setCloud).catch(() => setCloud(null));
  };
  useEffect(() => {
    refresh();
    // Recent senders arrive over time (someone DMs the bot) — poll so they surface to Allow.
    // The same poll also picks up sign-in / managed-connect completions from the browser flows.
    const t = setInterval(refresh, 5000);
    return () => clearInterval(t);
  }, []);

  return (
    <div className="space-y-3">
      <CloudAccountCard cloud={cloud} onChanged={refresh} />
      {/* Single-column list (owner call, 2026-07-03 #2): expanding a card just pushes rows
          down — a mixed-width grid reordered visually every time one opened. */}
      <div className="space-y-3">
        {connectors.map((c) => (
          <ConnectorRow
            key={c.name}
            c={c}
            cloud={cloud}
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
    </div>
  );
}

// Optional cloud sign-in: unlocks one-click managed connectors. Local-only /
// manual token setup keeps working without it (and stays available after).
function CloudAccountCard({
  cloud,
  onChanged,
}: {
  cloud: CloudStatus | null;
  onChanged: () => void;
}) {
  const [busy, setBusy] = useState(false);
  // Optimistic local echo of the toggle (the server refetch confirms it) so the
  // checkbox responds instantly instead of snapping back until status reloads.
  const [telemetry, setTelemetry] = useState<boolean | null>(null);

  const signIn = async () => {
    setBusy(true);
    await cloudLogin(); // sidecar opens the system browser; the tab poll picks up completion
    setTimeout(() => {
      setBusy(false);
      onChanged();
    }, 2500);
  };

  return (
    <div className={CARD + " p-3.5"} data-testid="cloud-account">
      <div className="flex items-center gap-3">
        <div className="min-w-0">
          <div className="font-semibold text-[14px]">OpenCoworker Cloud</div>
          {cloud?.signed_in ? (
            <div className="text-[12px] text-muted flex items-center gap-1.5">
              <span className="w-1.5 h-1.5 rounded-full bg-ok shrink-0" />
              {cloud.account || "signed in"}
            </div>
          ) : (
            <div className="text-[12px] text-muted">
              Sign in for one-click connectors. Manual token setup always works without it.
            </div>
          )}
        </div>
        <div className="ml-auto">
          {cloud?.signed_in ? (
            <button
              className={BTN_BORDERED}
              onClick={async () => {
                await cloudLogout();
                onChanged();
              }}
            >
              Sign out
            </button>
          ) : (
            <button className={BTN_ACCENT} onClick={signIn} disabled={busy}>
              {busy ? "Check your browser…" : "Sign in"}
            </button>
          )}
        </div>
      </div>
      {cloud?.signed_in && (
        <label className="flex items-start gap-2.5 mt-3 pt-3 border-t border-line select-none">
          <input
            type="checkbox"
            className="mt-0.5"
            checked={telemetry ?? cloud.telemetry_enabled !== false}
            data-testid="telemetry-toggle"
            onChange={async (e) => {
              setTelemetry(e.target.checked);
              await setCloudTelemetry(e.target.checked);
              onChanged();
            }}
          />
          <span>
            <span className="block text-[12.5px] text-ink">Help improve OpenCoworker</span>
            <span className="block text-[12px] text-muted">
              Tells us which coworker type was started and when — never your prompts, files,
              or connector data. Off means nothing is sent.
            </span>
          </span>
        </label>
      )}
    </div>
  );
}

function ConnectorRow({
  c,
  cloud,
  open,
  onToggleOpen,
  onChanged,
  onRefresh,
}: {
  c: Connector;
  cloud: CloudStatus | null;
  open: boolean;
  onToggleOpen: () => void;
  onChanged: () => void;
  onRefresh: () => void;
}) {
  return (
    <div className={CARD} data-testid={`connector-${c.name}`}>
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
      {open && !c.connected && <ConnectSetup c={c} cloud={cloud} onConnected={onChanged} />}
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

// Exported: also hosted inside the SourcesDrawer's connect-in-context child panel, so a
// recommended connector can be connected without leaving the session (owner ask, 2026-07-03).
export function ConnectSetup({
  c,
  cloud,
  onConnected,
}: {
  c: Connector;
  cloud: CloudStatus | null;
  onConnected: () => void;
}) {
  const [values, setValues] = useState<Record<string, string>>({});
  const [busy, setBusy] = useState(false);
  const [waiting, setWaiting] = useState(false); // managed flow: browser is open
  const [error, setError] = useState<string | null>(null);

  const submit = async () => {
    setBusy(true);
    setError(null);
    const res = await connectConnector(c.name, values);
    setBusy(false);
    if (res.ok) onConnected();
    else setError(res.error || "could not connect");
  };

  const oneClick = async () => {
    setError(null);
    const res = await connectManaged(c.name);
    // Completion arrives via the tab's poll: the broker form-POSTs the profile
    // to the sidecar, the connector flips to connected, this card closes itself.
    if (res.ok) setWaiting(true);
    else setError(res.error || "could not start managed connect");
  };

  return (
    <div className="border-t border-line px-3.5 py-3 space-y-3">
      {c.managed && (
        <div className="space-y-2" data-testid="managed-connect">
          {cloud?.signed_in ? (
            <button className={BTN_ACCENT} onClick={oneClick} disabled={waiting}>
              {waiting ? "Check your browser…" : `Connect ${c.title} with one click`}
            </button>
          ) : (
            <div className="text-[12.5px] text-muted">
              Sign in to OpenCoworker Cloud (above) to connect {c.title} with one click —
              or connect manually below.
            </div>
          )}
          {cloud?.signed_in && (
            <div className="text-[11.5px] text-faint">or connect manually:</div>
          )}
        </div>
      )}
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
