import type { SessionInfo, WsEvent } from "./types";

// Endpoint resolution order: runtime-injected globals (Tauri sets `window.__COWORKER_HTTP__`
// for its dynamically-chosen sidecar port) → Vite env → the 127.0.0.1:8765 dev default. This
// keeps a single codebase: browser `npm run dev` hits 8765; the desktop shell hits its sidecar.
const httpBase = (): string =>
  (globalThis as any).__COWORKER_HTTP__ ||
  (import.meta as any).env?.VITE_COWORKER_HTTP ||
  "http://127.0.0.1:8765";
const wsBase = (): string =>
  (globalThis as any).__COWORKER_WS__ ||
  (import.meta as any).env?.VITE_COWORKER_WS ||
  "ws://127.0.0.1:8765";

export interface Health {
  status: string;
  default_workspace: string | null;
  model: string;
}

export interface RecentWorkspace {
  path: string;
  name: string;
  exists: boolean;
}

export async function getHealth(): Promise<Health> {
  const res = await fetch(`${httpBase()}/v1/health`);
  return res.json();
}

export async function getRecentWorkspaces(): Promise<RecentWorkspace[]> {
  const res = await fetch(`${httpBase()}/v1/workspaces/recent`);
  return (await res.json()).workspaces ?? [];
}

export async function openWorkspace(
  path: string,
  create = false,
): Promise<{ path: string; ok: boolean; error?: string; git_branch?: string | null }> {
  const res = await fetch(`${httpBase()}/v1/workspaces/open`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path, create }),
  });
  return res.json();
}

export async function getSessions(workspace?: string): Promise<SessionInfo[]> {
  const q = workspace ? `?workspace=${encodeURIComponent(workspace)}` : "";
  const res = await fetch(`${httpBase()}/v1/sessions${q}`);
  return (await res.json()).sessions ?? [];
}

// A structured connector-delivered inbound message (§3.1). Attached to the user message it framed,
// for display only — the model still sees the framed `content`; this drives the ConnectorMessageCard.
export interface MessageSource {
  connector: string; // platform id, e.g. "slack"
  kind: "channel" | "dm";
  channel_id: string; // e.g. "C0BD7KZ1AH5"
  channel_name: string; // resolved; may equal the id (e.g. "#ocw-test")
  sender_id: string;
  sender_name: string; // resolved; may equal the id
  ts: number; // epoch seconds
  text: string; // the RAW message (what the card shows)
}

// A transcript message from GET /v1/sessions/{id}/messages. Kept permissive (open shape) because
// itemsFromMessages reads several role-specific fields; `source` is the optional connector sidecar.
export interface ConversationMessage {
  role: string;
  content?: any;
  tool_calls?: any[];
  tool_call_id?: string;
  source?: MessageSource;
  [key: string]: any;
}

export async function getSessionMessages(sessionId: string): Promise<ConversationMessage[]> {
  const res = await fetch(`${httpBase()}/v1/sessions/${sessionId}/messages`);
  return (await res.json()).messages ?? [];
}

export async function renameSession(sessionId: string, title: string): Promise<{ ok: boolean; error?: string }> {
  const res = await fetch(`${httpBase()}/v1/sessions/${encodeURIComponent(sessionId)}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title }),
  });
  return res.json();
}

export async function setSessionFlags(
  sessionId: string,
  flags: { pinned?: boolean; archived?: boolean },
): Promise<{ ok: boolean; error?: string }> {
  const res = await fetch(`${httpBase()}/v1/sessions/${encodeURIComponent(sessionId)}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(flags),
  });
  return res.json();
}

export async function deleteSession(sessionId: string): Promise<{ ok: boolean; error?: string }> {
  const res = await fetch(`${httpBase()}/v1/sessions/${encodeURIComponent(sessionId)}`, { method: "DELETE" });
  return res.json();
}

export interface ArtifactInfo {
  path: string;
  name: string;
  kind: "markdown" | "html" | "image" | "code" | "text" | string;
  size: number;
  modified_at: number;
}

export interface ArtifactContent {
  ok: boolean;
  error?: string;
  path: string;
  kind: string;
  content?: string;
  data_url?: string;
  truncated?: boolean;
}

export async function getArtifacts(sessionId: string): Promise<ArtifactInfo[]> {
  const res = await fetch(`${httpBase()}/v1/sessions/${encodeURIComponent(sessionId)}/artifacts`);
  return (await res.json()).artifacts ?? [];
}

export async function readArtifact(sessionId: string, path: string): Promise<ArtifactContent> {
  const q = new URLSearchParams({ path });
  const res = await fetch(`${httpBase()}/v1/sessions/${encodeURIComponent(sessionId)}/artifacts/read?${q.toString()}`);
  return res.json();
}

/** Show the artifact in the OS file manager ("reveal") or open it with its default app ("open"). */
export async function revealArtifact(
  sessionId: string,
  path: string,
  mode: "reveal" | "open" = "reveal",
): Promise<{ ok: boolean; error?: string }> {
  const res = await fetch(`${httpBase()}/v1/sessions/${encodeURIComponent(sessionId)}/artifacts/reveal`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path, mode }),
  });
  return res.json();
}

// -- session roots (orphan Cowork: scratch + added folders) -------------------
export interface RootInfo {
  path: string;
  writable: boolean;
  label: string;
  primary: boolean;
  exists: boolean;
}

export async function getRoots(sessionId: string): Promise<RootInfo[]> {
  const res = await fetch(`${httpBase()}/v1/sessions/${encodeURIComponent(sessionId)}/roots`);
  return (await res.json()).roots ?? [];
}

export async function addRoot(
  sessionId: string,
  path: string,
  writable: boolean,
): Promise<{ ok: boolean; error?: string; roots?: RootInfo[] }> {
  const res = await fetch(`${httpBase()}/v1/sessions/${encodeURIComponent(sessionId)}/roots`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path, writable }),
  });
  return res.json();
}

export async function removeRoot(
  sessionId: string,
  path: string,
): Promise<{ ok: boolean; error?: string; roots?: RootInfo[] }> {
  const q = new URLSearchParams({ path });
  const res = await fetch(
    `${httpBase()}/v1/sessions/${encodeURIComponent(sessionId)}/roots?${q.toString()}`,
    { method: "DELETE" },
  );
  return res.json();
}

// -- MCP servers --------------------------------------------------------------
export interface McpServer {
  name: string;
  enabled: boolean;
  transport: string;
  requires_approval: boolean;
  status: string;
  tool_count: number | null;
  config: Record<string, any>;
}

export async function getMcpServers(): Promise<McpServer[]> {
  const res = await fetch(`${httpBase()}/v1/mcp`);
  return (await res.json()).servers ?? [];
}

export async function addMcpServer(name: string, config: Record<string, any>) {
  const res = await fetch(`${httpBase()}/v1/mcp`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, config }),
  });
  return res.json();
}

export async function patchMcpServer(name: string, changes: Record<string, any>) {
  const res = await fetch(`${httpBase()}/v1/mcp/${encodeURIComponent(name)}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(changes),
  });
  return res.json();
}

export async function deleteMcpServer(name: string) {
  const res = await fetch(`${httpBase()}/v1/mcp/${encodeURIComponent(name)}`, { method: "DELETE" });
  return res.json();
}

export async function getMcpTools(
  name: string,
): Promise<{ ok: boolean; error?: string; tools: { name: string; description: string }[] }> {
  const res = await fetch(`${httpBase()}/v1/mcp/${encodeURIComponent(name)}/tools`);
  return res.json();
}

export async function reloadMcp() {
  const res = await fetch(`${httpBase()}/v1/mcp/reload`, { method: "POST" });
  return res.json();
}

// -- connectors ---------------------------------------------------------------
export interface ConnectorField {
  key: string;
  label: string;
  secret: boolean;
  required: boolean;
  help: string;
  placeholder: string;
}

export interface Connector {
  name: string;
  title: string;
  icon: string;
  blurb: string;
  auth: string;
  two_way: boolean;
  available: boolean;
  fields: ConnectorField[];
  instructions: string[];
  connected: boolean;
  account: string | null;
  enabled: boolean;
  brand_color: string; // hex brand color, e.g. "#611f69" (fallback gray "#6b7280")
  logo: string; // stable logo id keyed into the frontend registry (empty → fallback glyph)
  allowed_users: string[]; // the allow-list (managed inline in the Connectors tab)
  recent?: RecentSender[]; // recently-seen senders on a connected two-way connector
  tools: ConnectorTool[];
}

export interface ConnectorTool {
  name: string;
  label: string;
  kind: "read" | "write" | string;
  description: string;
  enabled: boolean;
  requires_approval: boolean;
}

export async function getConnectors(): Promise<Connector[]> {
  const res = await fetch(`${httpBase()}/v1/connectors`);
  return (await res.json()).connectors ?? [];
}

export async function connectConnector(
  name: string,
  fields: Record<string, string>,
): Promise<{ ok: boolean; account?: string; error?: string }> {
  const res = await fetch(`${httpBase()}/v1/connectors/${encodeURIComponent(name)}/connect`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ fields }),
  });
  return res.json();
}

export async function disconnectConnector(name: string): Promise<{ ok: boolean }> {
  const res = await fetch(`${httpBase()}/v1/connectors/${encodeURIComponent(name)}/disconnect`, {
    method: "POST",
  });
  return res.json();
}

export async function updateConnectorTools(
  name: string,
  enabled: Record<string, boolean>,
): Promise<{ ok: boolean; error?: string; tools?: Record<string, boolean> }> {
  const res = await fetch(`${httpBase()}/v1/connectors/${encodeURIComponent(name)}/tools`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ enabled }),
  });
  return res.json();
}

export interface AuditEvent {
  id: number;
  timestamp: string;
  session_id: string;
  agent: string;
  workspace: string;
  connector: string;
  tool: string;
  stage: string;
  status: string;
  approval: string;
  args: Record<string, any>;
  result_preview: string;
  reason: string;
  resource: string;
}

export async function getAudit(params: {
  limit?: number;
  session_id?: string;
  connector?: string;
  tool?: string;
} = {}): Promise<AuditEvent[]> {
  const q = new URLSearchParams();
  if (params.limit) q.set("limit", String(params.limit));
  if (params.session_id) q.set("session_id", params.session_id);
  if (params.connector) q.set("connector", params.connector);
  if (params.tool) q.set("tool", params.tool);
  const res = await fetch(`${httpBase()}/v1/audit${q.toString() ? "?" + q.toString() : ""}`);
  return (await res.json()).events ?? [];
}

export interface BrowserState {
  open: boolean;
  url: string;
  title: string;
  status: string;
  last_action: string;
  last_result: string;
  last_error: string;
  screenshot_data_url: string;
  updated_at: string | null;
  controls: any[];
}

export async function getBrowserState(): Promise<BrowserState> {
  const res = await fetch(`${httpBase()}/v1/browser/state`);
  return res.json();
}

export async function takeBrowserScreenshot(): Promise<BrowserState & { ok?: boolean; error?: string }> {
  const res = await fetch(`${httpBase()}/v1/browser/screenshot`, { method: "POST" });
  return res.json();
}

export async function closeBrowser(): Promise<{ ok?: boolean; error?: string }> {
  const res = await fetch(`${httpBase()}/v1/browser/close`, { method: "POST" });
  return res.json();
}

// -- settings (model API key, default model, onboarding) ----------------------
export interface SurfaceVisibility {
  cowork: boolean; // always true
  chat: boolean;
  code: boolean;
}

export interface ModelSettings {
  provider: string;
  model: string;
  models: string[];
  has_key: boolean;
  model_ready: boolean; // can the default model's provider actually run (any provider)?
  source: "env" | "store" | null;
  onboarded: boolean;
  surfaces: SurfaceVisibility;
  scratch_base: string;
  secrets_path: string;  // OS-native on-disk location the server reports (not hardcoded)
}

export async function setScratchBase(
  path: string,
): Promise<{ ok: boolean; error?: string; scratch_base?: string }> {
  const res = await fetch(`${httpBase()}/v1/settings/scratch-base`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path }),
  });
  return res.json();
}

export async function setSurfaces(
  flags: { chat?: boolean; code?: boolean },
): Promise<{ ok: boolean; surfaces: SurfaceVisibility }> {
  const res = await fetch(`${httpBase()}/v1/settings/surfaces`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(flags),
  });
  return res.json();
}

// -- Personas -----------------------------------------------------------------
export interface Persona {
  id: string;
  name: string;
  icon: string;
  tagline: string;
  needs_workspace: boolean;
  builtin: boolean;
  family: string;
  tools: string[];
  enabled: boolean;
  surfaced: boolean;
  default: boolean;
}

export interface PersonaConsent {
  id: string;
  name: string;
  description: string;
  tools: string[];
  risk: string[];
  connectors: boolean;
  mcp: string[];
  messaging: boolean;
  recommended_mode: string;
  recommended_models: string[];
  source: string | null;
  builtin: boolean;
}

export async function getPersonas(): Promise<Persona[]> {
  const res = await fetch(`${httpBase()}/v1/personas`);
  return (await res.json()).personas;
}

export async function updatePersona(
  id: string,
  body: { enabled?: boolean; surfaced?: boolean; default?: boolean },
): Promise<{ ok: boolean; personas?: Persona[]; error?: string }> {
  const res = await fetch(`${httpBase()}/v1/personas/${encodeURIComponent(id)}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return res.json();
}

export async function installPersona(
  body: { dir?: string; git_url?: string },
): Promise<{ ok: boolean; consent?: PersonaConsent[]; personas?: Persona[]; error?: string }> {
  const res = await fetch(`${httpBase()}/v1/personas/install`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return res.json();
}

// -- Persona detail + connection defaults (§5) --------------------------------
// A persona's declared recommendation (manifest `recommends`): a connector or MCP server it works
// best with, with a reason + tier (core/optional). `connected` is annotated server-side from the
// connector list so the detail page can show connect state without a second round-trip.
export interface PersonaRecommendation {
  kind: string; // "connector" | "mcp" | …
  ref: string; // connector id (e.g. "github") or mcp/server name
  reason: string;
  tier: string; // "core" | "optional"
  connected: boolean;
}

// A persona-default connection (the middle of the §4 hierarchy): for a connected connector, whether
// new sessions of this persona get it enabled by default.
export interface PersonaDefaultConnection {
  connector: string; // connector id
  enabled: boolean; // persona-default on/off
  connected: boolean; // is the account actually connected (else the toggle is disabled)
}

export interface PersonaDetail {
  id: string;
  name: string;
  icon: string;
  tagline: string;
  description: string;
  enabled: boolean; // persona on/off (shown in the picker)
  tools: string[];
  recommended_models: string[];
  default_permission_mode: string;
  workspace: string;
  recommends: PersonaRecommendation[];
  default_connections: PersonaDefaultConnection[];
}

export async function getPersonaDetail(id: string): Promise<PersonaDetail> {
  const res = await fetch(`${httpBase()}/v1/personas/${encodeURIComponent(id)}`);
  return res.json();
}

/** Set a persona-default connection (new sessions of this persona get it on/off by default). */
export async function setPersonaConnection(
  id: string,
  connector: string,
  enabled: boolean,
): Promise<{ ok: boolean; default_connections?: PersonaDefaultConnection[]; error?: string }> {
  const res = await fetch(`${httpBase()}/v1/personas/${encodeURIComponent(id)}/connections`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ connector, enabled }),
  });
  return res.json();
}

/** Enable/disable the persona (whether it surfaces in the new-session picker). */
export async function setPersonaEnabled(
  id: string,
  enabled: boolean,
): Promise<{ ok: boolean; personas?: Persona[]; error?: string }> {
  const res = await fetch(`${httpBase()}/v1/personas/${encodeURIComponent(id)}/enable`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ enabled }),
  });
  return res.json();
}

// -- Per-session connections (Sources bar + drawer, §6) -----------------------
// An effective-enabled connector for a session, with a short human detail (e.g. "#ocw-test · DMs").
// `enabled` reflects the session override/persona default so the drawer toggle shows correct state.
export interface SessionConnectedConnector {
  connector: string;
  enabled: boolean;
  detail: string;
}

// A persona-recommended connector not yet connected (drives the `⚠ N` attention count).
export interface SessionRecommendedConnector {
  connector: string;
  reason: string;
  tier: string;
  connected: boolean;
}

export interface SessionConnections {
  connected: SessionConnectedConnector[];
  recommended: SessionRecommendedConnector[];
  attention: number; // ⚠ count = recommended connectors not yet connected
}

export async function getSessionConnections(sessionId: string): Promise<SessionConnections> {
  const res = await fetch(`${httpBase()}/v1/sessions/${encodeURIComponent(sessionId)}/connections`);
  return res.json();
}

/**
 * Set a per-session connection override (mute/unmute a connector for THIS session). Pass
 * `clear: true` to drop the override and inherit the persona default again.
 */
export async function setSessionConnection(
  sessionId: string,
  connector: string,
  enabled: boolean,
  clear = false,
): Promise<{ ok: boolean; error?: string }> {
  const res = await fetch(`${httpBase()}/v1/sessions/${encodeURIComponent(sessionId)}/connections`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ connector, enabled, ...(clear ? { clear: true } : {}) }),
  });
  return res.json();
}

// -- Inbox + Unattended -------------------------------------------------------
export interface InboxItem {
  id: string;
  session_id: string;
  kind: "approval" | "question" | "notification" | "directory" | "plan";
  title: string;
  body: string;
  state: "pending" | "resolved";
  resolution: string | null;
  inbox: string;
  created_at: string;
  resolved_at: string | null;
  visibility?: "inline" | "inbox";
  // Question metadata (ask_user): quick-reply choices + a free-text escape.
  options?: string[];
  allow_text?: boolean;
  multi?: boolean;
  // Kind-specific payload (directory: {path, writable}; …).
  data?: Record<string, any>;
  // Originating-session context (server-joined) so the Inbox is self-contained.
  session_title?: string;
  session_agent?: string | null;
  session_workspace?: string | null;
  session_exists?: boolean;
}

export async function getInbox(sessionId?: string, state?: string): Promise<InboxItem[]> {
  const q = new URLSearchParams();
  if (sessionId) q.set("session_id", sessionId);
  if (state) q.set("state", state);
  const res = await fetch(`${httpBase()}/v1/inbox?${q.toString()}`);
  return (await res.json()).items;
}

export async function resolveInboxItem(
  id: string,
  resolution: string,
): Promise<{ ok: boolean }> {
  const res = await fetch(`${httpBase()}/v1/inbox/${encodeURIComponent(id)}/resolve`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ resolution }),
  });
  return res.json();
}

// -- channel subscriptions (view-only) ----------------------------------------
export interface Subscription {
  session_id: string;
  session_title: string;
  agent: string;
  channel: string;
  routing_target: string | null;
  collision: boolean; // inbound subscription == outbound Inbox routing on the same channel
}

export interface RecentChannel {
  channel: string;
  last_from: string | null;
  last_text: string | null;
}

export async function getSubscriptions(): Promise<Subscription[]> {
  const res = await fetch(`${httpBase()}/v1/subscriptions`);
  return (await res.json()).subscriptions ?? [];
}

// -- inbox routing (where Unattended approvals/questions get mirrored) ---------
export interface InboxBinding {
  name: string;
  channel: string | null; // platform, e.g. "slack" (null = in-app Inbox only)
  target: string; // chat_id, e.g. "C0BEJNCQQ8Y"
}

export async function getInboxRouting(): Promise<InboxBinding[]> {
  const res = await fetch(`${httpBase()}/v1/inbox/routing`);
  return (await res.json()).bindings ?? [];
}

export async function setInboxBinding(
  name: string,
  channel: string | null,
  target: string,
): Promise<{ ok: boolean; bindings?: InboxBinding[]; error?: string }> {
  const res = await fetch(`${httpBase()}/v1/inbox/routing/binding`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, channel, target }),
  });
  return res.json();
}

export interface UnroutedItem {
  source: string;
  sender: string;
  text: string;
  reason: string;
  ts: number;
}

export async function getUnrouted(): Promise<UnroutedItem[]> {
  const res = await fetch(`${httpBase()}/v1/unrouted`);
  return (await res.json()).items ?? [];
}

export async function getRecentChannels(): Promise<RecentChannel[]> {
  const res = await fetch(`${httpBase()}/v1/channels/recent`);
  return (await res.json()).channels ?? [];
}

export async function subscribeChannel(
  sessionId: string,
  channel: string,
): Promise<{ ok: boolean; channel?: string; error?: string }> {
  const res = await fetch(`${httpBase()}/v1/subscriptions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, channel }),
  });
  return res.json();
}

export async function unsubscribeChannel(
  sessionId: string,
  channel: string,
): Promise<{ ok: boolean; removed?: boolean }> {
  const res = await fetch(`${httpBase()}/v1/subscriptions/remove`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, channel }),
  });
  return res.json();
}

export async function getUnattended(sessionId: string): Promise<boolean> {
  const res = await fetch(
    `${httpBase()}/v1/sessions/${encodeURIComponent(sessionId)}/unattended`,
  );
  return (await res.json()).unattended;
}

export async function setUnattended(
  sessionId: string,
  unattended: boolean,
): Promise<{ ok: boolean; unattended: boolean }> {
  const res = await fetch(
    `${httpBase()}/v1/sessions/${encodeURIComponent(sessionId)}/unattended`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ unattended }),
    },
  );
  return res.json();
}

export async function getSettings(): Promise<ModelSettings> {
  const res = await fetch(`${httpBase()}/v1/settings`);
  return res.json();
}

export async function setModelKey(
  apiKey: string,
): Promise<{ ok: boolean; error?: string; has_key?: boolean; source?: string }> {
  const res = await fetch(`${httpBase()}/v1/settings/model-key`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ api_key: apiKey }),
  });
  return res.json();
}

export async function setDefaultModel(
  model: string,
): Promise<{ ok: boolean; error?: string; model?: string }> {
  const res = await fetch(`${httpBase()}/v1/settings/default-model`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model }),
  });
  return res.json();
}

export async function addModel(model: string): Promise<ModelSettings & { ok: boolean; error?: string }> {
  const res = await fetch(`${httpBase()}/v1/settings/models/add`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model }),
  });
  return res.json();
}

export async function removeModel(model: string): Promise<ModelSettings & { ok: boolean }> {
  const res = await fetch(`${httpBase()}/v1/settings/models/remove`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model }),
  });
  return res.json();
}

export async function setOnboarded(value: boolean): Promise<{ ok: boolean; onboarded: boolean }> {
  const res = await fetch(`${httpBase()}/v1/settings/onboarded`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ value }),
  });
  return res.json();
}

// -- model providers (OpenAI, Ollama, …) --------------------------------------
export interface ProviderField {
  key: string;
  label: string;
  secret: boolean;
  required: boolean;
  help: string;
  placeholder: string;
}

export interface ProviderInfo {
  name: string;
  title: string;
  needs_key: boolean;
  fields: ProviderField[];
  configured: boolean;
  values: Record<string, string>; // non-secret stored values (e.g. base_url), for prefilling
  suggested_models: string[]; // bare model-name suggestions for the "add model" datalist
  recommended_model: string | null; // pre-filled default for this provider (e.g. qwen3-coder:30b)
}

export async function getProviders(): Promise<ProviderInfo[]> {
  const res = await fetch(`${httpBase()}/v1/providers`);
  return res.json();
}

export async function setProvider(
  name: string,
  fields: Record<string, string>,
): Promise<{ ok: boolean; error?: string; provider?: string; recommended_model?: string | null }> {
  const res = await fetch(`${httpBase()}/v1/providers`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, fields }),
  });
  return res.json();
}

/** Live read-only credential check (does NOT save the key). Triggered by the user's "Test" click. */
export async function verifyProvider(
  name: string,
  fields: Record<string, string>,
): Promise<{ ok: boolean; error?: string }> {
  const res = await fetch(`${httpBase()}/v1/providers/verify`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, fields }),
  });
  return res.json();
}

/** Client-side provider guess from an API key's shape (mirrors the server's detect_provider). */
export function detectProvider(apiKey: string): string | null {
  const key = (apiKey || "").trim();
  if (!key) return null;
  if (key.startsWith("sk-ant-")) return "anthropic";
  if (key.startsWith("AIza")) return "gemini";
  if (key.startsWith("sk-") || key.startsWith("sk_")) return "openai";
  return null;
}

// -- super-agent --------------------------------------------------------------
export interface RecentSender {
  user_id: string;
  user_name: string | null;
  chat_id: string;
  chat_type: string;
  target: string;
  authorized: boolean;
}

// -- direct-message routing ---------------------------------------------------
export async function getDmRoute(): Promise<string | null> {
  const res = await fetch(`${httpBase()}/v1/messaging/dm-route`);
  return (await res.json()).dm_session ?? null;
}

export async function setDmRoute(sessionId: string): Promise<{ ok: boolean; dm_session: string | null }> {
  const res = await fetch(`${httpBase()}/v1/messaging/dm-route`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId }),
  });
  return res.json();
}

// -- automations (scheduled tasks) --------------------------------------------
export interface Automation {
  id: string;
  title: string;
  instructions: string;
  schedule: string;
  schedule_raw?: { kind: string; cron?: string | null; fire_at?: string | null; timezone?: string };
  workspace: string;
  agent: string;
  enabled: boolean;
  next_run: number | null;
  last_run: number | null;
  last_status: string | null;
  run_count: number;
  notify_on_completion: boolean;
  always_allowed: string[];
}

export interface AutomationRun {
  run_id: string;
  task_id: string;
  session_id: string;
  started_at: number;
  finished_at: number | null;
  status: string;
  result_text: string | null;
  artifacts: string[];
  error: string | null;
  trigger: string;
}

export async function getAutomations(): Promise<Automation[]> {
  const res = await fetch(`${httpBase()}/v1/automations`);
  return (await res.json()).tasks ?? [];
}

export async function createAutomation(payload: {
  title: string;
  instructions: string;
  cron?: string;
  fire_at?: string;
  timezone?: string;
}): Promise<{ ok: boolean; error?: string; task?: Automation }> {
  const res = await fetch(`${httpBase()}/v1/automations`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return res.json();
}

export async function getAutomation(id: string): Promise<{ task: Automation; runs: AutomationRun[] }> {
  const res = await fetch(`${httpBase()}/v1/automations/${encodeURIComponent(id)}`);
  return res.json();
}

export async function updateAutomation(id: string, changes: Record<string, any>) {
  const res = await fetch(`${httpBase()}/v1/automations/${encodeURIComponent(id)}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(changes),
  });
  return res.json();
}

export async function deleteAutomation(id: string) {
  const res = await fetch(`${httpBase()}/v1/automations/${encodeURIComponent(id)}`, { method: "DELETE" });
  return res.json();
}

export interface PreparedRun {
  ok: boolean;
  error?: string;
  run_id: string;
  session_id: string;
  workspace: string;
  agent: string;
  prompt: string;
}

/** Prepare a live manual run: returns the session to open + the opening prompt to send. */
export async function runAutomation(id: string): Promise<PreparedRun> {
  const res = await fetch(`${httpBase()}/v1/automations/${encodeURIComponent(id)}/run`, { method: "POST" });
  return res.json();
}

/** Mark a manual run complete after its first turn finished. */
export async function finalizeAutomationRun(id: string, runId: string) {
  const res = await fetch(
    `${httpBase()}/v1/automations/${encodeURIComponent(id)}/runs/${encodeURIComponent(runId)}/finalize`,
    { method: "POST" },
  );
  return res.json();
}

export async function allowUser(name: string, userId: string) {
  const res = await fetch(`${httpBase()}/v1/connectors/${encodeURIComponent(name)}/allow`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ user_id: userId }),
  });
  return res.json();
}

export async function disallowUser(name: string, userId: string) {
  const res = await fetch(`${httpBase()}/v1/connectors/${encodeURIComponent(name)}/disallow`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ user_id: userId }),
  });
  return res.json();
}

export type Handlers = {
  onEvent: (event: WsEvent) => void;
  onOpen?: () => void;
  onClose?: () => void;
};

export class Session {
  private ws: WebSocket;
  // Payloads sent before the socket finished opening, replayed on `onopen`. Belt-and-suspenders
  // against the first message being dropped if the user sends in the connect window.
  private outbox: object[] = [];

  constructor(sessionId: string, workspace: string, agent: string, handlers: Handlers) {
    const q = `?workspace=${encodeURIComponent(workspace)}&agent=${encodeURIComponent(agent)}`;
    this.ws = new WebSocket(`${wsBase()}/ws/session/${sessionId}${q}`);
    this.ws.onmessage = (e) => handlers.onEvent(JSON.parse(e.data));
    this.ws.onopen = () => {
      this.flush();
      handlers.onOpen?.();
    };
    this.ws.onclose = () => handlers.onClose?.();
  }

  private flush() {
    if (this.ws.readyState !== WebSocket.OPEN) return;
    const pending = this.outbox;
    this.outbox = [];
    for (const p of pending) this.ws.send(JSON.stringify(p));
  }

  private send(payload: object) {
    if (this.ws.readyState === WebSocket.OPEN) this.ws.send(JSON.stringify(payload));
    // Still connecting: queue and flush on open rather than silently dropping.
    else if (this.ws.readyState === WebSocket.CONNECTING) this.outbox.push(payload);
  }

  userMessage(text: string, attachments?: unknown[]) {
    this.send({ type: "user_message", text, ...(attachments?.length ? { attachments } : {}) });
  }

  approve(decision: string) {
    this.send({ type: "approval", decision });
  }

  // Reply to a `request_directory` prompt: grant a folder (with access level) or decline.
  respondDirectory(granted: boolean, path?: string, writable?: boolean) {
    this.send({ type: "directory_response", granted, ...(path ? { path } : {}), writable: !!writable });
  }

  // Reply to a `propose_plan` prompt: approve (choosing the execution mode) or reject with feedback.
  respondPlan(approved: boolean, mode?: string, feedback?: string) {
    this.send({
      type: "plan_response",
      approved,
      ...(mode ? { mode } : {}),
      ...(feedback ? { feedback } : {}),
    });
  }

  // Answer a live `ask_user` prompt (attended sessions; unattended ones answer via the Inbox).
  respondQuestion(answer: string) {
    this.send({ type: "question_response", answer });
  }

  interrupt() {
    this.send({ type: "interrupt" });
  }

  setMode(mode: string) {
    this.send({ type: "set_mode", mode });
  }

  setModel(model: string) {
    this.send({ type: "set_model", model });
  }

  close() {
    this.ws.close();
  }
}

