import { test as base, expect } from "@playwright/test";

// Hermetic API mock. Every /v1 request the GUI makes is fulfilled from the fixtures below (shapes
// mirrored from the real backend), and the event WebSocket is a SCRIPTED FAKE AGENT (ready on
// connect; user_message → turn_start/deltas/assistant_message/turn_done; "run a tool" triggers the
// approval flow), so specs run with no Python server and never touch real state. Mutations
// (sessions, personas, inbox, routing, channel subscriptions) are held in per-test in-memory state
// so add/remove/toggle reflect through the real UI on re-fetch.

const HEALTH = { status: "ok", default_workspace: null, model: "anthropic:claude-opus-4-8" };

const SETTINGS = {
  provider: "openai",
  model: "anthropic:claude-opus-4-8",
  models: ["anthropic:claude-opus-4-8", "gpt-5.5", "gpt-4o", "gpt-4o-mini", "o3-mini"],
  has_key: true,
  model_ready: true,
  source: "store",
  onboarded: true,
  experimental_connectors: false,
  surfaces: { cowork: true, chat: false, code: true },
  nav_layout: "grouped",
  scratch_base: "~/OpenCoworker",
  secrets_path: "/Users/test/.config/coworker/secrets.json",
  sessions_peek: 5,
  // Curated-matrix display names (subset — mirrors /v1/settings.model_labels).
  model_labels: {
    "anthropic:claude-opus-4-8": "Claude Opus 4.8 · Anthropic",
    "zai:glm-5.2": "GLM-5.2 · Z AI",
  },
};

const PERSONAS = {
  personas: [
    { id: "cowork", name: "OpenCoworker", icon: "cowork", tagline: "Produce a deliverable — research, analysis, scripts", needs_workspace: true, builtin: true, family: "knowledge", workspace: "deliverable", tools: ["files", "search"], enabled: true, surfaced: true, default: true },
    { id: "code", name: "Code", icon: "code", tagline: "Work in a codebase — files, git, shell", needs_workspace: true, builtin: true, family: "code", workspace: "git", tools: ["code_files", "git"], enabled: true, surfaced: true, default: false },
    { id: "chat", name: "Chat", icon: "chat", tagline: "Quick questions — no workspace", needs_workspace: false, builtin: true, family: "knowledge", workspace: "none", tools: [], enabled: true, surfaced: false, default: false },
    { id: "ops", name: "Ops Coworker", icon: "wrench", tagline: "Operate and investigate — runbooks, logs, infrastructure", needs_workspace: true, builtin: true, family: "knowledge", workspace: "deliverable", tools: ["files", "shell"], enabled: true, surfaced: true, default: false },
    // A non-builtin install (disabled pending consent — invisible to picker specs) so the
    // Personas page's delete/enable affordances have a target.
    { id: "acme-notes", name: "Acme Notes", icon: "pencil", tagline: "Acme's note-taking coworker", needs_workspace: true, builtin: false, family: "knowledge", workspace: "deliverable", tools: ["files"], enabled: false, surfaced: false, default: false },
  ],
};

// The boot-resume target (most recent updated_at) — existing specs open it by title.
const PINNED_SESSION = {
  session_id: "pinned-cowork-1",
  title: "Draft the launch note",
  workspace: "/Users/test/OpenCoworker/launch-note",
  agent: "cowork",
  model: "anthropic:claude-opus-4-8",
  mode: "interactive",
  updated_at: "2026-07-01 09:00:00",
  messages: 2,
  pinned: true,
  archived: false,
  attention: 0,
  liveness: "idle",
  subscriptions: [],
};

// Seven unpinned Coworker sessions: enough to exercise the sidebar peek cap (5) + "Show more (2)".
// wp-3 carries the pending Inbox approval below (attention badge parity). All OLDER than the
// pinned session so boot-resume stays deterministic.
const EXTRA_SESSIONS = Array.from({ length: 7 }, (_, i) => ({
  session_id: `wp-${i + 1}`,
  title: `Weekly plan ${i + 1}`,
  workspace: "",
  agent: "cowork",
  model: "anthropic:claude-opus-4-8",
  mode: "interactive",
  updated_at: `2026-06-2${8 - Math.min(i, 7)} 10:00:00`,
  messages: 3,
  pinned: false,
  archived: false,
  attention: i + 1 === 3 ? 1 : 0,
  liveness: "idle",
  subscriptions: [],
}));

// One Ops session (older than everything above so boot-resume stays deterministic) — the
// target for the disable-archives-conversations confirm flow on the Personas page.
const OPS_SESSION = {
  session_id: "ops-1",
  title: "Ops triage",
  workspace: "/Users/test/OpenCoworker/ops-triage",
  agent: "ops",
  model: "anthropic:claude-opus-4-8",
  mode: "interactive",
  updated_at: "2026-06-15 10:00:00",
  messages: 4,
  pinned: false,
  archived: false,
  attention: 0,
  liveness: "idle",
  subscriptions: [],
};

const CONNECTORS = {
  connectors: [
    { name: "browser", title: "Browser", icon: "B", blurb: "Headless browser.", auth: "none", two_way: false, available: true, brand_color: "#6b7280", logo: "", fields: [], instructions: [], connected: true, account: null, enabled: true, allowed_users: [], tools: [], managed: false, managed_profile: false },
    { name: "telegram", title: "Telegram", icon: "T", blurb: "Two-way Telegram messaging.", auth: "bot_token", two_way: true, available: true, brand_color: "#229ed9", logo: "telegram", fields: [{ key: "bot_token", label: "Bot token", secret: true, required: true, help: "", placeholder: "123456:ABC…" }], instructions: [], connected: false, account: null, enabled: false, allowed_users: [], tools: [], managed: false, managed_profile: false },
    // Managed-capable connector (one-click via cloud when signed in; manual paste otherwise).
    { name: "gmail", title: "Gmail", icon: "✉", blurb: "Search, summarize, draft, and send email.", auth: "oauth", two_way: false, available: true, brand_color: "#ea4335", logo: "gmail", fields: [{ key: "access_token", label: "OAuth access token", secret: true, required: true, help: "", placeholder: "" }], instructions: [], connected: false, account: null, enabled: false, allowed_users: [], tools: [], managed: true, managed_profile: false },
  ],
};

// Two pending items across two personas: drives the Inbox kind tabs, the persona filter chips
// (which only render with >1 persona), and resolve-removes-card. The question's session is NOT in
// the sessions list on purpose — the Inbox must be self-contained (server-joined context fields).
const INBOX_ITEMS = [
  {
    id: "inb-approval-1",
    session_id: "wp-3",
    kind: "approval",
    title: "Approve: run_shell",
    body: "rm -rf build/",
    state: "pending",
    resolution: null,
    inbox: "default",
    created_at: "2026-07-01 08:00:00",
    resolved_at: null,
    session_title: "Weekly plan 3",
    session_agent: "cowork",
    session_workspace: "",
    session_exists: true,
  },
  {
    id: "inb-question-1",
    session_id: "ops-1",
    kind: "question",
    title: "Which environment should I restart?",
    body: "",
    options: ["staging", "production"],
    allow_text: true,
    multi: false,
    state: "pending",
    resolution: null,
    inbox: "default",
    created_at: "2026-07-01 08:05:00",
    resolved_at: null,
    session_title: "Investigate alerts",
    session_agent: "ops",
    session_workspace: "",
    session_exists: true,
  },
];

// Mutable cloud sign-in state: POST /v1/cloud/login flips it (the real flow
// goes through the browser; the mock completes instantly), logout flips back.
export const CLOUD_STATE = {
  signed_in: false,
  account: "",
  user_id: "",
  telemetry_enabled: true,
};

const GALLERY_PERSONAS = [
  {
    slug: "sales",
    version: 1,
    name: "Sales Coworker",
    icon: "chart",
    tagline: "Research accounts, prep meetings, draft follow-ups",
    description: "A sales-focused coworker.",
    family: "knowledge",
    workspace: "deliverable",
    publisher: "OpenCoworker",
    recommended_connectors: ["hubspot", "gmail"],
    risk_summary: "Declarative manifest; no executable code.",
    featured: true,
  },
  {
    slug: "recruiter",
    version: 1,
    name: "Recruiter",
    icon: "search",
    tagline: "Sourcing summaries and scheduling loops",
    description: "A recruiting coworker.",
    family: "knowledge",
    workspace: "deliverable",
    publisher: "OpenCoworker",
    recommended_connectors: ["gmail"],
    risk_summary: "Declarative manifest; no executable code.",
    featured: false,
  },
];

// Persona detail (GET /v1/personas/:id) — SourcesDrawer/PersonaView read `recommends` and
// `default_connections` as arrays, so these must be present (not the catch-all {}).
const PERSONA_DETAIL = {
  id: "cowork",
  name: "OpenCoworker",
  icon: "cowork",
  tagline: "Produce a deliverable — research, analysis, scripts",
  description: "",
  enabled: true,
  tools: ["files", "search"],
  recommended_models: ["anthropic:claude-opus-4-8"],
  default_permission_mode: "interactive",
  workspace: "deliverable",
  recommends: [],
  default_connections: [],
};

const CONNECTIONS = {
  connected: [
    { connector: "browser", enabled: true, detail: "Browser" },
    { connector: "slack", enabled: true, detail: "Slack" },
  ],
  recommended: [
    { connector: "github", reason: "confirm deploys and inspect PRs", tier: "core", connected: false },
  ],
  attention: 1,
};

// One scheduled automation with a running run: its session id uses the real `__run__` convention
// so the session view's run banner (detection is id-based) can be exercised end-to-end.
const AUTOMATION = {
  id: "task-1",
  title: "Daily AI News",
  instructions: "Fetch the latest AI news and produce an HTML+Tailwind presentation.",
  schedule: "Every day at ~5:40 PM",
  schedule_raw: { kind: "cron", cron: "40 17 * * *", fire_at: null, timezone: "local" },
  workspace: "",
  agent: "cowork",
  enabled: true,
  next_run: Math.floor(Date.now() / 1000) + 3600,
  last_run: Math.floor(Date.now() / 1000) - 60,
  last_status: "running",
  run_count: 1,
  notify_on_completion: false,
  always_allowed: [],
};
const AUTOMATION_RUNS = [
  {
    run_id: "r1",
    task_id: "task-1",
    session_id: "__run__r1",
    started_at: Math.floor(Date.now() / 1000) - 60,
    finished_at: null,
    status: "running",
    result_text: null,
    artifacts: [],
    error: null,
    trigger: "schedule",
  },
];

const PRIMARY_ROOT = { path: "/Users/test/OpenCoworker/launch-note", writable: true, label: "scratch", primary: true, exists: true };
const baseName = (p: string) => p.split("/").filter(Boolean).pop() || p;

const PROVIDERS = [
  // openai: configured + used (drives the "Last used" sub-line and the status dot).
  { name: "openai", title: "OpenAI", needs_key: true, fields: [{ key: "api_key", label: "OpenAI API key", secret: true, required: true, help: "", placeholder: "sk-…" }], configured: true, values: {}, suggested_models: ["gpt-5.5"], key_set_at: "2026-06-12", last_used_at: Math.floor(Date.now() / 1000) - 7200 },
  // anthropic: configured but never used ("Not used yet").
  { name: "anthropic", title: "Claude (Anthropic)", needs_key: true, fields: [{ key: "api_key", label: "API key", secret: true, required: true, help: "", placeholder: "sk-…" }], configured: true, values: {}, suggested_models: ["claude-opus-4-8"], key_set_at: null, last_used_at: null },
  // zai: an OpenAI-compatible vendor — unconfigured, with a prefilled editable endpoint + blurb.
  { name: "zai", title: "Z AI (GLM)", needs_key: true, blurb: "Uses Z AI's OpenAI-compatible API — the endpoint is prefilled, just add your key.", fields: [{ key: "api_key", label: "Z AI API key", secret: true, required: true, help: "", placeholder: "" }, { key: "base_url", label: "Endpoint", secret: false, required: false, help: "Prefilled with Z AI's international endpoint.", placeholder: "https://api.z.ai/api/paas/v4", default: "https://api.z.ai/api/paas/v4" }], configured: false, values: {}, suggested_models: ["glm-5.2"], key_set_at: null, last_used_at: null },
];

/** Install the API + WebSocket mocks on a page. Returns handles for assertions/seed data. */
export async function mockApi(page: import("@playwright/test").Page) {
  const subscriptions: any[] = [
    // One existing subscription (a non-pinned session) so the Slack page's per-workspace
    // "Listening" row has an entry. Relay-mode channels are team-qualified (slack:T…/C…).
    { session_id: "wp-1", session_title: "Weekly plan 1", agent: "cowork", channel: "slack:T1DL/C0AAA111", channel_name: "ocw-test", routing_target: null, collision: false },
  ];
  // Parked unauthorized messages (§19) — mutable so Allow/Dismiss round-trip through the UI.
  // The relay is multi-workspace: parked items carry their team so the Slack page files them
  // under the right workspace card.
  const parked: any[] = [
    { id: "pk1", platform: "slack", chat_id: "C0AAA111", chat_name: "#ocw-test", user_id: "U0NEW", user_name: "Maya", chat_type: "channel", text: "hey ocw, can you summarize this thread?", ts: Date.now() / 1000 - 120, team_id: "T1DL" },
  ];
  // Slack connector — PER-TEST state (managed relay, two workspaces) so allow/disconnect
  // mutations never leak across tests sharing a worker. Backend parity: `workspaces` mirrors
  // the slack:team:* profiles, each with its OWN allow-list.
  const slackState = {
    connected: true,
    mode: "relay" as "" | "relay",
    account: "deeplearning.ai",
    allowed_users: [] as string[], // flat list (manual Socket Mode only)
    workspaces: [
      { team_id: "T1DL", account: "deeplearning.ai", allowed_users: [] as string[], allow_all: false, allowed_user_names: {} as Record<string, string | null> },
      { team_id: "T2AC", account: "acme-partners", allowed_users: [] as string[], allow_all: false, allowed_user_names: {} as Record<string, string | null> },
    ],
  };
  const slackConnector = () => ({
    name: "slack", title: "Slack", icon: "#", blurb: "Two-way Slack messaging.",
    auth: "bot_token", two_way: true, available: true, brand_color: "#611f69", logo: "slack",
    fields: [], instructions: [], connected: slackState.connected,
    account: slackState.account, enabled: slackState.connected,
    allowed_users: [...slackState.allowed_users], tools: [], managed: true,
    managed_profile: slackState.mode === "relay", mode: slackState.mode,
    workspaces: slackState.workspaces.map((w) => ({ ...w, allowed_users: [...w.allowed_users] })),
    unauthorized: parked.map((x) => ({ ...x })),
  });
  // Installed personas — mutable so enable/surface/delete round-trip through the UI.
  const personas: any[] = PERSONAS.personas.map((p) => ({ ...p }));
  // Sessions — mutable so archive (PATCH), rename (PATCH), and delete round-trip.
  const sessions: any[] = [
    { ...PINNED_SESSION },
    ...EXTRA_SESSIONS.map((s) => ({ ...s })),
    { ...OPS_SESSION },
  ];
  // Inbox items + the outbound routing binding — mutable for resolve + the inline Slack config.
  const inbox: any[] = INBOX_ITEMS.map((i) => ({ ...i }));
  const routing: { name: string; channel: string | null; target: string } = {
    name: "default",
    channel: null,
    target: "",
  };
  // Session roots — the primary (writable, non-removable) scratch plus any added folders. Mutable so
  // the RO/RW add/toggle round-trips through the real UI. POST upserts by path (a toggle re-adds).
  const roots: any[] = [{ ...PRIMARY_ROOT }];
  // Providers — mutable so save (POST) flips `configured` and stamps key_set_at, matching the
  // backend's set_provider. verify (POST) never mutates: it's a live read-only credential check.
  const providers: any[] = PROVIDERS.map((p) => ({ ...p }));
  // Automations — mutable so Run now appends a run, enable/disable toggles, and delete removes.
  const automations: any[] = [{ ...AUTOMATION }];
  const automationRuns: any[] = AUTOMATION_RUNS.map((r) => ({ ...r }));
  // Per-session unattended flag — mutable so the composer's "Send to Inbox" toggle persists and
  // the app reads it back (which is what gates parking approvals to the Inbox vs an inline card).
  const unattended: Record<string, boolean> = {};

  // Fresh cloud sign-in state per test (module state outlives a page).
  Object.assign(CLOUD_STATE, {
    signed_in: false,
    account: "",
    user_id: "",
    telemetry_enabled: true,
  });

  // The scripted fake agent behind the session WebSocket. Speaks the real event protocol
  // ({type, data}), so the full send → stream → render loop and the approval round-trip run
  // through the production code paths:
  //   · on connect: `ready`
  //   · user_message: turn_start (with input, exercising the foreground dedupe) → two
  //     assistant_deltas → assistant_message "Echo: <text>" → turn_done
  //   · a message containing "run a tool": tool_proposed + permission_required, then the turn
  //     SUSPENDS until the client's approval decision arrives (deny → skipped; else → ran)
  await page.routeWebSocket(/\/ws\/session\//, (ws) => {
    const send = (type: string, data: Record<string, unknown> = {}) =>
      ws.send(JSON.stringify({ type, data }));
    send("ready");
    ws.onMessage((raw) => {
      const msg = JSON.parse(String(raw));
      if (msg.type === "user_message") {
        send("turn_start", { input: msg.text });
        if (/run a tool/i.test(msg.text)) {
          send("tool_proposed", { name: "run_shell", arguments: { command: "ls" } });
          send("permission_required", {
            name: "run_shell",
            arguments: { command: "ls" },
            reason: "The coworker wants to run a command.",
          });
          return; // suspended on the approval
        }
        send("assistant_delta", { text: "Echo: " });
        send("assistant_delta", { text: msg.text });
        // Echo the model the message carried — pins the model-per-message contract (the
        // composer's visible model must ride on every user_message; 2026-07-04 fix).
        send("assistant_message", { text: `Echo: ${msg.text} [model=${msg.model || "none"}]` });
        send("turn_done");
      } else if (msg.type === "approval") {
        if (msg.decision === "deny") {
          send("tool_finished", { name: "run_shell", status: "denied" });
          send("assistant_message", { text: "Understood — skipped the command." });
        } else {
          send("tool_finished", { name: "run_shell", status: "done", result_preview: "README.md" });
          send("assistant_message", { text: "The command ran; 1 file found." });
        }
        send("turn_done");
      }
    });
  });

  await page.route("**/v1/**", async (route) => {
    const req = route.request();
    const p = new URL(req.url()).pathname;
    const m = req.method();
    const json = (body: unknown, status = 200) =>
      route.fulfill({ status, contentType: "application/json", body: JSON.stringify(body) });

    // session-scoped (id-agnostic — any session resolves to the same fixture)
    if (/\/v1\/sessions\/[^/]+\/connections$/.test(p)) return json(CONNECTIONS);
    if (/\/v1\/sessions\/[^/]+\/roots$/.test(p)) {
      if (m === "POST") {
        const b = req.postDataJSON();
        const existing = roots.find((r) => r.path === b.path);
        if (existing) existing.writable = !!b.writable;
        else roots.push({ path: b.path, writable: !!b.writable, label: baseName(b.path), primary: false, exists: true });
        return json({ ok: true, roots });
      }
      if (m === "DELETE") {
        const rp = new URL(req.url()).searchParams.get("path");
        const i = roots.findIndex((r) => r.path === rp && !r.primary);
        if (i >= 0) roots.splice(i, 1);
        return json({ ok: true, roots });
      }
      return json({ roots });
    }
    if (/\/v1\/sessions\/[^/]+\/messages$/.test(p)) return json({ messages: [] });
    if (/\/v1\/sessions\/[^/]+\/unattended$/.test(p)) {
      const id = decodeURIComponent(p.split("/").slice(-2)[0]);
      if (m === "POST") {
        unattended[id] = !!req.postDataJSON().unattended;
        return json({ ok: true, unattended: unattended[id] });
      }
      return json({ unattended: !!unattended[id] });
    }
    if (/\/v1\/sessions\/[^/]+$/.test(p)) {
      const id = decodeURIComponent(p.split("/").pop()!);
      const i = sessions.findIndex((s) => s.session_id === id);
      if (m === "PATCH") {
        if (i >= 0) Object.assign(sessions[i], req.postDataJSON());
        return json({ ok: true });
      }
      if (m === "DELETE") {
        if (i >= 0) sessions.splice(i, 1);
        return json({ ok: true });
      }
      return json(i >= 0 ? sessions[i] : PINNED_SESSION);
    }

    if (p.endsWith("/v1/health")) return json(HEALTH);
    if (p.endsWith("/v1/settings")) return json(SETTINGS);
    if (p.endsWith("/v1/workspaces/recent")) return json({ workspaces: [] });
    if (p.endsWith("/v1/workspaces/pick") && m === "POST") {
      return json({ ok: true, path: "/tmp/picked-folder" });
    }
    if (p.endsWith("/v1/workspaces/open") && m === "POST") {
      const b = req.postDataJSON();
      return json({ ok: true, path: b.path, git_branch: "main" });
    }
    // must precede the /v1/personas/{id} catch-all (install matches it too)
    if (p.endsWith("/v1/personas/install") && m === "POST") {
      const b = req.postDataJSON();
      if (b.gallery_slug) {
        return json(
          CLOUD_STATE.signed_in
            ? { ok: true, consent: [{ id: b.gallery_slug }], personas }
            : { ok: false, error: "gallery requires cloud sign-in" },
        );
      }
      return json({ ok: false, error: "unsupported in mock" });
    }
    if (/\/v1\/personas\/[^/]+$/.test(p) && m === "POST") {
      // Persona flag update (enabled/surfaced/default). Backend parity: enabling implies
      // surfacing (registry.set_enabled sets surfaced — the PM-invisible bug fix).
      const id = p.split("/").pop();
      const t = personas.find((x) => x.id === id);
      if (!t) return json({ ok: false, error: `unknown persona: ${id}` });
      const b = req.postDataJSON();
      if (b.default) personas.forEach((x) => (x.default = x.id === id));
      let archivedCount = 0;
      if (typeof b.enabled === "boolean") {
        t.enabled = b.enabled;
        if (b.enabled) t.surfaced = true;
        // Backend parity (disable-archives, §18): disabling archives the persona's real
        // sessions server-side, so its sidebar section disappears with it.
        if (!b.enabled) {
          for (const s of sessions) {
            if (s.agent === id && !s.archived && !s.session_id.startsWith("__")) {
              s.archived = true;
              archivedCount++;
            }
          }
        }
      }
      if (typeof b.surfaced === "boolean") t.surfaced = b.surfaced;
      return json({ ok: true, personas, archived_sessions: archivedCount });
    }
    if (/\/v1\/personas\/[^/]+$/.test(p) && m === "DELETE") {
      const id = p.split("/").pop();
      const i = personas.findIndex((x) => x.id === id && !x.builtin);
      if (i < 0) return json({ ok: false, error: `unknown persona: ${id}` });
      personas.splice(i, 1);
      return json({ ok: true, personas });
    }
    if (/\/v1\/personas\/[^/]+$/.test(p)) return json(PERSONA_DETAIL);
    if (p.endsWith("/v1/personas")) return json({ personas });
    if (p.endsWith("/v1/sessions")) return json({ sessions });
    if (/\/v1\/connectors\/slack\/unauthorized\/[^/]+$/.test(p) && m === "POST") {
      const id = p.split("/").pop();
      const i = parked.findIndex((x) => x.id === id);
      if (i < 0) return json({ ok: false, error: "unknown item" });
      const b = req.postDataJSON();
      const item = parked.splice(i, 1)[0];
      // Backend parity: allowing routes to the item's OWN workspace's list (ids are
      // workspace-scoped); a team-less item lands on the flat list (manual mode).
      if (b.action === "allow" || b.action === "allow_deliver") {
        const pool = item.team_id
          ? slackState.workspaces.find((w) => w.team_id === item.team_id)?.allowed_users
          : slackState.allowed_users;
        if (pool && !pool.includes(item.user_id)) pool.push(item.user_id);
      }
      return json({ ok: true });
    }
    // Per-workspace allow/disallow (team_id in the body) + the flat manual list without it.
    if (/\/v1\/connectors\/slack\/(allow|disallow)$/.test(p) && m === "POST") {
      const b = req.postDataJSON();
      const add = p.endsWith("/allow");
      const pool = b.team_id
        ? slackState.workspaces.find((w) => w.team_id === b.team_id)?.allowed_users
        : slackState.allowed_users;
      if (!pool) return json({ ok: false, error: "workspace not connected" });
      const i = pool.indexOf(b.user_id);
      if (add && i < 0) pool.push(b.user_id);
      if (!add && i >= 0) pool.splice(i, 1);
      return json({ ok: true, allowed_users: [...pool], team_id: b.team_id ?? null });
    }
    // Stop relaying one workspace; removing the last flips the connector off (backend parity).
    if (/\/v1\/connectors\/slack\/workspaces\/[^/]+\/disconnect$/.test(p) && m === "POST") {
      const teamId = decodeURIComponent(p.split("/").slice(-2)[0]);
      const i = slackState.workspaces.findIndex((w) => w.team_id === teamId);
      if (i < 0) return json({ ok: false, error: "workspace not connected" });
      slackState.workspaces.splice(i, 1);
      if (slackState.workspaces.length === 0) {
        slackState.connected = false;
        slackState.mode = "";
      }
      return json({ ok: true, remaining_workspaces: slackState.workspaces.length });
    }
    if (p.endsWith("/v1/connectors"))
      return json({
        connectors: [slackConnector(), ...CONNECTORS.connectors.map((c: any) => ({ ...c }))],
      });
    if (p.endsWith("/v1/cloud/status")) return json({ ...CLOUD_STATE });
    if (p.endsWith("/v1/cloud/login") && m === "POST") {
      Object.assign(CLOUD_STATE, { signed_in: true, account: "rohit@opencoworker.app", user_id: "usr_e2e" });
      return json({ ok: true });
    }
    if (p.endsWith("/v1/cloud/telemetry") && m === "POST") {
      CLOUD_STATE.telemetry_enabled = !!req.postDataJSON().enabled;
      return json({ ok: true, telemetry_enabled: CLOUD_STATE.telemetry_enabled });
    }
    if (p.endsWith("/v1/cloud/logout") && m === "POST") {
      Object.assign(CLOUD_STATE, { signed_in: false, account: "", user_id: "" });
      return json({ ok: true, signed_in: false });
    }
    if (/\/v1\/connectors\/[^/]+\/connect-managed$/.test(p) && m === "POST") {
      if (!CLOUD_STATE.signed_in) return json({ ok: false, error: "not signed in" });
      // Slack managed install = add a workspace. The real flow completes in the system
      // browser; the mock installs instantly so the page's poll picks it up.
      if (p.includes("/connectors/slack/")) {
        slackState.workspaces.push({ team_id: "T3NEW", account: "new-workspace", allowed_users: [], allow_all: false, allowed_user_names: {} });
        slackState.connected = true;
        slackState.mode = "relay";
      }
      return json({ ok: true });
    }
    if (p.endsWith("/v1/cloud/gallery")) {
      return json(
        CLOUD_STATE.signed_in
          ? { ok: true, personas: GALLERY_PERSONAS }
          : { ok: false, error: "gallery requires cloud sign-in", personas: [] },
      );
    }
    if (/\/v1\/cloud\/gallery\/[^/]+$/.test(p)) {
      if (!CLOUD_STATE.signed_in) return json({ ok: false, error: "gallery requires cloud sign-in" });
      const slug = p.split("/").pop();
      const cardBase = GALLERY_PERSONAS.find((g) => g.slug === slug) ?? GALLERY_PERSONAS[0];
      return json({
        ok: true,
        card: { ...cardBase, pitch_markdown: "**Walk into every call already knowing the account.**" },
        capabilities: {
          tools: ["files", "search", "todo"],
          risk: [],
          connectors: true,
          mcp: [],
          messaging: true,
          recommended_mode: "interactive",
          recommended_models: [],
        },
        recommends: [
          { kind: "connector", ref: "hubspot", reason: "read deals and contacts", tier: "core" },
        ],
      });
    }
    // provider credential check (read-only) — an api_key containing "bad" fails, else ok.
    if (p.endsWith("/v1/providers/verify") && m === "POST") {
      const key = String(req.postDataJSON()?.fields?.api_key || "");
      return /bad/i.test(key)
        ? json({ ok: false, error: "Invalid API key." })
        : json({ ok: true });
    }
    // save a provider key — flips `configured`, stamps key_set_at (backend set_provider parity).
    if (p.endsWith("/v1/providers") && m === "POST") {
      const b = req.postDataJSON();
      const prov = providers.find((x) => x.name === b.name);
      if (!prov) return json({ ok: false, error: `unknown provider: ${b.name}` });
      if (b.fields?.api_key) {
        prov.configured = true;
        prov.key_set_at = "2026-07-05";
      }
      if (b.fields?.base_url) prov.values = { ...prov.values, base_url: b.fields.base_url };
      return json({ ok: true, provider: b.name, recommended_model: null });
    }
    if (p.endsWith("/v1/providers")) return json(providers);
    if (p.endsWith("/v1/channels/recent"))
      return json({
        channels: [
          { channel: "slack:C0AAA111", name: "ocw-test", last_from: "amy", last_text: "standup at 10" },
          { channel: "slack:C0BBB222", last_from: "bob", last_text: "deploy failed" },
        ],
      });

    // inbox: pending items + the outbound routing binding (inline Slack config)
    if (/\/v1\/inbox\/[^/]+\/resolve$/.test(p) && m === "POST") {
      const id = decodeURIComponent(p.split("/")[p.split("/").length - 2]);
      const it = inbox.find((x) => x.id === id);
      if (it) {
        it.state = "resolved";
        it.resolution = req.postDataJSON().resolution;
      }
      return json({ ok: true });
    }
    if (p.endsWith("/v1/inbox/routing/binding") && m === "POST") {
      const b = req.postDataJSON();
      routing.channel = b.channel;
      routing.target = b.target;
      return json({ ok: true, bindings: [{ ...routing }] });
    }
    if (p.endsWith("/v1/inbox/routing")) return json({ bindings: [{ ...routing }] });
    if (p.endsWith("/v1/inbox")) {
      const q = new URL(req.url()).searchParams;
      const sid = q.get("session_id");
      const state = q.get("state");
      return json({
        items: inbox.filter(
          (i) => (!sid || i.session_id === sid) && (!state || i.state === state),
        ),
      });
    }

    // automations: one scheduled task with a running run (drives the Automations detail page
    // and the run-session banner + Back-to-runs flow). Mutable: Run now appends a run and opens
    // its live session; the enable toggle (PATCH) and delete (DELETE) round-trip through the UI.
    if (/\/v1\/automations\/[^/]+\/run$/.test(p) && m === "POST") {
      const id = p.split("/").slice(-2)[0];
      const task = automations.find((t) => t.id === id);
      if (!task) return json({ ok: false, error: "unknown task" });
      const runId = `r${automationRuns.length + 1}`;
      automationRuns.unshift({
        run_id: runId,
        task_id: id,
        session_id: `__run__${runId}`,
        started_at: Math.floor(Date.now() / 1000),
        finished_at: null,
        status: "running",
        result_text: null,
        artifacts: [],
        error: null,
        trigger: "manual",
      });
      return json({
        ok: true,
        run_id: runId,
        session_id: `__run__${runId}`,
        workspace: task.workspace,
        agent: task.agent,
        prompt: task.instructions,
      });
    }
    if (/\/v1\/automations\/[^/]+$/.test(p) && m === "GET") {
      const id = p.split("/").pop();
      const task = automations.find((t) => t.id === id) ?? automations[0];
      return json({ task, runs: automationRuns.filter((r) => r.task_id === task?.id) });
    }
    if (/\/v1\/automations\/[^/]+$/.test(p) && m === "PATCH") {
      const id = p.split("/").pop();
      const task = automations.find((t) => t.id === id);
      if (task) Object.assign(task, req.postDataJSON());
      return json({ ok: true, task });
    }
    if (/\/v1\/automations\/[^/]+$/.test(p) && m === "DELETE") {
      const id = p.split("/").pop();
      const i = automations.findIndex((t) => t.id === id);
      if (i >= 0) automations.splice(i, 1);
      return json({ ok: true });
    }
    if (p.endsWith("/v1/automations")) return json({ tasks: automations });
    if (p.endsWith("/v1/mcp")) return json({ servers: [] });
    if (p.endsWith("/v1/unrouted")) return json([]);

    // channel subscriptions — mutable so add/remove reflect through the UI
    if (p.endsWith("/v1/subscriptions") && m === "GET") return json({ subscriptions });
    if (p.endsWith("/v1/subscriptions") && m === "POST") {
      const b = req.postDataJSON();
      // Backend parity with resolve_channel: Copy-link URLs resolve to the id; bare #names
      // can't be looked up and are rejected with the same hint the server gives.
      const raw = String(b.channel || "").trim();
      if (raw.startsWith("#"))
        return json({
          ok: false,
          error:
            "Channel names can't be looked up — paste the channel ID (channel name ▸ About) or the channel's Copy-link URL.",
        });
      const link = raw.match(/slack\.com\/archives\/([A-Za-z0-9]+)/);
      const channel = link ? `slack:${link[1].toUpperCase()}` : raw;
      subscriptions.push({ session_id: b.session_id, session_title: "", agent: "", channel, routing_target: null, collision: false });
      return json({ ok: true, channel });
    }
    if (p.endsWith("/v1/subscriptions/remove") && m === "POST") {
      const b = req.postDataJSON();
      const i = subscriptions.findIndex((s) => s.session_id === b.session_id && s.channel === b.channel);
      if (i >= 0) subscriptions.splice(i, 1);
      return json({ ok: true });
    }

    // Anything else: an empty-but-valid body. GET list endpoints read `?? []`/`?? {}` fallbacks.
    return json({});
  });
}

// A `test` whose page has the API mocked before navigation.
export const test = base.extend({
  page: async ({ page }, use) => {
    await mockApi(page);
    await use(page);
  },
});

export { expect };
