import { test as base, expect } from "@playwright/test";

// Hermetic API mock. Every /v1 request the GUI makes is fulfilled from the fixtures below (shapes
// mirrored from the real backend), and the event WebSocket is stubbed, so specs run with no Python
// server and never touch real state. Mutations (channel subscribe/unsubscribe) are held in an
// in-memory list so the drill-down's add/remove reflect through the real UI on re-fetch.

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
};

const PERSONAS = {
  personas: [
    { id: "cowork", name: "OpenCoworker", icon: "cowork", tagline: "Produce a deliverable — research, analysis, scripts", needs_workspace: true, builtin: true, family: "knowledge", workspace: "deliverable", tools: ["files", "search"], enabled: true, surfaced: true, default: true },
    { id: "code", name: "Code", icon: "code", tagline: "Work in a codebase — files, git, shell", needs_workspace: true, builtin: true, family: "code", workspace: "git", tools: ["code_files", "git"], enabled: true, surfaced: true, default: false },
    { id: "chat", name: "Chat", icon: "chat", tagline: "Quick questions — no workspace", needs_workspace: false, builtin: true, family: "knowledge", workspace: "none", tools: [], enabled: true, surfaced: false, default: false },
    { id: "ops", name: "Ops Coworker", icon: "wrench", tagline: "Operate and investigate — runbooks, logs, infrastructure", needs_workspace: true, builtin: true, family: "knowledge", workspace: "deliverable", tools: ["files", "shell"], enabled: true, surfaced: true, default: false },
    // A non-builtin install (disabled pending consent — invisible to picker specs) so
    // the Personas page's delete affordance has a target.
    { id: "acme-notes", name: "Acme Notes", icon: "pencil", tagline: "Acme's note-taking coworker", needs_workspace: true, builtin: false, family: "knowledge", workspace: "deliverable", tools: ["files"], enabled: false, surfaced: false, default: false },
  ],
};

const PINNED_SESSION = {
  session_id: "pinned-cowork-1",
  title: "Draft the launch note",
  workspace: "/Users/test/OpenCoworker/launch-note",
  agent: "cowork",
  model: "anthropic:claude-opus-4-8",
  mode: "interactive",
  updated_at: "2026-06-30 01:22:23",
  messages: 2,
  pinned: true,
  archived: false,
  attention: 0,
  liveness: "idle",
  subscriptions: [],
};
const SESSIONS = { sessions: [PINNED_SESSION] };

const CONNECTORS = {
  connectors: [
    { name: "slack", title: "Slack", icon: "#", blurb: "Two-way Slack messaging.", auth: "bot_token", two_way: true, available: true, brand_color: "#611f69", logo: "slack", fields: [], instructions: [], connected: true, account: "acme", enabled: true, allowed_users: [], tools: [], managed: false, managed_profile: false },
    { name: "browser", title: "Browser", icon: "B", blurb: "Headless browser.", auth: "none", two_way: false, available: true, brand_color: "#6b7280", logo: "", fields: [], instructions: [], connected: true, account: null, enabled: true, allowed_users: [], tools: [], managed: false, managed_profile: false },
    { name: "telegram", title: "Telegram", icon: "T", blurb: "Two-way Telegram messaging.", auth: "bot_token", two_way: true, available: true, brand_color: "#229ed9", logo: "telegram", fields: [], instructions: [], connected: false, account: null, enabled: false, allowed_users: [], tools: [], managed: false, managed_profile: false },
    // Managed-capable connector (one-click via cloud when signed in; manual paste otherwise).
    { name: "gmail", title: "Gmail", icon: "✉", blurb: "Search, summarize, draft, and send email.", auth: "oauth", two_way: false, available: true, brand_color: "#ea4335", logo: "gmail", fields: [{ key: "access_token", label: "OAuth access token", secret: true, required: true, help: "", placeholder: "" }], instructions: [], connected: false, account: null, enabled: false, allowed_users: [], tools: [], managed: true, managed_profile: false },
  ],
};

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

const PRIMARY_ROOT = { path: "/Users/test/OpenCoworker/launch-note", writable: true, label: "scratch", primary: true, exists: true };
const baseName = (p: string) => p.split("/").filter(Boolean).pop() || p;

const PROVIDERS = [
  { name: "openai", title: "OpenAI", needs_key: true, fields: [{ key: "api_key", label: "OpenAI API key", secret: true, required: true, help: "", placeholder: "sk-…" }], configured: true, values: {}, suggested_models: ["gpt-5.5"] },
  { name: "anthropic", title: "Claude (Anthropic)", needs_key: true, fields: [{ key: "api_key", label: "API key", secret: true, required: true, help: "", placeholder: "sk-…" }], configured: true, values: {}, suggested_models: ["claude-opus-4-8"] },
];

/** Install the API + WebSocket mocks on a page. Returns handles for assertions/seed data. */
export async function mockApi(page: import("@playwright/test").Page) {
  const subscriptions: any[] = [];
  // Installed personas — mutable so the delete affordance round-trips through the UI.
  const personas: any[] = PERSONAS.personas.map((p) => ({ ...p }));
  // Session roots — the primary (writable, non-removable) scratch plus any added folders. Mutable so
  // the RO/RW add/toggle round-trips through the real UI. POST upserts by path (a toggle re-adds).
  const roots: any[] = [{ ...PRIMARY_ROOT }];

  // Fresh cloud sign-in state per test (module state outlives a page).
  Object.assign(CLOUD_STATE, {
    signed_in: false,
    account: "",
    user_id: "",
    telemetry_enabled: true,
  });

  // Stub the event WebSocket so the app's live channel doesn't error (no server behind it).
  await page.routeWebSocket(/.*/, () => {
    /* intercepted; left open with no messages */
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
    if (/\/v1\/sessions\/[^/]+$/.test(p)) return json(PINNED_SESSION);

    if (p.endsWith("/v1/health")) return json(HEALTH);
    if (p.endsWith("/v1/settings")) return json(SETTINGS);
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
    if (/\/v1\/personas\/[^/]+$/.test(p) && m === "DELETE") {
      const id = p.split("/").pop();
      const i = personas.findIndex((x) => x.id === id && !x.builtin);
      if (i < 0) return json({ ok: false, error: `unknown persona: ${id}` });
      personas.splice(i, 1);
      return json({ ok: true, personas });
    }
    if (/\/v1\/personas\/[^/]+$/.test(p)) return json(PERSONA_DETAIL);
    if (p.endsWith("/v1/personas")) return json({ personas });
    if (p.endsWith("/v1/sessions")) return json(SESSIONS);
    if (p.endsWith("/v1/connectors")) return json(CONNECTORS);
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
      return json(CLOUD_STATE.signed_in ? { ok: true } : { ok: false, error: "not signed in" });
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
    if (p.endsWith("/v1/providers")) return json(PROVIDERS);
    if (p.endsWith("/v1/channels/recent")) return json({ channels: [] });
    if (p.endsWith("/v1/inbox")) return json({ items: [] });
    if (p.endsWith("/v1/automations")) return json({ tasks: [] });
    if (p.endsWith("/v1/mcp")) return json({ servers: [] });
    if (p.endsWith("/v1/unrouted")) return json([]);

    // channel subscriptions — mutable so add/remove reflect through the UI
    if (p.endsWith("/v1/subscriptions") && m === "GET") return json({ subscriptions });
    if (p.endsWith("/v1/subscriptions") && m === "POST") {
      const b = req.postDataJSON();
      subscriptions.push({ session_id: b.session_id, session_title: "", agent: "", channel: b.channel, routing_target: null, collision: false });
      return json({ ok: true });
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
