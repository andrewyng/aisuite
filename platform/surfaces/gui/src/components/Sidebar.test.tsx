import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { Sidebar } from "./Sidebar";
import type { SessionInfo } from "../types";

// Hermetic fetch stub routing by URL substring + method; records calls for POST assertions.
type Call = { url: string; method: string; body: any };

function stubFetch(routes: { match: string; method?: string; json: any }[]) {
  const calls: Call[] = [];
  const fn = vi.fn(async (url: string, init?: RequestInit) => {
    const method = (init?.method || "GET").toUpperCase();
    calls.push({ url, method, body: init?.body ? JSON.parse(String(init.body)) : undefined });
    for (const r of routes) {
      if (url.includes(r.match) && (!r.method || r.method === method)) {
        return { ok: true, json: async () => r.json } as Response;
      }
    }
    return { ok: true, json: async () => ({}) } as Response;
  });
  vi.stubGlobal("fetch", fn);
  return calls;
}

const PERSONAS = {
  personas: [
    { id: "cowork", name: "OpenCoworker", icon: "cowork", tagline: "general assistant", family: "knowledge", enabled: true, surfaced: true, default: true },
    { id: "ops", name: "Ops", icon: "ops", tagline: "incidents, runbooks", family: "code", enabled: true, surfaced: true, default: false },
    { id: "code", name: "Code", icon: "code", tagline: "repository work", family: "code", enabled: true, surfaced: true, default: false },
    { id: "secret", name: "Disabled One", icon: "cowork", tagline: "off", family: "knowledge", enabled: false, surfaced: false, default: false },
  ],
};

const SESSIONS: SessionInfo[] = [
  { session_id: "s-ops-1", title: "incident watch", workspace: "/w", agent: "ops", model: "m", mode: "interactive", updated_at: "2026-06-29", messages: 2 },
  { session_id: "s-cowork-1", title: "hi there", workspace: "", agent: "cowork", model: "m", mode: "interactive", updated_at: "2026-06-29", messages: 1 },
];

const baseProps = {
  agent: "cowork",
  workspace: "",
  surfaces: { cowork: true, chat: false, code: false },
  sessions: SESSIONS,
  projects: [],
  activeSession: "s-cowork-1",
  onSwitchAgent: vi.fn(),
  onNewSession: vi.fn(),
  onSelectSession: vi.fn(),
  onNewProject: vi.fn(),
  onRenameSession: vi.fn(),
  onDeleteSession: vi.fn(),
  onArchiveSession: vi.fn(),
  onTogglePin: vi.fn(),
  onManage: vi.fn(),
  onOpenPersona: vi.fn(),
  onManagePersonas: vi.fn(),
  onOpenScheduled: vi.fn(),
  onOpenIntegrations: vi.fn(),
  onOpenAudit: vi.fn(),
  onOpenInbox: vi.fn(),
  scheduledActive: false,
  integrationsActive: false,
  auditActive: false,
  inboxActive: false,
};

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
  vi.clearAllMocks();
});

describe("Sidebar group/filter control", () => {
  it("choosing Persona persists via setNavLayout and switches to the per-persona accordion", async () => {
    const calls = stubFetch([
      { match: "/v1/personas", method: "GET", json: PERSONAS },
      { match: "/v1/settings", method: "GET", json: { nav_layout: "flat" } },
      { match: "/v1/settings/nav-layout", method: "POST", json: { ok: true, nav_layout: "grouped" } },
    ]);
    render(<Sidebar {...baseProps} />);

    // personas load drives the surfaces; the RECENT header's group/filter control is always present.
    const control = await screen.findByLabelText("Group and filter conversations");

    // Open the popover and choose "Group by → Persona".
    fireEvent.click(control);
    fireEvent.click(await screen.findByText("Persona"));

    // POSTs the new layout pref.
    await waitFor(() => {
      const post = calls.find((c) => c.method === "POST" && c.url.includes("/v1/settings/nav-layout"));
      expect(post).toBeTruthy();
      expect(post!.body).toMatchObject({ nav_layout: "grouped" });
    });

    // Close the popover (it stays open so you can group AND filter in one visit) before asserting
    // the accordion — otherwise "Ops" also matches the filter-by-coworker checkbox.
    fireEvent.click(control);

    // Grouped view = the per-persona accordion. The Ops header appears; expanding it lists its
    // session, and the header's gear (rightmost element) opens the persona detail page.
    const opsHeader = await screen.findByText("Ops");
    fireEvent.click(opsHeader);
    expect(screen.getByText("incident watch")).toBeTruthy();
    fireEvent.click(screen.getByTitle("About the Ops persona"));
    expect(baseProps.onOpenPersona).toHaveBeenCalledWith("ops");
  });
});

describe("New-session split button", () => {
  it("primary starts the last-used persona; the menu lists enabled personas + Manage personas…", async () => {
    stubFetch([
      { match: "/v1/personas", method: "GET", json: PERSONAS },
      { match: "/v1/settings", method: "GET", json: { nav_layout: "flat" } },
    ]);
    const { container } = render(<Sidebar {...baseProps} />);
    await screen.findByLabelText("Group and filter conversations");

    // Primary action → a new session with the current (last-used) persona.
    fireEvent.click(container.querySelector(".newsplit-primary")!);
    expect(baseProps.onNewSession).toHaveBeenCalledWith("cowork");

    // ▾ opens the persona menu: enabled personas appear, the disabled one does not, plus a manage entry.
    fireEvent.click(screen.getByLabelText("Choose a persona"));
    const menu = (await screen.findByText("Start a session as")).closest(".newsplit-menu") as HTMLElement;
    const w = within(menu);
    expect(w.getByText("Ops")).toBeTruthy();
    expect(w.getByText("Code")).toBeTruthy();
    expect(w.queryByText("Disabled One")).toBeNull();
    expect(w.getByText("Manage personas…")).toBeTruthy();

    // Selecting a persona starts a session as that persona.
    fireEvent.click(w.getByText("Ops"));
    expect(baseProps.onNewSession).toHaveBeenCalledWith("ops");

    // "Manage personas…" opens the persona management surface.
    fireEvent.click(screen.getByLabelText("Choose a persona"));
    fireEvent.click(await screen.findByText("Manage personas…"));
    expect(baseProps.onManagePersonas).toHaveBeenCalled();
  });
});
