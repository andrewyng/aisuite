import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { SourcesBar } from "./SourcesBar";

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

const CONNS = {
  connected: [
    { connector: "slack", enabled: true, detail: "#ocw-test · DMs" },
    { connector: "github", enabled: true, detail: "deeplearning-ai/platform" },
  ],
  recommended: [
    { connector: "datadog", reason: "pull the firing alerts", tier: "core", connected: false },
    { connector: "pagerduty", reason: "see who's on-call", tier: "optional", connected: false },
  ],
  attention: 2,
};

const CONNECTORS = {
  connectors: [
    { name: "slack", title: "Slack", logo: "slack", brand_color: "#611f69" },
    { name: "github", title: "GitHub", logo: "github", brand_color: "#1f2328" },
    { name: "datadog", title: "Datadog", logo: "datadog", brand_color: "#632ca6" },
  ],
};

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("SourcesBar", () => {
  it("shows the ⚠ N attention badge equal to `attention`", async () => {
    stubFetch([
      { match: "/connections", method: "GET", json: CONNS },
      { match: "/v1/connectors", method: "GET", json: CONNECTORS },
    ]);
    render(<SourcesBar sessionId="s1" />);
    expect(await screen.findByText("⚠ 2")).toBeTruthy();
  });

  it("opening the drawer lists connected (working toggles) + recommended", async () => {
    const calls = stubFetch([
      { match: "/connections", method: "GET", json: CONNS },
      { match: "/v1/connectors", method: "GET", json: CONNECTORS },
      { match: "/connections", method: "POST", json: { ok: true } },
    ]);
    render(<SourcesBar sessionId="s1" />);
    // open the drawer
    fireEvent.click(await screen.findByTitle("Manage this session's connections"));

    // connected section: header count + both connected connectors
    expect(await screen.findByText("Connected · 2")).toBeTruthy();
    expect(screen.getByText("#ocw-test · DMs")).toBeTruthy();
    // recommended connectors
    expect(screen.getByText("pull the firing alerts")).toBeTruthy();
    expect(screen.getAllByText("Connect").length).toBeGreaterThan(0);

    // toggling a connected connector POSTs a per-session override (mute).
    const switches = screen.getAllByRole("switch");
    fireEvent.click(switches[0]);
    await waitFor(() => {
      const post = calls.find(
        (c) => c.method === "POST" && c.url.includes("/sessions/s1/connections"),
      );
      expect(post).toBeTruthy();
      expect(post!.body).toMatchObject({ connector: "slack", enabled: false });
    });
  });

  it("renders nothing when there are no connections and no attention", async () => {
    stubFetch([
      { match: "/connections", method: "GET", json: { connected: [], recommended: [], attention: 0 } },
      { match: "/v1/connectors", method: "GET", json: CONNECTORS },
    ]);
    const { container } = render(<SourcesBar sessionId="s2" />);
    // give the async fetch a tick to resolve, then assert the bar stayed unrendered.
    await waitFor(() => expect(container.querySelector(".sources-bar")).toBeNull());
  });
});
