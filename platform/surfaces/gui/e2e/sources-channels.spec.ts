import { test, expect } from "./fixtures";

// Guards the per-session Slack channels drill-down (§14): the "Channels" affordance is gated to
// two-way connectors, opens a child panel, and add/remove round-trip through the subscribe APIs.
test("Slack channels drill-down: gating, add (auto-prefixed), remove", async ({ page }) => {
  await page.goto("/");

  // Open the pinned cowork session, then its Sources drawer.
  await page.getByText("Draft the launch note").first().click();
  await page.getByTitle("Manage this session's connections").click();

  const drawer = page.getByRole("dialog", { name: "Session connections" });
  await expect(drawer.getByText(/Connected · 2/)).toBeVisible();

  // Gating: only the two-way connector (Slack) gets a Channels affordance — not Browser.
  await expect(page.getByRole("button", { name: /Channels ·/ })).toHaveCount(1);
  await expect(page.getByRole("button", { name: /Channels · 0/ })).toBeVisible();

  // Drill in.
  await page.getByRole("button", { name: /Channels · 0/ }).click();
  await expect(page.getByText("Slack channels")).toBeVisible();
  await expect(page.getByText(/Not listening to any Slack channel yet/)).toBeVisible();

  // Add a bare channel id — the panel scopes it to the connector (→ "slack:C0123").
  await page.getByPlaceholder("slack:C0123 or #channel").fill("C0123");
  await page.getByRole("button", { name: "Add", exact: true }).click();
  await expect(page.getByText("slack:C0123")).toBeVisible();
  await expect(page.getByText(/Subscribed channels · 1/)).toBeVisible();

  // Remove it → back to the empty state.
  await page.getByTitle("Stop listening").click();
  await expect(page.getByText(/Not listening to any Slack channel yet/)).toBeVisible();

  // Back returns to the Sources list.
  await page.getByRole("button", { name: "Back to sources" }).click();
  await expect(drawer.getByText(/Connected · 2/)).toBeVisible();
});
