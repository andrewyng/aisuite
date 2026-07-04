import { test, expect } from "./fixtures";

// The Inbox (owner testing pass, 2026-07-03): kind tabs (All/Approvals/Questions), persona filter
// chips (only with >1 persona holding items), resolve-removes-card, and the INLINE Slack routing
// config — bound → Change/Stop; slack-connected-but-unbound → "Also send to a Slack channel →".

async function openInbox(page: import("@playwright/test").Page) {
  await page.goto("/");
  // The sidebar's Inbox row (its accessible name includes the attention badge count).
  await page.locator(".sidebar").getByRole("button", { name: /Inbox/ }).click();
  await expect(page.getByText("Approve: run_shell")).toBeVisible();
}

test("kind + persona filters narrow the pending list", async ({ page }) => {
  await openInbox(page);
  const question = "Which environment should I restart?";
  await expect(page.getByText(question)).toBeVisible();

  const filters = page.getByTestId("inbox-filters");
  await filters.getByRole("button", { name: "Approvals" }).click();
  await expect(page.getByText(question)).not.toBeVisible();
  await expect(page.getByText("Approve: run_shell")).toBeVisible();

  await filters.getByRole("button", { name: "Questions" }).click();
  await expect(page.getByText("Approve: run_shell")).not.toBeVisible();
  await expect(page.getByText(question)).toBeVisible();

  // Persona chips render because two personas hold items; filtering to Ops hides the cowork item.
  await filters.getByRole("button", { name: "All", exact: true }).click();
  await filters.getByRole("button", { name: "Ops", exact: true }).click();
  await expect(page.getByText("Approve: run_shell")).not.toBeVisible();
  await expect(page.getByText(question)).toBeVisible();
});

test("resolving an approval removes its card; question options resolve on click", async ({ page }) => {
  await openInbox(page);

  await page.getByRole("button", { name: "Approve", exact: true }).click();
  await expect(page.getByText("Approve: run_shell")).not.toBeVisible();

  // Single-select question: clicking an option resolves immediately.
  await page.getByRole("button", { name: "staging", exact: true }).click();
  await expect(page.getByText("Which environment should I restart?")).not.toBeVisible();
  await expect(page.getByText("Nothing pending.")).toBeVisible();
});

test("routing: inline Slack config binds a channel, then Stop clears it", async ({ page }) => {
  await openInbox(page);
  const line = page.getByTestId("inbox-routing");
  await expect(line).toContainText("Delivered here only");

  // Slack is connected in the fixture, so the inline configure affordance shows (no detour to
  // the global connectors page).
  await page.getByTestId("inbox-route-configure").click();
  await page.getByPlaceholder("slack:C0123 or #channel").fill("C0777");
  await line.getByRole("button", { name: "Set", exact: true }).click();

  await expect(line).toContainText("slack:C0777");
  await expect(line).toContainText("replies there resolve items here");

  await line.getByRole("button", { name: "Stop" }).click();
  await expect(line).toContainText("Delivered here only");
});
