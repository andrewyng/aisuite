// The dedicated Slack workspaces page (M3.5): multi-workspace list, Add workspace via
// managed connect, per-workspace disconnect (stop-relaying-only), and the manual
// Socket-Mode card (flat allow-list) so neither connect path regresses.
import { expect } from "@playwright/test";
import { test } from "./fixtures";

async function openSlackPage(page) {
  await page.goto("/");
  await page.getByRole("button", { name: /Settings & more/i }).click();
  await page.getByRole("button", { name: "Integrations", exact: true }).click();
  await page.getByRole("button", { name: "Slack workspaces" }).click();
}

test("lists every connected workspace with its own allow-list block", async ({ page }) => {
  await openSlackPage(page);
  await expect(page.getByTestId("slack-workspace-T1DL")).toContainText("deeplearning.ai");
  await expect(page.getByTestId("slack-workspace-T2AC")).toContainText("acme-partners");
  // each card carries its own allow-list section
  await expect(
    page.getByTestId("slack-workspace-T1DL").getByText("Allowed to message"),
  ).toBeVisible();
  await expect(
    page.getByTestId("slack-workspace-T2AC").getByText("Allowed to message"),
  ).toBeVisible();
});

test("Add workspace needs cloud sign-in; signed in it installs another workspace", async ({
  page,
}) => {
  await openSlackPage(page);
  // signed out: no Add button, a sign-in hint instead
  await expect(page.getByTestId("add-workspace")).toHaveCount(0);
  await expect(page.getByTestId("slack-workspaces")).toContainText("Sign in to OpenCoworker Cloud");

  // sign in via the Connectors tab card, then come back (the sub-nav button's
  // accessible name carries the connector-count badge, so match by prefix)
  await page.getByRole("button", { name: /^Connectors/ }).click();
  await page.getByTestId("cloud-account").getByRole("button", { name: "Sign in" }).click();
  await page.getByRole("button", { name: "Slack workspaces" }).click();

  await page.getByTestId("add-workspace").click();
  // the mock completes the browser install instantly; the page's refresh shows it
  await expect(page.getByTestId("slack-workspace-T3NEW")).toContainText("new-workspace");
  await expect(page.getByTestId("slack-workspace-T1DL")).toBeVisible(); // existing ones stay
});

test("disconnect removes one workspace and keeps the rest relaying", async ({ page }) => {
  await openSlackPage(page);
  await page.getByTestId("disconnect-workspace-T2AC").click();
  await expect(page.getByTestId("slack-workspace-T2AC")).toHaveCount(0);
  await expect(page.getByTestId("slack-workspace-T1DL")).toBeVisible();
});

test("manual Socket Mode: one card with the flat allow-list (no regression)", async ({
  page,
}) => {
  // Override the connectors payload AFTER mockApi so this test sees a manual-mode Slack
  // (routes registered later match first).
  await page.route("**/v1/connectors", (route) =>
    route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify({
        connectors: [
          {
            name: "slack", title: "Slack", icon: "#", blurb: "Two-way Slack messaging.",
            auth: "bot_token", two_way: true, available: true, brand_color: "#611f69",
            logo: "slack", fields: [], instructions: [], connected: true, account: "acme",
            enabled: true, allowed_users: ["U0OK"], allowed_user_names: { U0OK: "Rohit" },
            tools: [], managed: true, managed_profile: false, mode: "", workspaces: [],
            unauthorized: [],
          },
        ],
      }),
    }),
  );
  await openSlackPage(page);
  await expect(page.getByTestId("slack-mode-badge")).toContainText("Socket Mode");
  const card = page.getByTestId("slack-manual-card");
  await expect(card).toContainText("acme");
  await expect(card).toContainText("Rohit"); // flat allow-list chip, named
});
