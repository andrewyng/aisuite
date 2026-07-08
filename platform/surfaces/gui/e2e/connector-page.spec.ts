// Slack config moved to a dedicated Integrations tab (M3.5): the connector card is a
// COMPACT status row with "Manage workspaces →", and the §19 blocks — parked messages
// (Allow & deliver / Allow only / Dismiss) and "sessions listening" — live on the page,
// filed under the workspace they belong to (the relay is multi-workspace).
import { expect } from "@playwright/test";
import { test } from "./fixtures";

async function openSlackPage(page) {
  await page.goto("/");
  await page.getByRole("button", { name: /Settings & more/i }).click();
  await page.getByRole("button", { name: "Integrations", exact: true }).click();
  await page.getByRole("button", { name: "Slack workspaces" }).click();
}

test("connector card is compact: status + Manage workspaces link to the page", async ({
  page,
}) => {
  await page.goto("/");
  await page.getByRole("button", { name: /Settings & more/i }).click();
  await page.getByRole("button", { name: "Integrations", exact: true }).click();

  const card = page.getByTestId("connector-slack");
  await expect(card).toContainText("2 workspaces · managed relay");
  await page.getByTestId("manage-slack-workspaces").click();
  await expect(page.getByTestId("slack-workspaces")).toBeVisible();
  await expect(page.getByTestId("slack-mode-badge")).toContainText("managed relay");
});

test("parked message files under ITS workspace; Allow & deliver adds to that allow-list only", async ({
  page,
}) => {
  await openSlackPage(page);

  // pk1 belongs to T1DL — it renders in that workspace's card, not the other's.
  const block = page.getByTestId("unauthorized-slack-T1DL");
  await expect(block).toContainText("Messages from senders you haven't allowed · 1");
  await expect(block).toContainText("Maya");
  await expect(block).toContainText("in #ocw-test");
  await expect(block).toContainText("hey ocw, can you summarize this thread?");
  await expect(page.getByTestId("unauthorized-slack-T2AC")).toHaveCount(0);

  await page.getByTestId("parked-allow-deliver-pk1").click();
  await expect(page.getByTestId("unauthorized-slack-T1DL")).toHaveCount(0);
  // The sender lands on the T1DL allow-list; the sibling workspace stays empty.
  await expect(page.getByTestId("slack-workspace-T1DL")).toContainText("U0NEW");
  await expect(page.getByTestId("slack-workspace-T2AC")).not.toContainText("U0NEW");
});

test("parked message can be dismissed without allowing the sender", async ({ page }) => {
  await openSlackPage(page);
  await page.getByTestId("parked-dismiss-pk1").click();
  await expect(page.getByTestId("unauthorized-slack-T1DL")).toHaveCount(0);
  await expect(page.getByTestId("slack-workspace-T1DL")).not.toContainText("U0NEW");
});

test("sessions listening to Slack channels: listed with unsubscribe", async ({ page }) => {
  await openSlackPage(page);

  const block = page.getByTestId("listening-slack");
  await expect(block).toContainText("Sessions listening to Slack channels · 1");
  await expect(block).toContainText("Weekly plan 1");
  await expect(block).toContainText("#ocw-test"); // named channel (address in the tooltip)

  await block.getByTitle("Unsubscribe this session").click();
  await expect(block).toContainText("Sessions listening to Slack channels · 0");
  await expect(block).toContainText("None yet");
});
