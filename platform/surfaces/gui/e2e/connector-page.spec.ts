// The two-way connector's expanded card as the one-stop config surface (§19):
// parked messages from unallowed senders (Allow & deliver / Allow only / Dismiss)
// and the per-connector "sessions listening" list.
import { expect } from "@playwright/test";
import { test } from "./fixtures";

async function openSlackSettings(page) {
  await page.goto("/");
  await page.getByRole("button", { name: /Settings & more/i }).click();
  await page.getByRole("button", { name: "Integrations", exact: true }).click();
  await page.getByTestId("connector-slack").getByRole("button", { name: "Settings" }).click();
}

test("parked unauthorized message: shown with sender + text, Allow & deliver clears it", async ({
  page,
}) => {
  await openSlackSettings(page);

  const block = page.getByTestId("unauthorized-slack");
  await expect(block).toContainText("Messages from senders you haven't allowed · 1");
  await expect(block).toContainText("Maya");
  await expect(block).toContainText("in #ocw-test");
  await expect(block).toContainText("hey ocw, can you summarize this thread?");

  await page.getByTestId("parked-allow-deliver-pk1").click();
  // The item is resolved: the block disappears (it renders only when items exist) and the
  // sender lands on the allow-list.
  await expect(page.getByTestId("unauthorized-slack")).toHaveCount(0);
  await expect(page.getByTestId("connector-slack")).toContainText("U0NEW");
});

test("parked message can be dismissed without allowing the sender", async ({ page }) => {
  await openSlackSettings(page);
  await page.getByTestId("parked-dismiss-pk1").click();
  await expect(page.getByTestId("unauthorized-slack")).toHaveCount(0);
  await expect(page.getByTestId("connector-slack")).not.toContainText("U0NEW");
});

test("sessions listening to Slack channels: listed with unsubscribe", async ({ page }) => {
  await openSlackSettings(page);

  const block = page.getByTestId("listening-slack");
  await expect(block).toContainText("Sessions listening to Slack channels · 1");
  await expect(block).toContainText("Weekly plan 1");
  await expect(block).toContainText("#ocw-test"); // named channel (address in the tooltip)

  await block.getByTitle("Unsubscribe this session").click();
  await expect(block).toContainText("Sessions listening to Slack channels · 0");
  await expect(block).toContainText("None yet");
});
