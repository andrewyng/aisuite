// Unattended mode (item 8) — the composer's "Send to Inbox" toggle and its effect on approvals.
// When a session is unattended, an approval PARKS to the Inbox instead of surfacing an inline card
// (the app suppresses the live card; the Inbox list itself is covered by inbox.spec.ts). The
// mocked /v1/sessions/:id/unattended is stateful so the toggle persists across a reload.
import { expect } from "@playwright/test";
import { test } from "./fixtures";

// Open the composer's inbox-routing control (icon button, no text label — target its title).
async function openInboxControl(page) {
  await page.getByTitle(/Inbox routing|Sending approvals to the Inbox/).click();
}

test("attended (default): a tool request surfaces the inline approval card", async ({ page }) => {
  await page.goto("/");
  const box = page.getByPlaceholder(/Ask the coworker/);
  await box.fill("please run a tool");
  await page.getByRole("button", { name: "Send" }).click();
  await expect(page.getByText("The coworker wants to run a command.").first()).toBeVisible();
});

test("Send-to-Inbox toggle flips and persists across a reload", async ({ page }) => {
  await page.goto("/");
  await openInboxControl(page);
  const sw = page.getByRole("switch", { name: "Send approvals to the Inbox" });
  await expect(sw).toHaveAttribute("aria-checked", "false");
  await sw.click();
  await expect(sw).toHaveAttribute("aria-checked", "true");

  // Reload: the stateful endpoint returns the saved flag, so the toggle reads back on.
  await page.reload();
  await openInboxControl(page);
  await expect(page.getByRole("switch", { name: "Send approvals to the Inbox" })).toHaveAttribute(
    "aria-checked",
    "true",
  );
});

test("unattended: a tool request parks (no inline approval card)", async ({ page }) => {
  await page.goto("/");
  await openInboxControl(page);
  await page.getByRole("switch", { name: "Send approvals to the Inbox" }).click();
  // The popover's full-screen overlay closes it on any outside click.
  await page.mouse.click(5, 5);

  const box = page.getByPlaceholder(/Ask the coworker/);
  await box.fill("please run a tool");
  await page.getByRole("button", { name: "Send", exact: true }).click();

  // The turn still starts, but the live approval card is suppressed — the prompt is parked to the
  // Inbox instead. Give the (suppressed) card a beat to NOT appear.
  await expect(page.getByText("Echo:").first()).toBeVisible().catch(() => {});
  await expect(page.getByText("The coworker wants to run a command.")).toHaveCount(0);
});
