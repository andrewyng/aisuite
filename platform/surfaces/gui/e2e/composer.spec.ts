import { test, expect } from "./fixtures";

// Guards the three-control composer row (§22): send-gating (accent only with content), the "+"
// attach menu, and the Mode menu (permission options + the folded-in Send-to-Inbox toggle).
test("composer: send-gating, + attach menu, Mode menu", async ({ page }) => {
  await page.goto("/");
  await page.getByText("Draft the launch note").first().click();

  const box = page.getByPlaceholder(/Ask the coworker/);
  const send = page.getByRole("button", { name: "Send" });

  // Send is subtle grey when empty, accent once there's content, grey again when cleared.
  await expect(send).not.toHaveClass(/bg-accent/);
  await box.fill("hello there");
  await expect(send).toHaveClass(/bg-accent/);
  await box.fill("");
  await expect(send).not.toHaveClass(/bg-accent/);

  // "+" attach menu offers the three typed shortcuts.
  await page.getByRole("button", { name: "Attach" }).click();
  await expect(page.getByRole("button", { name: "Photo or image" })).toBeVisible();
  await expect(page.getByRole("button", { name: "PDF", exact: true })).toBeVisible();
  await expect(page.getByRole("button", { name: "Other files" })).toBeVisible();
  // Clicking the backdrop closes it.
  await page.locator(".fixed.inset-0.z-30").click();
  await expect(page.getByRole("button", { name: "Photo or image" })).toHaveCount(0);

  // Mode menu (workspace personas only): the five permission options with the current one
  // marked, plus the Unattended/send-to-Inbox toggle at the bottom (§22).
  await page.getByRole("button", { name: "Mode", exact: true }).click();
  const menu = page.getByTestId("mode-menu");
  await expect(menu.getByText("Discuss")).toBeVisible();
  await expect(menu.getByText("Explore read-only, propose a plan")).toBeVisible();
  // The current mode is marked with a ✓.
  await expect(menu.locator("button").filter({ hasText: "Ask for approval" })).toContainText("✓");
  await expect(menu.getByRole("switch", { name: "Send approvals to the Inbox" })).toBeVisible();
  // Picking an option closes the menu (and would flip the live engine's mode).
  await menu.getByText("Full access").click();
  await expect(page.getByTestId("mode-menu")).toHaveCount(0);
});
