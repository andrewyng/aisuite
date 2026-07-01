import { test, expect } from "./fixtures";

// Guards the composer control row: send-gating (accent only with content), the "+" attach menu, and
// the permission-mode dropdown.
test("composer: send-gating, + attach menu, mode dropdown", async ({ page }) => {
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

  // Permission-mode dropdown (workspace personas only) opens its options.
  await page.getByText("Ask for approval").click();
  await expect(page.getByText("Explore read-only, propose a plan")).toBeVisible();
});
