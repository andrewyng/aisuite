import { test, expect } from "./fixtures";

// Regression for the invisible-after-install bug (2026-07-03): enabling a persona in
// Settings ▸ Personas must surface it EVERYWHERE without a reload — the New-Session picker and
// the grouped sidebar — via the PERSONAS_CHANGED event (and backend enable-implies-surface).

test("enabling an installed persona surfaces it in picker + sidebar without reload", async ({
  page,
}) => {
  await page.goto("/");
  const sidebar = page.locator(".sidebar");

  // Disabled install: absent from the persona picker and the grouped sidebar.
  await page.getByLabel("Choose a persona").click();
  const menu = page.locator(".newsplit-menu");
  await expect(menu).toBeVisible();
  await expect(menu.getByText("Acme Notes")).toHaveCount(0);
  await page.locator(".fixed.inset-0.z-20").click(); // close via backdrop
  await expect(sidebar.getByText("Acme Notes")).toHaveCount(0);

  // Enable it on the Personas page.
  await page.getByRole("button", { name: /Settings & more/i }).click();
  await page.getByRole("button", { name: "Settings", exact: true }).click();
  await page.getByRole("button", { name: "Personas", exact: true }).click();
  const row = page.locator(".divide-y > div").filter({ hasText: "Acme Notes" });
  // Controlled checkbox: the DOM state flips only after the POST round-trip, so click + expect
  // (a plain .check() asserts the state synchronously and fails).
  const enabled = row.getByRole("checkbox", { name: "Enabled" });
  await enabled.click();
  await expect(enabled).toBeChecked();

  // No reload: the sidebar group and the picker both pick it up via PERSONAS_CHANGED.
  await expect(sidebar.getByText("Acme Notes")).toBeVisible();
  await page.getByLabel("Choose a persona").click();
  await expect(page.locator(".newsplit-menu").getByText("Acme Notes")).toBeVisible();
});
