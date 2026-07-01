import { test, expect } from "./fixtures";

// Guards the Settings-as-page refactor (§13): the ⚙ menu opens a full-page surface with a left
// sub-nav — not the retired modal — and each section renders.
test("Settings opens as a full page and navigates sections", async ({ page }) => {
  await page.goto("/");

  await page.getByRole("button", { name: /Settings & more/i }).click();
  await page.getByRole("button", { name: "Settings", exact: true }).click();

  // Full-page: left sub-nav + Appearance section (no modal backdrop).
  await expect(page.getByRole("heading", { name: "Appearance" })).toBeVisible();
  await expect(page.locator(".modal-backdrop")).toHaveCount(0);
  for (const label of ["Appearance", "Files", "Models", "Personas"]) {
    await expect(page.getByRole("button", { name: label, exact: true })).toBeVisible();
  }

  await page.getByRole("button", { name: "Files", exact: true }).click();
  await expect(page.getByText("Scratch location")).toBeVisible();

  await page.getByRole("button", { name: "Personas", exact: true }).click();
  await expect(page.getByText("Add personas")).toBeVisible();

  await page.getByRole("button", { name: "Models", exact: true }).click();
  await expect(page.getByText("API models")).toBeVisible();
});
