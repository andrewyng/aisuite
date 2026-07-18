import { test, expect } from "./fixtures";

// Guards the Settings-as-page refactor (§13): the ⚙ menu opens a full-page surface with a left
// sub-nav — not the retired modal — and each section renders.
test("Settings opens as a full page and navigates sections", async ({ page }) => {
  await page.goto("/");

  await page.getByTestId("account-row").click();
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

// Guards the Models pane's custom provider picker: rows carry a green "key set" dot and a
// Last-used sub-line; picking an OpenAI-compatible vendor prefills its endpoint + blurb.
test("Models: provider picker shows key status; vendor endpoint prefills", async ({ page }) => {
  await page.goto("/");
  await page.getByTestId("account-row").click();
  await page.getByRole("button", { name: "Settings", exact: true }).click();
  await page.getByRole("button", { name: "Models", exact: true }).click();

  await page.getByRole("button", { name: "Provider" }).click();
  const menu = page.getByRole("listbox", { name: "Provider" });
  // Sectioned: configured providers float to the top under "Ready to use"; the rest wait
  // under "Needs a key".
  await expect(menu.getByText("Ready to use")).toBeVisible();
  await expect(menu.getByText("Needs a key")).toBeVisible();
  await expect(menu.getByRole("option", { name: /OpenAI/ })).toContainText("Last used 2h ago");
  await expect(menu.getByRole("option", { name: /Claude/ })).toContainText("Not used yet");
  await expect(menu.getByTitle("Key set")).toHaveCount(2); // openai + anthropic dots; zai has none

  // Pick the vendor: blurb + prefilled (editable) endpoint render; status is not-connected.
  await menu.getByRole("option", { name: /Z AI/ }).click();
  await expect(page.getByText(/Uses Z AI's OpenAI-compatible API/)).toBeVisible();
  await expect(page.getByLabel("Endpoint")).toHaveValue("https://api.z.ai/api/paas/v4");
  await expect(page.getByText(/Not connected/)).toBeVisible();

  // Unconfigured providers still preview their curated models (read-only, matrix labels).
  const preview = page.getByTestId("model-preview");
  await expect(preview).toContainText("Included models");
  await expect(preview).toContainText("GLM-5.2 · Z AI");
});

// Token savings (owner ask, 2026-07-17): the card renders under Appearance with the PDF
// fallback segmented control + attach thresholds, and edits POST through.
test("Settings: Token savings card edits PDF fallback and thresholds", async ({ page }) => {
  await page.goto("/");
  await page.getByTestId("account-row").click();
  await page.getByRole("button", { name: "Settings", exact: true }).click();

  const card = page.getByTestId("token-savings-card");
  await expect(card).toBeVisible();
  await expect(card.getByText("Token savings")).toBeVisible();

  // Fallback mode: fixture says "text"; switching marks "Send page images" active.
  const seg = page.getByTestId("pdf-fallback");
  await expect(seg.getByRole("button", { name: "Extract text" })).toHaveClass(/active/);
  const [req] = await Promise.all([
    page.waitForRequest((r) => r.url().endsWith("/v1/settings/pdf") && r.method() === "POST"),
    seg.getByRole("button", { name: "Send page images" }).click(),
  ]);
  expect(req.postDataJSON()).toEqual({ pdf_fallback: "images" });
  await expect(seg.getByRole("button", { name: "Send page images" })).toHaveClass(/active/);

  // Thresholds: fixture starts at 2 pages / 10 MB; editing pages POSTs the clamped value.
  await expect(card.getByTestId("pdf-max-pages")).toHaveValue("2");
  await expect(card.getByTestId("pdf-max-mb")).toHaveValue("10");
  const [req2] = await Promise.all([
    page.waitForRequest((r) => r.url().endsWith("/v1/settings/pdf") && r.method() === "POST"),
    card.getByTestId("pdf-max-pages").fill("30"),
  ]);
  expect(req2.postDataJSON()).toEqual({ pdf_max_pages: 30 });
});
