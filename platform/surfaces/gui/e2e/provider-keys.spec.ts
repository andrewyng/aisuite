// Settings ▸ Models: the model-provider key flow Rohit tested by hand (add/verify/save keys for
// OpenAI/GLM/Anthropic/DeepSeek). Providers are seeded in three states (OpenAI configured+used,
// Anthropic configured-unused, Z AI unconfigured w/ a prefilled endpoint). The mock's POST
// /v1/providers flips `configured` on save; /verify is a read-only check that fails on a key
// containing "bad".
import { expect } from "@playwright/test";
import { test } from "./fixtures";

async function openModels(page) {
  await page.goto("/");
  await page.getByRole("button", { name: /Settings & more/i }).click();
  await page.getByRole("button", { name: "Settings", exact: true }).click();
  await page.getByRole("button", { name: "Models", exact: true }).click();
  // The API-models pane is the default sub-tab; the provider select is its anchor.
  await expect(page.getByRole("button", { name: "Provider" })).toBeVisible();
}

// Pick a provider from the custom SelectMenu (open the listbox, click the option by name).
async function selectProvider(page, label: string) {
  await page.getByRole("button", { name: "Provider" }).click();
  await page.getByRole("option", { name: new RegExp(label) }).click();
}

test("configured provider (OpenAI) shows connected + last-used", async ({ page }) => {
  await openModels(page);
  // OpenAI is the default selection and is configured + used.
  await expect(page.getByText(/● Connected/)).toBeVisible();
  await expect(page.getByText(/last used/)).toBeVisible();
});

test("unconfigured provider (Z AI) shows not-connected, prefilled endpoint, model preview", async ({
  page,
}) => {
  await openModels(page);
  await selectProvider(page, "Z AI");
  await expect(page.getByText(/● Not connected/)).toBeVisible();
  // The OpenAI-compatible endpoint comes prefilled from the field default.
  await expect(page.getByRole("textbox", { name: "Endpoint" })).toHaveValue(
    "https://api.z.ai/api/paas/v4",
  );
  // What a key unlocks is previewed even before the key is added.
  await expect(page.getByTestId("model-preview")).toContainText("Included models");
});

test("Test button: bad key fails, good key verifies — neither saves", async ({ page }) => {
  await openModels(page);
  await selectProvider(page, "Z AI");
  const key = page.locator('input[type="password"]');

  await key.fill("sk-bad-key");
  await page.getByRole("button", { name: "Test" }).click();
  await expect(page.getByText("Invalid API key.")).toBeVisible();

  await key.fill("sk-good-key");
  await page.getByRole("button", { name: "Test" }).click();
  await expect(page.getByText("✓ Key verified.")).toBeVisible();

  // Still unconfigured — Test never saves.
  await expect(page.getByText(/● Not connected/)).toBeVisible();
});

test("Save a key flips the provider to connected", async ({ page }) => {
  await openModels(page);
  await selectProvider(page, "Z AI");
  const key = page.locator('input[type="password"]');
  await key.fill("sk-glm-realkey");
  await page.getByRole("button", { name: "Save" }).click();

  // Confirmation names the local-only storage; the provider is now connected.
  await expect(page.getByText(/stored locally, never sent to the model|stored locally/)).toBeVisible();
  await expect(page.getByText(/● Connected/)).toBeVisible();
});
