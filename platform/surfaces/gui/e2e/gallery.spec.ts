// Settings ▸ Gallery: sign-in gated browsing + install of curated personas.
import { expect } from "@playwright/test";
import { test } from "./fixtures";

async function openGallery(page) {
  await page.goto("/");
  await page.getByRole("button", { name: /Settings & more/i }).click();
  await page.getByRole("button", { name: "Settings", exact: true }).click();
  await page.getByRole("button", { name: "Gallery", exact: true }).click();
}

test("signed out: gallery prompts for sign-in, local installs unaffected", async ({ page }) => {
  await openGallery(page);
  const prompt = page.getByTestId("gallery-signin");
  await expect(prompt).toContainText("needs a (free) cloud sign-in");
  await expect(prompt).toContainText("always works without an account");
  await expect(prompt.getByRole("button", { name: "Sign in" })).toBeVisible();
});

test("signed in: cards render and install reports the consent handoff", async ({ page }) => {
  await openGallery(page);
  await page.getByTestId("gallery-signin").getByRole("button", { name: "Sign in" }).click();

  const card = page.getByTestId("gallery-sales");
  await expect(card).toBeVisible({ timeout: 10_000 });
  await expect(card).toContainText("Sales Coworker");
  await expect(card).toContainText("hubspot");

  await card.getByRole("button", { name: "Install" }).click();
  await expect(page.getByTestId("gallery-cards")).toContainText(
    "review and enable it under Settings ▸ Personas",
  );
});
