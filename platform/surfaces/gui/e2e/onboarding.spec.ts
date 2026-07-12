// First-run onboarding (UX-DECISIONS §24): model → recipe → tips. Only step 1 gates;
// the recipe's consent line mints a §25 standing grant; "Start working" lands in a fresh
// session with the Session settings panel open. Entered here via the REPLAY path
// (Settings ▸ Appearance ▸ "Run setup again") — which is itself under test.
import { expect } from "@playwright/test";
import { test } from "./fixtures";

async function openOnboarding(page) {
  await page.goto("/");
  await page.getByTestId("account-row").click();
  await page.getByTestId("account-menu").getByRole("button", { name: "Settings" }).click();
  await page.getByRole("button", { name: "Run setup again" }).click();
  await expect(page.getByTestId("ob-step-model")).toBeVisible();
}

test("model step: configured provider fast-path; unconfigured gates until Test passes", async ({
  page,
}) => {
  await openOnboarding(page);

  // The configured provider (OpenAI in fixtures) is auto-picked: no re-typing a key. The
  // success state lives ON the Test button (no separate status line), and OpenAI's optional
  // endpoint hides behind the disclosure link even though it has no vendor default.
  await expect(page.getByTestId("ob-test")).toHaveText("✓ Connected");
  await expect(page.getByTestId("ob-field-base_url")).toHaveCount(0);
  await expect(page.getByTestId("ob-continue")).toBeEnabled();

  // Switching to an unconfigured vendor gates Continue (and the button reads Test again).
  // The picker is the same SelectMenu the Settings Models page uses (custom listbox).
  await page.getByRole("button", { name: "Provider" }).click();
  await page.getByRole("option", { name: "Z AI (GLM)" }).click();
  await expect(page.getByTestId("ob-test")).toHaveText("Test");
  await expect(page.getByTestId("ob-continue")).toBeDisabled();
  // The optional endpoint is an expert option: hidden behind "Configure custom endpoint",
  // prefilled with the vendor default once revealed.
  await expect(page.getByTestId("ob-field-base_url")).toHaveCount(0);
  await page.getByTestId("ob-endpoint-link").click();
  await expect(page.getByTestId("ob-field-base_url")).toHaveValue(/api\.z\.ai/);

  // Bad key fails the live check; a good key verifies and unlocks Continue.
  await page.getByTestId("ob-field-api_key").fill("bad-key");
  await page.getByTestId("ob-test").click();
  await expect(page.getByTestId("ob-continue")).toBeDisabled();
  await page.getByTestId("ob-field-api_key").fill("zk-good");
  await page.getByTestId("ob-test").click();
  await expect(page.getByTestId("ob-test")).toHaveText("✓ Connected");
  await expect(page.getByTestId("ob-continue")).toBeEnabled();
});

test("recipe: lazy single sign-in completes the pending connect; consent mints the grant; Start working opens the session panel", async ({
  page,
}) => {
  await openOnboarding(page);
  await page.getByTestId("ob-continue").click();
  await expect(page.getByTestId("ob-step-recipe")).toBeVisible();

  // Sales tab: Slack is already connected in fixtures; HubSpot isn't. No recipe yet.
  await page.getByTestId("ob-tab-sales").click();
  await expect(page.getByText("✓ Connected").first()).toBeVisible();
  await expect(page.getByTestId("ob-recipe")).toHaveCount(0);

  // Connect HubSpot while signed out → the ONE cloud pane appears; signing in finishes the
  // pending connect without another click.
  await page.getByTestId("ob-connect-hubspot").click();
  await expect(page.getByTestId("ob-cloudpane")).toBeVisible();
  await page.getByTestId("ob-cloud-signin").click();
  await expect(page.getByTestId("ob-recipe")).toBeVisible({ timeout: 15_000 });

  // The recipe assembles: channel picked BY NAME (the box shows #name; the raw address is
  // only the stored target), day+time cadence, and the §25 consent (pre-checked).
  const chan = page.locator('[data-testid="ob-channel"] input');
  await chan.click();
  await page.getByTestId("channel-suggestions").getByText("#ocw-test").click();
  await expect(chan).toHaveValue("#ocw-test");
  await expect(page.getByTestId("ob-consent")).toBeChecked();
  await page.getByTestId("ob-create").click();

  // Done step: recap card, then Start working lands in a session with the panel open.
  await expect(page.getByTestId("ob-recap")).toContainText("Pipeline digest");
  await expect(page.getByTestId("ob-recap")).toContainText("Mondays at 09:00");
  await page.getByTestId("ob-start").click();
  await expect(page.getByTestId("onboarding")).toHaveCount(0);
  await expect(page.getByRole("dialog", { name: "Session settings" })).toBeVisible();
});

test("everyday tab: read-only recipe carries disclosure, not a grant", async ({ page }) => {
  await openOnboarding(page);
  await page.getByTestId("ob-continue").click();
  await page.getByTestId("ob-tab-everyday").click();

  // Calendar + Gmail rows; no consent checkbox — reads never gate, the copy says so.
  await expect(page.getByText("Today's meetings and gaps")).toBeVisible();
  await expect(page.getByText("What arrived overnight")).toBeVisible();
  await expect(page.getByTestId("ob-consent")).toHaveCount(0);
});
