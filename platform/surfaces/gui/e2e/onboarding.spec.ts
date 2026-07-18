// First-run onboarding (UX-DECISIONS §24, restructured by §29): model → your tools → go.
// Only the model step gates; the tools page is a value-framed cloud sign-in (skippable — the
// lazy first-connect sign-in stays for skippers); the done page routes via two CTAs. The recipe
// machinery moved to the Automations quickstart (automations-quickstart.spec.ts). Entered here
// via the REPLAY path (Settings ▸ Appearance ▸ "Run setup again") — which is itself under test.
import { expect } from "@playwright/test";
import { test } from "./fixtures";

async function openOnboarding(page) {
  await page.goto("/");
  await page.getByTestId("account-row").click();
  await page.getByTestId("account-menu").getByRole("button", { name: "Settings" }).click();
  await page.getByRole("button", { name: "Run setup again" }).click();
  await expect(page.getByTestId("ob-step-model")).toBeVisible();
}

test("model step: configured provider fast-path; Continue verifies automatically", async ({
  page,
}) => {
  await openOnboarding(page);

  // The configured provider (OpenAI in fixtures) is auto-picked: no re-typing a key. The
  // Test affordance lives IN the key field (owner call, DMG #28) and reads ✓ when a key is
  // already saved; OpenAI's optional endpoint hides behind the disclosure link even though
  // it has no vendor default.
  await expect(page.getByTestId("ob-test")).toHaveText("✓");
  await expect(page.getByTestId("ob-field-base_url")).toHaveCount(0);
  await expect(page.getByTestId("ob-continue")).toBeEnabled();

  // Switching to an unconfigured vendor: Continue gates only on the key being FILLED — the
  // verify runs automatically on click (tester catch 2026-07-12: the manual Test-then-Continue
  // two-step read as a puzzle). The picker is the Settings-Models SelectMenu.
  await page.getByRole("button", { name: "Provider" }).click();
  await page.getByRole("option", { name: "Z AI (GLM)" }).click();
  await expect(page.getByTestId("ob-test")).toHaveText("Test");
  await expect(page.getByTestId("ob-continue")).toBeDisabled();
  // The optional endpoint is an expert option: hidden behind "Configure custom endpoint",
  // prefilled with the vendor default once revealed.
  await expect(page.getByTestId("ob-field-base_url")).toHaveCount(0);
  await page.getByTestId("ob-endpoint-link").click();
  await expect(page.getByTestId("ob-field-base_url")).toHaveValue(/api\.z\.ai/);

  // A bad key: Continue runs the check itself, fails, and STAYS on the step with the error.
  await page.getByTestId("ob-field-api_key").fill("bad-key");
  await expect(page.getByTestId("ob-continue")).toBeEnabled();
  await page.getByTestId("ob-continue").click();
  await expect(page.getByTestId("ob-step-model")).toBeVisible();
  await expect(page.getByTestId("ob-continue")).toBeEnabled();

  // A good key: the in-field Test flips to a tick, and Continue advances (it verifies
  // by itself too — Test stays optional).
  await page.getByTestId("ob-field-api_key").fill("zk-good");
  await page.getByTestId("ob-test").click();
  await expect(page.getByTestId("ob-test")).toHaveText("✓");
  await page.getByTestId("ob-continue").click();
  await expect(page.getByTestId("ob-step-tools")).toBeVisible();
});

test("model step: switching providers keeps unsaved input and shows the connected state", async ({
  page,
}) => {
  await openOnboarding(page);

  // The configured provider (OpenAI) says so ON the form — the stored key is never echoed
  // back, so without this line the empty password field read as "not set up" (owner
  // complaint 2026-07-16).
  await expect(page.getByTestId("ob-provider-connected")).toBeVisible();
  await expect(page.getByTestId("ob-field-api_key")).toHaveAttribute(
    "placeholder",
    /key saved/,
  );

  // Type a key on another vendor, peek back at OpenAI, return: the draft survives the
  // round trip (it used to be silently blanked) and OpenAI still reads connected.
  await page.getByRole("button", { name: "Provider" }).click();
  await page.getByRole("option", { name: "Z AI (GLM)" }).click();
  await expect(page.getByTestId("ob-provider-connected")).toHaveCount(0);
  await page.getByTestId("ob-field-api_key").fill("zk-draft");
  await page.getByRole("button", { name: "Provider" }).click();
  await page.getByRole("option", { name: "OpenAI" }).click();
  await expect(page.getByTestId("ob-provider-connected")).toBeVisible();
  await page.getByRole("button", { name: "Provider" }).click();
  await page.getByRole("option", { name: "Z AI (GLM)" }).click();
  await expect(page.getByTestId("ob-field-api_key")).toHaveValue("zk-draft");
});

test("tools page: out-of-band sign-in flips to the signed-in state; the automation CTA lands on the quickstart", async ({
  page,
}) => {
  await openOnboarding(page);
  await page.getByTestId("ob-continue").click();
  await expect(page.getByTestId("ob-step-tools")).toBeVisible();

  // §29's tools page (2026-07-16 owner redesign): the value is the headline, ONE primary
  // action — Sign in sits in the footer slot and Continue only replaces it once signed in.
  // The manual-keys path is spelled out inside the "Secure by design" card, and the sign-in
  // lands out-of-band (browser flow), flipping to the ✓ state.
  await expect(page.getByText("Secure by design")).toBeVisible();
  await expect(page.getByText("Prefer manual setup?")).toBeVisible();
  await expect(page.getByTestId("ob-continue-tools")).toHaveCount(0);
  await page.getByTestId("ob-cloud-signin").click();
  await expect(page.getByTestId("ob-tools-signedin")).toBeVisible({ timeout: 10_000 });
  await expect(page.getByTestId("ob-tools-signedin")).toContainText("rohit@openworker.com");
  await page.getByTestId("ob-continue-tools").click();

  // Done step: the automation CTA lands on the Automations quickstart.
  await expect(page.getByTestId("ob-step-done")).toBeVisible();
  await page.getByTestId("ob-cta-automation").click();
  await expect(page.getByTestId("onboarding")).toHaveCount(0);
  await expect(page.getByRole("heading", { name: "Automations" })).toBeVisible();
  // Fixtures seed a task, so the list isn't empty — the quickstart is one toggle away (a real
  // first run lands on the empty state, which shows it directly).
  await page.getByRole("button", { name: "+ New automation" }).click();
  await expect(page.getByText("Start from a template")).toBeVisible();
});

test("tools page skips cleanly; Start working lands in a session with the panel open", async ({
  page,
}) => {
  await openOnboarding(page);
  await page.getByTestId("ob-continue").click();
  await page.getByTestId("ob-tools-skip").click();
  await expect(page.getByTestId("ob-step-done")).toBeVisible();
  await page.getByTestId("ob-start").click();
  await expect(page.getByTestId("onboarding")).toHaveCount(0);
  // §32: "Start working" lands with the rail's Access section expanded (the drawer is gone).
  await expect(page.getByRole("region", { name: "Session access" })).toBeVisible();
});
