// Cloud sign-in + managed one-click connectors (Integrations ▸ Connectors).
// Product invariant under test: manual token setup is always present; managed
// one-click is an ADDITION that appears only when signed in.
import { expect } from "@playwright/test";
import { test } from "./fixtures";

async function openConnectors(page) {
  await page.goto("/");
  await page.getByRole("button", { name: /Settings & more/i }).click();
  await page.getByRole("button", { name: "Integrations", exact: true }).click();
  await expect(page.getByTestId("cloud-account")).toBeVisible();
}

test("signed out: account card offers sign-in; managed connector still connects manually", async ({
  page,
}) => {
  await openConnectors(page);

  const account = page.getByTestId("cloud-account");
  await expect(account).toContainText("Manual token setup always works");
  await expect(account.getByRole("button", { name: "Sign in" })).toBeVisible();

  // open the managed-capable connector: hint + manual fields, no one-click button
  const gmail = page.getByTestId("connector-gmail");
  await gmail.getByRole("button", { name: "Connect" }).click();
  const setup = page.getByTestId("managed-connect");
  await expect(setup).toContainText("Sign in to OpenCoworker Cloud");
  await expect(gmail.locator("input[type=password]")).toBeVisible(); // manual field rendered
  await expect(page.getByRole("button", { name: /one click/i })).toHaveCount(0);
});

test("signed in: one-click connect appears, manual fields remain", async ({ page }) => {
  await openConnectors(page);

  await page.getByTestId("cloud-account").getByRole("button", { name: "Sign in" }).click();
  await expect(page.getByTestId("cloud-account")).toContainText("rohit@opencoworker.app", {
    timeout: 10_000,
  });

  const gmail = page.getByTestId("connector-gmail");
  await gmail.getByRole("button", { name: "Connect", exact: true }).click();
  await expect(page.getByRole("button", { name: /Connect Gmail with one click/i })).toBeVisible();
  // the manual path must still be offered alongside
  await expect(page.getByTestId("managed-connect")).toContainText("or connect manually");

  // sign out flips back without breaking the card
  await page.getByRole("button", { name: "Sign out" }).click();
  await expect(
    page.getByTestId("cloud-account").getByRole("button", { name: "Sign in" }),
  ).toBeVisible();
});
