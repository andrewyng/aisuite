// Regression guard (shipped once, 2026-07-09): the cloud sign-in must be
// reachable by a FRESH user. The strip leads the Connectors page (it once sat
// below 25 rows — de-facto invisible), and every signed-out one-click pane
// carries a real Sign-in button, not a hint pointing at another page.
import { expect } from "@playwright/test";
import { test } from "./fixtures";

async function openConnectors(page) {
  await page.goto("/");
  await page.getByRole("button", { name: /Settings & more/i }).click();
  await page.getByRole("button", { name: "Integrations", exact: true }).click();
}

test("the cloud strip leads the connectors list (above the first row)", async ({
  page,
}) => {
  await openConnectors(page);
  const strip = await page.getByTestId("cloud-account").boundingBox();
  const firstRow = await page.getByTestId("connector-browser").boundingBox();
  expect(strip!.y).toBeLessThan(firstRow!.y);
  // Sign out is available right there once signed in.
  await page.getByTestId("cloud-account").getByRole("button", { name: "Sign in" }).click();
  await expect(page.getByTestId("cloud-account")).toContainText("rohit@opencoworker.app");
  await expect(
    page.getByTestId("cloud-account").getByRole("button", { name: "Sign out" }),
  ).toBeVisible();
});

test("signed-out one-click pane signs in inline, then connects", async ({ page }) => {
  await openConnectors(page);
  // Fresh user path: Available → Connect → the pane must offer sign-in itself.
  await page
    .getByTestId("connector-gmail")
    .getByRole("button", { name: "Connect", exact: true })
    .click();
  await page.getByTestId("inline-cloud-sign-in").click();
  // The mock signs in instantly; the section's poll re-renders the pane armed.
  await expect(
    page.getByRole("button", { name: /Connect Gmail with one click/i }),
  ).toBeVisible({ timeout: 10_000 });
});
