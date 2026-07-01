import { test, expect } from "./fixtures";

// Guards the per-session directory RO/RW gate (§ roots): the composer's folder popover lists the
// primary writable workspace, and adding a folder is gated read-only by default with an explicit
// "Allow writes" opt-in.
test("roots: add directories with the read-only / read-write gate", async ({ page }) => {
  await page.goto("/");
  await page.getByText("Draft the launch note").first().click();

  // Open the folder popover (icon-only trigger, titled by directory count).
  await page.getByTitle("1 directory the agent can use").click();
  await expect(page.getByText("Directories the agent can use")).toBeVisible();

  // The primary is the writable scratch workspace (Cowork shows it as "Temporary space").
  await expect(page.getByText("Temporary space")).toBeVisible();

  // Add a folder — the gate defaults to read-only (Allow writes OFF).
  await page.getByRole("button", { name: "Give access to a folder" }).click();
  const allowWrites = page.locator(".addfolder-write input[type=checkbox]");
  await expect(allowWrites).not.toBeChecked();
  await page.getByPlaceholder(/Choose or paste a folder path/).fill("/tmp/ro-data");
  await page.getByRole("button", { name: "Add", exact: true }).click();

  const roRow = page.locator(".root-row").filter({ hasText: "/tmp/ro-data" });
  await expect(roRow.getByRole("button", { name: "Read-only" })).toBeVisible();

  // Add another, this time opting into writes → it lands read-write.
  await page.getByRole("button", { name: "Give access to a folder" }).click();
  await page.getByPlaceholder(/Choose or paste a folder path/).fill("/tmp/rw-data");
  await page.locator(".addfolder-write input[type=checkbox]").check();
  await page.getByRole("button", { name: "Add", exact: true }).click();

  const rwRow = page.locator(".root-row").filter({ hasText: "/tmp/rw-data" });
  await expect(rwRow.getByRole("button", { name: "Read-write" })).toBeVisible();

  // Flip the read-only one to read-write via its access button (upsert re-add).
  await roRow.getByRole("button", { name: "Read-only" }).click();
  await expect(roRow.getByRole("button", { name: "Read-write" })).toBeVisible();
});
