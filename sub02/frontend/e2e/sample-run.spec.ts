import { expect, test } from "@playwright/test";
import { readFileSync } from "node:fs";

test("sample flares: run, see 33 rows, open the top flare, download the hand-off CSV", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByTestId("limitations-banner")).toBeVisible();

  await page.getByRole("button", { name: "Use the 33 sample TESS flares" }).click();
  await expect(page.getByText("33 sample flares loaded.")).toBeVisible();
  await page.getByRole("button", { name: "Run prediction" }).click();

  await expect(page).toHaveURL(/\/runs\/.+/);
  await expect(page.getByTestId("result-row")).toHaveCount(33);
  await expect(page.getByTestId("limitations-banner")).toBeVisible();

  // top-ranked flare: the table is sorted by probability, highest first
  await page.getByTestId("result-row").first().getByRole("button").click();
  const dialog = page.getByRole("dialog");
  await expect(dialog).toBeVisible();
  await expect(dialog.getByTestId("label-badge")).toBeVisible();
  await expect(dialog.getByTestId("reliability-badge")).toBeVisible();   // a label is never shown alone
  await expect(dialog.getByTestId("prob-text")).toContainText("%");
  await expect(dialog.getByText("Light curve")).toBeVisible();
  await page.keyboard.press("Escape");
  await expect(dialog).toBeHidden();

  const [download] = await Promise.all([
    page.waitForEvent("download"),
    page.getByRole("button", { name: "Download hand-off CSV" }).click(),
  ]);
  const csv = readFileSync(await download.path(), "utf-8").trim().split("\n");
  expect(csv[0]).toBe("flare_id,star_id,calibrated_probability,confidence_lower,confidence_upper,predicted_label,reliability");
  expect(csv).toHaveLength(34);                                          // header + 33 flares
});
