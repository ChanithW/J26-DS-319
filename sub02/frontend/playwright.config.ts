import { defineConfig } from "@playwright/test";

// The test runs against the real backend (with the trained bundle) and the dev frontend.
export default defineConfig({
  testDir: "./e2e",
  timeout: 90_000,
  retries: 0,
  reporter: "list",
  // PW_CHANNEL=chromium uses Playwright's own download; the default uses the Edge that ships with Windows.
  use: {
    baseURL: "http://localhost:5173",
    headless: true,
    acceptDownloads: true,
    channel: process.env.PW_CHANNEL === "chromium" ? undefined : (process.env.PW_CHANNEL ?? "msedge"),
  },
  webServer: [
    {
      command: "python -m uvicorn component2.api:app --port 8000",
      cwd: "..",
      url: "http://localhost:8000/api/v1/health",
      reuseExistingServer: true,
      timeout: 60_000,
      env: { DATABASE_URL: "sqlite:///./e2e_runs.db" },
    },
    {
      command: "npm run dev -- --port 5173",
      url: "http://localhost:5173",
      reuseExistingServer: true,
      timeout: 60_000,
    },
  ],
});
