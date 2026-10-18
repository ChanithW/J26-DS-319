import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { HttpResponse, http } from "msw";

import { AppRoutes } from "../src/App";
import { buildFlare, exampleFlare, parsePoints } from "../src/lib/manual";
import { API, flare, run, server } from "./mocks";
import { renderWithProviders } from "./utils";

const pairs = (n: number, start = -20, step = 2) =>
  Array.from({ length: n }, (_, i) => `${start + i * step}, ${(1 + i / 100).toFixed(3)}`).join("\n");

describe("parsePoints", () => {
  it("accepts commas, tabs, spaces and semicolons", () => {
    const r = parsePoints("1,2\n3\t4\n5 6\n7;8");
    expect(r.errors).toEqual([]);
    expect(r.time).toEqual([1, 3, 5, 7]);
    expect(r.flux).toEqual([2, 4, 6, 8]);
  });

  it("skips a text header row and blank lines", () => {
    const r = parsePoints("time,flux\n\n1,2\n3,4\n");
    expect(r.errors).toEqual([]);
    expect(r.time).toHaveLength(2);
  });

  it("reports the line number of a bad row", () => {
    const r = parsePoints("1,2\nabc,4\n5,6,7");
    expect(r.errors[0]).toMatch(/^Line 2:/);
    expect(r.errors[1]).toMatch(/^Line 3:/);
  });

  it("accepts scientific notation and negative numbers", () => {
    const r = parsePoints("-20, 1.5e-3\n0, 2E2");
    expect(r.errors).toEqual([]);
    expect(r.flux).toEqual([0.0015, 200]);
  });
});

describe("buildFlare", () => {
  const ok = { flareId: "F1", starId: "123", unit: "minutes" as const, tpeak: "", points: pairs(30) };

  it("builds a record, converting minutes from the peak to days with tpeak 0", () => {
    const r = buildFlare(ok);
    expect(r.errors).toEqual([]);
    expect(r.flare).toMatchObject({ flare_id: "F1", star_id: 123, tpeak: 0 });
    expect(r.flare!.time![0]).toBeCloseTo(-20 / 1440, 10);
    expect(r.flare!.flux).toHaveLength(30);
  });

  it("keeps BTJD times as they are and needs a peak time", () => {
    const bad = buildFlare({ ...ok, unit: "btjd", tpeak: "" });
    expect(bad.errors.join(" ")).toMatch(/peak time in BTJD/);
    const good = buildFlare({ ...ok, unit: "btjd", tpeak: "1500.5", points: pairs(30, 1499.9, 0.0014) });
    expect(good.errors).toEqual([]);
    expect(good.flare!.tpeak).toBe(1500.5);
  });

  it("rejects an empty id, a non-integer star id, duplicates and too few points", () => {
    expect(buildFlare({ ...ok, flareId: " " }).errors).toContain("Enter a flare id.");
    expect(buildFlare({ ...ok, starId: "12.5" }).errors.join(" ")).toMatch(/whole number/);
    expect(buildFlare(ok, ["F1"]).errors.join(" ")).toMatch(/already in the list/);
    expect(buildFlare({ ...ok, points: "1,2" }).errors.join(" ")).toMatch(/at least two/);
  });

  it("warns, but still allows, fewer than 20 points in the window", () => {
    const r = buildFlare({ ...ok, points: pairs(10) });
    expect(r.flare).toBeDefined();
    expect(r.warnings.join(" ")).toMatch(/Only 10 point/);
  });

  it("the demo flare is valid and has enough points", () => {
    const r = buildFlare(exampleFlare());
    expect(r.errors).toEqual([]);
    expect(r.warnings).toEqual([]);
  });
});

describe("manual entry on the Predict page", () => {
  const fill = async (id: string, points: string) => {
    const form = screen.getByTestId("manual-form");
    await userEvent.type(within(form).getByLabelText("Flare id"), id);
    await userEvent.type(within(form).getByLabelText("Star id (TIC number)"), "777");
    await userEvent.click(within(form).getByLabelText(/Brightness curve/));
    await userEvent.paste(points);
    await userEvent.click(within(form).getByRole("button", { name: "Add this flare" }));
  };

  it("shows the form beside the CSV upload", () => {
    renderWithProviders(<AppRoutes />, "/");
    expect(screen.getByTestId("dropzone")).toBeInTheDocument();
    expect(screen.getByTestId("manual-form")).toBeInTheDocument();
  });

  it("shows errors for bad input and does not add the flare", async () => {
    renderWithProviders(<AppRoutes />, "/");
    await userEvent.click(within(screen.getByTestId("manual-form")).getByRole("button", { name: "Add this flare" }));
    const alert = await screen.findByRole("alert");
    expect(alert).toHaveTextContent("Enter a flare id.");
    expect(alert).toHaveTextContent("Star id must be a whole number");
    expect(screen.queryByTestId("manual-list")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Run prediction" })).toBeDisabled();
  });

  it("adds flares to a list, lets you remove one, and runs the rest", async () => {
    let body: { flares: { flare_id: string; star_id: number }[]; save_run: boolean } | null = null;
    server.use(http.post(`${API}/predict`, async ({ request }) => {
      body = (await request.json()) as typeof body;
      return HttpResponse.json(run([flare()], { run_id: "abc" }));
    }));
    renderWithProviders(<AppRoutes />, "/");

    await fill("FLARE_A", pairs(30));
    await fill("FLARE_B", pairs(30));
    const list = await screen.findByTestId("manual-list");
    expect(list).toHaveTextContent("FLARE_A");
    expect(list).toHaveTextContent("FLARE_B");
    expect(screen.getByTestId("will-run")).toHaveTextContent("2 manually entered flares");

    await userEvent.click(screen.getByRole("button", { name: "Remove FLARE_A" }));
    expect(screen.getByTestId("will-run")).toHaveTextContent("1 manually entered flare");

    await userEvent.click(screen.getByRole("button", { name: "Run prediction" }));
    await waitFor(() => expect(body).not.toBeNull());
    expect(body!.flares).toHaveLength(1);
    expect(body!.flares[0]).toMatchObject({ flare_id: "FLARE_B", star_id: 777, tpeak: 0 });
    expect(body!.save_run).toBe(true);
    expect(await screen.findByTestId("summary-strip")).toBeInTheDocument();          // moved on to the results
  });

  it("rejects a duplicate flare id", async () => {
    renderWithProviders(<AppRoutes />, "/");
    await fill("SAME", pairs(30));
    await fill("SAME", pairs(30));
    expect(await screen.findByRole("alert")).toHaveTextContent("already in the list");
  });

  it("fills the demo flare and adds it", async () => {
    renderWithProviders(<AppRoutes />, "/");
    const form = screen.getByTestId("manual-form");
    await userEvent.click(within(form).getByRole("button", { name: "Fill with a demo flare" }));
    expect(within(form).getByLabelText("Flare id")).toHaveValue("EXAMPLE_FLARE");
    await userEvent.click(within(form).getByRole("button", { name: "Add this flare" }));
    expect(await screen.findByTestId("manual-list")).toHaveTextContent("EXAMPLE_FLARE · star 1 · 55 points");
  });

  it("the input chosen last is the one that runs", async () => {
    server.use(http.get(`${API}/samples`, () => HttpResponse.json(
      Array.from({ length: 3 }, (_, i) => ({ flare_id: `S${i}`, star_id: i, tpeak: 0, time: [0], flux: [1] })))));
    renderWithProviders(<AppRoutes />, "/");
    await fill("MINE", pairs(30));
    expect(screen.getByTestId("will-run")).toHaveTextContent("1 manually entered flare");
    await userEvent.click(screen.getByRole("button", { name: "Use the 33 sample TESS flares" }));
    await waitFor(() => expect(screen.getByTestId("will-run")).toHaveTextContent("the 3 sample TESS flares"));
  });
});

describe("manual form: hints, spacing warning and pasted spreadsheets", () => {
  const ok = { flareId: "F1", starId: "123", unit: "minutes" as const, tpeak: "" };

  it("shows the two helper hints", () => {
    renderWithProviders(<AppRoutes />, "/");
    const hints = screen.getByTestId("manual-hints");
    expect(hints).toHaveTextContent("Time 0 must be the flare's brightest point (within about 6 minutes).");
    expect(hints).toHaveTextContent("Include at least 3 points before -8 minutes (they set the background).");
  });

  it("warns, without blocking, when the spacing is not about 2 minutes", () => {
    const slow = buildFlare({ ...ok, points: pairs(30, -20, 10) });          // 10-minute steps
    expect(slow.flare).toBeDefined();
    expect(slow.warnings.join(" ")).toMatch(/spacing between your points is about 10\.0 minutes/);
    expect(slow.warnings.join(" ")).toMatch(/does not re-bin/);
    const fast = buildFlare({ ...ok, points: pairs(60, -20, 0.5) });         // 30-second steps
    expect(fast.warnings.join(" ")).toMatch(/about 30 seconds/);
    expect(buildFlare({ ...ok, points: pairs(30, -20, 2) }).warnings).toEqual([]);   // 2 minutes: no warning
    expect(buildFlare({ ...ok, points: pairs(30, -20, 2.4) }).warnings.join(" ")).not.toMatch(/spacing/);   // within tolerance
  });

  it("accepts a tab-separated spreadsheet paste through the form", async () => {
    renderWithProviders(<AppRoutes />, "/");
    const form = screen.getByTestId("manual-form");
    await userEvent.type(within(form).getByLabelText("Flare id"), "SHEET");
    await userEvent.type(within(form).getByLabelText("Star id (TIC number)"), "5");
    await userEvent.click(within(form).getByLabelText(/Brightness curve/));
    await userEvent.paste(Array.from({ length: 30 }, (_, i) => `${-20 + i * 2}\t${(1 + i / 100).toFixed(3)}`).join("\n"));
    await userEvent.click(within(form).getByRole("button", { name: "Add this flare" }));
    expect(await screen.findByTestId("manual-list")).toHaveTextContent("SHEET · star 5 · 30 points");
  });

  it("accepts tabs, spaces and commas mixed in one paste", () => {
    const r = parsePoints("-20\t1.0\n-18 1.1\n-16,1.2");
    expect(r.errors).toEqual([]);
    expect(r.time).toEqual([-20, -18, -16]);
  });
});
