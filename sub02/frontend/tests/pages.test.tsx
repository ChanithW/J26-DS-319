import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { HttpResponse, http } from "msw";

import { AppRoutes } from "../src/App";
import { API, server } from "./mocks";
import { renderWithProviders } from "./utils";

describe("limitations banner", () => {
  it.each(["/", "/runs", "/runs/abc", "/model", "/runs/latest"])("is present on %s", async (route) => {
    renderWithProviders(<AppRoutes />, route);
    const banner = await screen.findByTestId("limitations-banner");
    expect(banner).toHaveTextContent("Trained on solar flares.");
    expect(banner).toHaveTextContent("Accuracy on stars has not been validated.");
    expect(banner).toHaveTextContent("ranking score, not a chance of a CME");
    expect(within(banner).getByRole("link", { name: /limitations/i })).toHaveAttribute("href", "/model");
    expect(within(banner).queryByRole("button")).not.toBeInTheDocument();        // not dismissible
  });
});

describe("Model info page", () => {
  it("shows the metrics returned by the API", async () => {
    renderWithProviders(<AppRoutes />, "/model");
    const table = await screen.findByTestId("metrics-table");
    expect(table).toHaveTextContent("0.149");     // cross-validation TSS from the mocked API
    expect(table).toHaveTextContent("0.058");
    expect(screen.getByText("Trained on solar flares only.")).toBeInTheDocument();
    expect(screen.getByText("How to read an output")).toBeInTheDocument();
  });

  it("shows an error with a retry when the API fails", async () => {
    server.use(http.get(`${API}/model-info`, () => HttpResponse.json({ error: { code: "x", message: "boom" } }, { status: 500 })));
    renderWithProviders(<AppRoutes />, "/model");
    expect(await screen.findByRole("alert")).toHaveTextContent("boom");
    expect(screen.getByRole("button", { name: "Try again" })).toBeInTheDocument();
  });
});

describe("Predict page", () => {
  const upload = async (content = "a,b\n1,2\n") => {
    renderWithProviders(<AppRoutes />, "/");
    const file = new File([content], "bad.csv", { type: "text/csv" });
    await userEvent.upload(screen.getByLabelText("CSV file"), file);
    await userEvent.click(screen.getByRole("button", { name: "Run prediction" }));
  };

  it("shows CSV errors from the API next to the upload box", async () => {
    server.use(http.post(`${API}/predict/file`, () =>
      HttpResponse.json({ error: { code: "invalid_csv", message: "Missing column(s): flare_id.", details: null } }, { status: 422 })));
    await upload();
    const box = screen.getByTestId("dropzone");
    await waitFor(() => expect(within(box).getByRole("alert")).toHaveTextContent("Missing column(s): flare_id."));
  });

  it("keeps Run disabled until a file or the sample is chosen", () => {
    renderWithProviders(<AppRoutes />, "/");
    expect(screen.getByRole("button", { name: "Run prediction" })).toBeDisabled();
  });

  it("says so clearly when the API cannot be reached", async () => {
    server.use(http.post(`${API}/predict/file`, () => HttpResponse.error()));
    await upload();
    expect(await screen.findByText("The server can't be reached")).toBeInTheDocument();
  });

  it("loads the sample flares", async () => {
    server.use(http.get(`${API}/samples`, () => HttpResponse.json(
      Array.from({ length: 33 }, (_, i) => ({ flare_id: `TESS_${i}`, star_id: i, tpeak: 0, time: [0], flux: [1] })))));
    renderWithProviders(<AppRoutes />, "/");
    await userEvent.click(screen.getByRole("button", { name: "Use the 33 sample TESS flares" }));
    expect(await screen.findByText("33 sample flares loaded.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Run prediction" })).toBeEnabled();
  });
});

describe("Runs pages", () => {
  it("shows an empty state with no saved runs", async () => {
    renderWithProviders(<AppRoutes />, "/runs");
    expect(await screen.findByText("No saved runs yet")).toBeInTheDocument();
  });

  it("shows a saved run's results", async () => {
    renderWithProviders(<AppRoutes />, "/runs/abc");
    expect(await screen.findByTestId("summary-strip")).toBeInTheDocument();
    expect(screen.getAllByTestId("result-row")).toHaveLength(1);
  });

  it("shows a clear message for an unknown run", async () => {
    server.use(http.get(`${API}/runs/nope`, () => HttpResponse.json({ error: { code: "run_not_found", message: "That run does not exist." } }, { status: 404 })));
    renderWithProviders(<AppRoutes />, "/runs/nope");
    expect(await screen.findByText("This run is not available")).toBeInTheDocument();
  });
});
