import { screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { ResultsTable } from "../src/components/ResultsTable";
import { flare, notOk } from "./mocks";
import { renderWithProviders } from "./utils";

const results = [
  flare({ flare_id: "TESS_B", star_id: 20, calibrated_probability: 0.3, confidence_lower: 0.2, confidence_upper: 0.4, predicted_label: "eruptive", reliability: "uncertain" }),
  flare({ flare_id: "TESS_A", star_id: 10, calibrated_probability: 0.5, confidence_lower: 0.3, confidence_upper: 0.6, predicted_label: "eruptive", reliability: "confident" }),
  flare({ flare_id: "TESS_C", star_id: 30, calibrated_probability: 0.1, confidence_lower: 0.05, confidence_upper: 0.2, predicted_label: "confined", reliability: "confident" }),
  notOk("TESS_BAD", "too_few_points"),
];

const ids = () => screen.getAllByTestId("result-row").map((r) => within(r).getAllByRole("cell")[1].textContent);

describe("ResultsTable", () => {
  it("sorts by probability, highest first, by default", () => {
    renderWithProviders(<ResultsTable results={results} />);
    expect(ids()).toEqual(["TESS_A", "TESS_B", "TESS_C", "TESS_BAD"]);   // not-scored rows last
  });

  it("sorts on any column and toggles direction", async () => {
    renderWithProviders(<ResultsTable results={results} />);
    await userEvent.click(screen.getByRole("button", { name: /^Flare/ }));
    expect(ids()).toEqual(["TESS_A", "TESS_B", "TESS_BAD", "TESS_C"]);
    await userEvent.click(screen.getByRole("button", { name: /^Flare/ }));
    expect(ids()).toEqual(["TESS_C", "TESS_BAD", "TESS_B", "TESS_A"]);
    expect(screen.getByRole("columnheader", { name: /^Flare/ })).toHaveAttribute("aria-sort", "descending");
  });

  it("filters by label, reliability and status, and searches by id", async () => {
    renderWithProviders(<ResultsTable results={results} />);
    await userEvent.selectOptions(screen.getByLabelText("Filter by label"), "confined");
    expect(ids()).toEqual(["TESS_C"]);
    await userEvent.selectOptions(screen.getByLabelText("Filter by label"), "all");
    await userEvent.selectOptions(screen.getByLabelText("Filter by reliability"), "uncertain");
    expect(ids()).toEqual(["TESS_B"]);
    await userEvent.selectOptions(screen.getByLabelText("Filter by reliability"), "all");
    await userEvent.selectOptions(screen.getByLabelText("Filter by status"), "notok");
    expect(ids()).toEqual(["TESS_BAD"]);
    await userEvent.selectOptions(screen.getByLabelText("Filter by status"), "all");
    await userEvent.type(screen.getByRole("searchbox"), "30");
    expect(ids()).toEqual(["TESS_C"]);
    await userEvent.clear(screen.getByRole("searchbox"));
    await userEvent.type(screen.getByRole("searchbox"), "nothing-matches");
    expect(screen.getByText("No flares match these filters")).toBeInTheDocument();
  });

  it("greys out a not-scored row and shows the reason", () => {
    renderWithProviders(<ResultsTable results={results} />);
    const row = screen.getAllByTestId("result-row").find((r) => r.dataset.status === "too_few_points")!;
    expect(row.className).toMatch(/opacity-70/);
    expect(row).toHaveTextContent("fewer than 20 data points");
  });

  it("always shows the probability with its range", () => {
    renderWithProviders(<ResultsTable results={results} />);
    expect(screen.getByText("50% (30–60%)")).toBeInTheDocument();
    expect(screen.getByText("10% (5–20%)")).toBeInTheDocument();
    expect(screen.queryByText(/^\d+%$/)).not.toBeInTheDocument();         // no bare percentage anywhere
  });

  it("opens the detail drawer with label and reliability together, and moves with arrow keys", async () => {
    renderWithProviders(<ResultsTable results={results} />);
    await userEvent.click(screen.getByRole("button", { name: "TESS_A" }));
    const dialog = screen.getByRole("dialog");
    expect(within(dialog).getByTestId("label-badge")).toBeInTheDocument();
    expect(within(dialog).getByTestId("reliability-badge")).toBeInTheDocument();
    expect(within(dialog).getByTestId("prob-text")).toHaveTextContent("50% (30–60%)");
    expect(dialog).toHaveAccessibleName(/TESS_A/);
    await userEvent.keyboard("{ArrowRight}");
    expect(screen.getByRole("dialog")).toHaveAccessibleName(/TESS_B/);
    await userEvent.keyboard("{Escape}");
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });

  it("shows the warning explanations in the drawer", async () => {
    const warned = [flare({ flare_id: "W", warnings: ["out_of_range", "many_features_missing"], reliability: "uncertain" })];
    renderWithProviders(<ResultsTable results={warned} />);
    await userEvent.click(screen.getByRole("button", { name: "W" }));
    expect(screen.getByText(/looks unlike the flares the model was trained on/)).toBeInTheDocument();
    expect(screen.getByText(/Several shape features could not be measured/)).toBeInTheDocument();
  });

  it("paginates above 200 rows", async () => {
    const many = Array.from({ length: 250 }, (_, i) => flare({ flare_id: `F${i}`, star_id: i, calibrated_probability: 0.5 - i / 1000 }));
    renderWithProviders(<ResultsTable results={many} />);
    expect(screen.getAllByTestId("result-row")).toHaveLength(100);
    await userEvent.click(screen.getByRole("button", { name: "Next page" }));
    expect(screen.getByText(/Showing 101–200 of 250/)).toBeInTheDocument();
  });
});
