import { render, screen } from "@testing-library/react";

import { LabelBadge, LabelWithReliability, ReliabilityBadge } from "../src/components/Badges";
import { explainLabel } from "../src/lib/explain";
import { flare, notOk } from "./mocks";

describe("badges", () => {
  it.each([
    ["eruptive", "Predicted eruptive", "▲"],
    ["confined", "Predicted confined", "■"],
  ])("label %s has text and an icon", (label, text, icon) => {
    render(<LabelBadge label={label} />);
    const b = screen.getByTestId("label-badge");
    expect(b).toHaveTextContent(text);
    expect(b).toHaveTextContent(icon);
  });

  it.each([
    ["confident", "Confident", "●"],
    ["uncertain", "Uncertain", "○"],
  ])("reliability %s has text and an icon", (rel, text, icon) => {
    render(<ReliabilityBadge reliability={rel} />);
    const b = screen.getByTestId("reliability-badge");
    expect(b).toHaveTextContent(text);
    expect(b).toHaveTextContent(icon);
  });

  it.each([
    ["eruptive", "confident"], ["eruptive", "uncertain"], ["confined", "confident"], ["confined", "uncertain"],
  ] as const)("a %s label is never shown without its reliability (%s)", (label, rel) => {
    render(<LabelWithReliability result={{ predicted_label: label, reliability: rel }} />);
    expect(screen.getByTestId("label-badge")).toBeInTheDocument();
    expect(screen.getByTestId("reliability-badge")).toBeInTheDocument();
  });

  it("never uses detection or confirmation wording", () => {
    render(<LabelWithReliability result={{ predicted_label: "eruptive", reliability: "confident" }} />);
    expect(document.body.textContent).not.toMatch(/detected|confirmed|accurate/i);
  });
});

describe("explanations", () => {
  it("uses the fixed texts for every case", () => {
    expect(explainLabel(flare({ predicted_label: "eruptive", reliability: "confident" })))
      .toBe("The system gave a single answer: eruptive. It was trained on solar flares and has not been validated on stars.");
    expect(explainLabel(flare({ predicted_label: "confined", reliability: "confident" })))
      .toBe("The system gave a single answer: confined. It was trained on solar flares and has not been validated on stars.");
    expect(explainLabel(flare({ reliability: "uncertain" })))
      .toBe("The system could not rule out either answer for this flare. Treat the label as a weak lean, not a result.");
    expect(explainLabel(notOk("x", "too_few_points"))).toContain("fewer than 20 data points");
  });
});
