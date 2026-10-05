import type { FlareResult } from "../api/types";

const VALIDATION = "It was trained on solar flares and has not been validated on stars.";

/** The one-sentence explanation for a result's label and reliability. */
export function explainLabel(r: FlareResult): string {
  if (r.status !== "ok") return statusReason(r.status);
  if (r.reliability === "uncertain") {
    return "The system could not rule out either answer for this flare. Treat the label as a weak lean, not a result.";
  }
  if (r.predicted_label === "eruptive") return `The system gave a single answer: eruptive. ${VALIDATION}`;
  return `The system gave a single answer: confined. ${VALIDATION}`;
}

export const WARNING_TEXT: Record<string, string> = {
  many_features_missing:
    "Several shape features could not be measured for this flare, so its score is closer to an average flare.",
  out_of_range: "This flare looks unlike the flares the model was trained on. Do not rely on its output.",
};

export function statusReason(status: string): string {
  switch (status) {
    case "too_few_points":
      return "Not scored: fewer than 20 data points in the window.";
    case "no_peak":
      return "Not scored: no clear flare peak was found near the stated peak time.";
    case "invalid_input":
      return "Not scored: the input for this flare was incomplete or not valid.";
    default:
      return "";
  }
}
