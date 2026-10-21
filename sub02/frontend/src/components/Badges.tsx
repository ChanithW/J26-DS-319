import type { FlareResult } from "../api/types";
import { labelText } from "../lib/format";

const base = "inline-flex items-center gap-1 whitespace-nowrap rounded-full border px-2 py-0.5 text-xs font-medium";

/** Filled circle = confident, hollow circle = uncertain. Colour is never the only signal. */
export function ReliabilityBadge({ reliability }: { reliability: string | null | undefined }) {
  if (reliability !== "confident" && reliability !== "uncertain") return <span aria-label="no reliability">—</span>;
  const confident = reliability === "confident";
  return (
    <span
      data-testid="reliability-badge"
      className={`${base} ${confident
        ? "border-emerald-600 bg-emerald-50 text-emerald-900 dark:border-emerald-500 dark:bg-emerald-950 dark:text-emerald-200"
        : "border-amber-600 bg-amber-50 text-amber-900 dark:border-amber-500 dark:bg-amber-950 dark:text-amber-200"}`}
    >
      <span aria-hidden="true">{confident ? "●" : "○"}</span>
      {confident ? "Confident" : "Uncertain"}
    </span>
  );
}

/** "Predicted eruptive" / "Predicted confined": never "detected" or "confirmed". */
export function LabelBadge({ label }: { label: string | null | undefined }) {
  if (label !== "eruptive" && label !== "confined") return <span aria-label="no label">—</span>;
  const eruptive = label === "eruptive";
  return (
    <span
      data-testid="label-badge"
      className={`${base} ${eruptive
        ? "border-indigo-600 bg-indigo-50 text-indigo-900 dark:border-indigo-400 dark:bg-indigo-950 dark:text-indigo-200"
        : "border-slate-500 bg-slate-100 text-slate-800 dark:border-slate-500 dark:bg-slate-800 dark:text-slate-200"}`}
    >
      <span aria-hidden="true">{eruptive ? "▲" : "■"}</span>
      {labelText(label)}
    </span>
  );
}

/** A label is never shown alone: reliability always sits next to it. */
export function LabelWithReliability({ result }: { result: Pick<FlareResult, "predicted_label" | "reliability"> }) {
  return (
    <span className="inline-flex flex-wrap items-center gap-1">
      <LabelBadge label={result.predicted_label} />
      <ReliabilityBadge reliability={result.reliability} />
    </span>
  );
}
