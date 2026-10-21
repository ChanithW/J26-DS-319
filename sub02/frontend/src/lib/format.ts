import type { FlareResult } from "../api/types";

export const pct = (p: number): string => `${Math.round(p * 100)}%`;

/** One format everywhere: a probability is never shown without its range, e.g. "51% (32–59%)". */
export const probText = (p: number, lo: number, hi: number): string =>
  `${pct(p)} (${Math.round(lo * 100)}–${pct(hi)})`;

export function probWithRange(r: Pick<FlareResult, "calibrated_probability" | "confidence_lower" | "confidence_upper">): string {
  const { calibrated_probability: p, confidence_lower: lo, confidence_upper: hi } = r;
  if (p == null || lo == null || hi == null) return "—";
  return probText(p, lo, hi);
}

export const labelText = (label: string | null | undefined): string =>
  label === "eruptive" ? "Predicted eruptive" : label === "confined" ? "Predicted confined" : "—";

export const fixed = (v: unknown, digits = 3): string => (typeof v === "number" && isFinite(v) ? v.toFixed(digits) : "—");
export const percent = (v: unknown): string => (typeof v === "number" && isFinite(v) ? `${(v * 100).toFixed(1)}%` : "—");

export const formatDate = (iso: string): string => {
  const d = new Date(iso);
  return isNaN(d.getTime()) ? iso : d.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
};
