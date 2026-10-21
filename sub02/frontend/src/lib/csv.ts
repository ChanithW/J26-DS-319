import type { ExportFormat, FlareResult, Run } from "../api/types";

export const TEMPLATE_CSV =
  "flare_id,star_id,tpeak,time,flux\n" +
  "TESS_EXAMPLE,123456789,2458500.0,2458499.9861,1.0012\n" +
  "TESS_EXAMPLE,123456789,2458500.0,2458499.9875,1.0009\n" +
  "TESS_EXAMPLE,123456789,2458500.0,2458499.9889,1.0021\n";

const esc = (v: unknown): string => {
  if (v == null) return "";
  const s = Array.isArray(v) ? v.join(";") : String(v);
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
};

const HANDOFF = ["flare_id", "star_id", "calibrated_probability", "confidence_lower", "confidence_upper",
  "predicted_label", "reliability"] as const;
const FULL = ["flare_id", "star_id", "status", "calibrated_probability", "confidence_lower", "confidence_upper",
  "predicted_label", "reliability", "features_missing", "out_of_range", "warnings", "raw_score", "conformal_set"] as const;

export function toCsv(rows: FlareResult[], cols: readonly string[]): string {
  const lines = rows.map((r) => cols.map((c) => esc((r as Record<string, unknown>)[c])).join(","));
  return [cols.join(","), ...lines].join("\n") + "\n";
}

/** Client-side export, used for runs that were not saved (the API exports saved runs). */
export function exportLocally(run: Run, format: ExportFormat): { body: string; type: string; name: string } {
  if (format === "json") return { body: JSON.stringify(run, null, 2), type: "application/json", name: "run_unsaved.json" };
  if (format === "handoff_csv") {
    return { body: toCsv(run.results.filter((r) => r.status === "ok"), HANDOFF), type: "text/csv", name: "run_unsaved_handoff.csv" };
  }
  return { body: toCsv(run.results, FULL), type: "text/csv", name: "run_unsaved_full.csv" };
}

export function downloadBlob(body: string, type: string, name: string): void {
  const url = URL.createObjectURL(new Blob([body], { type }));
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  URL.revokeObjectURL(url);
}
