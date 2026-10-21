import { useEffect, useRef } from "react";

import type { FlareResult } from "../api/types";
import { WARNING_TEXT, explainLabel, statusReason } from "../lib/explain";
import { LabelWithReliability } from "./Badges";
import { LightCurveChart } from "./LightCurveChart";
import { ProbabilityBar } from "./ProbabilityBar";

interface Props {
  results: FlareResult[];
  index: number;
  onClose: () => void;
  onIndex: (i: number) => void;
}

/** Side drawer with one flare's detail. Arrow keys move between flares, Escape closes. */
export function FlareDrawer({ results, index, onClose, onIndex }: Props) {
  const r = results[index];
  const closeBtn = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    closeBtn.current?.focus();
  }, []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement | null)?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
      if (e.key === "Escape") onClose();
      else if (e.key === "ArrowRight" && index < results.length - 1) onIndex(index + 1);
      else if (e.key === "ArrowLeft" && index > 0) onIndex(index - 1);
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [index, results.length, onClose, onIndex]);

  if (!r) return null;
  const ok = r.status === "ok" && r.calibrated_probability != null;

  return (
    <div className="fixed inset-0 z-40 flex justify-end bg-black/40" onClick={onClose}>
      <aside
        role="dialog"
        aria-modal="true"
        aria-label={`Flare ${r.flare_id ?? ""} detail`}
        className="h-full w-full max-w-xl overflow-y-auto bg-white p-4 shadow-xl dark:bg-slate-900"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-start justify-between gap-2">
          <div>
            <h2 className="text-lg font-semibold">{r.flare_id ?? "Unnamed flare"}</h2>
            <p className="text-sm text-slate-600 dark:text-slate-400">
              Star {r.star_id ?? "—"} · flare {index + 1} of {results.length}
            </p>
          </div>
          <button ref={closeBtn} type="button" className="btn" onClick={onClose} aria-label="Close detail (Escape)">
            Close
          </button>
        </div>

        <div className="mt-2 flex gap-2">
          <button type="button" className="btn" disabled={index === 0} onClick={() => onIndex(index - 1)}>
            ← Previous flare
          </button>
          <button type="button" className="btn" disabled={index === results.length - 1} onClick={() => onIndex(index + 1)}>
            Next flare →
          </button>
        </div>

        {!ok ? (
          <p className="mt-4 rounded-md border border-slate-300 bg-slate-100 p-3 text-sm dark:border-slate-700 dark:bg-slate-800">
            {statusReason(r.status) || "This flare was not scored."}
          </p>
        ) : (
          <div className="mt-4 space-y-5">
            <section aria-label="Answer">
              <LabelWithReliability result={r} />
              <p className="mt-2 text-sm" data-testid="explanation">{explainLabel(r)}</p>
              {r.warnings.map((w) => (
                <p key={w} className="mt-2 rounded-md border border-amber-300 bg-amber-50 p-2 text-sm text-amber-950 dark:border-amber-800 dark:bg-amber-950 dark:text-amber-100">
                  {WARNING_TEXT[w] ?? w}
                </p>
              ))}
            </section>

            <section aria-label="Ranking score">
              <ProbabilityBar p={r.calibrated_probability!} lo={r.confidence_lower!} hi={r.confidence_upper!} />
              <p className="mt-1 text-xs text-slate-600 dark:text-slate-400">
                The range reflects variation in the solar training data only, not the solar-to-stellar gap.
              </p>
            </section>

            {r.curve && (
              <section aria-label="Light curve">
                <h3 className="mb-1 text-sm font-semibold">Light curve</h3>
                <LightCurveChart curve={r.curve} />
              </section>
            )}

            <section aria-label="Features">
              <h3 className="mb-1 text-sm font-semibold">Shape features</h3>
              <table className="w-full text-sm">
                <tbody>
                  {Object.entries(r.features ?? {}).map(([k, v]) => (
                    <tr key={k} className="border-b border-slate-100 dark:border-slate-800">
                      <th scope="row" className="py-1 pr-3 text-left font-normal text-slate-600 dark:text-slate-400">{k}</th>
                      <td className="py-1">{v == null ? "—" : Number(v).toFixed(3)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <p className="mt-2 text-sm">
                Features missing: <strong>{r.features_missing ?? 0}</strong> of 13 · Outside training range:{" "}
                <strong>{r.out_of_range ? "yes" : "no"}</strong>
              </p>
            </section>
          </div>
        )}
      </aside>
    </div>
  );
}
