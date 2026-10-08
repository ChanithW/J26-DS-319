import { useModelInfo } from "../api/hooks";
import { ErrorNotice, Skeleton } from "../components/Feedback";
import { fixed, formatDate, percent } from "../lib/format";

type Obj = Record<string, unknown>;
const obj = (v: unknown): Obj => (v && typeof v === "object" ? (v as Obj) : {});

// Every number shown on this page comes from the API; only the wording is fixed.
const MEASURES: { key: string; label: string; fmt: (v: unknown) => string; meaning: string }[] = [
  { key: "tss", label: "TSS", fmt: (v) => fixed(v), meaning: "Hit rate + clear rate − 1. 0 is no skill, 1 is perfect. The main measure here, because most flares are confined." },
  { key: "auc", label: "AUC", fmt: (v) => fixed(v), meaning: "How well the score ranks eruptive flares above confined ones. 0.5 is chance." },
  { key: "hit_rate", label: "Hit rate", fmt: percent, meaning: "Share of eruptive flares labelled eruptive." },
  { key: "clear_rate", label: "Clear rate", fmt: percent, meaning: "Share of confined flares labelled confined." },
  { key: "balanced_accuracy", label: "Balanced accuracy", fmt: percent, meaning: "Average of hit rate and clear rate." },
  { key: "accuracy", label: "Plain accuracy", fmt: percent, meaning: "Share labelled correctly. Misleading here: always answering “confined” scores higher." },
];

export function ModelInfoPage() {
  const { data, isLoading, error, refetch } = useModelInfo();
  if (isLoading) return <Skeleton lines={6} />;
  if (error || !data) return <ErrorNotice title="Model information could not be loaded" message={error?.message ?? "Unknown error."} onRetry={() => refetch()} />;

  const m = obj(data.metrics);
  const cv = obj(m.cv5_grouped), held = obj(m["solar_test_raw_0.5"]), conf = obj(m.solar_test_conformal_weighted), split = obj(m.split);

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-semibold">Model information</h1>
        <p className="text-sm text-slate-600 dark:text-slate-400">
          Version {data.bundle_version} · created {formatDate(data.created)} · training data {data.data_hash}
        </p>
      </div>

      <section aria-labelledby="limits">
        <h2 id="limits" className="mb-2 text-lg font-semibold">Limitations</h2>
        <ul className="list-disc space-y-1 pl-5 text-sm">
          {data.limitations.map((l) => <li key={l}>{l}</li>)}
        </ul>
      </section>

      <section aria-labelledby="skill">
        <h2 id="skill" className="mb-2 text-lg font-semibold">How well it works on the Sun</h2>
        <p className="mb-2 text-sm text-slate-700 dark:text-slate-300">
          These numbers measure solar flares only. They say nothing about accuracy on stars, which has not been measured.
        </p>
        <div className="overflow-x-auto rounded-lg border border-slate-200 dark:border-slate-800">
          <table className="w-full min-w-[40rem] bg-white dark:bg-slate-900" data-testid="metrics-table">
            <caption className="sr-only">Skill measures on solar flares</caption>
            <thead className="border-b border-slate-200 dark:border-slate-800">
              <tr>
                <th scope="col" className="th">Measure</th>
                <th scope="col" className="th">Cross-validation (all solar flares)</th>
                <th scope="col" className="th">Held-out solar test</th>
                <th scope="col" className="th">What it means</th>
              </tr>
            </thead>
            <tbody>
              {MEASURES.map((r) => (
                <tr key={r.key} className="border-b border-slate-100 dark:border-slate-800">
                  <th scope="row" className="td text-left font-medium">{r.label}</th>
                  <td className="td">{r.fmt(cv[r.key])}</td>
                  <td className="td">{r.fmt(held[r.key])}</td>
                  <td className="td text-slate-700 dark:text-slate-300">{r.meaning}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section aria-labelledby="more">
        <h2 id="more" className="mb-2 text-lg font-semibold">Other checks</h2>
        <dl className="grid gap-3 text-sm sm:grid-cols-2">
          <div className="card"><dt className="font-medium">Reliability flag coverage (target 90%)</dt>
            <dd>Eruptive {percent(conf.coverage_eruptive)} · confined {percent(conf.coverage_confined)}</dd></div>
          <div className="card"><dt className="font-medium">Held-out flares with a single answer</dt>
            <dd>{percent(conf.single_label_share)} (those answers were right {percent(conf.single_label_accuracy)} of the time)</dd></div>
          <div className="card"><dt className="font-medium">Brier score</dt>
            <dd>{fixed(m.brier_test)} against {fixed(m.brier_test_base_rate_baseline)} for always answering the base rate. The probability is a ranking score, not a chance.</dd></div>
          <div className="card"><dt className="font-medium">Label cut-off (eruptive share of the calibration flares)</dt>
            <dd>{fixed(m.base_rate_cal)}. A flare is labelled eruptive when its probability is at or above this, not 0.5.</dd></div>
          <div className="card"><dt className="font-medium">Solar flares used</dt>
            <dd>{String(m.n_solar_with_features ?? "—")} ({String(m.n_eruptive ?? "—")} eruptive) · train {String(split.train ?? "—")} / calibration {String(split.calibration ?? "—")} / test {String(split.test ?? "—")}</dd></div>
        </dl>
      </section>

      <section aria-labelledby="read">
        <h2 id="read" className="mb-2 text-lg font-semibold">How to read an output</h2>
        <ul className="list-disc space-y-2 pl-5 text-sm">
          <li><strong>Ranking score.</strong> Higher means the flare looks more like the eruptive solar flares. It is for ordering flares, not a chance of a CME. The range shows how much it moves with the solar training data only.</li>
          <li><strong>Predicted eruptive / confined.</strong> A label, not a detection. It always comes with a reliability flag.</li>
          <li><strong>Confident</strong> (●) means the system could rule out one of the two answers. <strong>Uncertain</strong> (○) means it could not; treat the label as a weak lean. Most stellar flares are uncertain, and that is shown, not hidden.</li>
          <li><strong>Warnings.</strong> Missing features or a flare unlike the training flares make the answer uncertain.</li>
          <li><strong>No rates.</strong> The share labelled eruptive depends strongly on how solar and stellar flares are aligned, so no stellar CME rate can be read from these outputs.</li>
        </ul>
      </section>
    </div>
  );
}
