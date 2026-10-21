import { useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";

import { ApiError } from "../api/client";
import { exportUrl } from "../api/client";
import { LATEST, useDeleteRun, useRun } from "../api/hooks";
import type { ExportFormat, Run } from "../api/types";
import { EmptyState, ErrorNotice, Skeleton } from "../components/Feedback";
import { ResultsTable } from "../components/ResultsTable";
import { downloadBlob, exportLocally } from "../lib/csv";
import { formatDate } from "../lib/format";

function SummaryStrip({ run }: { run: Run }) {
  const s = run.summary;
  const items: [string, number][] = [
    ["Flares", s.n_flares], ["Scored", s.n_ok], ["Predicted eruptive", s.n_eruptive], ["Predicted confined", s.n_confined],
    ["Confident", s.n_confident], ["Uncertain", s.n_uncertain],
  ];
  return (
    <dl className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6" data-testid="summary-strip">
      {items.map(([k, v]) => (
        <div key={k} className="card py-3">
          <dt className="text-xs text-slate-600 dark:text-slate-400">{k}</dt>
          <dd className="text-2xl font-semibold">{v}</dd>
        </div>
      ))}
    </dl>
  );
}

export function RunPage() {
  const { runId } = useParams();
  const navigate = useNavigate();
  const { data: run, isLoading, error, refetch } = useRun(runId);
  const del = useDeleteRun();
  const [confirming, setConfirming] = useState(false);
  const saved = !!run?.run_id;

  const download = (fmt: ExportFormat) => {
    if (!run) return;
    if (run.run_id) {
      const a = document.createElement("a");
      a.href = exportUrl(run.run_id, fmt);
      a.click();
    } else {
      const f = exportLocally(run, fmt);
      downloadBlob(f.body, f.type, f.name);
    }
  };

  if (isLoading) return <Skeleton lines={5} />;
  if (error || !run) {
    const notFound = runId === LATEST || (error instanceof ApiError && error.status === 404);
    return notFound ? (
      <EmptyState title="This run is not available">
        It may have been deleted, or it was not saved. <Link className="underline" to="/">Start a new prediction</Link>.
      </EmptyState>
    ) : (
      <ErrorNotice title="The run could not be loaded" message={error?.message ?? "Unknown error."} onRetry={() => refetch()} />
    );
  }

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h1 className="text-2xl font-semibold">{run.name || (saved ? "Saved run" : "Prediction results")}</h1>
          <p className="text-sm text-slate-600 dark:text-slate-400">
            {formatDate(run.created_at)} · model version {run.model_version}
            {!saved && " · this run was not saved"}
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          <button type="button" className="btn" onClick={() => download("handoff_csv")}>Download hand-off CSV</button>
          <button type="button" className="btn" onClick={() => download("full_csv")}>Download full CSV</button>
          {saved && !confirming && <button type="button" className="btn" onClick={() => setConfirming(true)}>Delete run</button>}
        </div>
      </div>

      {confirming && (
        <div role="alertdialog" aria-label="Confirm delete" className="card flex flex-wrap items-center gap-3 border-red-300 dark:border-red-800">
          <p className="text-sm">Delete this saved run? This cannot be undone.</p>
          <button type="button" className="btn btn-primary" disabled={del.isPending}
            onClick={() => del.mutate(run.run_id!, { onSuccess: () => navigate("/runs") })}>
            Yes, delete
          </button>
          <button type="button" className="btn" onClick={() => setConfirming(false)}>Cancel</button>
          {del.error && <span className="text-sm text-red-700">{del.error.message}</span>}
        </div>
      )}

      <SummaryStrip run={run} />
      {run.summary.n_ok < run.summary.n_flares && (
        <p role="status" className="text-sm text-slate-700 dark:text-slate-300">
          {run.summary.n_flares - run.summary.n_ok} flare(s) could not be scored. They are greyed out below with the reason.
        </p>
      )}
      <ResultsTable results={run.results} />
    </div>
  );
}
