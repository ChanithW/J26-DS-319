import { useState } from "react";
import { Link } from "react-router-dom";

import { useDeleteRun, useRuns } from "../api/hooks";
import { EmptyState, ErrorNotice, Skeleton } from "../components/Feedback";
import { formatDate } from "../lib/format";

export function RunsPage() {
  const { data, isLoading, error, refetch } = useRuns();
  const del = useDeleteRun();
  const [confirm, setConfirm] = useState<string | null>(null);

  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-semibold">Saved runs</h1>
      {isLoading && <Skeleton lines={4} />}
      {error && <ErrorNotice title="Saved runs could not be loaded" message={error.message} onRetry={() => refetch()} />}
      {data && data.length === 0 && (
        <EmptyState title="No saved runs yet">
          <Link className="underline" to="/">Run a prediction</Link> and keep “Save this run” ticked.
        </EmptyState>
      )}
      {data && data.length > 0 && (
        <div className="overflow-x-auto rounded-lg border border-slate-200 dark:border-slate-800">
          <table className="w-full min-w-[40rem] bg-white dark:bg-slate-900">
            <caption className="sr-only">Saved runs, newest first</caption>
            <thead className="border-b border-slate-200 dark:border-slate-800">
              <tr>
                {["Name", "Date", "Model", "Flares", "Predicted eruptive", "Confident", ""].map((h) => (
                  <th key={h || "actions"} scope="col" className="th">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {data.map((r) => (
                <tr key={r.run_id} className="border-b border-slate-100 dark:border-slate-800">
                  <td className="td font-medium">{r.name || "Untitled run"}</td>
                  <td className="td">{formatDate(r.created_at)}</td>
                  <td className="td">v{r.model_version}</td>
                  <td className="td">{r.n_flares}</td>
                  <td className="td">{r.n_eruptive}</td>
                  <td className="td">{r.n_confident}</td>
                  <td className="td">
                    <span className="flex gap-2">
                      <Link className="btn" to={`/runs/${r.run_id}`}>Open</Link>
                      {confirm === r.run_id ? (
                        <>
                          <button type="button" className="btn btn-primary" onClick={() => del.mutate(r.run_id, { onSuccess: () => setConfirm(null) })}>Confirm delete</button>
                          <button type="button" className="btn" onClick={() => setConfirm(null)}>Cancel</button>
                        </>
                      ) : (
                        <button type="button" className="btn" onClick={() => setConfirm(r.run_id)}>Delete</button>
                      )}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
