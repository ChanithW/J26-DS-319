import { useState } from "react";
import { useNavigate } from "react-router-dom";

import { ApiError } from "../api/client";
import { LATEST, usePredict, useSamples } from "../api/hooks";
import type { FlareInput } from "../api/types";
import { ErrorNotice } from "../components/Feedback";
import { FileDropzone } from "../components/FileDropzone";

export function PredictPage() {
  const navigate = useNavigate();
  const [file, setFile] = useState<File | null>(null);
  const [samples, setSamples] = useState<FlareInput[] | null>(null);
  const [runName, setRunName] = useState("");
  const [saveRun, setSaveRun] = useState(true);
  const predict = usePredict();
  const loadSamples = useSamples();

  const chooseFile = (f: File | null) => { setFile(f); setSamples(null); predict.reset(); };
  const useSample = () => loadSamples.mutate(undefined, { onSuccess: (s) => { setSamples(s); setFile(null); predict.reset(); } });

  const run = () =>
    predict.mutate(
      { file, flares: samples, saveRun, runName: runName.trim() || null },
      { onSuccess: (r) => navigate(`/runs/${r.run_id ?? LATEST}`) },
    );

  const err = predict.error instanceof ApiError ? predict.error : null;
  const fileError = err && file && ["invalid_csv", "file_too_large", "too_many_flares", "too_many_points"].includes(err.code) ? err.message : null;
  const otherError = predict.error && !fileError ? predict.error : null;
  const ready = !!file || !!samples;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold">Run a prediction</h1>
        <p className="mt-1 max-w-3xl text-sm text-slate-700 dark:text-slate-300">
          Upload brightness curves of stellar flares, or try the 33 sample TESS flares. For each flare the system gives a ranking
          score with a range, a predicted label, and a reliability flag.
        </p>
      </div>

      <section className="card space-y-4" aria-label="Choose input">
        <h2 className="text-base font-semibold">1. Choose the flares</h2>
        <FileDropzone file={file} onFile={chooseFile} error={fileError} />
        <p className="text-center text-sm text-slate-600 dark:text-slate-400">or</p>
        <div className="text-center">
          <button type="button" className="btn" onClick={useSample} disabled={loadSamples.isPending}>
            {loadSamples.isPending ? "Loading samples…" : "Use the 33 sample TESS flares"}
          </button>
          {samples && <p className="mt-2 text-sm font-medium" role="status">{samples.length} sample flares loaded.</p>}
          {loadSamples.error && (
            <div className="mt-2 text-left"><ErrorNotice title="Samples unavailable" message={loadSamples.error.message} onRetry={useSample} /></div>
          )}
        </div>
      </section>

      <section className="card space-y-3" aria-label="Run options">
        <h2 className="text-base font-semibold">2. Run</h2>
        <label className="block max-w-sm text-sm">
          <span className="mb-1 block font-medium">Run name (optional)</span>
          <input className="input w-full" value={runName} onChange={(e) => setRunName(e.target.value)} placeholder="e.g. TESS sample, October" />
        </label>
        <label className="flex items-center gap-2 text-sm">
          <input type="checkbox" checked={saveRun} onChange={(e) => setSaveRun(e.target.checked)} />
          Save this run so I can open it later
        </label>
        <button type="button" className="btn btn-primary" onClick={run} disabled={!ready || predict.isPending}>
          {predict.isPending ? "Running…" : "Run prediction"}
        </button>
        {!ready && <p className="text-sm text-slate-600 dark:text-slate-400">Choose a file or the sample first.</p>}
        {otherError && (
          <ErrorNotice
            title={err?.unreachable ? "The server can't be reached" : "The prediction could not be run"}
            message={otherError.message}
            onRetry={ready ? run : undefined}
          />
        )}
      </section>
    </div>
  );
}
