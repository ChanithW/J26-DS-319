import { useState } from "react";
import { useNavigate } from "react-router-dom";

import { ApiError } from "../api/client";
import { LATEST, usePredict, useSamples } from "../api/hooks";
import type { FlareInput } from "../api/types";
import { ErrorNotice } from "../components/Feedback";
import { FileDropzone } from "../components/FileDropzone";
import { ManualFlareForm } from "../components/ManualFlareForm";

type Source = "file" | "sample" | "manual";

export function PredictPage() {
  const navigate = useNavigate();
  const [file, setFile] = useState<File | null>(null);
  const [samples, setSamples] = useState<FlareInput[] | null>(null);
  const [manual, setManual] = useState<FlareInput[]>([]);
  const [source, setSource] = useState<Source | null>(null);   // the input chosen last is the one that runs
  const [runName, setRunName] = useState("");
  const [saveRun, setSaveRun] = useState(true);
  const predict = usePredict();
  const loadSamples = useSamples();

  const chooseFile = (f: File | null) => { setFile(f); setSource(f ? "file" : manual.length ? "manual" : samples ? "sample" : null); predict.reset(); };
  const useSample = () => loadSamples.mutate(undefined, { onSuccess: (s) => { setSamples(s); setSource("sample"); predict.reset(); } });
  const addManual = (f: FlareInput) => { setManual((m) => [...m, f]); setSource("manual"); predict.reset(); };
  const removeManual = (i: number) => setManual((m) => m.filter((_, j) => j !== i));

  const ready = (source === "file" && !!file) || (source === "sample" && !!samples) || (source === "manual" && manual.length > 0);

  const run = () =>
    predict.mutate(
      { file: source === "file" ? file : null, flares: source === "sample" ? samples : source === "manual" ? manual : null,
        saveRun, runName: runName.trim() || null },
      { onSuccess: (r) => navigate(`/runs/${r.run_id ?? LATEST}`) },
    );

  const err = predict.error instanceof ApiError ? predict.error : null;
  const fileError = err && source === "file" && ["invalid_csv", "file_too_large", "too_many_flares", "too_many_points"].includes(err.code) ? err.message : null;
  const otherError = predict.error && !fileError ? predict.error : null;

  const willRun =
    source === "file" && file ? `the CSV file “${file.name}”`
    : source === "sample" && samples ? `the ${samples.length} sample TESS flares`
    : source === "manual" && manual.length ? `${manual.length} manually entered flare${manual.length > 1 ? "s" : ""}`
    : null;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold">Run a prediction</h1>
        <p className="mt-1 max-w-3xl text-sm text-slate-700 dark:text-slate-300">
          Upload brightness curves of stellar flares, enter a flare by hand, or try the 33 sample TESS flares. For each flare the
          system gives a ranking score with a range, a predicted label, and a reliability flag.
        </p>
      </div>

      <section aria-label="Choose input" className="space-y-4">
        <h2 className="text-base font-semibold">1. Choose the flares</h2>
        <div className="grid gap-4 lg:grid-cols-2">
          <div className="card space-y-3" aria-label="Upload a CSV file">
            <h3 className="text-sm font-semibold">Upload a CSV file</h3>
            <FileDropzone file={file} onFile={chooseFile} error={fileError} />
          </div>
          <div className="card space-y-3" aria-label="Enter a flare manually">
            <h3 className="text-sm font-semibold">Or enter a flare manually</h3>
            <ManualFlareForm existingIds={manual.map((m) => m.flare_id)} onAdd={addManual} />
          </div>
        </div>

        {manual.length > 0 && (
          <div className="card" aria-label="Manually entered flares" data-testid="manual-list">
            <div className="mb-2 flex items-center justify-between">
              <h3 className="text-sm font-semibold">Manually entered flares ({manual.length})</h3>
              <button type="button" className="btn" onClick={() => { setManual([]); if (source === "manual") setSource(null); }}>Remove all</button>
            </div>
            <ul className="divide-y divide-slate-100 text-sm dark:divide-slate-800">
              {manual.map((m, i) => (
                <li key={m.flare_id} className="flex items-center justify-between gap-2 py-2">
                  <span><strong>{m.flare_id}</strong> · star {m.star_id} · {m.time?.length ?? 0} points</span>
                  <button type="button" className="btn" onClick={() => removeManual(i)} aria-label={`Remove ${m.flare_id}`}>Remove</button>
                </li>
              ))}
            </ul>
          </div>
        )}

        <div className="text-center">
          <p className="mb-2 text-sm text-slate-600 dark:text-slate-400">or</p>
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
        <p className="text-sm text-slate-600 dark:text-slate-400" data-testid="will-run">
          {willRun ? <>This will run on <strong>{willRun}</strong> (the input you chose last).</> : "Choose a file, add a flare, or pick the sample first."}
        </p>
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
