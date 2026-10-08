import { useRef, useState } from "react";

import { TEMPLATE_CSV, downloadBlob } from "../lib/csv";
import { ErrorNotice } from "./Feedback";

/** Drag-and-drop (or click) CSV chooser. API errors about the file are shown inside this box. */
export function FileDropzone({ file, onFile, error }: {
  file: File | null;
  onFile: (f: File | null) => void;
  error?: string | null;
}) {
  const input = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);

  return (
    <div data-testid="dropzone">
      <div
        onDragOver={(e) => { e.preventDefault(); setOver(true); }}
        onDragLeave={() => setOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setOver(false);
          onFile(e.dataTransfer.files[0] ?? null);
        }}
        className={`rounded-lg border-2 border-dashed p-6 text-center ${over ? "border-accent bg-indigo-50 dark:bg-indigo-950" : "border-slate-300 dark:border-slate-700"}`}
      >
        <p className="text-sm">Drag a CSV file here, or</p>
        <button type="button" className="btn mt-2" onClick={() => input.current?.click()}>
          Choose a CSV file
        </button>
        <input
          ref={input}
          type="file"
          accept=".csv,text/csv"
          aria-label="CSV file"
          className="sr-only"
          onChange={(e) => onFile(e.target.files?.[0] ?? null)}
        />
        {file && <p className="mt-3 text-sm font-medium">Selected: {file.name}</p>}
      </div>
      {error && <div className="mt-2"><ErrorNotice title="This file can't be used" message={error} /></div>}
      <details className="mt-3 text-sm text-slate-700 dark:text-slate-300">
        <summary className="cursor-pointer font-medium">What should the CSV look like?</summary>
        <p className="mt-2">
          One row per light-curve point, with the columns <code>flare_id</code>, <code>star_id</code>, <code>tpeak</code>,{" "}
          <code>time</code>, <code>flux</code>. The first three repeat on every row of a flare. <code>time</code> and{" "}
          <code>tpeak</code> are in TESS days (BTJD = Julian date − 2457000); <code>flux</code> can be on any positive scale.
        </p>
        <button type="button" className="mt-2 underline" onClick={() => downloadBlob(TEMPLATE_CSV, "text/csv", "component2_template.csv")}>
          Download the CSV template
        </button>
      </details>
    </div>
  );
}
