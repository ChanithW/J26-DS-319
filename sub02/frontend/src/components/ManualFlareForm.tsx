import { useState } from "react";

import type { FlareInput } from "../api/types";
import { type ManualForm, type TimeUnit, buildFlare, exampleFlare } from "../lib/manual";

const EMPTY: ManualForm = { flareId: "", starId: "", unit: "minutes", tpeak: "", points: "" };

/** Enter one flare by hand: ids plus pasted time/flux values. Added flares go to the list on the page. */
export function ManualFlareForm({ existingIds, onAdd }: { existingIds: string[]; onAdd: (f: FlareInput) => void }) {
  const [form, setForm] = useState<ManualForm>(EMPTY);
  const [errors, setErrors] = useState<string[]>([]);
  const [warnings, setWarnings] = useState<string[]>([]);
  const [added, setAdded] = useState<string | null>(null);
  const set = <K extends keyof ManualForm>(k: K, v: ManualForm[K]) => { setForm((f) => ({ ...f, [k]: v })); setAdded(null); };

  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    const r = buildFlare(form, existingIds);
    setErrors(r.errors);
    setWarnings(r.warnings);
    if (r.flare) {
      onAdd(r.flare);
      setAdded(r.flare.flare_id);
      setForm({ ...EMPTY, unit: form.unit });
    }
  };

  return (
    <form onSubmit={submit} className="space-y-3" aria-label="Enter a flare manually" data-testid="manual-form" noValidate>
      <div className="grid gap-3 sm:grid-cols-2">
        <label className="block text-sm">
          <span className="mb-1 block font-medium">Flare id</span>
          <input className="input w-full" value={form.flareId} onChange={(e) => set("flareId", e.target.value)} placeholder="e.g. TESS_294750180" />
        </label>
        <label className="block text-sm">
          <span className="mb-1 block font-medium">Star id (TIC number)</span>
          <input className="input w-full" inputMode="numeric" value={form.starId} onChange={(e) => set("starId", e.target.value)} placeholder="e.g. 294750180" />
        </label>
      </div>

      <div className="grid gap-3 sm:grid-cols-2">
        <label className="block text-sm">
          <span className="mb-1 block font-medium">Time is given as</span>
          <select className="input w-full" value={form.unit} onChange={(e) => set("unit", e.target.value as TimeUnit)}>
            <option value="minutes">Minutes from the flare peak</option>
            <option value="btjd">TESS days (BTJD)</option>
          </select>
        </label>
        {form.unit === "btjd" && (
          <label className="block text-sm">
            <span className="mb-1 block font-medium">Flare peak time (BTJD)</span>
            <input className="input w-full" inputMode="decimal" value={form.tpeak} onChange={(e) => set("tpeak", e.target.value)} placeholder="e.g. 1500.1234" />
          </label>
        )}
      </div>

      <label className="block text-sm">
        <span className="mb-1 block font-medium">Brightness curve: one “time, flux” pair per line</span>
        <textarea
          className="input h-40 w-full font-mono text-xs"
          value={form.points}
          onChange={(e) => set("points", e.target.value)}
          placeholder={"-20, 1.000\n-18, 1.001\n-16, 1.000\n…"}
          spellCheck={false}
        />
        <span className="hint mt-1 block">
          Paste two columns from a spreadsheet (separated by a comma, a tab or a space), or type them. Use 2-minute data
          covering −20 to +90 minutes around the peak (at least 20 points). Flux can be on any positive scale.
        </span>
        <span className="hint mt-1 block" data-testid="manual-hints">
          {form.unit === "minutes" ? "Time 0 must be the flare's brightest point (within about 6 minutes)."
            : "The peak time must be the flare's brightest point (within about 6 minutes)."}
          {" "}Include at least 3 points before -8 minutes (they set the background).
        </span>
      </label>

      {errors.length > 0 && (
        <div role="alert" className="rounded-md border border-red-300 bg-red-50 p-3 text-sm text-red-900 dark:border-red-800 dark:bg-red-950 dark:text-red-100">
          <p className="font-semibold">This flare can't be added yet</p>
          <ul className="mt-1 list-disc pl-5">{errors.map((m) => <li key={m}>{m}</li>)}</ul>
        </div>
      )}
      {warnings.length > 0 && (
        <div role="status" className="rounded-md border border-amber-300 bg-amber-50 p-3 text-sm text-amber-950 dark:border-amber-800 dark:bg-amber-950 dark:text-amber-100">
          <ul className="list-disc pl-5">{warnings.map((m) => <li key={m}>{m}</li>)}</ul>
        </div>
      )}
      {added && <p role="status" className="text-sm font-medium">Added “{added}” to the list below.</p>}

      <div className="flex flex-wrap gap-2">
        <button type="submit" className="btn btn-primary">Add this flare</button>
        <button type="button" className="btn" onClick={() => { setForm(exampleFlare()); setErrors([]); setWarnings([]); setAdded(null); }}>
          Fill with a demo flare
        </button>
        <button type="button" className="btn" onClick={() => { setForm(EMPTY); setErrors([]); setWarnings([]); setAdded(null); }}>
          Clear the form
        </button>
      </div>
      <p className="hint">The demo flare is synthetic, only to try the form. It is not real data.</p>
    </form>
  );
}
