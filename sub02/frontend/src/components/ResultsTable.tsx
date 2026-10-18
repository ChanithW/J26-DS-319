import { useMemo, useState } from "react";

import type { FlareResult } from "../api/types";
import { statusReason } from "../lib/explain";
import { probWithRange } from "../lib/format";
import { LabelBadge, ReliabilityBadge } from "./Badges";
import { EmptyState } from "./Feedback";
import { FlareDrawer } from "./FlareDrawer";

type SortKey = "rank" | "flare_id" | "star_id" | "label" | "probability" | "reliability" | "missing" | "status";
type Row = FlareResult & { rank: number | null };

const PAGE_SIZE = 100;
const PAGINATE_ABOVE = 200;

const COLUMNS: { key: SortKey; title: string }[] = [
  { key: "rank", title: "Rank" },
  { key: "flare_id", title: "Flare" },
  { key: "star_id", title: "Star" },
  { key: "label", title: "Label" },
  { key: "probability", title: "Ranking score (5–95% range)" },
  { key: "reliability", title: "Reliability" },
  { key: "missing", title: "Missing features" },
  { key: "status", title: "Status" },
];

const value = (r: Row, k: SortKey): string | number | null => {
  switch (k) {
    case "rank": return r.rank;
    case "flare_id": return r.flare_id ?? null;
    case "star_id": return r.star_id ?? null;
    case "label": return r.predicted_label ?? null;
    case "probability": return r.calibrated_probability ?? null;
    case "reliability": return r.reliability ?? null;
    case "missing": return r.features_missing ?? null;
    case "status": return r.status;
  }
};

export function ResultsTable({ results }: { results: FlareResult[] }) {
  const [sort, setSort] = useState<{ key: SortKey; dir: "asc" | "desc" }>({ key: "probability", dir: "desc" });
  const [label, setLabel] = useState("all");
  const [reliability, setReliability] = useState("all");
  const [status, setStatus] = useState("all");
  const [search, setSearch] = useState("");
  const [page, setPage] = useState(0);
  const [open, setOpen] = useState<number | null>(null);

  const rows: Row[] = useMemo(() => {
    const ok = results.filter((r) => r.calibrated_probability != null)
      .sort((a, b) => (b.calibrated_probability as number) - (a.calibrated_probability as number));
    const rank = new Map(ok.map((r, i) => [r, i + 1]));
    return results.map((r) => ({ ...r, rank: rank.get(r) ?? null }));
  }, [results]);

  const visible = useMemo(() => {
    const q = search.trim().toLowerCase();
    const f = rows.filter((r) =>
      (label === "all" || r.predicted_label === label) &&
      (reliability === "all" || r.reliability === reliability) &&
      (status === "all" || (status === "ok" ? r.status === "ok" : r.status !== "ok")) &&
      (!q || `${r.flare_id ?? ""} ${r.star_id ?? ""}`.toLowerCase().includes(q)));
    const dir = sort.dir === "asc" ? 1 : -1;
    return [...f].sort((a, b) => {
      const x = value(a, sort.key), y = value(b, sort.key);
      if (x == null && y == null) return 0;
      if (x == null) return 1;                         // empty values always last
      if (y == null) return -1;
      return (typeof x === "number" && typeof y === "number" ? x - y : String(x).localeCompare(String(y))) * dir;
    });
  }, [rows, label, reliability, status, search, sort]);

  const paged = visible.length > PAGINATE_ABOVE;
  const pages = Math.max(1, Math.ceil(visible.length / PAGE_SIZE));
  const shown = paged ? visible.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE) : visible;

  const toggle = (key: SortKey) => {
    setSort((s) => (s.key === key ? { key, dir: s.dir === "asc" ? "desc" : "asc" } : { key, dir: key === "probability" ? "desc" : "asc" }));
    setPage(0);
  };

  const resetPage = <T,>(set: (v: T) => void) => (v: T) => { set(v); setPage(0); };

  return (
    <div>
      <div className="mb-3 flex flex-wrap items-end gap-3">
        <label className="text-sm">
          <span className="mb-1 block text-xs font-medium">Label</span>
          <select className="input" value={label} onChange={(e) => resetPage(setLabel)(e.target.value)} aria-label="Filter by label">
            <option value="all">All labels</option>
            <option value="eruptive">Predicted eruptive</option>
            <option value="confined">Predicted confined</option>
          </select>
        </label>
        <label className="text-sm">
          <span className="mb-1 block text-xs font-medium">Reliability</span>
          <select className="input" value={reliability} onChange={(e) => resetPage(setReliability)(e.target.value)} aria-label="Filter by reliability">
            <option value="all">All</option>
            <option value="confident">Confident</option>
            <option value="uncertain">Uncertain</option>
          </select>
        </label>
        <label className="text-sm">
          <span className="mb-1 block text-xs font-medium">Status</span>
          <select className="input" value={status} onChange={(e) => resetPage(setStatus)(e.target.value)} aria-label="Filter by status">
            <option value="all">All</option>
            <option value="ok">Scored</option>
            <option value="notok">Not scored</option>
          </select>
        </label>
        <label className="min-w-[12rem] flex-1 text-sm">
          <span className="mb-1 block text-xs font-medium">Search flare or star id</span>
          <input className="input w-full" type="search" value={search} onChange={(e) => resetPage(setSearch)(e.target.value)} placeholder="e.g. TESS_294 or 294750180" />
        </label>
      </div>

      {visible.length === 0 ? (
        <EmptyState title="No flares match these filters">Clear a filter or change the search text.</EmptyState>
      ) : (
        <div className="overflow-x-auto rounded-lg border border-slate-200 dark:border-slate-800">
          <table className="w-full min-w-[56rem] bg-white dark:bg-slate-900">
            <caption className="sr-only">Prediction results, one row per flare</caption>
            <thead className="border-b border-slate-200 dark:border-slate-800">
              <tr>
                {COLUMNS.map((c) => (
                  <th key={c.key} scope="col" className="th"
                    aria-sort={sort.key === c.key ? (sort.dir === "asc" ? "ascending" : "descending") : "none"}>
                    <button type="button" onClick={() => toggle(c.key)} className="font-semibold uppercase tracking-wide hover:underline">
                      {c.title}
                      <span aria-hidden="true">{sort.key === c.key ? (sort.dir === "asc" ? " ↑" : " ↓") : ""}</span>
                    </button>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {shown.map((r, i) => {
                const notOk = r.status !== "ok";
                return (
                  <tr key={`${r.flare_id}-${i}`} data-testid="result-row" data-status={r.status}
                    className={`cursor-pointer border-b border-slate-100 hover:bg-slate-50 dark:border-slate-800 dark:hover:bg-slate-800 ${notOk ? "bg-slate-100 italic text-slate-600 dark:bg-slate-800 dark:text-slate-300" : ""}`}
                    onClick={() => setOpen(visible.indexOf(r))}>
                    <td className="td">{r.rank ?? "—"}</td>
                    <td className="td">
                      <button type="button" className="font-medium underline" onClick={(e) => { e.stopPropagation(); setOpen(visible.indexOf(r)); }}>
                        {r.flare_id ?? "—"}
                      </button>
                    </td>
                    <td className="td">{r.star_id ?? "—"}</td>
                    <td className="td">{notOk ? "—" : <LabelBadge label={r.predicted_label} />}</td>
                    <td className="td whitespace-nowrap">{notOk ? "—" : probWithRange(r)}</td>
                    <td className="td">{notOk ? "—" : <ReliabilityBadge reliability={r.reliability} />}</td>
                    <td className="td">{r.features_missing ?? "—"}</td>
                    <td className="td">{notOk ? <span title={statusReason(r.status)}>{statusReason(r.status)}</span> : "Scored"}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {paged && (
        <div className="mt-3 flex items-center justify-between text-sm">
          <span>Showing {page * PAGE_SIZE + 1}–{Math.min(visible.length, (page + 1) * PAGE_SIZE)} of {visible.length}</span>
          <span className="flex gap-2">
            <button type="button" className="btn" disabled={page === 0} onClick={() => setPage(page - 1)}>Previous page</button>
            <button type="button" className="btn" disabled={page >= pages - 1} onClick={() => setPage(page + 1)}>Next page</button>
          </span>
        </div>
      )}

      {open != null && <FlareDrawer results={visible} index={open} onClose={() => setOpen(null)} onIndex={setOpen} />}
    </div>
  );
}
