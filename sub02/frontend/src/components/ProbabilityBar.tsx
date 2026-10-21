import { pct, probText } from "../lib/format";

/** 0-100% bar with the point value and a 5-95% whisker. The text always accompanies the graphic. */
export function ProbabilityBar({ p, lo, hi }: { p: number; lo: number; hi: number }) {
  const at = (v: number) => `${Math.min(100, Math.max(0, v * 100))}%`;
  return (
    <div>
      <p className="text-lg font-semibold" data-testid="prob-text">
        {probText(p, lo, hi)}
      </p>
      <div
        className="relative mt-2 h-4 rounded bg-slate-200 dark:bg-slate-800"
        role="img"
        aria-label={`Ranking score ${pct(p)}, 5 to 95 percent range ${pct(lo)} to ${pct(hi)}`}
      >
        <div className="absolute top-1/2 h-0.5 -translate-y-1/2 bg-slate-500" style={{ left: at(lo), width: `calc(${at(hi)} - ${at(lo)})` }} />
        <div className="absolute top-0 h-4 w-0.5 bg-slate-500" style={{ left: at(lo) }} />
        <div className="absolute top-0 h-4 w-0.5 bg-slate-500" style={{ left: at(hi) }} />
        <div className="absolute top-[-3px] h-[22px] w-1.5 rounded bg-accent dark:bg-accent-dark" style={{ left: `calc(${at(p)} - 3px)` }} />
      </div>
      <div className="mt-1 flex justify-between text-xs text-slate-600 dark:text-slate-400" aria-hidden="true">
        <span>0%</span>
        <span>100%</span>
      </div>
      <p className="mt-1 text-xs text-slate-600 dark:text-slate-400">Ranking score, not a chance of a CME.</p>
    </div>
  );
}
