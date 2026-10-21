import { CartesianGrid, ComposedChart, Line, ReferenceArea, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import type { Curve } from "../api/types";

function summarise(curve: Curve): string {
  const n = curve.t_min.length;
  const near = (t: number) => {
    let best = 0;
    curve.t_min.forEach((x, i) => { if (Math.abs(x - t) < Math.abs(curve.t_min[best] - t)) best = i; });
    return curve.g[best];
  };
  return `Normalised brightness curve with ${n} points from ${Math.min(...curve.t_min).toFixed(0)} to ` +
    `${Math.max(...curve.t_min).toFixed(0)} minutes from the peak. It is 1 at the peak, about ${near(10).toFixed(2)} ten minutes ` +
    `later and about ${near(30).toFixed(2)} thirty minutes later (0 is the background level).`;
}

export function LightCurveChart({ curve }: { curve: Curve }) {
  const data = curve.t_min.map((t, i) => ({ t, g: curve.g[i] }));
  return (
    <figure>
      <div className="h-64 w-full" aria-hidden="true">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={data} margin={{ top: 20, right: 12, bottom: 24, left: 4 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#94a3b8" strokeOpacity={0.3} />
            <ReferenceArea x1={-20} x2={-8} fill="#64748b" fillOpacity={0.15} label={{ value: "background", fontSize: 11, position: "insideTop" }} />
            <XAxis dataKey="t" type="number" domain={[-20, 90]} ticks={[-20, 0, 20, 40, 60, 80]} tick={{ fontSize: 11 }}
              label={{ value: "minutes from peak", position: "insideBottom", offset: -12, fontSize: 12 }} />
            <YAxis domain={[-0.1, 1.1]} ticks={[0, 0.25, 0.5, 0.75, 1]} tick={{ fontSize: 11 }} width={44}
              label={{ value: "brightness above background (peak = 1)", angle: -90, position: "insideLeft", offset: 12, fontSize: 11, style: { textAnchor: "middle" } }} />
            <ReferenceLine x={0} stroke="#4f46e5" strokeDasharray="4 2" label={{ value: "peak", fontSize: 11, position: "top" }} />
            <Tooltip formatter={(v: number) => v.toFixed(3)} labelFormatter={(t: number) => `${t} min`} />
            <Line dataKey="g" stroke="#4f46e5" strokeWidth={1} dot={{ r: 2 }} isAnimationActive={false} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
      <figcaption className="mt-1 text-xs text-slate-600 dark:text-slate-400">{summarise(curve)}</figcaption>
      <details className="mt-2 text-xs">
        <summary className="cursor-pointer font-medium">Show the data as a table</summary>
        <div className="mt-1 max-h-40 overflow-auto">
          <table className="w-full text-left">
            <thead><tr><th className="pr-4">Minutes from peak</th><th>Brightness above background</th></tr></thead>
            <tbody>{data.map((d) => <tr key={d.t}><td>{d.t}</td><td>{d.g.toFixed(3)}</td></tr>)}</tbody>
          </table>
        </div>
      </details>
    </figure>
  );
}
