import type { FlareInput } from "../api/types";

export type TimeUnit = "minutes" | "btjd";

export interface ManualForm {
  flareId: string;
  starId: string;
  unit: TimeUnit;
  tpeak: string;     // only used when unit is "btjd"
  points: string;    // one "time, flux" pair per line
}

export interface ParsedPoints {
  time: number[];
  flux: number[];
  errors: string[];
}

const NUM = /^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$/;

/** Parse pasted "time flux" lines (comma, tab, space or semicolon separated). A text header line is skipped. */
export function parsePoints(text: string): ParsedPoints {
  const time: number[] = [];
  const flux: number[] = [];
  const errors: string[] = [];
  const lines = text.split(/\r?\n/);
  let seenData = false;
  lines.forEach((raw, i) => {
    const line = raw.trim();
    if (!line) return;
    const parts = line.split(/[\s,;]+/).filter(Boolean);
    const numeric = parts.every((p) => NUM.test(p));
    if (!numeric && !seenData && parts.length >= 2 && parts.every((p) => !NUM.test(p))) return;   // header row
    if (parts.length !== 2 || !numeric) {
      if (errors.length < 5) errors.push(`Line ${i + 1}: expected two numbers (time and flux), found "${line.slice(0, 40)}".`);
      return;
    }
    seenData = true;
    time.push(Number(parts[0]));
    flux.push(Number(parts[1]));
  });
  return { time, flux, errors };
}

export interface BuildResult {
  flare?: FlareInput;
  errors: string[];
  warnings: string[];
}

/** Check the form and build the request record. Time in minutes from the peak is stored as days with tpeak = 0. */
export function buildFlare(f: ManualForm, existingIds: string[] = []): BuildResult {
  const errors: string[] = [];
  const warnings: string[] = [];

  const flareId = f.flareId.trim();
  if (!flareId) errors.push("Enter a flare id.");
  else if (existingIds.includes(flareId)) errors.push(`A flare called "${flareId}" is already in the list. Use a different id.`);

  const starText = f.starId.trim();
  if (!/^\d+$/.test(starText)) errors.push("Star id must be a whole number (the TIC number).");

  let tpeak = 0;
  if (f.unit === "btjd") {
    if (!NUM.test(f.tpeak.trim())) errors.push("Enter the flare peak time in BTJD (Julian date − 2457000).");
    else tpeak = Number(f.tpeak.trim());
  }

  const pts = parsePoints(f.points);
  errors.push(...pts.errors);
  if (!pts.errors.length && pts.time.length < 2) errors.push("Enter at least two data points (time and flux).");
  if (errors.length) return { errors, warnings };

  const days = f.unit === "minutes" ? pts.time.map((m) => m / 1440) : pts.time;
  const inWindow = days.filter((d) => {
    const m = (d - tpeak) * 1440;
    return m >= -20 && m <= 90;
  }).length;
  if (inWindow < 20) {
    warnings.push(`Only ${inWindow} point(s) fall between −20 and +90 minutes from the peak. At least 20 are needed, otherwise the flare will not be scored.`);
  }
  if (pts.flux.some((v) => v <= 0)) warnings.push("Some flux values are zero or negative. Flux should be on a positive scale.");
  const spacing = medianSpacingMinutes(days);
  if (spacing !== null && (spacing < 1.5 || spacing > 2.5)) {
    const shown = spacing < 1 ? `${(spacing * 60).toFixed(0)} seconds` : `${spacing.toFixed(1)} minutes`;
    warnings.push(`The spacing between your points is about ${shown}. The model expects 2-minute data and does not re-bin, so the result may not be reliable.`);
  }

  return { flare: { flare_id: flareId, star_id: Number(starText), tpeak, time: days, flux: pts.flux }, errors, warnings };
}

/** Median gap between consecutive points, in minutes (times in days); null if it cannot be worked out. */
export function medianSpacingMinutes(days: number[]): number | null {
  const sorted = [...days].sort((a, b) => a - b);
  const gaps: number[] = [];
  for (let i = 1; i < sorted.length; i++) {
    const g = (sorted[i] - sorted[i - 1]) * 1440;
    if (g > 1e-9) gaps.push(g);
  }
  if (!gaps.length) return null;
  gaps.sort((a, b) => a - b);
  const mid = Math.floor(gaps.length / 2);
  return gaps.length % 2 ? gaps[mid] : (gaps[mid - 1] + gaps[mid]) / 2;
}

/** A synthetic demo flare (fast rise, exponential decay) so the form can be tried without real data. */
export function exampleFlare(): ManualForm {
  const lines: string[] = [];
  for (let t = -20; t < 90; t += 2) {
    const flux = t < -4 ? 1 : t < 0 ? 1 + (0.5 * (t + 4)) / 4 : 1 + 0.5 * Math.exp(-t / 12);
    lines.push(`${t}, ${flux.toFixed(4)}`);
  }
  return { flareId: "EXAMPLE_FLARE", starId: "1", unit: "minutes", tpeak: "", points: lines.join("\n") };
}
