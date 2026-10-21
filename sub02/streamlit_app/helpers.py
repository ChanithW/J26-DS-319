"""Display rules shared by all views: formats, fixed explanation texts, CSV export, table building."""
from __future__ import annotations

import io
import math
import re

import pandas as pd

VALIDATION = "It was trained on solar flares and has not been validated on stars."

WARNING_TEXT = {
    "many_features_missing":
        "Several shape features could not be measured for this flare, so its score is closer to an average flare.",
    "out_of_range": "This flare looks unlike the flares the model was trained on. Do not rely on its output.",
}

TEMPLATE_CSV = (
    "flare_id,star_id,tpeak,time,flux\n"
    "TESS_EXAMPLE,123456789,2458500.0,2458499.9861,1.0012\n"
    "TESS_EXAMPLE,123456789,2458500.0,2458499.9875,1.0009\n"
    "TESS_EXAMPLE,123456789,2458500.0,2458499.9889,1.0021\n"
)

HANDOFF_COLUMNS = ["flare_id", "star_id", "calibrated_probability", "confidence_lower", "confidence_upper",
                   "predicted_label", "reliability"]
FULL_COLUMNS = ["flare_id", "star_id", "status", "calibrated_probability", "confidence_lower", "confidence_upper",
                "predicted_label", "reliability", "features_missing", "out_of_range", "warnings", "raw_score",
                "conformal_set"]


def pct(p: float) -> str:
    return f"{round(p * 100)}%"


def prob_text(p, lo, hi) -> str:
    """One format everywhere: a probability is never shown without its range, e.g. '51% (32–59%)'."""
    if p is None or lo is None or hi is None:
        return "—"
    return f"{pct(p)} ({round(lo * 100)}–{pct(hi)})"


def label_text(label) -> str:
    """'Predicted eruptive' / 'Predicted confined' (never 'detected' or 'confirmed'), with an icon."""
    return {"eruptive": "▲ Predicted eruptive", "confined": "■ Predicted confined"}.get(label, "—")


def reliability_text(rel) -> str:
    """Filled circle = confident, hollow circle = uncertain; colour is never the only signal."""
    return {"confident": "● Confident", "uncertain": "○ Uncertain"}.get(rel, "—")


def label_with_reliability(label, rel) -> str:
    """A label is never shown alone."""
    return f"{label_text(label)} · {reliability_text(rel)}"


def status_reason(status: str) -> str:
    return {
        "too_few_points": "Not scored: fewer than 20 data points in the window.",
        "no_peak": "Not scored: no clear flare peak was found near the stated peak time.",
        "invalid_input": "Not scored: the input for this flare was incomplete or not valid.",
    }.get(status, "")


def explain_label(r: dict) -> str:
    if r.get("status") != "ok":
        return status_reason(r.get("status", ""))
    if r.get("reliability") == "uncertain":
        return "The system could not rule out either answer for this flare. Treat the label as a weak lean, not a result."
    word = "eruptive" if r.get("predicted_label") == "eruptive" else "confined"
    return f"The system gave a single answer: {word}. {VALIDATION}"


def results_table(results: list[dict]) -> pd.DataFrame:
    """One row per flare for display. Rank is by probability among scored flares."""
    scored = sorted((r for r in results if r.get("calibrated_probability") is not None),
                    key=lambda r: -r["calibrated_probability"])
    rank = {id(r): i + 1 for i, r in enumerate(scored)}
    rows = []
    for r in results:
        ok = r.get("status") == "ok"
        rows.append({
            "Rank": rank.get(id(r)),
            "Flare": r.get("flare_id"),
            "Star": r.get("star_id"),
            "Label": label_text(r.get("predicted_label")) if ok else "—",
            "Ranking score (5–95% range)": prob_text(r.get("calibrated_probability"), r.get("confidence_lower"),
                                                      r.get("confidence_upper")) if ok else "—",
            "Reliability": reliability_text(r.get("reliability")) if ok else "—",
            "Missing features": r.get("features_missing"),
            "Status": "Scored" if ok else status_reason(r.get("status", "")),
            "_p": r.get("calibrated_probability"),
            "_ok": ok,
        })
    return pd.DataFrame(rows)


def to_csv(results: list[dict], cols: list[str]) -> str:
    rows = []
    for r in results:
        row = {c: r.get(c) for c in cols}
        if "warnings" in row:
            row["warnings"] = ";".join(row["warnings"] or [])
        rows.append(row)
    buf = io.StringIO()
    pd.DataFrame(rows, columns=cols).to_csv(buf, index=False, lineterminator="\n")
    return buf.getvalue()


def handoff_csv(results: list[dict]) -> str:
    return to_csv([r for r in results if r.get("status") == "ok"], HANDOFF_COLUMNS)


def full_csv(results: list[dict]) -> str:
    return to_csv(results, FULL_COLUMNS)


def curve_summary(curve: dict) -> str:
    t, g = curve["t_min"], curve["g"]

    def near(x: float) -> float:
        return g[min(range(len(t)), key=lambda i: abs(t[i] - x))]

    return (f"Normalised brightness curve with {len(t)} points from {min(t):.0f} to {max(t):.0f} minutes from the peak. "
            f"It is 1 at the peak, about {near(10):.2f} ten minutes later and about {near(30):.2f} thirty minutes later "
            "(0 is the background level).")


def fmt_num(v, digits: int = 3) -> str:
    return f"{v:.{digits}f}" if isinstance(v, (int, float)) and v == v else "—"


def fmt_pct(v) -> str:
    return f"{v * 100:.1f}%" if isinstance(v, (int, float)) and v == v else "—"


# ----------------------------------------------------------------- manual entry
_NUM = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")


def parse_points(text: str):
    """Parse pasted "time flux" lines (comma, tab, space or semicolon separated). A text header row is skipped.

    Returns (time, flux, errors).
    """
    time, flux, errors = [], [], []
    seen_data = False
    for i, raw in enumerate(text.splitlines()):
        line = raw.strip()
        if not line:
            continue
        parts = [p for p in re.split(r"[\s,;]+", line) if p]
        numeric = all(_NUM.match(p) for p in parts)
        if not numeric and not seen_data and len(parts) >= 2 and not any(_NUM.match(p) for p in parts):
            continue                                              # header row
        if len(parts) != 2 or not numeric:
            if len(errors) < 5:
                errors.append(f'Line {i + 1}: expected two numbers (time and flux), found "{line[:40]}".')
            continue
        seen_data = True
        time.append(float(parts[0]))
        flux.append(float(parts[1]))
    return time, flux, errors


def build_flare(flare_id: str, star_id: str, unit: str, tpeak: str, points: str, existing_ids=()):
    """Check the manual-entry fields. Returns (flare_or_None, errors, warnings).

    ``unit`` is "minutes" (time = minutes from the peak, stored as days with tpeak 0) or "btjd".
    """
    errors, warnings = [], []
    flare_id = (flare_id or "").strip()
    if not flare_id:
        errors.append("Enter a flare id.")
    elif flare_id in existing_ids:
        errors.append(f'A flare called "{flare_id}" is already in the list. Use a different id.')
    if not re.fullmatch(r"\d+", (star_id or "").strip()):
        errors.append("Star id must be a whole number (the TIC number).")
    peak = 0.0
    if unit == "btjd":
        if not _NUM.match((tpeak or "").strip()):
            errors.append("Enter the flare peak time in BTJD (Julian date − 2457000).")
        else:
            peak = float(tpeak.strip())
    time, flux, perrors = parse_points(points or "")
    errors.extend(perrors)
    if not perrors and len(time) < 2:
        errors.append("Enter at least two data points (time and flux).")
    if errors:
        return None, errors, warnings
    days = [m / 1440 for m in time] if unit == "minutes" else time
    inside = sum(1 for d in days if -20 <= (d - peak) * 1440 <= 90)
    if inside < 20:
        warnings.append(f"Only {inside} point(s) fall between −20 and +90 minutes from the peak. At least 20 are needed, "
                        "otherwise the flare will not be scored.")
    if any(v <= 0 for v in flux):
        warnings.append("Some flux values are zero or negative. Flux should be on a positive scale.")
    spacing = median_spacing_minutes(days)
    if spacing is not None and not 1.5 <= spacing <= 2.5:
        shown = f"{spacing * 60:.0f} seconds" if spacing < 1 else f"{spacing:.1f} minutes"
        warnings.append(f"The spacing between your points is about {shown}. The model expects 2-minute data and does "
                        "not re-bin, so the result may not be reliable.")
    flare = {"flare_id": flare_id, "star_id": int(star_id.strip()), "tpeak": peak, "time": days, "flux": flux}
    return flare, errors, warnings


def median_spacing_minutes(days):
    """Median gap between consecutive points, in minutes (times in days); None if it cannot be worked out."""
    s = sorted(days)
    gaps = sorted(g for g in ((b - a) * 1440 for a, b in zip(s, s[1:])) if g > 1e-9)
    if not gaps:
        return None
    mid = len(gaps) // 2
    return gaps[mid] if len(gaps) % 2 else (gaps[mid - 1] + gaps[mid]) / 2


def example_points() -> str:
    """A synthetic demo flare (fast rise, exponential decay), only to try the form. Not real data."""
    lines = []
    for t in range(-20, 90, 2):
        f = 1.0 if t < -4 else 1 + 0.5 * (t + 4) / 4 if t < 0 else 1 + 0.5 * math.exp(-t / 12)
        lines.append(f"{t}, {f:.4f}")
    return "\n".join(lines)

