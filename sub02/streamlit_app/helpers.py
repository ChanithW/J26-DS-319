"""Display rules shared by all views: formats, fixed explanation texts, CSV export, table building."""
from __future__ import annotations

import io

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
