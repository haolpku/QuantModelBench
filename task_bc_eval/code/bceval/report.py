"""Markdown rendering for the rubric tables.

Numbers that are missing stay missing: `fmt` renders `None`/NaN as an em dash
rather than `0.000`, so a dropped component is never mistaken for a zero score.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

MISSING = "—"


def fmt(value, digits: int = 3, pct: bool = False, signed: bool = False) -> str:
    if value is None:
        return MISSING
    try:
        val = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(val):
        return MISSING
    if pct:
        val *= 100.0
    text = f"{val:{'+' if signed else ''}.{digits}f}"
    return text + "%" if pct else text


def table(headers: Sequence[str], rows: Iterable[Sequence], align: Sequence[str] | None = None) -> str:
    """GitHub-flavoured markdown table. `align` entries are 'l', 'c' or 'r'."""
    align = list(align or ["l"] + ["r"] * (len(headers) - 1))
    sep = {"l": "---", "c": ":-:", "r": "---:"}
    body = [
        "| " + " | ".join(str(h) for h in headers) + " |",
        "|" + "|".join(sep.get(a, "---") for a in align) + "|",
    ]
    for row in rows:
        cells = list(row) + [MISSING] * (len(headers) - len(row))
        body.append("| " + " | ".join(MISSING if c is None else str(c) for c in cells) + " |")
    return "\n".join(body)


def ci_text(lo, hi, digits: int = 3, pct: bool = False) -> str:
    if lo is None or hi is None:
        return MISSING
    return f"[{fmt(lo, digits, pct)}, {fmt(hi, digits, pct)}]"


def score_band(score100) -> str:
    """Rubric section 6 bands, on the 0-100 scale."""
    if score100 is None or not math.isfinite(float(score100)):
        return MISSING
    val = float(score100)
    if val >= 80:
        return "80-100 强"
    if val >= 60:
        return "60-80 有价值但有短板"
    if val >= 40:
        return "40-60 接近 baseline"
    return "0-40 证据不足"


def one_pager(name: str, outcome: dict, process: dict, evidence: dict) -> str:
    """Rubric section 6 / report-version section 6 single-method summary block."""
    dropped = process.get("dropped") or []
    dropped_note = f"（已剔除并重归一化：{', '.join(dropped)}）" if dropped else ""
    lines = [
        f"Method: {name}",
        "",
        "Outcome",
        f"  Absolute Quality:  {fmt(outcome.get('q_abs_100'), 1)}/100",
        f"  Baseline Uplift:   {fmt(outcome.get('q_uplift_100'), 1)}/100",
        f"  Outcome Score:     {fmt(outcome.get('outcome_100'), 1)}/100",
        "",
        "Process",
        f"  Reliability:       {fmt(process.get('reliability_100'), 1)}/100",
        f"  Stability:         {fmt(process.get('stability_100'), 1)}/100",
        f"  LLM-as-Judge:      {fmt(process.get('judge_100'), 1)}/100",
        f"  Efficiency:        {fmt(process.get('efficiency_100'), 1)}/100",
        f"  Process Score:     {fmt(process.get('process_100'), 1)}/100{dropped_note}",
        "",
        "Evidence",
        f"  Valid cases/origins: {evidence.get('cases', MISSING)} / {evidence.get('origins', MISSING)}",
        f"  ImprovedRate:        {fmt(evidence.get('improved_rate'), 1, pct=True)}",
        f"  Active IR:           {fmt(evidence.get('active_ir'), 3)}",
        f"  95% block-bootstrap CI: {evidence.get('ci', MISSING)}",
        f"  Pareto status:       {evidence.get('pareto', MISSING)}",
    ]
    return "```text\n" + "\n".join(lines) + "\n```"


def pareto_front(points: dict[str, tuple[float, float]]) -> set[str]:
    """Keys not dominated on both axes (higher is better on each)."""
    front = set()
    for key, (o_a, p_a) in points.items():
        if not (math.isfinite(o_a) and math.isfinite(p_a)):
            continue
        dominated = any(
            key != other
            and math.isfinite(o_b) and math.isfinite(p_b)
            and o_b >= o_a and p_b >= p_a
            and (o_b > o_a or p_b > p_a)
            for other, (o_b, p_b) in points.items()
        )
        if not dominated:
            front.add(key)
    return front


def scatter_ascii(points: dict[str, tuple[float, float]], width: int = 52, height: int = 18) -> str:
    """Plain-text (O, P) scatter so the decision plot survives in markdown."""
    live = {k: v for k, v in points.items() if all(math.isfinite(x) for x in v)}
    if not live:
        return "(no scorable methods)"
    xs = [v[0] for v in live.values()]
    ys = [v[1] for v in live.values()]
    x_lo, x_hi = min(xs), max(xs)
    y_lo, y_hi = min(ys), max(ys)
    x_span = (x_hi - x_lo) or 1.0
    y_span = (y_hi - y_lo) or 1.0
    grid = [[" "] * width for _ in range(height)]
    labels = []
    for i, (key, (x, y)) in enumerate(sorted(live.items())):
        mark = chr(ord("a") + i) if i < 26 else "*"
        col = min(width - 1, int((x - x_lo) / x_span * (width - 1)))
        row = height - 1 - min(height - 1, int((y - y_lo) / y_span * (height - 1)))
        grid[row][col] = mark
        labels.append(f"  {mark} = {key}  (O={x:.1f}, P={y:.1f})")
    body = [f"P {y_hi:6.1f} |" + "".join(r) for r in grid[:1]]
    body += ["        |" + "".join(r) for r in grid[1:-1]]
    body += [f"  {y_lo:6.1f} |" + "".join(grid[-1])]
    body.append("        +" + "-" * width)
    body.append(f"         {x_lo:.1f}" + " " * max(1, width - 12) + f"{x_hi:.1f}  O")
    return "```text\n" + "\n".join(body + [""] + labels) + "\n```"
