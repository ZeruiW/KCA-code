"""Turn evaluate() results into paper-ready LaTeX, following the paper's conventions:
mean$\\pm$std, best **bold**, second \\underline, n/a kept as n/a, arrows in headers.

Writes results.json (raw) and tables.tex (formatted rows) so the numbers can be
\\input or pasted over the \\tbd cells in ../paper/sec/4_experiments.tex.
"""
from __future__ import annotations

import json
from typing import Optional

# lower-is-better metrics (for bold/underline selection and arrows)
LOWER_BETTER = {"deletion_auc", "sufficiency"}
ARROW = {
    "deletion_auc": r"$\downarrow$", "insertion_auc": r"$\uparrow$",
    "comprehensiveness": r"$\uparrow$", "sufficiency": r"$\downarrow$",
    "sensitivity_n": r"$\uparrow$", "pg_moving": r"$\uparrow$", "ebpg_moving": r"$\uparrow$",
}


def _fmt(mean: float, std: float, decimals: int = 3) -> str:
    return f"{mean:.{decimals}f}$\\pm${std:.{decimals}f}"


def _rank_best_second(results: dict, method_names: list[str], metric: str):
    """Return (best_method, second_method) by metric, honouring direction; skip n/a."""
    scored = [(m, results[m][metric]["mean"]) for m in method_names
              if "na" not in results[m] and metric in results[m]]
    if not scored:
        return None, None
    scored.sort(key=lambda kv: kv[1], reverse=(metric not in LOWER_BETTER))
    best = scored[0][0]
    second = scored[1][0] if len(scored) > 1 else None
    return best, second


def format_cell(results: dict, method: str, metric: str, best: str, second: str,
                decimals: int = 3) -> str:
    r = results.get(method, {})
    if "na" in r or metric not in r:
        return r"\emph{n/a}"
    cell = _fmt(r[metric]["mean"], r[metric]["std"], decimals)
    if method == best:
        return r"\textbf{" + cell + "}"
    if method == second:
        return r"\underline{" + cell + "}"
    return cell


def latex_table_body(results: dict, method_rows: list[tuple[str, str]],
                     metric_cols: list[str], decimals: int = 3) -> str:
    """method_rows: list of (method_key, latex_label). metric_cols: metric keys in order."""
    bests = {mc: _rank_best_second(results, [m for m, _ in method_rows], mc) for mc in metric_cols}
    lines = []
    for key, label in method_rows:
        cells = [format_cell(results, key, mc, bests[mc][0], bests[mc][1], decimals)
                 for mc in metric_cols]
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    return "\n".join(lines)


def header_row(metric_cols: list[str], names: dict[str, str]) -> str:
    cols = [names.get(mc, mc) + ARROW.get(mc, "") for mc in metric_cols]
    return "Method & " + " & ".join(cols) + r" \\"


def save(results: dict, out_dir: str, method_rows: list[tuple[str, str]],
         metric_cols: list[str], stem: str = "tab_faithfulness"):
    import os
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{stem}.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    body = latex_table_body(results, method_rows, metric_cols)
    with open(os.path.join(out_dir, f"{stem}.tex"), "w") as fh:
        fh.write(body + "\n")
    return body
