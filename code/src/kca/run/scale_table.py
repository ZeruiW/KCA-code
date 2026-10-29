"""Appendix table: the paper's headline numbers at the two training scales.

The main text reports the full-scale backbones (official width, 2048 points per frame,
trained on an A100); the laptop-scale run (reduced width for P4Transformer and Mamba4D,
1024 points, trained on an Apple M4) is the second training scale. Both `numbers.tex`
files are written by `kca.run.paper_tables`; this writes the table body.

Run:  python -m kca.run.scale_table --laptop results/msr/numbers_laptop.tex \
          --full ../paper/tables/numbers.tex --out ../paper/tables/tab_scale.tex
"""
from __future__ import annotations

import argparse

from kca.run.compare_full import read_macros

# (row label, macro, kind): kind "p" renders a p-value macro as $p...$, "acc" adds %
ROWS = [
    ("Test accuracy, P4Transformer (\\%)", "BBaccPFT", "acc"),
    ("Test accuracy, PSTNet (\\%)", "BBaccPST", "acc"),
    ("Test accuracy, Mamba4D (\\%)", "BBaccMBA", "acc"),
    None,
    ("P4T Del.: \\method\\ (context)", "NumPFTKcaCtxDel", ""),
    ("P4T Del.: region occlusion", "NumPFTOcclDel", ""),
    ("P4T Del.: Integrated Gradients", "NumPFTIGDel", ""),
    ("P4T Del.: context vs.\\ occlusion", "NumPKcaCtxOcclDel", "p"),
    ("P4T Del.: context vs.\\ IG", "NumPKcaCtxIGDel", "p"),
    ("PSTNet Del.: context / occlusion", ("NumPSTKcaCtxDel", "NumPSTOcclDel"), ""),
    ("Mamba4D Del.: context / occlusion", ("NumMBAKcaCtxDel", "NumMBAOcclDel"), ""),
    ("Mamba4D Del.: context vs.\\ IG", "NumPMBAKcaCtxIGDel", "p"),
    ("P4T $\\Smot$ EBPG: context / IG", ("NumPFTKcaCtxSmotEBPG", "NumPFTIGEBPG"), ""),
    None,
    ("Crossover P4T: motion / shape task", ("NumXMotShare", "NumXGeoShare"), ""),
    ("Crossover PSTNet: motion / shape task", ("NumXPstMotShare", "NumXPstGeoShare"), ""),
    ("Crossover Mamba4D: motion / shape task", ("NumXMbaMotShare", "NumXMbaGeoShare"), ""),
    None,
    ("Randomised model: \\method\\ / Grad$\\times$Input", ("NumSanKca", "NumSanGxI"), ""),
]


def cell(mac: dict[str, str], key, kind: str) -> str:
    keys = key if isinstance(key, tuple) else (key,)
    vals = [mac.get(k) for k in keys]
    if any(v is None for v in vals):
        return "--"
    if kind == "p":
        return f"$p{vals[0]}$"
    return " / ".join(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--laptop", required=True)
    ap.add_argument("--full", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    lap, full = read_macros(a.laptop), read_macros(a.full)
    lines = []
    for r in ROWS:
        if r is None:
            lines.append("\\midrule")
            continue
        label, key, kind = r
        lines.append(f"{label} & {cell(lap, key, kind)} & {cell(full, key, kind)}\\\\")
    open(a.out, "w").write("\n".join(lines) + "\n")
    print(f"[scale table] -> {a.out}")


if __name__ == "__main__":
    main()
