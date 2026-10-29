"""Compare two generations of the paper's numbers (laptop-scale vs full-scale) and check
whether each claim the text makes still holds.

Inputs are two `numbers.tex` files written by `kca.run.paper_tables` (every number the
text quotes, as macros). Writes to --out:
  numbers_compare.csv   every macro: old, new, new - old
  REPORT.md             backbone accuracies, the claim checks on both versions, headline
                        numbers side by side, and the lines of the text that describe the
                        laptop-scale setup (to be rewritten for the full-scale version)

Run:  python -m kca.run.compare_full --old ../paper/tables/numbers.tex \
          --new <paper_full>/tables/numbers.tex --paper <paper_full> --out <report dir>
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import re

PUBLISHED = {"PFT": 90.9, "PST": 91.2, "MBA": 92.7}
BB = {"PFT": "P4Transformer", "PST": "PSTNet", "MBA": "Mamba4D"}
XB = {"PFT": "", "PST": "Pst", "MBA": "Mba"}              # crossover macro infix
PB = {"PFT": "", "PST": "PST", "MBA": "MBA"}              # p-value macro infix


def read_macros(path: str) -> dict[str, str]:
    pat = re.compile(r"\\providecommand\{\\([A-Za-z]+)\}\{(.*)\}\s*$")
    out = {}
    for line in open(path):
        m = pat.match(line.strip())
        if m:
            out[m.group(1)] = m.group(2)
    return out


def num(v: str | None) -> float | None:
    """'0.178' / '$-$0.01' / '{=}0.0043' / '{<}10^{-3}' -> float (the bound for '<')."""
    if v is None:
        return None
    v = v.replace("$-$", "-").replace("{=}", "")
    if v.startswith("{<}10^{-"):
        return 10.0 ** -int(re.findall(r"\d+", v)[-1])
    try:
        return float(v)
    except ValueError:
        return None


XACC = 60   # binary crossover tasks: a model below this accuracy has not learned its cue


def claims() -> list[tuple[str, list[str], callable]]:
    """(statement, macros shown, test on the numbers). A test gets a lookup g(name)."""
    C = []
    for b in ("PFT", "PST", "MBA"):
        C.append((f"{BB[b]}: accuracy within 2 points of the published value", [f"BBacc{b}"],
                  lambda g, b=b: g(f"BBacc{b}") >= PUBLISHED[b] - 2))
    C.append(("P4T: KCA-ctx deletion AUC below region occlusion, significant after Bonferroni correction (p < 0.0125)",
              ["NumPFTKcaCtxDel", "NumPFTOcclDel", "NumPKcaCtxOcclDel"],
              lambda g: g("NumPFTKcaCtxDel") < g("NumPFTOcclDel") and g("NumPKcaCtxOcclDel") < 0.0125))
    for b in ("PFT", "PST", "MBA"):
        p = PB[b]
        C.append((f"{BB[b]}: KCA-ctx deletion AUC significantly below IG (p < 0.05)",
                  [f"Num{b}KcaCtxDel", f"Num{b}IGDel", f"NumP{p}KcaCtxIGDel"],
                  lambda g, b=b, p=p: g(f"Num{b}KcaCtxDel") < g(f"Num{b}IGDel") and g(f"NumP{p}KcaCtxIGDel") < 0.05))
        C.append((f"{BB[b]}: S_mot EBPG on par with or better than IG (not significantly worse)",
                  [f"Num{b}KcaCtxSmotEBPG", f"Num{b}IGEBPG", f"NumP{p}KcaCtxIGEBPG"],
                  lambda g, b=b, p=p: g(f"Num{b}KcaCtxSmotEBPG") >= g(f"Num{b}IGEBPG") or g(f"NumP{p}KcaCtxIGEBPG") >= 0.05))
    for b in ("PST", "MBA"):
        C.append((f"{BB[b]}: KCA-ctx deletion AUC on par with or better than region occlusion (not significantly worse)",
                  [f"Num{b}KcaCtxDel", f"Num{b}OcclDel", f"NumP{b}KcaCtxOcclDel"],
                  lambda g, b=b: g(f"Num{b}KcaCtxDel") <= g(f"Num{b}OcclDel") or g(f"NumP{b}KcaCtxOcclDel") >= 0.05))
    # a crossover share is only meaningful if both binary task models learned their task (chance = 50 %)
    for b in ("PFT", "PST", "MBA"):
        x = XB[b]
        C.append((f"Crossover {BB[b]}: both task models learned (accuracy ≥ {XACC}%), motion share of the motion task above that of the shape task, 95% CI of the gap excludes 0",
                  [f"NumX{x}MotAcc", f"NumX{x}GeoAcc", f"NumX{x}MotShare", f"NumX{x}GeoShare", f"NumX{x}GapLo", f"NumX{x}GapHi"],
                  lambda g, x=x: min(g(f"NumX{x}MotAcc"), g(f"NumX{x}GeoAcc")) >= XACC
                  and g(f"NumX{x}MotShare") > g(f"NumX{x}GeoShare") and g(f"NumX{x}GapLo") > 0))
    for S in ("One", "Two"):
        C.append((f"Crossover P4T seed {'1' if S == 'One' else '2'}: both task models learned (accuracy ≥ {XACC}%), same direction",
                  [f"NumXSeed{S}MotAcc", f"NumXSeed{S}GeoAcc", f"NumXSeed{S}MotShare", f"NumXSeed{S}GeoShare"],
                  lambda g, S=S: min(g(f"NumXSeed{S}MotAcc"), g(f"NumXSeed{S}GeoAcc")) >= XACC
                  and g(f"NumXSeed{S}MotShare") > g(f"NumXSeed{S}GeoShare")))
    C.append(("Method ranking consistent across backbones (Kendall τ > 0.5)", ["NumTauPSTPFT", "NumTauPSTMBA", "NumTauPFTMBA"],
              lambda g: min(g("NumTauPSTPFT"), g("NumTauPSTMBA"), g("NumTauPFTMBA")) > 0.5))
    C.append(("Sanity: after full randomization the KCA rank correlation ≈ 0 (absolute value < 0.2), Grad×Input still > 0.5", ["NumSanKca", "NumSanGxI"],
              lambda g: abs(g("NumSanKca")) < 0.2 and g("NumSanGxI") > 0.5))
    C.append(("Ablation: at m=24 the context estimator repairs the collapse of local necessity", ["NumAblCtxFineDel", "NumAblLocalFineDel"],
              lambda g: g("NumAblCtxFineDel") < g("NumAblLocalFineDel")))
    C.append(("displace probe: freeze changes the prediction more often than trajectory reversal", ["NumDspFreezeFlip", "NumDspReverseFlip"],
              lambda g: g("NumDspFreezeFlip") > g("NumDspReverseFlip")))
    return C


def check(test, mac):
    def g(k):
        v = num(mac.get(k))
        if v is None:
            raise KeyError(k)
        return v
    try:
        return "✓" if test(g) else "✗"
    except KeyError:
        return "n/a"


HEADLINE = [("BBaccPFT", "P4T accuracy %"), ("BBaccPST", "PSTNet accuracy %"), ("BBaccMBA", "Mamba4D accuracy %"),
            ("NumPFTKcaCtxDel", "P4T KCA-ctx Del"), ("NumPFTOcclDel", "P4T region occlusion Del"),
            ("NumPFTIGDel", "P4T IG Del"), ("NumPKcaCtxOcclDel", "P4T p(KCA-ctx vs occlusion)"),
            ("NumPSTKcaCtxDel", "PSTNet KCA-ctx Del"), ("NumPSTOcclDel", "PSTNet region occlusion Del"),
            ("NumMBAKcaCtxDel", "Mamba4D KCA-ctx Del"), ("NumMBAOcclDel", "Mamba4D region occlusion Del"),
            ("NumXMotShare", "Crossover P4T motion-task share"), ("NumXGeoShare", "Crossover P4T shape-task share"),
            ("NumXPstMotShare", "Crossover PSTNet motion"), ("NumXPstGeoShare", "Crossover PSTNet shape"),
            ("NumXMbaMotShare", "Crossover Mamba4D motion"), ("NumXMbaGeoShare", "Crossover Mamba4D shape"),
            ("NumPFTKcaTime", "P4T KCA-local s/clip"), ("NumPFTKcaCtxTime", "P4T KCA-ctx s/clip")]

# text that describes the laptop-scale setup and must be rewritten for full-scale numbers
SETUP_TEXT = re.compile(r"laptop|reduced width|Widths are reduced|1024 points|dim 256|batch 8|50/30/20|Apple|\bCPU\b|"
                        r"fall short|matches its published", re.I)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--old", required=True, help="numbers.tex of the current (laptop-scale) paper")
    p.add_argument("--new", required=True, help="numbers.tex regenerated from the full-scale results")
    p.add_argument("--paper", required=True, help="paper directory whose sec/*.tex to scan")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    old, new = read_macros(a.old), read_macros(a.new)

    with open(os.path.join(a.out, "numbers_compare.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["macro", "laptop", "full", "full_minus_laptop"])
        for k in sorted(set(old) | set(new)):
            o, n = num(old.get(k)), num(new.get(k))
            w.writerow([k, old.get(k, ""), new.get(k, ""), "" if o is None or n is None else f"{n - o:+.4g}"])

    L = ["# KCA: laptop scale vs full scale — comparison of results", ""]
    L += ["## 1. Do the paper's claims still hold?", "", "| Claim | Laptop | Full scale | Values (laptop → full scale) |", "|---|---|---|---|"]
    n_broken = 0
    for text, keys, test in claims():
        co, cn = check(test, old), check(test, new)
        n_broken += cn == "✗"
        vals = "; ".join(f"`{k}` {old.get(k, '–')} → {new.get(k, '–')}" for k in keys)
        L.append(f"| {text} | {co} | {'**✗**' if cn == '✗' else cn} | {vals} |")
    L += ["", f"Claims that no longer hold at full scale: **{n_broken}** (✗). Each of them needs the text rewritten, or the experiment re-checked.", ""]

    L += ["## 2. Key numbers", "", "| Quantity | Laptop | Full scale |", "|---|---|---|"]
    L += [f"| {lab} (`{k}`) | {old.get(k, '–')} | {new.get(k, '–')} |" for k, lab in HEADLINE]
    changed = sum(1 for k in set(old) & set(new) if old[k] != new[k])
    L += ["", f"{len(set(old) | set(new))} number macros in total, {changed} of them changed; present on one side only: "
          f"{len(set(old) - set(new))} laptop, {len(set(new) - set(old))} full scale. Full list in `numbers_compare.csv`.", ""]
    missing = sorted(set(old) - set(new))
    if missing:
        L += ["Missing from the full-scale results (the text will show a red ??): " + ", ".join(f"`{k}`" for k in missing), ""]

    L += ["## 3. Text to rewrite (sentences describing the laptop training setup)", "",
          "These sentences describe the laptop-scale setup (width, number of points, hardware, run time) and must be rewritten by hand for the full-scale numbers:", ""]
    for f in sorted(glob.glob(os.path.join(a.paper, "sec", "*.tex"))):
        for i, line in enumerate(open(f), 1):
            if SETUP_TEXT.search(line) and not line.lstrip().startswith("%"):
                hits = sorted({m.group(0) for m in SETUP_TEXT.finditer(line)})
                L.append(f"- `sec/{os.path.basename(f)}:{i}` — matches: {', '.join(hits)}")
    L += ["", "Also: `tab_cost` now holds GPU timings (the laptop version was an M4 CPU), so the Cost paragraph of the main text and the table note in the appendix must change together.", ""]
    with open(os.path.join(a.out, "REPORT.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"[report] {n_broken} claim(s) no longer hold -> {os.path.join(a.out, 'REPORT.md')}")


if __name__ == "__main__":
    main()
