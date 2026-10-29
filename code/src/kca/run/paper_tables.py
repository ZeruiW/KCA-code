"""Make the paper's LaTeX tables from the per-clip benchmark records.

Reads results/msr/<backbone>/raw.jsonl (kca.run.msr_benchmark), results/msr/crossover/
crossover.json (kca.run.msr_crossover), results/msr/ablation/ablation.json
(kca.run.msr_ablation) and writes table bodies to paper/tables/*.tex, plus a JSON digest.

Conventions: mean ± std over clips; best bold, second underlined (floors excluded from
the ranking); * = KCA significantly better than the strongest baseline (paired
two-sided Wilcoxon over clips, Bonferroni-corrected over the table's metrics, p < 0.05).

Run:  python -m kca.run.paper_tables --results results/msr --paper ../paper
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from scipy.stats import kendalltau, wilcoxon

FAITH = [("deletion_auc", "Del.", -1), ("comprehensiveness", "Comp.", +1), ("insertion_auc", "Ins.", +1),
         ("sufficiency", "Suff.", -1)]
FAITH_SUPP = FAITH + [("sensitivity_n", "Sens-$n$", +1)]
NAMES = {
    "random": "Random (floor)", "geometry_edge": "Geometry-edge (floor)",
    "motion_prior": "Motion-prior (floor)", "grad_x_input": r"Grad$\times$Input",
    "smoothgrad": r"SmoothGrad~\cite{smilkov2017smoothgrad}",
    "integrated_gradients": r"Integrated Grad.~\cite{sundararajan2017ig}",
    "grad_cam_3d": r"3D Grad-CAM~\cite{selvaraju2017gradcam}",
    "point_drop": r"Region occlusion~\cite{zheng2019pointcloudsaliency}",
    "occam": r"OccAM~\cite{schinagl2022occam}", "extremal": r"Extremal Pert.~\cite{fong2019extremal}",
    "attn_rollout": r"Attn.\ Rollout~\cite{abnar2020rollout}",
    "transformer_attr": r"Transformer Attr.~\cite{chefer2021transformer}",
    "staa4d": r"Raw attention (last layer)",
    "kca": r"\textbf{\method} (local)", "kca_ctx": r"\textbf{\method} (context)",
}
FLOORS = {"random", "geometry_edge", "motion_prior"}
OURS = {"kca", "kca_ctx"}


def load(path):
    with open(path) as fh:
        return [json.loads(l) for l in fh if l.strip()]


def column(recs, method, metric):
    """Per-clip values (nan where n/a or missing), aligned across methods by clip."""
    return np.array([r["methods"].get(method, {}).get(metric, np.nan) for r in recs], dtype=float)


def fmt(m, s, digits=3):
    v = f"{m:.{digits}f}"
    if float(v) == 0:
        v = f"{0:.{digits}f}"                                # no "-0.000" for a mean that rounds to zero
    v = "$-$" + v[1:] if v.startswith("-") else v            # a real minus sign, not a hyphen
    return v + (rf"{{\scriptsize$\pm${s:.{digits - 1}f}}}" if s is not None else "")


def ranked_cells(recs, methods, metrics, digits=3, std=True):
    """{method: [cell,...]} with bold/underline/significance markers."""
    cells = {m: [] for m in methods}
    for key, _, sign in metrics:
        vals = {m: column(recs, m, key) for m in methods}
        means = {m: np.nanmean(v) if np.isfinite(v).any() else np.nan for m, v in vals.items()}
        cand = [m for m in methods if m not in FLOORS and np.isfinite(means[m])]
        order = sorted(cand, key=lambda m: -sign * means[m])
        base = [m for m in order if m not in OURS]
        strongest = base[0] if base else None
        for m in methods:
            v = vals[m]
            if not np.isfinite(means[m]):
                cells[m].append(r"\emph{n/a}")
                continue
            c = fmt(means[m], np.nanstd(v) if std else None, digits)
            shown = lambda x: round(means[x], digits)              # ranks as the reader sees them: ties share a mark
            if order and shown(m) == shown(order[0]):
                c = rf"\textbf{{{c}}}"
            elif len(order) > 1 and shown(m) == shown(order[1]):
                c = rf"\underline{{{c}}}"
            if m in OURS and strongest is not None:
                ok = np.isfinite(v) & np.isfinite(vals[strongest])
                d = sign * (v[ok] - vals[strongest][ok])
                if ok.sum() > 5 and np.any(d != 0):
                    p = wilcoxon(v[ok], vals[strongest][ok]).pvalue
                    if p * len(metrics) < 0.05 and np.median(d) > 0:
                        c += r"$^\ast$"
            cells[m].append(c)
    return cells


def table_faithfulness(recs, methods, path, metrics=FAITH):
    cells = ranked_cells(recs, methods, metrics)
    lines = []
    first = next((m for m in methods if m in OURS), None)
    for m in methods:
        if m == first:
            lines.append(r"\midrule")
        lines.append(f"{NAMES[m]} & " + " & ".join(cells[m]) + r"\\")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def _with_loc(recs):
    """Add 'loc_ebpg': EBPG of the moving GT, of Smot for KCA (as in the localisation table)."""
    return [{**r, "methods": {m: {**d, "loc_ebpg": d.get("smot_ebpg" if m in OURS else "ebpg", np.nan)}
                              for m, d in r["methods"].items()}} for r in recs]


def table_cross(by_bb, methods, path):
    """Del. and moving-GT EBPG per backbone + pairwise Kendall tau of the method ranking (by Del.)."""
    bbs = list(by_bb)
    cells = {m: [] for m in methods}
    for bb in bbs:
        c = ranked_cells(_with_loc(by_bb[bb]), methods, [FAITH[0], ("loc_ebpg", "EBPG", +1)], std=False)
        for m in methods:
            cells[m] += c[m]
    lines = []
    first = next((m for m in methods if m in OURS), None)
    for m in methods:
        if m == first:
            lines.append(r"\midrule")
        lines.append(f"{NAMES[m]} & " + " & ".join(cells[m]) + r"\\")
    taus = {}
    common = [m for m in methods if all(np.isfinite(np.nanmean(column(by_bb[b], m, "deletion_auc")))
                                        for b in bbs if np.isfinite(column(by_bb[b], m, "deletion_auc")).any())
              and all(np.isfinite(column(by_bb[b], m, "deletion_auc")).any() for b in bbs)]
    for i in range(len(bbs)):
        for j in range(i + 1, len(bbs)):
            a = [np.nanmean(column(by_bb[bbs[i]], m, "deletion_auc")) for m in common]
            b = [np.nanmean(column(by_bb[bbs[j]], m, "deletion_auc")) for m in common]
            taus[f"{bbs[i]}~{bbs[j]}"] = float(kendalltau(a, b).statistic)
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return {"kendall_tau": taus, "common_methods": common}


def table_localization(recs, cross, methods, path):
    """PG/EBPG vs moving GT (Smot for KCA), Sgeo PG vs static GT, and the crossover shares.
    Best non-floor value per GT column in bold."""
    cols = {}
    for m in methods:
        pg = column(recs, m, "smot_pg" if m in OURS else "pg")
        eb = column(recs, m, "smot_ebpg" if m in OURS else "ebpg")
        cols[m] = (np.nanmean(pg), np.nanmean(eb))
    best = [max((cols[m][j] for m in methods if m not in FLOORS), default=None) for j in (0, 1)]
    lines = []
    for m in methods:
        cells = []
        for j in (0, 1):
            c = fmt(cols[m][j], None, 2)
            cells.append(rf"\textbf{{{c}}}" if m not in FLOORS and best[j] is not None and abs(cols[m][j] - best[j]) < 1e-9 else c)
        spg = column(recs, m, "sgeo_pg") if m in OURS else None
        cells.append(fmt(np.nanmean(spg), None, 2) if spg is not None else "--")
        if m in OURS and cross:
            cells += [fmt(cross[t][m]["arm_motion_share"], None, 2) if t in cross else "--" for t in ("motion", "geometry")]
        else:
            cells += ["--", "--"]
        if m == next((o for o in methods if o in OURS), None):
            lines.append(r"\midrule")
        lines.append(f"{NAMES[m]} & " + " & ".join(cells) + r"\\")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


MKEY = {"kca": "Kca", "kca_ctx": "KcaCtx", "point_drop": "Occl", "occam": "Occam", "integrated_gradients": "IG",
        "grad_x_input": "GxI", "extremal": "Extremal", "motion_prior": "MotionPrior", "random": "Random",
        "geometry_edge": "GeoEdge", "smoothgrad": "SmoothGrad", "grad_cam_3d": "GradCAM", "attn_rollout": "Rollout",
        "transformer_attr": "TAttr", "staa4d": "STAA"}
QKEY = {"deletion_auc": "Del", "insertion_auc": "Ins", "comprehensiveness": "Comp", "sufficiency": "Suff",
        "sensitivity_n": "Sens", "ebpg": "EBPG", "pg": "PG", "smot_ebpg": "SmotEBPG", "smot_pg": "SmotPG",
        "sgeo_pg": "SgeoPG", "time_s": "Time", "abs_dint_over_abs_nec": "Dint", "n_forward": "Fwd"}
BKEY = {"p4transformer": "PFT", "pstnet": "PST", "mamba4d": "MBA"}


SEED_WORDS = {1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five"}
STAB_METHODS = ["random", "motion_prior", "grad_x_input", "integrated_gradients", "point_drop", "occam",
                "attn_rollout", "kca", "kca_ctx"]
XACC_MIN = 0.60   # a binary crossover task model below this accuracy has not learned its cue (chance = 0.5)


def table_seeds(results, path):
    """P4T crossover over training seeds: seed 0 is `crossover/`, seed n is `crossover_seed<n>/`."""
    rows, out = [], {"total": 0, "trained": 0, "same_dir": 0, "ci_excl0": 0}
    for seed in [0] + sorted(SEED_WORDS):
        f = os.path.join(results, "crossover" if seed == 0 else f"crossover_seed{seed}", "crossover.json")
        if not os.path.exists(f):
            continue
        c = json.load(open(f))
        if not all("kca" in c.get(t, {}) for t in ("motion", "geometry")):
            continue
        am, ag = c["motion"]["test_acc"], c["geometry"]["test_acc"]
        sm, sg = c["motion"]["kca"]["arm_motion_share"], c["geometry"]["kca"]["arm_motion_share"]
        gap = c.get("ci_kca", {}).get("gap", [float("nan")] * 2)
        trained = min(am, ag) >= XACC_MIN
        out["total"] += 1
        out["trained"] += trained
        out["same_dir"] += trained and sm > sg
        out["ci_excl0"] += trained and gap[0] > 0
        note = "" if trained else r" \emph{(not trained)}"
        rows.append(f"{seed}{note} & {100 * am:.1f} / {100 * ag:.1f} & {sm:.2f} / {sg:.2f} & "
                    f"[{fmt(gap[0], None, 2)}, {fmt(gap[1], None, 2)}]\\\\")
    if rows:
        with open(path, "w") as fh:
            fh.write("\n".join(rows) + "\n")
    return out


def table_stability(path_jsonl, path):
    """Explanation stability under re-sampling (kca.run.msr_stability): mean over clips of the rank
    correlation with sampling 0 (point level, part level) and of the two-sampling averages."""
    recs = load(path_jsonl)
    if not recs:
        return {}
    keys = (("rank_corr", "point"), ("part_corr", "part"), ("avg2_rank_corr", "avg_point"), ("avg2_part_corr", "avg_part"))
    out = {"n_clips": len(recs), "samplings": recs[0]["samplings"], "methods": {}}
    pred_same = [p == r["pred"][0] for r in recs for p in r["pred"][1:]]
    out["pred_same"] = float(np.mean(pred_same))
    for m in STAB_METHODS:
        d = {}
        for k, short in keys:
            v = [np.nanmean(r[k][m]) if isinstance(r[k].get(m), list) else r[k].get(m, np.nan)
                 for r in recs if r.get(k, {}).get(m) not in (None, [])]
            v = np.array(v, dtype=float)
            if np.isfinite(v).any():
                d[short] = float(np.nanmean(v))
        if d:
            out["methods"][m] = d
    for m in ("kca", "kca_ctx"):                     # spread of the whole-body motion share across samplings
        sd = [np.nanstd(r["motion_share"][m]) for r in recs if r.get("motion_share", {}).get(m)]
        if sd:
            out[f"share_sd_{m}"] = float(np.mean(sd))
    lines = []
    for m in STAB_METHODS:
        d = out["methods"].get(m)
        if not d:
            continue
        if m == "kca":
            lines.append(r"\midrule")
        lines.append(f"{NAMES[m]} & " + " & ".join(fmt(d[s], None, 2) if s in d else "--"
                                                    for s in ("point", "part", "avg_point", "avg_part")) + r"\\")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return out


def write_numbers(summary, ckpt_root, results, path):
    """Every number quoted in the text, as LaTeX macros (so text and tables agree)."""
    mac = {}
    for b, k in BKEY.items():
        f = os.path.join(ckpt_root, {"p4transformer": "p4t"}.get(b, b), "summary.json")
        if os.path.exists(f):
            mac[f"BBacc{k}"] = f"{100 * json.load(open(f))['last_acc']:.1f}"
    for b, s in summary.items():
        if b not in BKEY:
            continue
        for m, d in s.items():
            if m not in MKEY or not isinstance(d, dict):
                continue
            for q, v in d.items():
                if q in QKEY and isinstance(v, dict):
                    dig = 1 if q in ("time_s", "n_forward") else (2 if q in ("pg", "ebpg", "smot_ebpg", "smot_pg", "sgeo_pg", "abs_dint_over_abs_nec") else 3)
                    mac[f"Num{BKEY[b]}{MKEY[m]}{QKEY[q]}"] = f"{v['mean']:.{dig}f}"
            if "motion_share_on_moving_gt" in d:
                mac[f"Num{BKEY[b]}{MKEY[m]}MotShareMoving"] = f"{d['motion_share_on_moving_gt']:.2f}"
                mac[f"Num{BKEY[b]}{MKEY[m]}MotShareAll"] = f"{d['motion_share_overall']:.2f}"
        mac[f"Num{BKEY[b]}Clips"] = str(s.get("_n_clips", ""))
        if "_n_five" in s:
            mac[f"Num{BKEY[b]}ClipsFive"] = str(s["_n_five"])
        for k, v in s.get("_tests", {}).items():
            mac[f"Num{k}"] = (f"{{=}}{v:.2f}" if v >= 0.1 else f"{{=}}{v:.2g}") if v >= 1e-3 else r"{<}10^{-3}"
        mac[f"Num{BKEY[b]}SubsetAcc"] = f"{100 * s.get('_accuracy_on_subset', float('nan')):.0f}"
    st = summary.get("stability")
    if st:
        mac["NumStabClips"], mac["NumStabSamplings"] = str(st["n_clips"]), str(st["samplings"])
        mac["NumStabPredSame"] = f"{100 * st['pred_same']:.0f}"
        for m, d in st["methods"].items():
            for k, K in (("point", "Point"), ("part", "Part"), ("avg_point", "AvgPoint"), ("avg_part", "AvgPart")):
                if k in d:
                    mac[f"NumStab{MKEY[m]}{K}"] = fmt(d[k], None, 2)
        for m in ("kca", "kca_ctx"):
            if f"share_sd_{m}" in st:
                mac[f"NumStab{MKEY[m]}ShareSD"] = f"{st[f'share_sd_{m}']:.2f}"
    sd = summary.get("seeds")
    if sd and sd["total"]:
        mac["NumXSeedsTotal"], mac["NumXSeedsTrained"] = str(sd["total"]), str(sd["trained"])
        mac["NumXSeedsSameDir"], mac["NumXSeedsCIExcl"] = str(sd["same_dir"]), str(sd["ci_excl0"])
    gm = summary.get("gt_missing", {})
    if "p4transformer" in gm:
        mac["NumGTMissing"] = str(gm["p4transformer"])
        mac["NumGTValid"] = str(summary["p4transformer"]["_n_clips"] - gm["p4transformer"])
    if "pstnet" in gm:
        mac["NumGTMissingCross"] = str(gm["pstnet"])
    g = summary.get("gt")
    if g:
        mac["NumGTFitMedian"] = f"{100 * g['fit_median']:.0f}"
        mac["NumGTFitPten"] = f"{100 * g['fit_p10']:.0f}"
        mac["NumGTMovingPct"] = f"{100 * g['moving']:.0f}"
        mac["NumGTStaticPct"] = f"{100 * g['static']:.0f}"
    cr = summary.get("cross")
    if cr:
        for pair, tau in cr.get("kendall_tau", {}).items():
            x, y = pair.split("~")
            mac[f"NumTau{BKEY[x]}{BKEY[y]}"] = f"{tau:.2f}"
        mac["NumCrossClips"] = str(cr.get("n_clips", ""))
    cx = summary.get("crossover")
    if cx:
        for task, K in (("motion", "Mot"), ("geometry", "Geo")):
            if task in cx:
                mac[f"NumX{K}Acc"] = f"{100 * cx[task]['test_acc']:.1f}"
                mac[f"NumX{K}Share"] = f"{cx[task]['kca']['arm_motion_share']:.2f}"
                mac[f"NumX{K}ShareCtx"] = f"{cx[task]['kca_ctx']['arm_motion_share']:.2f}"
                mac[f"NumX{K}PriorEBPG"] = f"{cx[task]['motion_prior_ebpg_arm']:.2f}"
                mac[f"NumX{K}NecEBPG"] = f"{cx[task]['kca']['nec_ebpg_arm']:.2f}"
        mac["NumXTest"] = str(cx.get("n_test", ""))
        mac["NumXTrain"] = str(cx.get("n_train", ""))
        if "motion" in cx and "per_clip_pos" in cx["motion"].get("kca", {}):
            mac["NumXEval"] = str(len(cx["motion"]["kca"]["per_clip_pos"]))
        for est, K in (("kca", ""), ("kca_ctx", "Ctx")):
            ci = cx.get(f"ci_{est}")
            if ci:
                mac[f"NumXGap{K}Lo"], mac[f"NumXGap{K}Hi"] = (f"{v:.2f}" for v in ci["gap"])
                mac[f"NumXMot{K}Lo"], mac[f"NumXMot{K}Hi"] = (f"{v:.2f}" for v in ci["motion"])
                mac[f"NumXGeo{K}Lo"], mac[f"NumXGeo{K}Hi"] = (f"{v:.2f}" for v in ci["geometry"])
    thick = os.path.join(results, "crossover", "crossover_beta3.json")
    if os.path.exists(thick):
        t = json.load(open(thick)).get("geometry")
        if t:
            mac["NumXThickAcc"] = f"{100 * t['test_acc']:.1f}"
            mac["NumXThickShare"] = f"{t['kca']['arm_motion_share']:.2f}"
            mac["NumXThickShareSigned"] = f"{t['kca'].get('arm_motion_share_signed', float('nan')):.2f}"
    for seed, S in SEED_WORDS.items():   # P4T crossover retrained with other seeds
        s1 = os.path.join(results, f"crossover_seed{seed}", "crossover.json")
        if not os.path.exists(s1):
            continue
        c1 = json.load(open(s1))
        for task, K in (("motion", "Mot"), ("geometry", "Geo")):
            if task in c1:
                mac[f"NumXSeed{S}{K}Share"] = f"{c1[task]['kca']['arm_motion_share']:.2f}"
                mac[f"NumXSeed{S}{K}Acc"] = f"{100 * c1[task]['test_acc']:.1f}"
        if c1.get("ci_kca"):
            mac[f"NumXSeed{S}GapLo"], mac[f"NumXSeed{S}GapHi"] = (f"{v:.2f}" for v in c1["ci_kca"]["gap"])
    for sub, B in (("crossover_pstnet", "Pst"), ("crossover_mamba4d", "Mba")):   # the same test on other backbones
        xp = os.path.join(results, sub, "crossover.json")
        if not os.path.exists(xp):
            continue
        c = json.load(open(xp))
        for task, K in (("motion", "Mot"), ("geometry", "Geo")):
            if task in c and "kca" in c[task]:
                mac[f"NumX{B}{K}Acc"] = f"{100 * c[task]['test_acc']:.1f}"
                mac[f"NumX{B}{K}Share"] = f"{c[task]['kca']['arm_motion_share']:.2f}"
                mac[f"NumX{B}{K}ShareCtx"] = f"{c[task]['kca_ctx']['arm_motion_share']:.2f}"
                mac[f"NumX{B}{K}ShareSigned"] = f"{c[task]['kca'].get('arm_motion_share_signed', float('nan')):.2f}"
                mac[f"NumX{B}{K}PriorEBPG"] = f"{c[task]['motion_prior_ebpg_arm']:.2f}"
        for est, K in (("kca", ""), ("kca_ctx", "Ctx")):
            ci = c.get(f"ci_{est}")
            if ci:
                mac[f"NumX{B}Gap{K}Lo"], mac[f"NumX{B}Gap{K}Hi"] = (f"{v:.2f}" for v in ci["gap"])
                mac[f"NumX{B}Mot{K}Lo"], mac[f"NumX{B}Mot{K}Hi"] = (f"{v:.2f}" for v in ci["motion"])
                mac[f"NumX{B}Geo{K}Lo"], mac[f"NumX{B}Geo{K}Hi"] = (f"{v:.2f}" for v in ci["geometry"])
    thin = os.path.join(results, "crossover", "crossover_thin.json")
    if os.path.exists(thin) and "geometry" in json.load(open(thin)):
        mac["NumXThinAcc"] = f"{100 * json.load(open(thin))['geometry']['test_acc']:.0f}"
    if cx and "motion" in cx:
        mac["NumXMotShareSigned"] = f"{cx['motion']['kca'].get('arm_motion_share_signed', float('nan')):.2f}"
        if "geometry" in cx:
            mac["NumXGeoShareSigned"] = f"{cx['geometry']['kca'].get('arm_motion_share_signed', float('nan')):.2f}"
    ctrl = os.path.join(os.path.dirname(results.rstrip("/")), "controlled", "crossover.json")
    if os.path.exists(ctrl):
        c = json.load(open(ctrl))
        for key, K in (("motion", "CtrlMot"), ("shape", "CtrlShape"), ("motion_ctx", "CtrlMotCtx"), ("shape_ctx", "CtrlShapeCtx")):
            if key in c:
                mac[f"Num{K}"] = f"{c[key]['odd_motion_fraction_pooled']:.2f}"
    cab = os.path.join(os.path.dirname(results.rstrip("/")), "controlled", "ablations.json")
    if os.path.exists(cab):
        C = json.load(open(cab))
        if "bg_symmetric" in C:
            mac["NumCtrlBgShape"] = f"{100 * C['bg_symmetric']['shape_odd_motion_share']:.0f}"
            mac["NumCtrlProxyDint"] = f"{C['proxy_symmetric'].get('motion_dint_over_nec', float('nan')):.2f}"
            mac["NumCtrlBgDint"] = f"{C['bg_symmetric'].get('shape_dint_over_nec', float('nan')):.2f}"
    dp = os.path.join(os.path.dirname(results.rstrip("/")), "controlled", "direction_probe.json")
    if os.path.exists(dp):
        d = json.load(open(dp))
        mac["NumCtrlDspFlip"] = f"{100 * d['displace_flip_to_opposite']['mean']:.0f}"
        mac["NumCtrlFrzFlip"] = f"{100 * d['freeze_flip_to_opposite']['mean']:.0f}"
    ab = os.path.join(results, "ablation", "ablation.json")
    if os.path.exists(ab):
        A = json.load(open(ab))
        V = A.get("ablation", {})
        for key, K in (("local_m24", "LocalFine"), ("ctx_m24", "CtxFine"), ("local_m12", "Local"), ("ctx_m12", "Ctx"),
                       ("local_m12_bg", "LocalBg"), ("ctx_m12_bg", "CtxBg"), ("local_m12_motionfirst", "LocalMotionFirst"),
                       ("local_m12_nn", "LocalNn"), ("ctx_m12_S64", "CtxSsixtyfour"),
                       ("ctx_m12_S512", "CtxSfivetwelve"), ("local_m12_t0mid", "LocalMid"), ("ctx_m12_t0mid", "CtxMid")):
            if key in V:
                mac[f"NumAbl{K}Del"] = f"{V[key]['deletion_auc']:.3f}"
                mac[f"NumAbl{K}Share"] = f"{V[key]['motion_share_moving']:.2f}"
                mac[f"NumAbl{K}EBPG"] = f"{V[key]['smot_ebpg']:.2f}"
        if A.get("sanity"):
            mac["NumSanKca"] = f"{A['sanity'][-1]['kca_rankcorr']:.2f}"
            mac["NumSanGxI"] = f"{A['sanity'][-1]['gradxinput_rankcorr']:.2f}"
            mac["NumSanHeadKca"] = f"{A['sanity'][0]['kca_rankcorr']:.2f}"
        if A.get("displace"):
            for k, K in (("freeze_flip", "DspFreezeFlip"), ("reverse_flip", "DspReverseFlip"),
                         ("freeze_drop", "DspFreezeDrop"), ("reverse_drop", "DspReverseDrop")):
                v = A["displace"][k]
                mac[f"Num{K}"] = f"{100 * v:.0f}" if "flip" in k else f"{v:.2f}"
    with open(path, "w") as fh:
        fh.write("% auto-generated by kca.run.paper_tables -- do not edit\n")
        for k, v in sorted(mac.items()):
            if v.startswith("-"):                   # text-mode minus sign (p-value macros never start with "-")
                v = "$-$" + v[1:]
            fh.write(f"\\providecommand{{\\{k}}}{{{v}}}\n")
    return mac


ABL_ROWS = [("local_m6", r"local, $m{=}6$"), ("local_m12", r"local, $m{=}12$ (default)"), ("local_m24", r"local, $m{=}24$"),
            ("ctx_m6", r"context, $m{=}6$"), ("ctx_m12", r"context, $m{=}12$ (default)"), ("ctx_m24", r"context, $m{=}24$"),
            ("local_m12_bg", r"local, removal instead of proxy"), ("ctx_m12_bg", r"context, removal instead of proxy"),
            ("local_m12_nn", r"local, donor fill instead of proxy"), ("local_m12_motionfirst", r"local, motion-first order"),
            ("ctx_m12_S64", r"context, $S{=}64$"), ("ctx_m12_S128", r"context, $S{=}128$"), ("ctx_m12_S512", r"context, $S{=}512$"),
            ("ctx_m12_q0.75", r"context, $q{=}0.75$"), ("local_m12_t0mid", r"local, anchor $t_0$ = middle frame"),
            ("ctx_m12_t0mid", r"context, anchor $t_0$ = middle frame")]


def table_ablation(ab_path, path):
    A = json.load(open(ab_path)).get("ablation", {})
    lines = []
    for key, lab in ABL_ROWS:
        if key not in A:
            continue
        r = A[key]
        if key == "ctx_m6":
            lines.append(r"\midrule")
        if key == "local_m12_bg":
            lines.append(r"\midrule")
        cells = [f"{r['deletion_auc']:.3f}", f"{r['smot_ebpg']:.2f}", f"{r['motion_share_moving']:.2f}",
                 "--" if r["dint_over_nec"] != r["dint_over_nec"] else f"{r['dint_over_nec']:.2f}",
                 f"{r['n_forward']:.0f}", f"{r['time_s']:.1f}"]
        lines.append(f"{lab} & " + " & ".join(cells) + r"\\")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def table_controlled_abl(path_json, path):
    A = json.load(open(path_json))
    rows = [("proxy_symmetric", "static proxy (default), symmetric"), ("proxy_motion_first", "static proxy, motion-first"),
            ("bg_symmetric", "deletion, symmetric"), ("bg_motion_first", "deletion, motion-first"),
            ("nn_symmetric", "donor fill, symmetric"), ("nn_motion_first", "donor fill, motion-first")]
    lines = [r"\begin{tabular}{l ccc}", r"\toprule", r"$(0,0)$ corner, order & mot. & shape & Del.$\downarrow$\\", r"\midrule"]
    for k, lab in rows:
        if k not in A:
            continue
        r = A[k]
        dl = r.get("deletion_auc", {}).get("mean", float("nan"))
        lines.append(f"{lab} & {r['motion_odd_motion_share']:.2f} & {r['shape_odd_motion_share']:.2f} & {dl:.3f}" + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def table_cost(recs, methods, path):
    lines = []
    for m in methods:
        t = column(recs, m, "time_s")
        if not np.isfinite(t).any():
            continue
        lines.append(f"{NAMES[m]} & {np.nanmean(t):.2f}" + r"\\")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


LOC_KEYS = ("pg", "ebpg", "smot_pg", "smot_ebpg", "sgeo_pg", "sgeo_ebpg", "smot_moving", "sgeo_moving")


def skeleton_complete(data, clip, frames=24):
    """True if every frame of the clip's centred window has a skeleton. Some MSR-Action3D videos store
    all-zero joints for a run of frames; the nearest-bone labels (moving/static ground truth) are then
    meaningless, so such clips are left out of every ground-truth-based number."""
    d = np.load(os.path.join(data, clip), allow_pickle=True)
    n = len(d["point_clouds"])
    s = max(0, (n - frames) // 2)
    return bool((np.abs(d["joints"][s:s + frames]).sum((1, 2)) > 0).all())


def gt_stats(recs, data, fit_units=25.0):
    """Skeleton ground truth on the clips the benchmark explained (same sampling as the
    benchmark): share of points within `fit_units` raw depth units of their nearest bone
    (skeleton fit; 1 unit is about 5 mm), and the share of points labelled moving / static."""
    from ..data.msr_action3d import BONES, MSRAction3D
    npts = recs[0].get("num_points", 1024) if recs else 1024
    ds = MSRAction3D(data, "test", num_points=npts, region_method="tracked", region_k=12)
    names = [os.path.basename(f) for f in ds.files]
    fit, mov, sta = [], [], []
    for r in recs:
        if r.get("gt_missing"):
            continue
        i = names.index(r["clip"])
        b = ds.get(i)
        pts = b.clip.points[0].numpy() * 300.0                            # back to raw units
        J = np.load(ds.files[i], allow_pickle=True)["joints"][ds._window(len(np.load(ds.files[i], allow_pickle=True)["point_clouds"]))]
        a_, b_ = J[:, [x[0] for x in BONES]], J[:, [x[1] for x in BONES]]  # (T,nb,3)
        ab = b_ - a_
        t = np.clip(((pts[:, :, None] - a_[:, None]) * ab[:, None]).sum(-1)
                    / np.maximum((ab ** 2).sum(-1), 1e-9)[:, None], 0, 1)  # (T,N,nb)
        d = np.linalg.norm(pts[:, :, None] - (a_[:, None] + t[..., None] * ab[:, None]), axis=-1).min(-1)
        fit.append(float((d <= fit_units).mean()))
        mov.append(float(b.moving_gt.float().mean())); sta.append(float(b.static_gt.float().mean()))
    return {"n": len(fit), "fit_median": float(np.median(fit)), "fit_p10": float(np.percentile(fit, 10)),
            "moving": float(np.mean(mov)), "static": float(np.mean(sta)), "points": npts}


def positive_shares(recs, data, regions=12):
    """Motion share of the POSITIVE parts, Smot+ / (Smot+ + Sgeo+), on moving-GT points and
    overall, from the stored per-region scores and the (deterministic) clips/regions/GT."""
    from ..data.msr_action3d import MSRAction3D
    npts = recs[0].get("num_points", 1024) if recs else 1024      # the clips the benchmark explained
    ds = MSRAction3D(data, "test", num_points=npts, region_method="tracked", region_k=regions)
    names = [os.path.basename(f) for f in ds.files]
    out = {}
    for m in ("kca", "kca_ctx"):
        mv = [0.0, 0.0]; al = [0.0, 0.0]
        per_class: dict[int, list[float]] = {}
        for r in recs:
            if m not in r["methods"] or "region_smot" not in r["methods"][m]:
                continue
            b = ds.get(names.index(r["clip"]))
            reg = b.clip.regions[0]
            c_all = [0.0, 0.0]
            for rid, (a_, g_) in enumerate(zip(r["methods"][m]["region_smot"], r["methods"][m]["region_sgeo"])):
                n_mv = 0 if r.get("gt_missing") else int(((reg == rid) & b.moving_gt[0]).sum())
                n_all = int((reg == rid).sum())
                mv[0] += max(a_, 0) * n_mv; mv[1] += max(g_, 0) * n_mv
                al[0] += max(a_, 0) * n_all; al[1] += max(g_, 0) * n_all
                c_all[0] += max(a_, 0) * n_all; c_all[1] += max(g_, 0) * n_all
            if sum(c_all) > 0:
                per_class.setdefault(r["label"], []).append(c_all[0] / sum(c_all))
        if mv[0] + mv[1] > 0:
            out[m] = {"moving": mv[0] / (mv[0] + mv[1]), "overall": al[0] / (al[0] + al[1]),
                      "per_class": {c: float(np.mean(v)) for c, v in sorted(per_class.items())}}
    return out


def table_per_class(recs, shares, path):
    """Per action: motion share of KCA (context) and deletion AUC of KCA vs region occlusion."""
    from ..data.msr_action3d import ACTIONS
    lines = []
    pc = shares.get("kca_ctx", {}).get("per_class", {})
    for c in sorted({r["label"] for r in recs}):
        sub = [r for r in recs if r["label"] == c]
        dk = np.nanmean([r["methods"]["kca_ctx"]["deletion_auc"] for r in sub if "kca_ctx" in r["methods"]])
        do = np.nanmean([r["methods"]["point_drop"]["deletion_auc"] for r in sub if "point_drop" in r["methods"]])
        acc = np.mean([r["label"] == r["pred"] for r in sub])
        name = ACTIONS[c].replace("&", r"\&")
        lines.append(f"{name} & {pc.get(c, float('nan')):.2f} & {dk:.3f} & {do:.3f} & {100 * acc:.0f}" + r"\\")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def digest(recs, methods):
    out = {}
    for m in methods:
        d = {}
        for key in ["deletion_auc", "insertion_auc", "comprehensiveness", "sufficiency", "sensitivity_n",
                    "pg", "ebpg", "smot_pg", "smot_ebpg", "sgeo_pg", "time_s", "n_forward",
                    "abs_dint_over_abs_nec"]:
            v = column(recs, m, key)
            if np.isfinite(v).any():
                d[key] = {"mean": float(np.nanmean(v)), "std": float(np.nanstd(v)), "n": int(np.isfinite(v).sum())}
        if m in OURS:
            sm = column(recs, m, "smot_moving"); sg = column(recs, m, "sgeo_moving")
            d["motion_share_on_moving_gt"] = float(np.nansum(sm) / (np.nansum(sm) + np.nansum(sg)))
            st, gt = column(recs, m, "smot_total"), column(recs, m, "sgeo_total")
            d["motion_share_overall"] = float(np.nansum(st) / (np.nansum(st) + np.nansum(gt)))
        out[m] = d
    out["_accuracy_on_subset"] = float(np.mean([r["label"] == r["pred"] for r in recs]))
    out["_n_clips"] = len(recs)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results/msr")
    p.add_argument("--paper", default="../paper")
    p.add_argument("--ckpt-root", default="~/kca_work/ckpt")
    p.add_argument("--data", default="~/kca_work/data/msr", help="npz dir (for positive-part motion shares)")
    a = p.parse_args()
    tdir = os.path.join(a.paper, "tables")
    os.makedirs(tdir, exist_ok=True)
    summary = {}
    bbs = [b for b in ("pstnet", "p4transformer", "mamba4d") if os.path.exists(os.path.join(a.results, b, "raw.jsonl"))]
    recs = {b: load(os.path.join(a.results, b, "raw.jsonl")) for b in bbs}
    n_gt_missing = {}
    if a.data:                                     # no ground truth where the skeleton is missing
        for b in bbs:
            n_gt_missing[b] = 0
            for r in recs[b]:
                if not skeleton_complete(os.path.expanduser(a.data), r["clip"]):
                    r["gt_missing"] = True
                    n_gt_missing[b] += 1
                    for d in r["methods"].values():
                        for k in LOC_KEYS:
                            d.pop(k, None)
    cross_path = os.path.join(a.results, "crossover", "crossover.json")
    cross = json.load(open(cross_path)) if os.path.exists(cross_path) else None
    if "p4transformer" in recs:
        main_methods = [m for m in ["random", "geometry_edge", "motion_prior", "grad_x_input", "smoothgrad",
                                    "integrated_gradients", "grad_cam_3d", "point_drop", "occam", "extremal",
                                    "attn_rollout", "transformer_attr", "staa4d", "kca", "kca_ctx"]
                        if any(m in r["methods"] for r in recs["p4transformer"])]
        table_faithfulness(recs["p4transformer"], main_methods, os.path.join(tdir, "tab_faithfulness.tex"))
        table_faithfulness(recs["p4transformer"], main_methods, os.path.join(tdir, "tab_faithfulness_supp.tex"), FAITH_SUPP)
        loc_methods = [m for m in ["grad_x_input", "integrated_gradients", "grad_cam_3d", "point_drop", "occam",
                                   "attn_rollout", "motion_prior", "kca", "kca_ctx"]
                       if m in main_methods]
        table_localization(recs["p4transformer"], cross, loc_methods, os.path.join(tdir, "tab_localization.tex"))
    for b in bbs:
        summary[b] = digest(recs[b], sorted({m for r in recs[b] for m in r["methods"]}))
        if a.data:
            sh = positive_shares(recs[b], os.path.expanduser(a.data))
            for m, v in sh.items():
                summary[b][m]["motion_share_on_moving_gt"] = v["moving"]
                summary[b][m]["motion_share_overall"] = v["overall"]
                summary[b][m]["motion_share_per_class"] = v["per_class"]
            if b == "p4transformer":
                table_per_class(recs[b], sh, os.path.join(tdir, "tab_per_class.tex"))
                summary["gt"] = gt_stats(recs[b], os.path.expanduser(a.data))
    if len(bbs) > 1:
        cm = ["random", "integrated_gradients", "point_drop", "occam", "extremal", "attn_rollout", "staa4d",
              "kca", "kca_ctx"]
        cm = [m for m in cm if any(m in r["methods"] for b in bbs for r in recs[b])]
        common = set.intersection(*[{r["clip"] for r in recs[b]} for b in bbs])     # same clips everywhere
        sub = {b: sorted([r for r in recs[b] if r["clip"] in common], key=lambda r: r["clip"]) for b in bbs}
        summary["cross"] = table_cross(sub, cm, os.path.join(tdir, "tab_cross.tex"))
        summary["cross"]["n_clips"] = len(common)
        summary["cross"]["accuracy_on_subset"] = {b: float(np.mean([r["label"] == r["pred"] for r in sub[b]])) for b in bbs}
    if cross:
        summary["crossover"] = cross
    ab = os.path.join(a.results, "ablation", "ablation.json")
    if os.path.exists(ab):
        table_ablation(ab, os.path.join(tdir, "tab_ablation.tex"))
    cab = os.path.join(os.path.dirname(a.results.rstrip("/")), "controlled", "ablations.json")
    if os.path.exists(cab) and "proxy_symmetric" in json.load(open(cab)):
        table_controlled_abl(cab, os.path.join(tdir, "tab_controlled_abl.tex"))
    summary["gt_missing"] = n_gt_missing
    stab = os.path.join(a.results, "stability", "stability.jsonl")
    if os.path.exists(stab):
        summary["stability"] = table_stability(stab, os.path.join(tdir, "tab_stability.tex"))
    summary["seeds"] = table_seeds(a.results, os.path.join(tdir, "tab_seeds.tex"))
    if "p4transformer" in recs:
        table_cost(recs["p4transformer"], main_methods, os.path.join(tdir, "tab_cost.tex"))
    with open(os.path.join(a.results, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    for b in bbs:                                    # paired tests quoted in the text (P4T: no prefix)
        R = recs[b]
        pre = "P" + ("" if b == "p4transformer" else BKEY[b])
        tests = {}
        for x, y, key, K in (("kca_ctx", "point_drop", "deletion_auc", "KcaCtxOcclDel"),
                             ("kca_ctx", "integrated_gradients", "deletion_auc", "KcaCtxIGDel"),
                             ("kca", "point_drop", "deletion_auc", "KcaOcclDel"),
                             ("kca_ctx", "integrated_gradients", "smot_ebpg/ebpg", "KcaCtxIGEBPG"),
                             ("kca", "integrated_gradients", "smot_ebpg/ebpg", "KcaIGEBPG")):
            kx, ky = key.split("/") if "/" in key else (key, key)
            u, v = column(R, x, kx), column(R, y, ky)
            ok = np.isfinite(u) & np.isfinite(v)
            if ok.sum() > 5:
                tests[pre + K] = float(wilcoxon(u[ok], v[ok]).pvalue)
        if b == "p4transformer":                     # the same tests on the first five clips per action
            five = {}
            for r in sorted(R, key=lambda r: r["clip"]):
                five.setdefault(r["label"], [])
                if len(five[r["label"]]) < 5:
                    five[r["label"]].append(r["clip"])
            keep = {c for v in five.values() for c in v}
            R5 = [r for r in R if r["clip"] in keep]
            for x, y, K in (("kca_ctx", "point_drop", "KcaCtxOcclDelFive"), ("kca_ctx", "integrated_gradients", "KcaCtxIGDelFive")):
                u, v = column(R5, x, "deletion_auc"), column(R5, y, "deletion_auc")
                ok = np.isfinite(u) & np.isfinite(v)
                tests[pre + K] = float(wilcoxon(u[ok], v[ok]).pvalue)
            summary[b]["_n_five"] = len(R5)
        summary[b]["_tests"] = tests
    mac = write_numbers(summary, os.path.expanduser(a.ckpt_root), a.results, os.path.join(tdir, "numbers.tex"))
    # every number macro the text uses but no result defines yet -> a visible red "??"
    import glob as _glob
    import re as _re
    used = set()
    for f in _glob.glob(os.path.join(a.paper, "sec", "*.tex")) + _glob.glob(os.path.join(a.paper, "fig", "*.tex")):
        used |= set(_re.findall(r"\\((?:Num|BBacc)[A-Za-z]+)", open(f).read()))
    missing = sorted(used - set(mac))
    with open(os.path.join(tdir, "defaults.tex"), "w") as fh:
        fh.write("% auto-generated: macros used in the text with no measured value yet\n")
        for k in missing:
            fh.write(f"\\providecommand{{\\{k}}}{{\\textcolor{{red}}{{??}}}}\n")
    if missing:
        print("[numbers] not yet measured:", ", ".join(missing))
    print(json.dumps({b: {m: {k: round(v["mean"], 3) for k, v in d.items() if isinstance(v, dict) and k in
                              ("deletion_auc", "insertion_auc", "smot_ebpg", "ebpg")}
                          for m, d in s.items() if isinstance(d, dict)} for b, s in summary.items() if b in bbs},
                     indent=1)[:4000])


if __name__ == "__main__":
    main()
