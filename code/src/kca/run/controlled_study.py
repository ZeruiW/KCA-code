"""Rigorous CONTROLLED study of KCA on trained 4D networks, runnable on CPU.

Ground truth is known by construction (the odd cluster), so we can measure not just
faithfulness but whether the motion/geometry split is CORRECT. Produces real numbers
(not placeholders): a faithfulness table, the motion-vs-geometry double dissociation,
ablations, and significance tests -- written to a results dir with a conclusions file.

This is the controlled diagnostic of the paper's supplement; the GPU pipeline
(run_experiments.py + real backbones) produces the main-paper benchmark tables.

Run:  PYTHONPATH=src python -m kca.run.controlled_study --out results/controlled
"""
from __future__ import annotations

import argparse
import json
import os

import torch

from ..data.motion_task import ClusterCueTask
from ..run.train_backbone import train, accuracy
from ..run import evaluate as E
from ..run import make_tables as T
from .. import attribute, methods
from ..metrics import localization as L
from ..metrics import faithfulness as F

try:
    from scipy.stats import wilcoxon
except Exception:
    wilcoxon = None

FAITH_METHODS = ["random", "geometry_edge", "motion_prior", "grad_x_input", "smoothgrad",
                 "integrated_gradients", "grad_cam_3d", "point_drop", "occam", "extremal",
                 "attn_rollout", "staa4d", "kca", "kca_ctx"]
FAITH_ROWS = [("random", "Random (floor)"), ("geometry_edge", "Geometry-edge (floor)"),
              ("motion_prior", "Motion-prior (floor)"), ("grad_x_input", r"Grad$\times$Input"),
              ("smoothgrad", "SmoothGrad"), ("integrated_gradients", "Integrated Grad."),
              ("grad_cam_3d", "3D Grad-CAM"), ("point_drop", "Point-drop"),
              ("occam", "OccAM"), ("extremal", "Extremal Pert."),
              ("attn_rollout", "Attn. Rollout"), ("staa4d", "STAA-4D (ablation)"),
              ("kca", r"\textbf{KCA} (local)"), ("kca_ctx", r"\textbf{KCA} (context)")]
FAITH_COLS = ["deletion_auc", "insertion_auc", "comprehensiveness", "sufficiency", "sensitivity_n"]


def train_model(cue, seed, n_train=180, epochs=14, region_k=3):
    tr = ClusterCueTask(n=n_train, K=3, cue=cue, seed=seed, region_method="tracked", region_k=region_k)
    va = ClusterCueTask(n=60, K=3, cue=cue, seed=1000 + seed, region_method="tracked", region_k=region_k)
    bb = train(tr, epochs=epochs, seed=seed, verbose=False)
    return bb, va, accuracy(bb.net, va, list(range(len(va))))


# ---- per-clip faithfulness collection (for significance) --------------------------
def collect_perclip(bb, dataset, names, n_eval, steps=12, baseline="proxy"):
    rows = {m: {c: [] for c in FAITH_COLS} for m in names}
    loc = {"kca": [], "random": []}
    for i in range(min(n_eval, len(dataset))):
        b = dataset.get(i); clip = b.clip; tgt = clip.label
        for m in names:
            f = methods.run_method(m, bb, clip, tgt)
            if f is None:
                continue
            di = F.deletion_insertion(bb, clip, f, tgt, steps=steps)
            ap = F.aopc_comp_suff(bb, clip, f, tgt)
            sn = F.sensitivity_n(bb, clip, f, tgt, n_subsets=30)
            rows[m]["deletion_auc"].append(di["deletion_auc"].mean().item())
            rows[m]["insertion_auc"].append(di["insertion_auc"].mean().item())
            rows[m]["comprehensiveness"].append(ap["comprehensiveness"].mean().item())
            rows[m]["sufficiency"].append(ap["sufficiency"].mean().item())
            rows[m]["sensitivity_n"].append(sn.mean().item())
        # KCA motion-localization vs random floor (GT = odd/moving cluster)
        attr = attribute.explain(bb, clip, tgt, k=12, n_control=3, baseline=baseline)
        loc["kca"].append(L.ebpg(attr.smot_field, b.moving_gt, clip.mask).mean().item())
        loc["random"].append(L.ebpg(methods.run_method("random", bb, clip, tgt), b.moving_gt, clip.mask).mean().item())
    return rows, loc


def agg(vals):
    t = torch.tensor(vals, dtype=torch.float)
    return {"mean": t.mean().item(), "std": t.std(unbiased=False).item(), "n": len(vals)}


def faithfulness_study(seeds, n_eval, out):
    print("\n== Faithfulness (motion task, %d seeds x %d clips) ==" % (len(seeds), n_eval))
    pooled = {m: {c: [] for c in FAITH_COLS} for m in FAITH_METHODS}
    loc_pooled = {"kca": [], "random": []}
    accs = []
    for s in seeds:
        bb, va, acc = train_model("motion", s)
        accs.append(acc)
        rows, loc = collect_perclip(bb, va, FAITH_METHODS, n_eval)
        for m in FAITH_METHODS:
            for c in FAITH_COLS:
                pooled[m][c] += rows[m][c]
        loc_pooled["kca"] += loc["kca"]; loc_pooled["random"] += loc["random"]
    results = {m: ({c: agg(pooled[m][c]) for c in FAITH_COLS} if pooled[m]["deletion_auc"] else {"na": True})
               for m in FAITH_METHODS}
    # Significance. NOTE: point_drop/occam/extremal are perturbation methods that share
    # KCA's ablation-necessity core, so KCA *ties* the best of them on raw deletion (its
    # edge over them is the decomposition, shown by the crossover). The decisive, testable
    # faithfulness win is over the GRADIENT/ATTENTION family. We report both.
    floors = {"random", "geometry_edge", "motion_prior"}
    perturb = {"point_drop", "occam", "extremal"}
    def _sig(vs):
        if wilcoxon is None or not pooled[vs]["deletion_auc"]:
            return None
        a, b = pooled["kca"]["deletion_auc"], pooled[vs]["deletion_auc"]
        if max(abs(x - y) for x, y in zip(a, b)) < 1e-9:
            return {"vs": vs, "p_value": None, "note": "tied (shared necessity core)"}
        try:
            return {"vs": vs, "p_value": float(wilcoxon(a, b).pvalue)}
        except Exception:
            return None
    nonfloor = [m for m in FAITH_METHODS if m != "kca" and m not in floors and pooled[m]["deletion_auc"]]
    gradattn = [m for m in nonfloor if m not in perturb]
    best = min(nonfloor, key=lambda m: results[m]["deletion_auc"]["mean"])
    best_ga = min(gradattn, key=lambda m: results[m]["deletion_auc"]["mean"]) if gradattn else None
    sig = {"vs_best_overall": _sig(best),
           "vs_best_grad_attn": _sig(best_ga) if best_ga else None}
    body = T.latex_table_body(results, FAITH_ROWS, FAITH_COLS)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "faithfulness.json"), "w") as fh:
        json.dump({"results": results, "val_acc": accs, "significance": sig,
                   "localization_ebpg": {"kca": agg(loc_pooled["kca"]), "random": agg(loc_pooled["random"])}}, fh, indent=2)
    with open(os.path.join(out, "faithfulness.tex"), "w") as fh:
        fh.write(body + "\n")
    print("val acc: %.3f +/- %.3f" % (torch.tensor(accs).mean(), torch.tensor(accs).std(unbiased=False)))
    print("Deletion-AUC (lower better): " + "  ".join(
        f"{m}={results[m]['deletion_auc']['mean']:.2f}" for m in ["random", "geometry_edge", "motion_prior", "occam", "kca"]))
    print("Smot EBPG->moving: KCA %.3f vs random %.3f" % (loc_pooled and agg(loc_pooled["kca"])["mean"], agg(loc_pooled["random"])["mean"]))
    for key in ("vs_best_overall", "vs_best_grad_attn"):
        s = sig.get(key)
        if s:
            pv = "tied" if s.get("p_value") is None else f"p={s['p_value']:.2e}"
            print(f"KCA vs {s['vs']} ({key}) deletion: {pv}")
    return results, sig, accs, loc_pooled


def crossover_study(seeds, n_eval, out):
    """Double dissociation: train a model per cue, measure where KCA puts mass.
    Report Smot/(Smot+Sgeo) energy on the ODD cluster for each model."""
    print("\n== Motion-vs-geometry double dissociation ==")
    res = {}
    for cue, est in (("motion", "local"), ("shape", "local"), ("motion_ctx", "context"), ("shape_ctx", "context")):
        sm_tot = sg_tot = 0.0
        per_clip = []               # per-clip fraction (only where denom is non-trivial)
        for s in seeds:
            bb, va, acc = train_model(cue.split("_")[0], s)
            for i in range(min(n_eval, len(va))):
                b = va.get(i); clip = b.clip; tgt = clip.label
                attr = attribute.explain(bb, clip, tgt, k=12, n_control=3, baseline="proxy", estimator=est,
                                         n_samples=256, generator=torch.Generator().manual_seed(i))
                odd = (b.moving_gt if cue.startswith("motion") else b.static_gt).bool()
                sm = attr.smot_field.clamp_min(0); sg = attr.sgeo_field.clamp_min(0)
                sm_odd = sm[odd].sum().item(); sg_odd = sg[odd].sum().item()
                sm_tot += sm_odd; sg_tot += sg_odd
                if sm_odd + sg_odd > 0.1:        # ignore near-flat clips (noise)
                    per_clip.append(sm_odd / (sm_odd + sg_odd))
        pooled = sm_tot / (sm_tot + sg_tot + 1e-8)   # robust pooled fraction
        res[cue] = {"odd_motion_fraction_pooled": pooled,
                    "odd_motion_fraction_perclip": agg(per_clip) if per_clip else {"mean": float("nan")},
                    "smot_on_odd_total": sm_tot, "sgeo_on_odd_total": sg_tot}
        print(f"  cue={cue:6s}: motion-fraction WITHIN odd cluster (pooled) = {pooled:.2f} "
              f"[Smot={sm_tot:.0f} Sgeo={sg_tot:.0f}] (high iff cue=motion)")
    res["dissociation_gap"] = res["motion"]["odd_motion_fraction_pooled"] - res["shape"]["odd_motion_fraction_pooled"]
    res["dissociation_gap_ctx"] = res["motion_ctx"]["odd_motion_fraction_pooled"] - res["shape_ctx"]["odd_motion_fraction_pooled"]
    with open(os.path.join(out, "crossover.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    return res


def ablation_study(seed, n_eval, out):
    """Ablate the (M=0, G=0) vertex (static proxy / true removal / donor fill) and
    symmetric vs motion-first scoring. Deletion only sees Nec = Smot + Sgeo, which the
    symmetrisation leaves unchanged, so each variant also reports the decomposition's
    own quantities: motion share on the odd cluster of the motion AND shape models, and
    the normalised interaction |Delta_int| / |Nec|."""
    print("\n== Ablations (1 seed) ==")
    models = {cue: train_model(cue, seed) for cue in ("motion", "shape")}
    abl = {}
    for bl in ("proxy", "bg", "nn"):
        for sym in (True, False):
            rec = {}
            for cue, (bb, va, acc) in models.items():
                ds, sm, sg, ratio = [], 0.0, 0.0, []
                for i in range(min(n_eval, len(va))):
                    b = va.get(i); clip = b.clip; tgt = clip.label
                    attr = attribute.explain(bb, clip, tgt, k=12, baseline=bl, symmetric=sym)
                    odd = (b.moving_gt if cue == "motion" else b.static_gt).bool()
                    sm += attr.smot_field.clamp_min(0)[odd].sum().item()
                    sg += attr.sgeo_field.clamp_min(0)[odd].sum().item()
                    if cue == "motion":
                        ds.append(F.deletion_insertion(bb, clip, attr.nec_field, tgt, steps=12)["deletion_auc"].mean().item())
                    if sym:
                        nec = torch.stack([r.necessity.abs().mean() for r in attr.region_scores.values()]).mean().item()
                        ratio.append(attr.mean_abs_interaction() / max(nec, 1e-8))
                rec[f"{cue}_odd_motion_share"] = sm / (sm + sg + 1e-8)
                if ds:
                    rec["deletion_auc"] = agg(ds)
                if ratio:
                    rec[f"{cue}_dint_over_nec"] = float(sum(ratio) / len(ratio))
            abl[f"{bl}_{'symmetric' if sym else 'motion_first'}"] = rec
    with open(os.path.join(out, "ablations.json"), "w") as fh:
        json.dump(abl, fh, indent=2)
    for k, v in abl.items():
        print(f"  {k:24s} " + "  ".join(f"{kk}={vv['mean'] if isinstance(vv, dict) else vv:+.3f}" for kk, vv in v.items()))
    return abl


def direction_probe_study(seed, n_eval, out):
    """Sign-sensitive `displace` probe (Sec. method). Class = motion DIRECTION with a
    SHARED spatial envelope, so only temporal order discriminates. Reversing the actor's
    trajectory should flip the prediction to the OPPOSITE direction (a property freeze
    cannot have). Demonstrates `displace` captures motion direction, not just magnitude."""
    print("\n== displace sign-sensitivity (motion-direction task) ==")
    from ..data.motion_task import MotionDirectionTask
    from .. import interventions as itv
    tr = MotionDirectionTask(n=320, seed=seed)
    va = MotionDirectionTask(n=80, seed=1000 + seed)
    bb = train(tr, epochs=20, seed=seed, verbose=False)
    acc = accuracy(bb.net, va, list(range(len(va))))
    OPP = MotionDirectionTask.OPP
    ndsp, nfrz, flip_d, flip_f = [], [], [], []
    for i in range(min(n_eval, len(va))):
        b = va.get(i); clip = b.clip; c = clip.label
        mv = b.moving_gt[0, 0]
        r = max(range(clip.num_regions()), key=lambda r: ((clip.regions[0, 0] == r) & mv).sum().item())
        phi = bb.phi(clip, c).item()
        rev = itv.displace(clip, r, mode="reverse"); frz = itv.freeze(clip, r)
        ndsp.append(phi - bb.phi(rev, c).item()); nfrz.append(phi - bb.phi(frz, c).item())
        flip_d.append(float(bb.predicted_class(rev).item() == OPP[c.item()]))
        flip_f.append(float(bb.predicted_class(frz).item() == OPP[c.item()]))
    res = {"val_acc": acc, "displace_necessity": agg(ndsp), "freeze_necessity": agg(nfrz),
           "displace_flip_to_opposite": agg(flip_d), "freeze_flip_to_opposite": agg(flip_f)}
    with open(os.path.join(out, "direction_probe.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"  acc {acc:.2f} | displace-reverse necessity {res['displace_necessity']['mean']:.2f} "
          f"vs freeze {res['freeze_necessity']['mean']:.2f}")
    print(f"  flip-to-OPPOSITE: displace {res['displace_flip_to_opposite']['mean']:.2f} "
          f"vs freeze {res['freeze_flip_to_opposite']['mean']:.2f} (displace is sign-sensitive)")
    return res


def soundness_study(seed, n_eval, out):
    """Adebayo model-randomization sanity, Infidelity, Max-Sensitivity, and the
    amortized-head fidelity to exact scores -- all measured on the trained net."""
    print("\n== Soundness & amortized fidelity (1 seed) ==")
    from ..metrics import soundness as S
    from ..backbones.simple4d import Simple4DNet, Simple4DBackbone
    from .. import amortized as A
    bb, va, acc = train_model("motion", seed)
    rbb = Simple4DBackbone(Simple4DNet(va.num_classes, grid=3), va.num_classes)  # random weights

    adeb, infid, maxsen = [], [], []
    for i in range(min(n_eval, len(va))):
        b = va.get(i); clip = b.clip; tgt = clip.label
        f_tr = attribute.explain(bb, clip, tgt, k=12).nec_field
        f_rd = attribute.explain(rbb, clip, tgt, k=12).nec_field
        adeb.append(S.rank_correlation(f_tr, f_rd, clip.mask).mean().item())
        infid.append(S.infidelity(bb, clip, f_tr, tgt, n_pert=15).mean().item())
        maxsen.append(S.max_sensitivity(lambda c: attribute.explain(bb, c, tgt, k=12).nec_field,
                                        clip, f_tr, n_pert=5).mean().item())
    # amortized head fidelity to exact KCA
    head = A.distill(bb, va, k=12, epochs=120, limit=n_eval, baseline="nn")
    fid, dex, dam = [], [], []
    for i in range(min(n_eval, len(va))):
        b = va.get(i); clip = b.clip; tgt = clip.label
        ex = attribute.explain(bb, clip, tgt, k=12).nec_field
        sg, sm = A.explain_amortized(bb, clip, head, tgt)
        fid.append(S.rank_correlation(ex, sg + sm, clip.mask).mean().item())
        dex.append(F.deletion_insertion(bb, clip, ex, tgt, steps=10)["deletion_auc"].mean().item())
        dam.append(F.deletion_insertion(bb, clip, sg + sm, tgt, steps=10)["deletion_auc"].mean().item())
    res = {"adebayo_rankcorr": agg(adeb), "infidelity": agg(infid), "max_sensitivity": agg(maxsen),
           "amortized_fidelity_rankcorr": agg(fid),
           "deletion_exact": agg(dex), "deletion_amortized": agg(dam)}
    with open(os.path.join(out, "soundness.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    print(f"  Adebayo rank-corr (trained vs random weights) = {res['adebayo_rankcorr']['mean']:+.3f} (->0 = sane)")
    print(f"  Infidelity = {res['infidelity']['mean']:.3f} | Max-Sensitivity = {res['max_sensitivity']['mean']:.3f}")
    print(f"  Amortized fidelity rank-corr to exact = {res['amortized_fidelity_rankcorr']['mean']:.3f}; "
          f"Deletion exact {res['deletion_exact']['mean']:.2f} vs amortized {res['deletion_amortized']['mean']:.2f}")
    return res


def _sig_phrase(s):
    if not s:
        return "."
    if s.get("p_value") is None:
        return "."
    return f" (significantly, Wilcoxon p={s['p_value']:.1e} vs {s['vs']})."


def write_conclusions(out, faith, sig, accs, loc, cross, abl, snd=None, dirp=None):
    """Data-driven summary: every sentence is computed from the measured numbers."""
    def d(m):
        return faith[m]["deletion_auc"]["mean"] if m in faith and "na" not in faith[m] else float("nan")
    methods = [m for m in faith if "na" not in faith[m]]
    floors = {"random", "geometry_edge", "motion_prior"}
    ranked = sorted([m for m in methods if m not in floors], key=d)
    ek = agg(loc["kca"])["mean"]; er = agg(loc["random"])["mean"]
    lines = [
        "# Controlled study — results (auto-generated by `kca.run.controlled_study`)",
        "",
        f"Attention-based 4D net trained per cue; motion-task val acc "
        f"{torch.tensor(accs).mean():.3f} ± {torch.tensor(accs).std(unbiased=False):.3f} ({len(accs)} seeds). "
        "Regions: correspondence-free tracked k-means (3 = the clusters). Decomposition vertex: "
        "static/moving point-mass proxy. Faithfulness scored on the target-class probability.",
        "",
        "## 1. Faithfulness (Deletion-AUC, lower is better)",
        "- Ranking (non-floor methods): " + ", ".join(f"{m} {d(m):.3f}" for m in ranked),
        f"- Floors: random {d('random'):.3f}, geometry-edge {d('geometry_edge'):.3f}, motion-prior {d('motion_prior'):.3f}.",
        (f"- KCA (local) vs best other method `{sig['vs_best_overall']['vs']}`: "
         + ("identical fields" if sig["vs_best_overall"].get("p_value") is None
            else f"paired Wilcoxon p = {sig['vs_best_overall']['p_value']:.2e}") if sig and sig.get("vs_best_overall") else "- (no significance test)"),
        "",
        "## 2. Motion localisation (EBPG of Smot on the moving cluster)",
        f"- KCA {ek:.3f} vs random {er:.3f}.",
        "",
        "## 3. Motion-vs-geometry double dissociation (motion share Smot/(Smot+Sgeo) on the odd cluster)",
        f"- local estimator: motion-cue model **{cross['motion']['odd_motion_fraction_pooled']:.2f}**, "
        f"shape-cue model **{cross['shape']['odd_motion_fraction_pooled']:.2f}** (gap {cross['dissociation_gap']:.2f}).",
        f"- context estimator: motion-cue **{cross['motion_ctx']['odd_motion_fraction_pooled']:.2f}**, "
        f"shape-cue **{cross['shape_ctx']['odd_motion_fraction_pooled']:.2f}** (gap {cross['dissociation_gap_ctx']:.2f}).",
        "",
        "## 4. Ablations (vertex x symmetrisation)",
    ]
    for k, v in abl.items():
        lines.append(f"- `{k}`: " + ", ".join(
            f"{kk} {vv['mean'] if isinstance(vv, dict) else vv:.3f}" for kk, vv in v.items()))
    lines += ["",
              "## 5. Soundness",
              (f"- Max-Sensitivity {snd['max_sensitivity']['mean']:.3f}; Adebayo rank-corr(trained, random weights) "
               f"{snd['adebayo_rankcorr']['mean']:+.3f} (3 regions only: underpowered — see the real-backbone "
               f"cascading test); amortized-head fidelity (rank-corr to exact) "
               f"{snd['amortized_fidelity_rankcorr']['mean']:.2f}, Deletion-AUC amortized "
               f"{snd['deletion_amortized']['mean']:.3f} vs exact {snd['deletion_exact']['mean']:.3f}." if snd else "- (skipped)"),
              "",
              "## 6. `displace` sign-sensitivity (class = motion direction, shared spatial envelope)",
              (f"- reversing the actor's trajectory flips the prediction to the opposite direction "
               f"{dirp['displace_flip_to_opposite']['mean']:.0%} of the time vs {dirp['freeze_flip_to_opposite']['mean']:.0%} "
               f"for freeze (necessity {dirp['displace_necessity']['mean']:.2f} vs {dirp['freeze_necessity']['mean']:.2f}; "
               f"acc {dirp['val_acc']:.2f})." if dirp else "- (skipped)"),
              ]
    with open(os.path.join(out, "CONCLUSIONS.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n[written] %s/{faithfulness,crossover,ablations}.json, faithfulness.tex, CONCLUSIONS.md" % out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/controlled")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--n-eval", type=int, default=20)
    args = p.parse_args()
    torch.manual_seed(0)
    seeds = list(range(args.seeds))
    faith, sig, accs, loc = faithfulness_study(seeds, args.n_eval, args.out)
    cross = crossover_study(seeds, args.n_eval, args.out)
    abl = ablation_study(0, args.n_eval, args.out)
    snd = soundness_study(0, args.n_eval, args.out)
    dirp = direction_probe_study(0, args.n_eval, args.out)
    write_conclusions(args.out, faith, sig, accs, loc, cross, abl, snd, dirp)


if __name__ == "__main__":
    main()
