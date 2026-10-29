"""Evaluation harness: methods x metrics over a dataset -> aggregated results.

Produces the numbers that populate Tabs. 1-4 of the paper. Handles 'n/a' (a method
that returns None, e.g. attention baselines on attention-free backbones) and
aggregates mean +/- std across clips.

Usage (programmatic):
    res = evaluate(model, dataset, ["kca", "occam", ...])
    res['kca']['deletion_auc'] -> {'mean':..., 'std':..., 'n':...}
"""
from __future__ import annotations

from typing import Optional

import torch

from ..backbones.base import Backbone
from ..data.base import Dataset, Batch
from .. import methods as M
from ..metrics import faithfulness as F, localization as L


def _aggregate(vals: list[float]) -> dict:
    t = torch.tensor(vals, dtype=torch.float)
    return {"mean": t.mean().item(), "std": t.std(unbiased=False).item(), "n": len(vals)}


@torch.no_grad()
def _faithfulness_for_field(model, clip, field_t, target, steps) -> dict:
    di = F.deletion_insertion(model, clip, field_t, target, steps=steps)
    aopc = F.aopc_comp_suff(model, clip, field_t, target)
    sn = F.sensitivity_n(model, clip, field_t, target)
    return {
        "deletion_auc": di["deletion_auc"].mean().item(),
        "insertion_auc": di["insertion_auc"].mean().item(),
        "comprehensiveness": aopc["comprehensiveness"].mean().item(),
        "sufficiency": aopc["sufficiency"].mean().item(),
        "sensitivity_n": sn.mean().item(),
    }


def evaluate_field_fn(model, dataset, field_fn, steps=20, limit=None,
                      do_localization=False, gt="moving") -> dict:
    """Evaluate an arbitrary per-clip field producer (e.g. the amortized head) with the
    same metrics as evaluate(). field_fn(model, clip, target) -> (B,T,N) or None.
    Returns {metric: {mean,std,n}}."""
    acc: dict[str, list[float]] = {}
    for batch in dataset.batches(limit=limit):
        clip = batch.clip
        target = clip.label if clip.label is not None else model.predicted_class(clip)
        f = field_fn(model, clip, target)
        if f is None:
            continue
        res = _faithfulness_for_field(model, clip, f, target, steps)
        gt_mask = batch.moving_gt if gt == "moving" else batch.static_gt
        if do_localization and gt_mask is not None:
            res["pg_moving"] = L.pointing_game(f, gt_mask, clip.mask).mean().item()
            res["ebpg_moving"] = L.ebpg(f, gt_mask, clip.mask).mean().item()
        for k, v in res.items():
            acc.setdefault(k, []).append(v)
    return {k: _aggregate(v) for k, v in acc.items()}


def evaluate(
    model: Backbone,
    dataset: Dataset,
    method_names: list[str],
    steps: int = 20,
    limit: Optional[int] = None,
    do_localization: bool = False,
) -> dict:
    """Return {method: {metric: {mean,std,n}}}. 'n/a' methods get marked na=True.
    Each non-na method gets the full faithfulness battery (+localization if requested)."""
    per_method_metric: dict[str, dict[str, list[float]]] = {m: {} for m in method_names}
    na_methods: set[str] = set()

    for batch in dataset.batches(limit=limit):
        clip = batch.clip
        target = clip.label if clip.label is not None else model.predicted_class(clip)
        for name in method_names:
            f = M.run_method(name, model, clip, target)
            if f is None:
                na_methods.add(name)
                continue
            res = _faithfulness_for_field(model, clip, f, target, steps)
            if do_localization and batch.moving_gt is not None:
                res["pg_moving"] = L.pointing_game(f, batch.moving_gt, clip.mask).mean().item()
                res["ebpg_moving"] = L.ebpg(f, batch.moving_gt, clip.mask).mean().item()
            for k, v in res.items():
                per_method_metric[name].setdefault(k, []).append(v)

    out: dict = {}
    for name in method_names:
        if name in na_methods and not per_method_metric[name]:
            out[name] = {"na": True}
            continue
        out[name] = {k: _aggregate(v) for k, v in per_method_metric[name].items()}
    return out


def kendall_consistency(results_by_backbone: dict, metric: str = "deletion_auc") -> dict:
    """Given {backbone: evaluate(...)}, compute per-method Kendall tau of the method
    ranking across backbone pairs (the cross-backbone consistency headline)."""
    from ..metrics.consistency import kendall_tau
    backbones = list(results_by_backbone)
    methods_common = set.intersection(*[
        {m for m, r in results_by_backbone[b].items() if "na" not in r} for b in backbones])
    methods_common = sorted(methods_common)
    vecs = {}
    for b in backbones:
        vecs[b] = torch.tensor([results_by_backbone[b][m][metric]["mean"] for m in methods_common])
    taus = {}
    for i in range(len(backbones)):
        for j in range(i + 1, len(backbones)):
            taus[f"{backbones[i]}~{backbones[j]}"] = kendall_tau(vecs[backbones[i]], vecs[backbones[j]])
    return {"methods": methods_common, "tau": taus}
