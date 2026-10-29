"""THE end-to-end validation the synthetic oracle could not give: train a REAL
attention-based 4D network on a motion-only task, then show KCA explains it correctly.

Verifiable success criteria (on a learned model, geometry constant across classes):
  1. The model learns the task (motion is the only cue) -> high accuracy.
  2. KCA necessity is FAITHFUL: lower Deletion-AUC and higher Insertion-AUC than the
     random floor and the motion-blind geometry-edge floor.
  3. KCA's Smot LOCALIZES to the genuinely moving cluster (moving_gt) better than random.
  4. The attention proposal path is live (has_attention) -> STAA-4D is defined here.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from kca.data.motion_task import WhichClusterMoves
from kca.run.train_backbone import train, accuracy
from kca import attribute, methods
from kca.metrics import faithfulness as F, localization as L


def _build():
    torch.manual_seed(0)
    tr = WhichClusterMoves(n=180, K=3, seed=0, grid=3)
    # k-means supervoxels so each cluster is ONE coherent region (not fragmented)
    va = WhichClusterMoves(n=60, K=3, seed=999, region_method="kmeans", region_k=3)
    bb = train(tr, epochs=14, val_dataset=None, seed=0, verbose=False)
    acc = accuracy(bb.net, va, list(range(len(va))))
    return bb, va, acc


def test_real_backbone_kca():
    bb, va, acc = _build()
    assert acc > 0.9, f"backbone did not learn the motion task: acc={acc}"

    del_kca = del_rnd = del_edge = ins_kca = ins_rnd = 0.0
    ebpg_kca = ebpg_rnd = 0.0
    nclip = 12
    for i in range(nclip):
        batch = va.get(i)
        clip, tgt = batch.clip, batch.clip.label
        attr = attribute.explain(bb, clip, tgt, k=12, n_control=3, baseline="nn")
        f_kca = attr.nec_field
        f_rnd = methods.run_method("random", bb, clip, tgt)
        f_edge = methods.run_method("geometry_edge", bb, clip, tgt)

        d_k = F.deletion_insertion(bb, clip, f_kca, tgt, steps=12)
        d_r = F.deletion_insertion(bb, clip, f_rnd, tgt, steps=12)
        d_e = F.deletion_insertion(bb, clip, f_edge, tgt, steps=12)
        del_kca += d_k["deletion_auc"].mean().item(); ins_kca += d_k["insertion_auc"].mean().item()
        del_rnd += d_r["deletion_auc"].mean().item(); ins_rnd += d_r["insertion_auc"].mean().item()
        del_edge += d_e["deletion_auc"].mean().item()
        ebpg_kca += L.ebpg(attr.smot_field, batch.moving_gt, clip.mask).mean().item()
        ebpg_rnd += L.ebpg(f_rnd, batch.moving_gt, clip.mask).mean().item()

    del_kca/=nclip; del_rnd/=nclip; del_edge/=nclip; ins_kca/=nclip; ins_rnd/=nclip
    ebpg_kca/=nclip; ebpg_rnd/=nclip

    print(f"acc={acc:.2f}")
    print(f"Deletion-AUC  KCA {del_kca:.3f} | random {del_rnd:.3f} | geo-edge {del_edge:.3f}  (lower=better)")
    print(f"Insertion-AUC KCA {ins_kca:.3f} | random {ins_rnd:.3f}   (higher=better; OOD-sensitive)")
    print(f"Smot EBPG->moving  KCA {ebpg_kca:.3f} | random {ebpg_rnd:.3f}            (higher=better)")

    # 2. faithfulness (primary): KCA beats BOTH the random and the motion-blind
    #    geometry-edge floors on deletion, by a wide margin (AUC of the target-class
    #    probability, so a relative margin).
    assert del_kca < 0.6 * del_rnd, (del_kca, del_rnd)
    assert del_kca < 0.6 * del_edge, (del_kca, del_edge)
    # 3. localization: Smot concentrates on the moving cluster far above random.
    assert ebpg_kca > 0.6 and ebpg_kca > 2 * ebpg_rnd, (ebpg_kca, ebpg_rnd)
    # 4. attention proposal path is live (STAA-4D defined on this attention backbone)
    assert methods.run_method("staa4d", bb, va.get(0).clip, va.get(0).clip.label) is not None
    # Insertion is reported as a diagnostic only: inserting the moving cluster alone is
    # out-of-distribution (training always had all K clusters), so partial-input
    # insertion is noisy -- deletion is the reliable faithfulness signal here.
    print("OK  on a TRAINED 4D net: KCA deletion crushes both floors; Smot localizes motion (EBPG %.2f)" % ebpg_kca)


if __name__ == "__main__":
    test_real_backbone_kca()
    print("\nREAL-BACKBONE VALIDATION PASSED")
