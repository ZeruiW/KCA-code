"""End-to-end pipeline on CPU: dataset -> evaluate (methods x metrics) -> LaTeX tables,
plus the amortized head train + O(1) inference. Proves the full harness runs so that
on the GPU host only the real dataset + real backbone need swapping in.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

from kca.data.synthetic_dataset import SyntheticDataset
from kca.backbones.synthetic import SyntheticBackbone
from kca.run import evaluate as E
from kca.run import make_tables as T
from kca import amortized as A


def test_evaluate_and_tables(tmp=None):
    ds = SyntheticDataset(n=6, T=6, N=64, grid=3)
    model = SyntheticBackbone(a=1.0, b=2.0)
    method_names = ["random", "geometry_edge", "integrated_gradients", "occam",
                    "staa4d", "kca"]
    res = E.evaluate(model, ds, method_names, steps=8, do_localization=True, limit=6)

    # every non-na method has the 5 faithfulness metrics aggregated over 6 clips
    for m in ["random", "kca", "occam"]:
        assert "deletion_auc" in res[m] and res[m]["deletion_auc"]["n"] == 6, (m, res[m])
    # staa4d is attention-only -> n/a on the SSM oracle
    assert res["staa4d"].get("na") is True, res["staa4d"]

    # LaTeX generation: best bold, second underline, n/a preserved
    rows = [("random", "Random (floor)"), ("geometry_edge", "Geometry-edge (floor)"),
            ("integrated_gradients", "Integrated Grad."), ("occam", "OccAM"),
            ("staa4d", "STAA-4D"), ("kca", r"\textbf{KCA}")]
    cols = ["deletion_auc", "insertion_auc", "comprehensiveness", "sufficiency", "sensitivity_n"]
    body = T.latex_table_body(res, rows, cols)
    assert r"\emph{n/a}" in body                  # staa4d row
    assert r"\textbf{" in body                    # a best cell exists
    out_dir = os.path.join(os.path.dirname(__file__), "_out")
    T.save(res, out_dir, rows, cols, stem="tab_faithfulness_demo")
    print("OK  evaluate(): 6 methods over 6 clips; staa4d n/a; tables.tex written")
    print("    sample row (KCA):", body.splitlines()[-1][:90])


def test_cross_backbone_kendall():
    ds = SyntheticDataset(n=4, T=6, N=48, grid=3)
    method_names = ["random", "integrated_gradients", "occam", "kca"]
    r1 = E.evaluate(SyntheticBackbone(a=1.0, b=2.0), ds, method_names, steps=6, limit=4)
    r2 = E.evaluate(SyntheticBackbone(a=0.5, b=3.0), ds, method_names, steps=6, limit=4)
    cons = E.kendall_consistency({"bb1": r1, "bb2": r2}, metric="deletion_auc")
    assert "bb1~bb2" in cons["tau"]
    print("OK  cross-backbone Kendall tau(bb1,bb2) on deletion =", round(cons["tau"]["bb1~bb2"], 3))


def test_amortized_head():
    ds = SyntheticDataset(n=8, T=6, N=64, grid=3)
    model = SyntheticBackbone(a=1.0, b=2.0)
    head = A.distill(model, ds, k=8, epochs=150, limit=8, baseline="bg")
    batch = ds.get(0)
    sgeo, smot = A.explain_amortized(model, batch.clip, head)
    assert sgeo.shape == batch.clip.mask.shape and smot.shape == batch.clip.mask.shape
    # amortized motion should still land more on moving GT than static (sanity)
    mv, st = batch.moving_gt, batch.static_gt
    assert smot[mv].mean() >= smot[st].mean() - 1e-3, (smot[mv].mean().item(), smot[st].mean().item())
    print("OK  amortized head trained; O(1) fields shape %s; smot moving %.3f vs static %.3f"
          % (tuple(smot.shape), smot[mv].mean(), smot[st].mean()))


if __name__ == "__main__":
    test_evaluate_and_tables()
    test_cross_backbone_kendall()
    test_amortized_head()
    print("\nALL PIPELINE TESTS PASSED")
