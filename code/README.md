# KCA — code

Implementation of **Kinematic Causal Attribution** (the paper in `../paper/`): interventional
attribution for 4D point-cloud video models that splits each spacetime region's importance
into a **motion** term and a **geometry** term. Everything talks to the model through one
`Backbone` interface, so the same code explains the official PSTNet (convolution),
P4Transformer (attention) and Mamba4D (state-space) models.

Runbook: **`SETUP.md`** (data → training → benchmarks → tables/figures → paper).
One command: `../reproduce.sh` (CPU checks), `../reproduce.sh --msr` (all real-data results).
Results digest: **`results/RESULTS.md`**.

## What the method does (and the three design decisions the code enforces)
1. **Correspondence-free.** Real point-cloud video re-samples every frame, so there is no
   point identity across frames. Regions are *tracked* per-frame point sets
   (`regions.tracked_regions`), and every operator rewrites only those sets
   (`interventions.py`): `freeze` fills R_t with R_{t0}; `displace` replays R_{π(t)}.
   `tests/test_correspondence_free.py` checks that shuffling each frame's point order
   leaves the scores unchanged (and that the old index-aligned freeze fails on such data).
2. **A proper (motion, geometry) cube.** For each region: X (moving, real shape),
   `freeze` (static, real shape), `abl-geo` (moving shape-free proxy on the centroid
   path), `ablate` with the default `proxy` baseline (static proxy). Every edge changes
   exactly ONE factor, so the order-symmetric scores (`scores.py`) measure motion and
   geometry and nothing else; completeness Smot + Sgeo = Nec is exact and the interaction
   Delta_int is reported. (Deleting the region at the (0,0) corner mixes presence into
   the motion term — measured in the ablation.)
3. **Two estimators.** `local` (3m+1 passes, the classic "intervene on this region
   alone") and `context` (S passes; regress phi on random joint states of all regions →
   context-averaged / Banzhaf-type scores, robust when a structure spans several
   regions). Both are batched through `interventions.compose_states`.

## Layout
```
src/kca/
  types.py, regions.py, interventions.py, scores.py, attribute.py   the method
  proposal.py, amortized.py                     optional region pruning / O(1) head (controlled study only)
  backbones/base.py                             the Backbone interface (model-agnostic boundary)
  backbones/official.py                         official P4Transformer / PSTNet / Mamba4D, isolated imports,
                                                strict checkpoint loading, attention/Grad-CAM hooks
  backbones/cpu_ops.py                          pure-PyTorch FPS / ball query / grouping / KNN / Mamba / RMSNorm
  backbones/cuda_fast.py                        optional CUDA FPS kernel + mamba_ssm selective scan (same semantics)
  backbones/{p4transformer,pstnet,mamba4d}.py   thin aliases of OfficialBackbone
  backbones/simple4d.py, synthetic.py           controlled-study network and test oracles
  baselines/                                    floors, gradient, perturbation, attention baselines
  metrics/                                      faithfulness (probability-scored), localization, soundness
  data/prepare_msr.py                           MSR-Action3D depth + skeleton -> point-cloud npz with joints
  data/msr_action3d.py                          explanation loader with skeleton moving/static GT
  data/motion_task.py                           controlled tasks (motion / shape / direction cue)
  run/train_official.py                         train the official backbones (CUDA, CPU or MPS)
  run/msr_benchmark.py                          every method x metric on real clips (per-clip JSONL)
  run/msr_crossover.py                          real-data motion/geometry crossover (same arm, two cues)
  run/msr_ablation.py                           ablations, cascading-randomisation sanity, displace probe
  run/controlled_study.py                       controlled experiments (ground truth by construction)
  run/paper_tables.py, paper_figures.py         LaTeX tables / numbers / PDF figures for the paper
  run/compare_full.py                           laptop vs full-scale numbers: claim checks, text to rewrite
  run/cvpr_benchmark.py                         legacy orchestrator (synthetic dry-run wiring check)
tests/                                          7 CPU test suites (see SETUP.md §5)
```
