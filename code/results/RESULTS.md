# KCA — results digest (2026-09-26)

> **2026-10-22: the paper's main text now uses the full-scale run** (official width, 2048 points,
> A100; `results/msr_full/`, produced by `colab/KCA_fullscale_colab.ipynb`). This digest describes the
> laptop-scale run below (`results/msr/`), which is the paper's second training scale (appendix,
> `kca.run.scale_table` -> `paper/tables/tab_scale.tex`). Claim-by-claim comparison and the adoption
> decision: `FULLSCALE_ADOPTION.md`.

Every number here is produced by the scripts in `src/kca/run/` from per-clip records
(`results/msr/*/raw.jsonl`, `results/msr/crossover/`, `results/msr/ablation/`,
`results/controlled/`); the paper's tables and in-text numbers are generated from the
same files (`kca.run.paper_tables` → `paper/tables/`). Rerun: `../reproduce.sh --msr`.

**These numbers supersede the June results** (`results/controlled_old/`, `controlled_s5/`,
`analysis/`), which were produced before three fixes: (1) interventions and regions
assumed per-index point identity across frames — false on real point-cloud video;
(2) the (0,0) cube corner deleted the region, so a static region's presence leaked into
S_mot (shape-task motion share 0.49 instead of 0.00); (3) faithfulness was scored on raw
logits (not comparable across clips) instead of the target-class probability.

## Setup
MSR-Action3D (MeteorNet point clouds, 24 frames × 1024 points, subjects 1–5 train / 6–10
test), 20-joint skeleton lifted into the same coordinates (median 96 % of points within
12 cm of a bone) → moving/static ground truth. Official PSTNet / P4Transformer / Mamba4D
code, trained from scratch on an Apple M4 (pure-PyTorch ops, reduced width for P4T and
Mamba4D, AdamW + cosine; time-channel init scaled 1/L; clips centred).

| backbone | params | test video acc | published (full width, 2048 pts) |
|---|---|---|---|
| P4Transformer | 2.25 M | **90.6 %** | 90.9 |
| PSTNet | 8.26 M | **87.5 %** | 91.2 |
| Mamba4D | 1.92 M | **83.6 %** (best epoch 88.9 %; we report and use the last epoch) | 92.7 |

## 1. Faithfulness — P4Transformer, 100 test clips (5 per action)
Deletion AUC of the target-class probability (lower is better):

| method | Del ↓ | Comp ↑ |
|---|---|---|
| **KCA (context)** | **0.178** | **0.447** |
| region occlusion | 0.219 | 0.426 |
| KCA (local) | 0.261 | 0.401 |
| Integrated Gradients | 0.268 | 0.322 |
| transformer attribution | 0.419 | 0.192 |
| attention rollout | 0.484 | 0.161 |
| raw attention (last layer) | 0.542 | 0.111 |
| motion prior (floor) | 0.547 | 0.206 |
| 3D Grad-CAM / Grad×Input / SmoothGrad | 0.637 / 0.667 / 0.675 | ≤ 0.08 |
| extremal / OccAM | 0.733 / 0.848 | ≤ 0.05 |
| random (floor) | 0.915 | 0.00 |

- KCA-context vs region occlusion: paired Wilcoxon p = 0.0043 (KCA better; significant
  after Bonferroni over the table's 4 metrics — the * in Tab. 3); vs Integrated Gradients:
  p < 1e-3. (With the first 60 clips it was a tie, p = 0.26.)
- Region occlusion IS KCA's local necessity with a deletion corner (identical by
  construction); the deletion metric favours it over KCA-local (p = 0.022); the context
  estimator more than recovers the gap.
- Random point dropout barely changes these models (random deletion curve stays near
  the full-clip probability), so OccAM-style point masking and soft-mask extremal
  perturbation carry little signal, and keep-based metrics (insertion, sufficiency) reward
  spatially scattered rankings (random insertion AUC 0.914). Read deletion/comprehensiveness.

## 2. Localisation (skeleton moving-GT, P4Transformer, 100 clips)
| method | PG | EBPG |
|---|---|---|
| Integrated Gradients | **0.92** | **0.58** |
| KCA S_mot (local / context) | 0.57 / 0.58 | 0.56 / 0.54 |
| region occlusion | 0.52 | 0.45 |
| Grad×Input | 0.63 | 0.39 |
| motion prior (floor) | 0.85 | 0.27 |

KCA-context S_mot EBPG vs IG: p = 0.093 (tie; local vs IG p = 0.78); vs region
occlusion p < 1e-7. IG wins the Pointing Game: region-constant maps have no within-region
ranking. On moving-GT points 0.80 of KCA's positive importance is motion, vs 0.68 over the
whole body; per action (whole body, context estimator) it ranges from 0.45 (horizontal arm
wave) and 0.46 (jogging) to 0.76–0.79 (bend, forward kick, draw tick, side kick, tennis
serve) — see the supplementary per-action table.

## 3. Motion vs geometry — the crossover (decisive test)
Two binary tasks from real MSR clips, classes differing only in the right arm; one model
per task; KCA explains the class whose defining property is present (40 test clips).

| backbone | task | test acc | arm motion share (local) | 95 % CI | context |
|---|---|---|---|---|---|
| P4Transformer | motion (arm replayed at 25 % speed) | 85.5 % | **0.81** | [0.73, 0.88] | 0.91 |
| P4Transformer | shape (arm thickened 3× about its bones) | 92.2 % | **0.39** | [0.31, 0.47] | 0.39 |
| P4Transformer | gap | | **0.42** | [0.31, 0.52] | 0.52 |
| PSTNet | motion | 79.5 % | **0.85** | [0.81, 0.89] | 0.87 |
| PSTNet | shape | 90.8 % | **0.23** | [0.18, 0.28] | 0.24 |
| PSTNet | gap | | **0.61** | [0.55, 0.68] | 0.63 |
| Mamba4D | motion | 86.0 % | **0.89** | [0.86, 0.92] | 0.89 |
| Mamba4D | shape | 97.2 % | **0.54** | [0.49, 0.60] | 0.54 |
| Mamba4D | gap | | **0.35** | [0.29, 0.41] | 0.35 |

The model-blind motion prior marks the arm identically in both tasks (EBPG 0.23 / 0.23,
every backbone). The split moves with the cue on all three backbones (gap CIs exclude 0);
it is weakest on Mamba4D, whose shape model still draws about half of the arm's importance
from motion — in line with its motion-heavy maps on MSR (whole-body motion share 0.74). Signed-sum shares agree in direction (P4T 0.88 vs 0.37; PSTNet 1.10 vs
−0.01 — outside [0, 1] because the other term's sum over the arm is slightly negative).
Variants kept for the record (P4T): explaining the *original* class in the shape task
(its evidence is the absence of the cue) gave sign-convention-dependent shares (positive
0.95 / signed 0.31, `crossover_beta3.json`); thinning the arm was not learnable (chance,
`crossover_thin.json`). PSTNet: `crossover_pstnet/` (trained on CPU with `--train-only`,
explained on MPS with `--eval-only --explain-device mps`); Mamba4D: `crossover_mamba4d/` (trained and
explained on MPS).

## 4. Controlled study (ground truth by construction; 3 seeds × 20 clips)
- Double dissociation, motion share on the odd cluster: **1.00 (motion cue) vs 0.00
  (shape cue)**, local and context estimators alike; |Δint|/|Nec| = 0.02.
- With a deletion (or donor-fill) (0,0) corner the shape model's motion share becomes
  0.49 and |Δint|/|Nec| 0.99 — the presence leak the proxy corner removes.
- `displace` (trajectory reversal) flips a direction classifier to the opposite class in
  90 % of clips, `freeze` in 0 %.
- Deletion: region occlusion 0.392, KCA-context 0.399, KCA-local 0.419 (occlusion better
  than local, p = 0.034), Grad×Input 0.482, …, random 0.907.

## 5. Cost (M4 CPU, per clip, P4Transformer, mean over 100 clips)
KCA local 3.0 s (37 passes), KCA context 19.5 s (257 passes), OccAM 10.6 s, IG 4.9 s,
SmoothGrad 4.0 s, extremal 4.9 s, region occlusion 1.3 s. (PSTNet / Mamba4D on MPS, shared
GPU: KCA local 9.0 / 6.0 s, context 52.7 / 30.7 s.)

## 6. Ablation (P4Transformer, 40 clips; deletion AUC ↓, Smot EBPG ↑, motion share on moving GT, |Δint|/|Nec|)
| variant | Del | EBPG | share | Δint/Nec | passes | s |
|---|---|---|---|---|---|---|
| local m=6 / 12 / 24 | 0.245 / 0.250 / **0.432** | 0.46 / 0.58 / 0.58 | 0.71 / 0.77 / 0.85 | 0.55 / 0.63 / 0.71 | 19 / 37 / 73 | 2.1 / 3.4 / 5.6 |
| context m=6 / 12 / 24 | 0.189 / 0.200 / **0.228** | 0.43 / 0.57 / 0.56 | 0.70 / 0.79 / 0.85 | 0.48 / 0.55 / 0.71 | 257 | 25 / 24 / 22 |
| local, deletion corner | 0.226 | 0.54 | 0.73 | 0.77 | 37 | 3.3 |
| context, deletion corner | **0.155** | 0.51 | 0.75 | 0.61 | 257 | 20 |
| local, donor fill | 0.228 | 0.54 | 0.73 | 0.79 | 37 | 3.0 |
| local, motion-first order | 0.250 | 0.57 | **0.86** | – | 25 | 2.3 |
| context S = 64 / 128 / 512 | 0.214 / 0.184 / 0.202 | 0.52 / 0.56 / 0.58 | 0.79 / 0.79 / 0.80 | | 65 / 129 / 513 | 6 / 11 / 36 |
| context q = 0.75 | 0.221 | 0.59 | 0.77 | 0.61 | 257 | 18 |
| local, anchor t0 = middle frame | 0.255 | 0.61 | 0.72 | 0.61 | 37 | 3.1 |
| context, anchor t0 = middle frame | 0.157 | 0.59 | 0.74 | 0.55 | 257 | 21 |

- Redundancy on real data: at m = 24 the local necessity collapses (0.432) and the
  context estimator repairs it (0.228).
- The deletion corner helps the deletion metric (context 0.155 is the best of all) but
  weakens Smot localisation and raises the interaction; on the controlled task it makes
  a pure-shape model look 49 % motion-driven — the reason the default is the proxy corner.
- Motion-first scoring absorbs the interaction into Smot (share 0.86 vs 0.77 symmetric).
- Anchor frame (freeze / static proxy at the middle instead of the first frame): split nearly
  unchanged (local motion share 0.72 vs 0.77) and faithfulness not hurt (context deletion
  0.157 vs 0.200) — the first-frame default is not a tuned choice.
- Interactions are large on the real model (|Δint|/|Nec| 0.5–0.8) vs 0.02 on the
  controlled network — hence symmetric scores + a measured residual.

## 7. Sanity and the displace probe (P4Transformer)
- Cascading randomisation (head first, 7 blocks, 20 clips): KCA's rank correlation with
  the trained model's map drops to −0.11 once the head is re-initialised and stays ≈ 0
  (−0.01 with all blocks random). Grad×Input keeps 0.72 with all blocks random — mostly
  input magnitude, not the model.
- On the top-Smot region: freezing changes the prediction in 48 % of clips, reversing
  the trajectory in 18 % (logit drops 5.2 vs 1.7): real MSR decisions depend more on
  whether a part moves than on which way.

## 8. Crossover robustness
Second training seed: motion task 72.2 % acc, arm motion share 0.73; shape task 87.5 %,
share 0.45; gap 95 % CI [0.16, 0.40] (seed 0: 0.81 vs 0.39, [0.31, 0.52]).

## 9. Cross-backbone (40 common test clips, 2 per action)
Deletion AUC ↓ / moving-GT EBPG ↑ (S_mot for KCA):

| method | PSTNet | P4Transformer | Mamba4D |
|---|---|---|---|
| KCA (context) | **0.191** / 0.45 | **0.200** / 0.57 | 0.245 / 0.49 |
| KCA (local) | 0.229 / 0.41 | 0.250 / 0.58 | 0.299 / **0.49** |
| region occlusion | 0.222 / 0.40 | 0.226 / 0.47 | **0.240** / 0.40 |
| Integrated Gradients | 0.264 / **0.46** | 0.254 / **0.60** | 0.318 / 0.48 |
| extremal / OccAM (Del only) | 0.912 / 0.804 | 0.729 / 0.844 | 0.633 / 0.803 |
| random (floor) | 0.902 / 0.17 | 0.908 / 0.17 | 0.875 / 0.17 |

- KCA-context vs region occlusion (Del): PSTNet p = 0.55, Mamba4D p = 0.46 (ties); vs IG:
  p = 0.018 and p < 1e-3 (KCA better). S_mot EBPG vs IG: p = 0.72 / 0.33 (ties); vs
  region occlusion: p = 0.09 / 0.003.
- Kendall τ of the method ranking by Del: PSTNet–P4T 0.81, PSTNet–Mamba4D 0.71,
  P4T–Mamba4D 0.90.
- Every informative map puts less mass on the moving parts for PSTNet and Mamba4D than
  for P4Transformer; qualitatively both spread motion importance over the torso
  (Mamba4D's strongest motion region for *high arm wave* is the torso).
- Whole-body motion share (context): PSTNet 0.59, P4T 0.68, Mamba4D 0.74; on moving
  parts 0.70 / 0.80 / 0.85.
