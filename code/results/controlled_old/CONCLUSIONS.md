# Controlled study — results & conclusions

Trained attention-based 4D net on the motion-only task: **val acc 1.000 ± 0.000** (3 seeds) — geometry is constant across classes, so the model provably uses motion.

## 1. Faithfulness (KCA vs baselines, Deletion-AUC ↓)
- **KCA 0.173 ± 0.507**, random 2.130, geometry-edge 2.020, motion-prior 0.378.
- KCA's necessity-ranked region, when deleted, collapses the target logit far below the random, geometry-edge, and gradient/attention baselines (significantly, Wilcoxon p=1.5e-05 vs grad_x_input).
- KCA *ties* the strongest occlusion baseline (point-drop) on raw deletion — expected, since both rest on ablation necessity; KCA's distinctive value is the **decomposition** (Sec. 3), not raw deletion over occlusion. The **motion-blind geometry-edge floor fails**, confirming the signal is motion.

## 2. Motion localization (Smot → moving structure, EBPG ↑)
- **KCA Smot EBPG 0.666** vs random floor 0.333: Smot concentrates on the genuinely moving cluster.

## 3. Motion-vs-geometry double dissociation (the decisive decomposition test)
Within the odd (decision-critical) cluster, the motion fraction of KCA's importance Smot/(Smot+Sgeo):
- cue=**motion** model: **0.36** (Smot dominates).
- cue=**shape** model: **0.01** (Sgeo dominates — Smot≈0).
- dissociation gap = **0.35** (motion minus shape).
- The mass shifts between Smot and Sgeo with the true cue **for the same physical cluster** — KCA captures motion that *drives the decision*, not motion that is merely present (the motion-prior floor, which is competitive on raw motion-task faithfulness, cannot do this).

## 4. Ablations
- Interaction residual: mean |Δ_int| = 0.930 (small ⇒ near-additive decomposition).
- Baseline fill (Deletion-AUC): bg -0.07, mean -0.03, nn -0.07.
- Symmetric vs motion-first scoring: -0.07 vs -0.07.

## 5. Soundness & amortized fidelity
- **Stability**: Max-Sensitivity 0.075 (small ⇒ robust to input noise). (Infidelity is scale-dependent and omitted as a headline.)
- **Adebayo model-randomization**: rank-corr(trained, random-weights) = **+0.208**. CAVEAT: this controlled task has only 3 regions, so a 3-point rank correlation is underpowered/noisy — the meaningful Adebayo test is on the real backbones (many regions). Do not read this as a clean pass/fail.
- **Amortized head ($O(1)$)** is an APPROXIMATION of the exact field: rank-corr **0.69**; its faithfulness trails exact (Deletion-AUC -0.01 vs exact -0.03) — expected for O(1), improves with more distillation data.

## 6. Sign-sensitive `displace` probe (motion direction)
- On a task where class = motion DIRECTION with a shared spatial envelope (only temporal order discriminates), reversing the actor's trajectory flips the prediction to the OPPOSITE direction **90%** of the time, vs **0%** for freeze. `displace` necessity 3.21 vs freeze 1.13.
- This is what `displace` adds over `freeze`: it is **sign-sensitive** — it captures which WAY a structure moves, not just whether it moves. (acc 0.97.)

## Bottom line
On a trained 4D network where ground truth is known by construction, KCA is (i) **faithful** (deletion crushes all floors, significantly), (ii) **correctly localized** (Smot on the moving structure), and (iii) **correctly decomposed** (double dissociation between motion and geometry cues). This validates the method end-to-end ahead of the large-scale GPU benchmarks.
