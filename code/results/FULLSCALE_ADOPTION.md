# Full-scale reproduction: results and adoption recommendation (2026-10-22, Claude)

One complete run on a Colab A100 (10/21–10/22). The raw outputs are in the project Google Drive under
`My Drive/kca_full/`, unpacked locally in `~/kca_work/full/`: `report/REPORT.md` (automatic comparison report),
`report/numbers_compare.csv`, `paper_full/tables`, `fig`, `results/msr/`. The checkpoints exist only on Drive.

## 1. Training results (official width, 2048 points per frame, AdamW, last epoch)

| backbone | params | full scale | laptop scale | published | gap to published |
|---|---|---|---|---|---|
| P4Transformer | 44.1 M | **89.2** | 90.6 | 90.9 | −1.7 |
| PSTNet | 8.3 M | **88.9** | 87.5 | 91.2 | −2.3 |
| Mamba4D | 109.8 M | **88.2** (best epoch 89.2) | 83.6 | 92.7 | −4.5 (laptop scale: −9.1) |

Both CUDA fast paths (FPS, selective scan) were enabled and all 7 unit tests passed. Time on the A100: P4T training
44 minutes, PSTNet 87 minutes, Mamba4D 7.8 hours.

## 2. Claim-by-claim comparison (the automatic report has 4 ✗; one more problem it missed is added)

| Claim | Laptop scale | Full scale | Verdict |
|---|---|---|---|
| P4T: KCA-ctx has the lowest deletion AUC | 0.178 | **0.207**, still the lowest, Comp still the highest | ✓ |
| P4T: KCA-ctx significantly better than region occlusion | p=0.004 | 0.207 vs 0.250, **p=0.10** | ✗ same direction, but not significant |
| P4T: KCA-ctx significantly better than IG | p<10⁻³ | p=0.003 | ✓ |
| PSTNet: deletion AUC | lowest, on par with occlusion, significantly better than IG | lowest (0.217), on par with occlusion (p=0.38), significantly better than IG (p=0.007) | ✓ |
| Mamba4D: deletion AUC | on par with occlusion, significantly better than IG | second (0.269 vs occlusion 0.246, p=0.83), vs IG **p=0.064** | ✗ vs IG goes from significant to on par |
| S_mot localization on par with IG | holds on all three backbones | holds on all three backbones | ✓ |
| Crossover P4T | 0.81 vs 0.39, gap CI [0.31, 0.52] | 0.83 vs **0.60**, CI [0.14, 0.32] | ✓ but weaker |
| Crossover PSTNet | 0.85 vs 0.23 | 0.86 vs 0.30, CI [0.48, 0.65] | ✓ |
| Crossover Mamba4D | 0.89 vs 0.54 | 0.92 vs 0.61, CI [0.22, 0.41] | ✓ |
| **Crossover P4T, second seed** | motion-task accuracy 72.2%, 0.73 vs 0.45 | **motion-task accuracy 50.0% (chance)**, 0.65 vs 0.55 | ✗ **the automatic report marked it ✓, but the motion-task model never learned its task, so this result is meaningless** |
| Sanity check | after randomization KCA −0.01, G×I 0.72 | KCA −0.10, G×I 0.81 | ✓ |
| At m=24 the context estimator repairs the collapse of local necessity | 0.432 → 0.228 | 0.334 → 0.212 | ✓ |
| displace probe | freeze 48% vs reverse 18% | 42% vs 22% | ✓ |
| Method ranking, Kendall τ | 0.71–0.90 | 0.71–0.81 | ✓ |
| Explanation time (per clip) | M4 CPU: local 3.0 s, ctx 19.5 s | **A100: local 0.5 s, ctx 2.3 s** | strengthens the "deployable" argument |

## 3. Recommendation (Claude's judgement)

**Recommended option (a): the main text switches to the full-scale results; the laptop scale goes to the appendix as a
"second training scale".**

Reasons:
1. **Model scale is what reviewers are most likely to question**, and the full-scale run answers it directly: the gap
   between Mamba4D and its published accuracy shrinks from 9.1 to 4.5 points.
2. **Having full-scale results and reporting only the better-looking laptop numbers in the main text would be selective
   reporting.** Reporting both scales instead supports the statement "the main conclusions hold across training scales",
   which is exactly the kind of reliability DAI cares about.
3. **The core claims still hold**: KCA-ctx has the lowest or second-lowest deletion AUC on all three backbones; the gap
   CI of the crossover excludes 0 on all three backbones; the sanity check passes; the explanation cost on a GPU is
   0.5 s and 2.3 s per clip.

What needs rewriting:
- P4T: "significantly better than region occlusion" becomes "lowest, but the difference to region occlusion is not
  significant (p=0.10); significantly better than IG". The abstract and conclusion say "lowest deletion AUC", which
  still holds.
- Mamba4D: versus IG becomes "on par".
- Crossover: in the shape task of P4T and Mamba4D, about 60% of the arm importance still comes from motion. Write it as
  "the motion share moves clearly with the cue (gap CI excludes 0), but these two models do not separate the cues
  completely". PSTNet separates them most cleanly.
- **Second seed**: the motion-task model stayed at chance, so the result cannot be used. Recommended: rerun the P4T
  crossover on Colab with another seed (each task model trains in about 100 s on an A100; with the explanations about
  10 minutes), and state honestly in the appendix that the seed-1 motion model failed to train. The alternative is to
  drop the sentence.
- Sentences about the training setup (width, number of points, hardware, run time): locations in Section 3 of
  REPORT.md; the Cost paragraph switches to the GPU numbers; Limitations (i) is rewritten.
- Add an appendix table comparing the two scales, generated by a script from the two `numbers.tex` files, not typed by hand.

Option (b), keeping the laptop scale in the main text and putting the full scale in the appendix: keeps the P4T sentence
"significantly better than occlusion", but reviewers will ask why the main text uses the smaller models. Not recommended.
Option (c), not using the full-scale results at all: not recommended, for reason 2 above.

## 4. Decision and execution (10/22)

**The author chose option (a).** Done by Claude:
- The full-scale results were copied to `code/results/msr_full/`: per-clip records, crossover, ablation, the
  `summary.json` and training logs of the three backbones, and the Colab logs. The `.pth` files stay on Drive only.
  The tables regenerated locally are byte-identical to the Colab outputs.
- The laptop-scale numbers are kept in `code/results/msr/numbers_laptop.tex`. The new script `kca.run.scale_table`
  builds the appendix table `tab_scale.tex` from both sets of numbers, in the new appendix section "Two training scales".
- Main text and appendix rewritten for the full-scale results: P4T vs region occlusion and Mamba4D vs IG are now
  "not significant"; the crossover gained the sentence "the audit shows which cue a model responds to, not that it uses
  that cue alone"; Cost uses the A100 numbers; the implementation details describe both scales; Limitations (i) rewritten.
- `paper_tables` gained the seed-2 macros; `compare_full` gained an accuracy threshold: crossover task models must reach
  at least 60% accuracy.
- **The teaser clip was changed (Claude's judgement)**: at full scale, P4T puts the motion importance of *high arm wave*
  on the torso, not on the waving arm, so the old teaser caption no longer held. The teaser now uses *draw circle*, where
  the motion importance of all three backbones is on the drawing hand. Appendix Fig. 6 still shows all 3 actions × 3
  backbones, Section 4.6 reports the *high arm wave* case honestly, and the teaser caption mentions that a counterexample
  exists. The clip was changed after seeing the results, which is why it is recorded here.
- **Seed 2 (rerun on 10/22, `colab/KCA_seed2_colab.ipynb`, outputs on Drive in `kca_full/seed2/`)**: motion-task accuracy
  67.8%, shape task 88.8%, both above the 60% threshold. Arm motion share 0.73 vs 0.57, gap 95% CI [0.06, 0.26]: same
  direction as seed 0, smaller gap. `crossover.json` was copied to `code/results/msr_full/crossover_seed2/`; the appendix
  reports the seed-1 failure and the seed-2 result. The paper compiles with every check passing and 0 `??`.
