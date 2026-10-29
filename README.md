# KCA (Kinematic Causal Attribution) — code, results and Colab notebooks

Companion code for the paper *Kinematic Causal Attribution: Black-Box Explanations of Motion versus Geometry for 4D
Point-Cloud Video Models* (AAAI-27 Workshop on Deployable AI, DAI 2027; double-blind, so please keep this material
private until the review is over). KCA explains a 4D point-cloud video model by intervening on its input
(ablate / freeze / displace) and splits each spacetime region's importance into a **motion** term and a **geometry**
term. Backbones: P4Transformer, PSTNet, Mamba4D; dataset: MSR-Action3D.

The paper itself is edited on Overleaf from the separate package `KCA_overleaf.zip`. The `paper/` folder here is the
same source, included so that the scripts below can regenerate the tables and figures in place; upload regenerated
`paper/tables/*.tex` and `paper/fig/*.pdf` files to Overleaf afterwards.

## Contents
```
README.md                     this file
reproduce.sh                  one-command reproduction (see below)
tools/check_paper.sh          compile paper/ with pdfLaTeX + BibTeX and check the AAAI page limit, ??, references, overfull boxes
code/                         the implementation; start with code/README.md (method and layout) and code/SETUP.md (runbook)
  src/kca/                    method, backbones, baselines, metrics, data loaders, experiment runners
  tests/                      7 CPU test suites
  configs/                    example configuration
  results/RESULTS.md          digest of every result
  results/FULLSCALE_ADOPTION.md  laptop vs full-scale comparison and why the paper uses the full-scale run
  results/msr_full/           full-scale run (official width, 2048 points, A100): per-clip records, crossover (seeds 0-5,
                              PSTNet, Mamba4D, control variants), ablation, stability, training summaries and all Colab logs
                              -> the numbers in the paper's main text
  results/msr/                laptop-scale run (reduced width, Apple M4) -> the paper's second training scale (appendix)
  results/controlled*/        controlled study on synthetic networks (ground truth by construction)
  results/analysis/           extra analyses (k sweep, amortized head, bar charts)
colab/                        the three Colab notebooks of the full-scale run, ENGLISH COPIES WITH THEIR RECORDED OUTPUTS
  KCA_fullscale_colab.ipynb   train the 3 backbones at official scale + every real-data experiment + post-processing (10/21-22)
  KCA_seed2_colab.ipynb       P4Transformer crossover, seed 2 (10/22)
  KCA_extend_colab.ipynb      full test set benchmark, re-sampling stability, crossover seeds 3-5 (10/22)
  make_bundle.sh              packs the code, paper and data the notebooks need (upload to Drive: MyDrive/kca_full/upload/)
paper/                        LaTeX source of the paper (same as KCA_overleaf.zip)
```

About the notebooks: they are the executed notebooks saved on Google Drive after the actual runs, with every Chinese
markdown cell, comment and printed message translated to English. The code is unchanged (checked token by token) and
every output is the recorded one; only output lines that contained Chinese text were translated. Paths such as
`~/Documents/4DXAI` and `~/kca_work` refer to the original author's machine. The saved copy of
`KCA_extend_colab.ipynb` ends after its Section 2 (the later runs finished, see the note in its first cell), and
Section 8 of `KCA_fullscale_colab.ipynb` still compiles the paper the way it was built at that time (tectonic,
8-page CVPR format); the current paper needs pdfLaTeX and the 7-page AAAI limit.

## Not included (too large or regenerable)
- **MSR-Action3D data**: public dataset. `./reproduce.sh --msr` downloads it (MeteorNet's link) and converts it with
  `code/src/kca/data/prepare_msr.py` into 567 point-cloud `.npz` files with skeleton joints; the conversion is deterministic.
- **The three official model repositories** (P4Transformer, PSTNet, Mamba4D): cloned by `reproduce.sh`; the commits
  used are listed in the paper's appendix ("Data, models and code behind each result").
- **Trained weights (`.pth`)**: the full-scale checkpoints (P4Transformer 177 MB, PSTNet 33 MB, Mamba4D 439 MB, each
  `best.pth` and `last.pth`; the paper uses `last.pth`) and the crossover task models are kept on the project Google Drive
  (`My Drive/kca_full/ckpt/` and `.../results/msr/crossover*/`); ask the author for access. Their training summaries
  and logs are in `code/results/msr_full/ckpt/` and `code/results/msr_full/logs/`.

## Quick start
Requirements: Python ≥ 3.10 (3.12 used), PyTorch; for the paper, pdfLaTeX (TeX Live or TinyTeX). The script creates a
virtual environment and keeps data and checkpoints in `$KCA_WORK` (default `~/kca_work`, outside any cloud-synced folder).
```bash
./reproduce.sh --quick     # environment + 7 CPU test suites + dry run (a few minutes)
./reproduce.sh             # + controlled study, its figures, tables from the full-scale results, compile the paper
./reproduce.sh --msr       # + the whole laptop-scale real-data pipeline (download, convert, train, benchmark; ~15-20 h on an M4)
```
The full-scale run needs a GPU: follow `colab/KCA_fullscale_colab.ipynb` (A100, roughly 13-15 hours in total, most of it Mamba4D training).
Regenerating the paper's tables and figures from the stored full-scale records (`code/results/msr_full/`) needs only
the MSR `.npz` files, no GPU and no checkpoints: this is step 6b of `reproduce.sh` (details in `code/SETUP.md`, Section 4).

## Rule for the paper
Every table body and every number quoted in the text comes from `code/src/kca/run/paper_tables.py` (into
`paper/tables/`), and the result figures from `paper_figures.py` / `make_controlled_figures.py` (into `paper/fig/`).
Do not type numbers into the LaTeX by hand: change the code or the records, regenerate, then upload the regenerated files.
