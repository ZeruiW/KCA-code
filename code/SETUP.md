# Runbook — from raw data to every number in the paper

Everything below was run end-to-end on an Apple M4 laptop (CPU + MPS) on 2026-09-26.
On a CUDA machine the same commands work with `--device cuda` (and optionally the
official full-width configs, `--scale official`). One command does it all:
`./reproduce.sh --msr` (see the repo root).

Keep the working directory **out of iCloud/Dropbox** (`KCA_WORK=~/kca_work` by default):
venvs, cloned repos and datasets are tens of thousands of small files, and iCloud
"optimize storage" evicts files (reads then hang until re-downloaded).

## 0. Environment
```bash
uv venv ~/kca_work/venv --python 3.12
uv pip install --python ~/kca_work/venv/bin/python torch numpy scipy matplotlib einops timm termcolor gdown
export PYTHONPATH=$PWD/src PYTHONDONTWRITEBYTECODE=1 PYTORCH_ENABLE_MPS_FALLBACK=1
PY=~/kca_work/venv/bin/python
```
No CUDA extension has to be compiled: `kca/backbones/cpu_ops.py` provides pure-PyTorch
FPS / ball query / grouping / KNN / Mamba selective scan / RMSNorm with the CUDA kernels'
semantics, and `kca/backbones/official.py` imports each official repo in isolation.
(On a GPU host you may still install the official ops — `P4Transformer/modules-pytorch-1.8.1`
(not `modules/`), PSTNet `modules/`, Mamba4D's docker image — they are used automatically
when CUDA is available.)

## 1. Data: MSR-Action3D + skeleton ground truth (no approval needed)
```bash
mkdir -p ~/kca_work/data/msr_raw && cd ~/kca_work/data/msr_raw
~/kca_work/venv/bin/gdown 1djwAK3oZTAIFbCz531eClxINmsZgGO_H -O msr.zip   # MeteorNet's link, 64.6 MB
unzip -o msr.zip && mkdir -p depth skel2d
bsdtar -xf Depth.rar -C depth && bsdtar -xf "MSRAction3DSkeleton(20joints).rar" -C skel2d
cd -
$PY -m kca.data.prepare_msr --depth ~/kca_work/data/msr_raw/depth/Depth \
    --skeleton "$HOME/kca_work/data/msr_raw/skel2d/MSRAction3DSkeleton(20joints)" --out ~/kca_work/data/msr
```
567 `.npz` files (13 s): `point_clouds` is byte-compatible with the official loaders
(MeteorNet projection); `joints` holds the 20-joint skeleton lifted into the same
coordinates (a median 96 % of the points lie within 12 cm of a bone). Split: subjects 1–5 train (270
videos), 6–10 test (297; 287 have ≥ 24 frames).

## 2. Official backbones (no pretrained weights exist — train from scratch)
```bash
mkdir -p ~/kca_work/ext && cd ~/kca_work/ext
git clone --depth 1 https://github.com/hehefan/P4Transformer
git clone --depth 1 https://github.com/hehefan/Point-Spatio-Temporal-Convolution
git clone --depth 1 https://github.com/IRMVLab/Mamba4D
cd -
$PY -m kca.run.train_official --backbone p4transformer --data ~/kca_work/data/msr --out ~/kca_work/ckpt/p4t     --epochs 50 --clips-per-video 5
$PY -m kca.run.train_official --backbone pstnet        --data ~/kca_work/data/msr --out ~/kca_work/ckpt/pstnet  --epochs 30 --clips-per-video 3
$PY -m kca.run.train_official --backbone mamba4d       --data ~/kca_work/data/msr --out ~/kca_work/ckpt/mamba4d --epochs 20 --clips-per-video 3
```
Verified facts about the repos (their `train-msr.py`): P4Transformer `models/msr.py`
class `P4Transformer` (radius 0.7, nsamples 32, 24 frames); PSTNet
`models/sequence_classification.py` class `MSRAction` (radius 0.5, nsamples 9);
Mamba4D `models/msr.py` class `MAMBA4D` (its `--intra False` path does not construct).
All three save `{'model': model_without_ddp.state_dict(), 'args': ...}`;
`OfficialBackbone(name, ckpt)` loads both those and ours, strictly.

Two things were needed for from-scratch training on a laptop budget (`--scale reduced`,
AdamW 3e-4 + cosine; the official SGD recipe needs its ~17k iterations): the absolute
time channel of the positional embedding dominates every token at initialisation, so
its initial weights are scaled by 1/L, and clips are centred (stored in the checkpoint
as `preproc`, re-applied at explanation time). Reached (24 frames, 1024 points):
P4Transformer 90.6 %, PSTNet 87.5 %, Mamba4D 83.6 % test video accuracy (last epoch; no
test-set model selection).

## 3. Benchmarks → tables and figures
```bash
$PY -m kca.run.msr_benchmark --backbone p4transformer --ckpt ~/kca_work/ckpt/p4t/last.pth \
    --data ~/kca_work/data/msr --out results/msr/p4transformer --per-class 5 --device cpu
for b in pstnet mamba4d; do          # on MPS/CUDA (Mamba4D's scan is far too slow on CPU)
  $PY -m kca.run.msr_benchmark --backbone $b --ckpt ~/kca_work/ckpt/$b/last.pth --data ~/kca_work/data/msr \
      --out results/msr/$b --per-class 2 --metrics delins --methods random,integrated_gradients,point_drop,occam,extremal,kca,kca_ctx
done
$PY -m kca.run.msr_crossover --data ~/kca_work/data/msr --out results/msr/crossover --device cpu
$PY -m kca.run.msr_crossover --data ~/kca_work/data/msr --out results/msr/crossover_seed1 --device cpu --seed 1
# the same crossover on PSTNet: train on CPU (or GPU), explain on MPS/CUDA
$PY -m kca.run.msr_crossover --backbone pstnet --data ~/kca_work/data/msr --out results/msr/crossover_pstnet --device cpu --threads 4 --train-only
$PY -m kca.run.msr_crossover --backbone pstnet --data ~/kca_work/data/msr --out results/msr/crossover_pstnet --eval-only --explain-device mps
# and on Mamba4D (train on the GPU: its scan is far too slow on CPU)
$PY -m kca.run.msr_crossover --backbone mamba4d --data ~/kca_work/data/msr --out results/msr/crossover_mamba4d --device mps --train-only
$PY -m kca.run.msr_crossover --backbone mamba4d --data ~/kca_work/data/msr --out results/msr/crossover_mamba4d --eval-only --explain-device mps
$PY -m kca.run.msr_ablation  --ckpt ~/kca_work/ckpt/p4t/last.pth --data ~/kca_work/data/msr --out results/msr/ablation
$PY -m kca.run.controlled_study --out results/controlled --seeds 3 --n-eval 20
$PY -m kca.run.paper_tables  --results results/msr --paper ../paper          # -> paper/tables/*.tex
$PY -m kca.run.paper_figures --results results/msr --data ~/kca_work/data/msr --fig ../paper/fig
bash ../tools/check_paper.sh        # pdfLaTeX + BibTeX (AAAI-27 style), page and ?? checks
```
`msr_benchmark` writes one JSON line per clip (resumable; `--per-class 5` later extends
the same run). Costs on the M4 CPU: ~2.5 min per clip for all 15 methods on
P4Transformer (KCA local ≈ 3 s, context ≈ 20 s per clip). Do not run a latency-bound
benchmark on MPS while a training job occupies the GPU: its small kernels starve behind
the training command buffers.

## 4. Full-scale (GPU) version of the paper's numbers
Colab: `../colab/KCA_fullscale_colab.ipynb` runs everything below plus the post-processing
(tables, numbers, figures, PDF and a laptop-vs-full-scale report, `kca.run.compare_full`);
`../colab/make_bundle.sh` packs the code, paper and data it needs.

By hand: export `KCA_PURE_OPS=1` (the pure-PyTorch ops on CUDA too, with the checked fast
paths of `kca/backbones/cuda_fast.py`: a compiled FPS kernel and `mamba_ssm`'s selective
scan) and `KCA_MAMBA_CHECKPOINT=1` (recompute Mamba layers in backward; official-width
Mamba4D needs it). Train with `--device cuda --scale official --points 2048 --resume`, and
pass `--points 2048` to `msr_benchmark` / `msr_ablation` and `--scale official --points
2048 --reuse` to `msr_crossover` (the benchmark records `num_points` per clip, and
`paper_tables` / `paper_figures` rebuild the same clips from it). Expected published
accuracies: P4Transformer 90.9, PSTNet 91.2, Mamba4D 92.7 (MSR, 24 frames).

## 5. Tests
```bash
for t in test_decomposition test_symmetrization test_correspondence_free test_metrics_baselines \
         test_pipeline_e2e test_real_backbone test_displace_probe; do $PY tests/$t.py; done
```
`test_correspondence_free` guards the property real data needs: shuffling every frame's
point order must not change the scores, and freeze/displace/abl-geo must act on tracked
region SETS (the index-aligned freeze fails that test on correspondence-free data).
