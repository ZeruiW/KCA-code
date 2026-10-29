#!/usr/bin/env bash
# ============================================================================
# KCA (4D-XAI) — one-command reproduction.
#
#   ./reproduce.sh            env + tests + controlled study + figures + paper
#   ./reproduce.sh --quick    env + tests + dry-run only
#   ./reproduce.sh --msr      ... plus the REAL-DATA pipeline on MSR-Action3D:
#                             download + convert (depth -> points, skeleton GT),
#                             train the three official backbones, benchmark every
#                             method, crossover, ablation, tables, figures.
#                             (Laptop scale: roughly 15-20 h on an Apple M4, mostly training.)
#
# Knobs (env vars):
#   KCA_WORK=~/kca_work       venv, data, checkpoints, cloned repos (keep it OUT of
#                             iCloud/Dropbox: tens of thousands of small files)
#   SEEDS=3 NEVAL=20          controlled-study scale
#   DEVICE=mps|cuda|cpu       real-data device (default: best available)
# ============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE="$SCRIPT_DIR/code"
PAPER="$SCRIPT_DIR/paper"
WORK="${KCA_WORK:-$HOME/kca_work}"
VENV="$WORK/venv"
PY="$VENV/bin/python"
SEEDS="${SEEDS:-3}"
NEVAL="${NEVAL:-20}"
MODE="${1:-full}"
export PYTHONPATH="$CODE/src" PYTHONDONTWRITEBYTECODE=1 PYTORCH_ENABLE_MPS_FALLBACK=1 MPLBACKEND=Agg

cd "$CODE"
hr() { printf '\n\033[1m==== %s ====\033[0m\n' "$1"; }
ok() { printf '\033[32m✓ %s\033[0m\n' "$1"; }
die() { printf '\033[31m✗ %s\033[0m\n' "$1"; exit 1; }

# ---------------------------------------------------------------------------
hr "1  Environment ($VENV)"
mkdir -p "$WORK"
if [ ! -x "$PY" ]; then
  if command -v uv >/dev/null 2>&1; then uv venv "$VENV" --python 3.12 -q || die "uv venv failed"
  else python3 -m venv "$VENV" || die "venv creation failed (need Python >= 3.10)"; fi
fi
PKGS="torch numpy scipy matplotlib einops timm termcolor gdown"
if command -v uv >/dev/null 2>&1; then uv pip install --python "$PY" -q $PKGS || die "pip install failed"
else "$VENV/bin/pip" install --quiet $PKGS || die "pip install failed"; fi
ok "python: $("$PY" --version 2>&1) | torch: $("$PY" -c 'import torch;print(torch.__version__)')"

# ---------------------------------------------------------------------------
hr "2  Verification tests"
TESTS=(test_decomposition test_symmetrization test_correspondence_free test_metrics_baselines
       test_pipeline_e2e test_real_backbone test_displace_probe)
fail=0
for t in "${TESTS[@]}"; do
  if "$PY" "tests/$t.py" >"$WORK/test_$t.log" 2>&1; then
    ok "$t  $(grep -hoE 'OK .*' "$WORK/test_$t.log" | head -1)"
  else
    printf '\033[31m✗ %s (see %s)\033[0m\n' "$t" "$WORK/test_$t.log"; tail -5 "$WORK/test_$t.log"; fail=1
  fi
done
[ "$fail" = 0 ] || die "some tests failed"
"$PY" -m kca.run.cvpr_benchmark --dry-run --out "$WORK/cvpr_dry" >/dev/null 2>&1 \
  && ok "legacy orchestrator dry-run" || die "dry-run failed"
[ "$MODE" = "--quick" ] && { hr "DONE (--quick)"; exit 0; }

# ---------------------------------------------------------------------------
hr "3  Controlled study ($SEEDS seeds x $NEVAL clips) + figures"
"$PY" -m kca.run.controlled_study --out results/controlled --seeds "$SEEDS" --n-eval "$NEVAL" 2>&1 \
  | grep -vE "RuntimeWarning|z = |Matplotlib is building" || die "controlled study failed"
"$PY" -m kca.run.make_controlled_figures --out "$PAPER/fig" >/dev/null 2>&1 && ok "controlled figures"

# ---------------------------------------------------------------------------
if [ "$MODE" = "--msr" ]; then
  hr "4  Real data: MSR-Action3D (depth + skeleton) -> point-cloud npz"
  RAW="$WORK/data/msr_raw"; NPZ="$WORK/data/msr"
  if [ ! -d "$NPZ" ] || [ "$(ls "$NPZ" | wc -l)" -lt 567 ]; then
    mkdir -p "$RAW" && cd "$RAW"
    [ -f msr.zip ] || "$VENV/bin/gdown" 1djwAK3oZTAIFbCz531eClxINmsZgGO_H -O msr.zip || die "download failed (MeteorNet's MSR-Action3D link)"
    unzip -o -q msr.zip && mkdir -p depth skel2d \
      && bsdtar -xf Depth.rar -C depth && bsdtar -xf "MSRAction3DSkeleton(20joints).rar" -C skel2d || die "extract failed (needs bsdtar)"
    cd "$CODE"
    "$PY" -m kca.data.prepare_msr --depth "$RAW/depth/Depth" --skeleton "$RAW/skel2d/MSRAction3DSkeleton(20joints)" --out "$NPZ" || die "conversion failed"
  fi
  ok "npz: $(ls "$NPZ" | wc -l | tr -d ' ') videos"

  hr "5  Official backbones (cloned, trained from scratch: no public weights exist)"
  mkdir -p "$WORK/ext" && cd "$WORK/ext"
  for r in hehefan/P4Transformer hehefan/Point-Spatio-Temporal-Convolution IRMVLab/Mamba4D; do
    [ -d "$(basename $r)" ] || git clone --depth 1 -q "https://github.com/$r.git" || die "clone $r failed"
  done
  cd "$CODE"
  DEV="${DEVICE:-}"; DARG=""; [ -n "$DEV" ] && DARG="--device $DEV"
  C="$WORK/ckpt"
  [ -f "$C/p4t/summary.json" ]    || "$PY" -m kca.run.train_official --backbone p4transformer --data "$NPZ" --out "$C/p4t" --epochs 50 --clips-per-video 5 $DARG
  [ -f "$C/pstnet/summary.json" ] || "$PY" -m kca.run.train_official --backbone pstnet --data "$NPZ" --out "$C/pstnet" --epochs 30 --clips-per-video 3 $DARG
  [ -f "$C/mamba4d/summary.json" ]|| "$PY" -m kca.run.train_official --backbone mamba4d --data "$NPZ" --out "$C/mamba4d" --epochs 20 --clips-per-video 3 $DARG
  for b in p4t pstnet mamba4d; do ok "$b: test video acc $("$PY" -c "import json;print(round(json.load(open('$C/$b/summary.json'))['last_acc'],3))")"; done

  hr "6  Benchmarks, crossover, ablation"
  # CPU for the latency-bound P4Transformer benchmark; MPS/CUDA for Mamba4D and PSTNet
  "$PY" -m kca.run.msr_benchmark --backbone p4transformer --ckpt "$C/p4t/last.pth" --data "$NPZ" --out results/msr/p4transformer --per-class 5 --device cpu
  for b in pstnet mamba4d; do
    "$PY" -m kca.run.msr_benchmark --backbone $b --ckpt "$C/$b/last.pth" --data "$NPZ" --out results/msr/$b --per-class 2 \
      --metrics delins --methods random,integrated_gradients,point_drop,occam,extremal,kca,kca_ctx $DARG
  done
  "$PY" -m kca.run.msr_crossover --data "$NPZ" --out results/msr/crossover --device cpu              # beta 3, seed 0
  "$PY" -m kca.run.msr_crossover --data "$NPZ" --out results/msr/crossover_seed1 --device cpu --seed 1
  "$PY" -m kca.run.msr_crossover --backbone pstnet --data "$NPZ" --out results/msr/crossover_pstnet --train-only $DARG
  EXDEV="${DEVICE:-$("$PY" -c 'import torch;print("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")')}"
  "$PY" -m kca.run.msr_crossover --backbone pstnet --data "$NPZ" --out results/msr/crossover_pstnet --eval-only --explain-device "$EXDEV"
  "$PY" -m kca.run.msr_crossover --backbone mamba4d --data "$NPZ" --out results/msr/crossover_mamba4d --train-only --device "$EXDEV"   # GPU only
  "$PY" -m kca.run.msr_crossover --backbone mamba4d --data "$NPZ" --out results/msr/crossover_mamba4d --eval-only --explain-device "$EXDEV"
  "$PY" -m kca.run.msr_ablation --ckpt "$C/p4t/last.pth" --data "$NPZ" --out results/msr/ablation --device cpu
  # laptop scale = the paper's second training scale (appendix): keep only its numbers
  LAP="$(mktemp -d)"; mkdir -p "$LAP/tables"
  "$PY" -m kca.run.paper_tables --results results/msr --paper "$LAP" --ckpt-root "$C" --data "$NPZ" \
    && cp "$LAP/tables/numbers.tex" results/msr/numbers_laptop.tex && ok "laptop-scale numbers -> results/msr/numbers_laptop.tex"
fi

# The paper's main text uses the full-scale run (colab/KCA_fullscale_colab.ipynb -> results/msr_full/)
cd "$CODE"; NPZ="${NPZ:-$WORK/data/msr}"
if [ -d results/msr_full ] && [ -d "$NPZ" ]; then
  hr "6b Paper tables and figures from the full-scale results"
  "$PY" -m kca.run.paper_tables --results results/msr_full --paper "$PAPER" --ckpt-root results/msr_full/ckpt \
    --data "$NPZ" >/dev/null && ok "tables -> paper/tables/"
  "$PY" -m kca.run.scale_table --laptop results/msr/numbers_laptop.tex --full "$PAPER/tables/numbers.tex" \
    --out "$PAPER/tables/tab_scale.tex" && ok "two-scale table -> paper/tables/tab_scale.tex"
  "$PY" -m kca.run.paper_figures --results results/msr_full --controlled results/controlled/crossover.json \
    --data "$NPZ" --fig "$PAPER/fig" && ok "figures -> paper/fig/"
fi

# ---------------------------------------------------------------------------
hr "7  Compile the paper"
# AAAI-27 style: pdfLaTeX + BibTeX (aaai2027.sty refuses tectonic/XeTeX); TinyTeX, see paper/README.md
bash "$SCRIPT_DIR/tools/check_paper.sh" "$PAPER" && ok "paper/main.pdf" || echo "paper check failed (see output above)"
hr "DONE"
