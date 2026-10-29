#!/usr/bin/env bash
# Pack what colab/KCA_fullscale_colab.ipynb needs into ~/kca_work/colab_upload/:
#   kca_bundle.tar.gz  code (src, tests, configs, controlled-study results), the paper sources
#                      (its tables/numbers.tex = the laptop-scale numbers to compare against),
#                      and the three official repos at the commits used so far
#   msr_npz.tar.gz     the MSR-Action3D point-cloud npz files behind every laptop result
# Then upload both to Google Drive: MyDrive/kca_full/upload/
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="${KCA_WORK:-$HOME/kca_work}"
OUT="$WORK/colab_upload"
STAGE="$OUT/stage"
rm -rf "$STAGE" && mkdir -p "$STAGE/code/results/msr" "$STAGE/ext"
brctl download "$ROOT/code" >/dev/null 2>&1 || true        # iCloud may have evicted files
brctl download "$ROOT/paper" >/dev/null 2>&1 || true
X=(--exclude __pycache__ --exclude .DS_Store)
rsync -a "${X[@]}" --exclude _out "$ROOT/code/src" "$ROOT/code/tests" "$ROOT/code/configs" \
      "$ROOT/code/README.md" "$ROOT/code/SETUP.md" "$STAGE/code/"
rsync -a "${X[@]}" "$ROOT/code/results/controlled" "$ROOT/code/results/RESULTS.md" "$STAGE/code/results/"
cp "$ROOT/code/results/msr/summary.json" "$STAGE/code/results/msr/"
rsync -a "${X[@]}" --exclude '*.log' --exclude '*.blg' --exclude '*.aux' "$ROOT/paper/" "$STAGE/paper/"
for r in P4Transformer Point-Spatio-Temporal-Convolution Mamba4D; do
  echo "$r $(git -C "$WORK/ext/$r" rev-parse HEAD)" >> "$STAGE/ext/COMMITS.txt"
  rsync -a "${X[@]}" --exclude .git --exclude imgs "$WORK/ext/$r" "$STAGE/ext/"
done
tar -czf "$OUT/kca_bundle.tar.gz" -C "$STAGE" code paper ext
[ "$(ls "$WORK/data/msr" | wc -l)" -ge 567 ] || { echo "expected 567 npz in $WORK/data/msr"; exit 1; }
tar -czf "$OUT/msr_npz.tar.gz" -C "$WORK/data" msr
rm -rf "$STAGE"
ls -lh "$OUT"
echo "Upload both files to Google Drive: MyDrive/kca_full/upload/"
