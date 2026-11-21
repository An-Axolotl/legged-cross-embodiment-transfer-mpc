#!/usr/bin/env bash
# Reproduces every Part A number, plot and video.
#   conda env create -f environment/trajopt.yml && conda activate xfer-trajopt && bash trajopt/run_jumps.sh
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT=$ROOT/results/jump
mkdir -p "$OUT"
cd "$ROOT/trajopt"
for r in anymal talos go1 b1 hyq; do
  python jump.py $r        --out "$OUT"     # stock robot
  python jump.py $r --same --out "$OUT"     # 20 lb, 38 cm thigh, 33 cm shank, 80 N·m limit
  python render_jump.py $r        --res "$OUT" --out "$ROOT/media"   # full-mesh videos / GIFs
  python render_jump.py $r --same --res "$OUT" --out "$ROOT/media"
done
python jump.py anymal --fwddyn --out "$OUT"  # forward-dynamics transcription, for comparison
python jump.py talos  --fwddyn --out "$OUT"
python render_jump.py anymal --fwddyn --res "$OUT" --out "$ROOT/media"
python render_jump.py talos  --fwddyn --res "$OUT" --out "$ROOT/media"
