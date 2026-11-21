#!/usr/bin/env bash
# Reproduces every Part B number and video (see analysis/run_experiments.py), then the analysis.
#   source deps/env.sh && bash mpc_transfer/run_transfers.sh
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
python "$ROOT/analysis/run_experiments.py" "$@"
python "$ROOT/analysis/analyze.py"
