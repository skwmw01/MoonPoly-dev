#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BIN_PATH="${BIN_PATH:-$REPO_ROOT/build/benchmarks/cpp/moonpoly_profile_fp16_predictor}"
RAW_CSV="${RAW_CSV:-$REPO_ROOT/benchmarks/data/fp16_predictor/fp16_predictor_eval_raw.csv}"
OUTDIR="${OUTDIR:-$REPO_ROOT/artifacts/benchmarks/fp16_predictor_eval}"
WARMUP="${WARMUP:-5}"
ITERS="${ITERS:-20}"
M_LIST="${M_LIST:-16,64,256}"
N_LIST="${N_LIST:-1024,4096}"
K_LIST="${K_LIST:-1024,4096}"
MAX_SHAPES="${MAX_SHAPES:-0}"

exec python3 "$REPO_ROOT/benchmarks/tools/plot_fp16_predictor_eval.py" \
  --run-bin \
  --bin "$BIN_PATH" \
  --raw-csv "$RAW_CSV" \
  --outdir "$OUTDIR" \
  --warmup "$WARMUP" \
  --iters "$ITERS" \
  --m-list "$M_LIST" \
  --n-list "$N_LIST" \
  --k-list "$K_LIST" \
  --max-shapes "$MAX_SHAPES" \
  "$@"
