#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BIN_PATH="${BIN_PATH:-$REPO_ROOT/build/benchmarks/cpp/moonpoly_bench_fp16_cutlass_default}"
WARMUP="${WARMUP:-5}"
ITERS="${ITERS:-20}"

if [ "$#" -gt 0 ] && [[ "$1" == -* ]]; then
  exec python3 "$REPO_ROOT/benchmarks/tools/benchmark_cutlass_default.py" \
    --bin "$BIN_PATH" --pass-inner-loop-args --warmup "$WARMUP" --iters "$ITERS" "$@"
fi

M="${1:-4096}"
N="${2:-4096}"
K="${3:-4096}"

EXTRA_ARGS=()
if [ "$#" -gt 3 ]; then
  EXTRA_ARGS=("${@:4}")
fi

exec python3 "$REPO_ROOT/benchmarks/tools/benchmark_cutlass_default.py" \
  --bin "$BIN_PATH" \
  --m "$M" --n "$N" --k "$K" \
  --warmup "$WARMUP" --iters "$ITERS" \
  --pass-inner-loop-args \
  "${EXTRA_ARGS[@]}"
