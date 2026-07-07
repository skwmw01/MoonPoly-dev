#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.10/site-packages/torch/lib:${LD_LIBRARY_PATH:-}"
fi

M="${M:-8}"
N="${N:-10240}"
K="${K:-5120}"
MONO_KIND="${MONO_KIND:-generic}"
MONO_PID="${MONO_PID:-96}"
MONO_SPLIT_K="${MONO_SPLIT_K:-1}"
SPLITK_KIND="${SPLITK_KIND:-forced}"
SPLITK_PID="${SPLITK_PID:-286}"
SPLIT_K="${SPLIT_K:-4}"
CUDA_DEV="${CUDA_VISIBLE_DEVICES:-1}"
OUT_DIR="${OUT_DIR:-artifacts/ncu_pattern3_qwen_m8n10240k5120_pid286_sk4}"

METRICS="gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,smsp__warps_active.avg.per_cycle_active,sm__cycles_elapsed.avg"

mkdir -p "$OUT_DIR"

echo "[info] ROOT_DIR=$ROOT_DIR"
echo "[info] shape=($M,$N,$K) mono=${MONO_KIND}:${MONO_PID}/${MONO_SPLIT_K} splitk=${SPLITK_KIND}:${SPLITK_PID}/${SPLIT_K}"
echo "[info] output dir=$OUT_DIR"

echo "[step] timing summary"
CUDA_VISIBLE_DEVICES="$CUDA_DEV" python benchmarks/tools/profile_pattern3_splitk_ablation.py \
  --m "$M" --n "$N" --k "$K" \
  --mono-kind "$MONO_KIND" --mono-pid "$MONO_PID" --mono-split-k "$MONO_SPLIT_K" \
  --splitk-kind "$SPLITK_KIND" --splitk-pid "$SPLITK_PID" --splitk "$SPLIT_K" \
  --warmup 100 --iters 1000 \
  --out-json "$OUT_DIR/timing_summary.json"

for backend in cublas mono splitk; do
  echo "[step] ncu backend=$backend"
  CUDA_VISIBLE_DEVICES="$CUDA_DEV" \
  ncu --target-processes all --csv --page raw \
    --metrics "$METRICS" \
    python benchmarks/tools/profile_pattern3_splitk_ablation.py \
      --m "$M" --n "$N" --k "$K" \
      --mono-kind "$MONO_KIND" --mono-pid "$MONO_PID" --mono-split-k "$MONO_SPLIT_K" \
      --splitk-kind "$SPLITK_KIND" --splitk-pid "$SPLITK_PID" --splitk "$SPLIT_K" \
      --backend "$backend" --quiet-metadata \
    > "$OUT_DIR/${backend}.csv"

  python benchmarks/tools/parse_pattern3_ncu_csv.py \
    "$OUT_DIR/${backend}.csv" \
    --out-json "$OUT_DIR/${backend}_summary.json"
done

cat <<EOF
[done]
Timing summary:
  $OUT_DIR/timing_summary.json
NCU raw CSV:
  $OUT_DIR/cublas.csv
  $OUT_DIR/mono.csv
  $OUT_DIR/splitk.csv
NCU parsed summaries:
  $OUT_DIR/cublas_summary.json
  $OUT_DIR/mono_summary.json
  $OUT_DIR/splitk_summary.json
EOF
