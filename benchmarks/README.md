# benchmarks

This directory is reserved for reproducible benchmark entry points and curated
benchmark input data. Generated figures, one-off profiling outputs, and ad-hoc
kernel-search scripts should not live here.

## Layout

1. `tools/`: runnable benchmark/profile/analysis entry points
- `benchmark_cutlass_default.py`: default CUTLASS GEMM comparison runner
- `benchmark_ttft_tpot_vllm.py`: vLLM TTFT/TPOT end-to-end benchmark driver
- `benchmark_ttft_length_sweep_vllm.py`: TTFT prompt-length sweep
- `benchmark_irregular_ttft_vllm.py`: irregular-length TTFT sweep
- `profile_ttft_breakdown_vllm.py`: vLLM CUDA-kernel breakdown profiler
- `run_fp16_predictor_eval.sh`: one-command FP16 predictor evaluation
- `plot_fp16_predictor_eval.py`: predictor accuracy plots and summary
- `process_llm_gemm.py`, `export_llm_gemm_shapes.py`,
  `merge_llm_gemm_families.py`: LLM GEMM shape extraction/aggregation
- `run_pattern3_ncu_case.sh`, `profile_pattern3_splitk_ablation.py`,
  `parse_pattern3_ncu_csv.py`: Pattern-3 Split-K profiling workflow
- `paper_reproduce_core.py`: non-vLLM Online reproduction entry point for
  operator-level GEMM, FP16 predictor accuracy, and Pattern-3 timing

2. `data/`: curated benchmark input/intermediate data
- `data/core/`: generic GEMM CSV data
- `data/fp16_predictor/`: FP16 predictor raw CSV
- `data/llm_gemm/`: Llama/Qwen GEMM shape summaries
- `data/legacy/`: historical CSV kept only for traceability

3. root CUDA sources
- `gemm_baselines.cu`: low-level cuBLAS/CUTLASS FP16 RCC benchmark source

Example:

```bash
nvcc -std=c++17 benchmarks/gemm_baselines.cu \
  -I3rdparty/cutlass/include -lcublas -o /tmp/gemm_baselines

/tmp/gemm_baselines \
  --backend both \
  --shapes benchmarks/data/core/gemm.csv \
  --warmup 5 --iters 20
```

The old hardcoded shape arrays from `cublas_gemm.cu` and `cutlass_gemm.cu`
are stored as `data/core/gemm.csv`. LLM-specific subsets can be passed
explicitly, for example `data/llm_gemm/llm_gemm_bench_select.csv`.

4. `case_studies/`: focused micro-kernel composition studies
- `case_studies/pattern2_twin_gemm/`: Pattern-2 TwinGemm case study.
  It splits the GEMM output `N` dimension into two regions and dispatches the
  two regions to different CUTLASS micro-kernels in one fused launch.

Build and run:

```bash
cmake --build build --target moonpoly_case_pattern2_twin_gemm

./build/benchmarks/case_studies/pattern2_twin_gemm/moonpoly_case_pattern2_twin_gemm \
  1024 4096 4096 24
```

5. documentation
- `VLLM_E2E_TEST_GUIDE.md`: vLLM end-to-end testing guide
- `tools/MOONPOLY_DECODE_PROFILE.md`: decode linear-profile guide

## Outputs And Archives

Benchmark outputs should go under `artifacts/benchmarks/` by default.

For the paper core path without vLLM integration, use:

```bash
python benchmarks/tools/paper_reproduce_core.py \
  --outdir artifacts/paper_core \
  --stages predictor,operator,pattern3 \
  --max-shapes 3
```

If the shell Python is not the same Python used to build the `moonpoly`
extension, pass the interpreter explicitly:

```bash
python benchmarks/tools/paper_reproduce_core.py \
  --python /path/to/python \
  --outdir artifacts/paper_core \
  --stages predictor,operator,pattern3 \
  --max-shapes 3
```

Use `--max-shapes 0` only for a full operator sweep. Use
`--run-predictor-bin` when regenerating the FP16 predictor raw CSV instead of
reading the curated data under `benchmarks/data/fp16_predictor/`.

Ad-hoc scripts and old generated outputs have been moved out of this directory
to `experiments/benchmark_archive/`, which is ignored by git:
- `ad_hoc_tools/`: old tile-search/template-generation scripts
- `generated_outputs/`: regenerated figures and temporary predictor outputs
- `paper_revision_outputs/`: paper revision tables/figures/traces
- `legacy_gpu_runs/`: old one-off GPU runs such as 4090 experiments
