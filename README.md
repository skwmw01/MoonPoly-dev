# MoonPoly

MoonPoly is a two-stage micro-kernel polymerization framework for dynamic-shape
GEMM. This repository includes FP16/FP32/INT8 row-column-column GEMM kernels,
checked-in runtime selectors, integration patches, and Online benchmark scripts.

## Layout

- `moonpoly/core/`: CUDA/CUTLASS kernels and runtime dispatch.
- `moonpoly/generated/`: generated runtime assets consumed by `moonpoly/core/`.
- `benchmarks/tools/`: profiling, plotting, and reproduction scripts.
- `benchmarks/data/`: curated predictor and LLM GEMM shape data.
- `tests/cpp/`, `tests/python/`: correctness and smoke tests.
- `integrations/vllm/v0.10.0/`: vLLM Linear patch.
- `artifacts/`: ignored output directory for local runs.

## Build

Requirements: CUDA 12.x, CUTLASS under `3rdparty/cutlass`, PyTorch in the active
Python environment, and an SM80-class GPU for the paper path.

Vendored CUTLASS base commit:
`76c96b0be35cb263debe3e3d8418b80911a544ab` (`76c96b0b`, CUTLASS upstream).
MoonPoly-specific CUTLASS changes are recorded under `integrations/cutlass/`.

```bash
cmake -S . -B /tmp/moonpoly-build -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/moonpoly-build -j
python setup.py build_ext --inplace
```

```bash
# If pip build isolation cannot see torch.
pip install -e . --no-build-isolation
```

```bash
# If CUTLASS is not under 3rdparty/cutlass.
export CUTLASS_DIR=/path/to/cutlass
cmake -S . -B /tmp/moonpoly-build -DCMAKE_BUILD_TYPE=Release
```

## Core Reproduction

```bash
CUDA_VISIBLE_DEVICES=0 python benchmarks/tools/paper_reproduce_core.py \
  --python "$(which python)" \
  --outdir artifacts/paper_core \
  --stages predictor,operator,pattern3 \
  --max-shapes 3
```

Use `--max-shapes 0` for a full operator sweep.

## Runtime Selection

Default selectors use exported per-pid `k_iter` fitted predictors for
FP16/FP32/INT8. Analytic costs remain as a fallback and can be forced with:

```bash
MOONPOLY_DISABLE_FITTED_COST=1 /tmp/moonpoly-build/benchmarks/cpp/moonpoly_profile_fp32_rcc 4096 4096 4096
MOONPOLY_DISABLE_FITTED_COST=1 /tmp/moonpoly-build/benchmarks/cpp/moonpoly_profile_int8_rcc 4096 4096 4096
```

FP16 paper elite set: `0-39,400,407,413,425,439`. Pids `400+` are split-K.

## Smoke Tests

```bash
cmake --build /tmp/moonpoly-build --target \
  test_fp16_row_col_col test_fp16_pattern2_twin_gemm \
  test_fp32_row_col_col test_int8_row_col_col -j

CUDA_VISIBLE_DEVICES=0 ctest --test-dir /tmp/moonpoly-build --output-on-failure
```

The registered CTest targets are:

```text
moonpoly.fp16_rcc
moonpoly.fp16_pattern2
moonpoly.fp32_rcc
moonpoly.int8_rcc
```

Additional Python smoke tests:

```bash
CUDA_VISIBLE_DEVICES=0 python tests/python/test_fp16_splitk_runtime.py \
  --m 8 --n 1024 --k 1024 --pid 400 --split-ks 1,2,4

CUDA_VISIBLE_DEVICES=0 python tests/python/test_int8_scaled_mm_correctness.py --quick

bash benchmarks/tools/run_fp16_predictor_eval.sh
bash benchmarks/tools/run_pattern3_ncu_case.sh
```

## vLLM Patch

Apply from a vLLM v0.10.0 checkout:

```bash
MOONPOLY_ROOT=/path/to/moonpoly
git apply "$MOONPOLY_ROOT/integrations/vllm/v0.10.0/moonpoly_linear.patch"
```

Run vLLM with MoonPoly Linear:

```bash
CUDA_VISIBLE_DEVICES=0 \
VLLM_USE_MOONPOLY_LINEAR=1 \
VLLM_USE_MOONPOLY_CPP_SELECTOR=1 \
python -m vllm.entrypoints.openai.api_server \
  --model /path/to/model \
  --dtype float16 \
  --host 127.0.0.1 --port 8009 \
  --enforce-eager
```

TTFT/TPOT script: `benchmarks/tools/benchmark_ttft_tpot_vllm.py`.

## Citation

If you use MoonPoly, cite:

```bibtex
@article{zhang2026moonpoly,
  title = {MoonPoly: Bridging Code Generation and Adaptive Execution via Micro-Kernel Polymerization for Optimizing Dynamic-Shape Tensor Operators},
  author = {Zhang, Yangyu and Li, Guangli and Yu, Feng and Luo, Fan and Sun, Qianqi and Wang, Xueying and Zhao, Jiacheng and Cui, Huimin and Feng, Xiaobing and Xue, Jingling},
  journal = {ACM Transactions on Architecture and Code Optimization},
  year = {2026},
  publisher = {ACM},
  doi = {10.1145/3818618}
}

@inproceedings{yu2024optimizing,
  title = {Optimizing Dynamic-Shape Neural Networks on Accelerators via On-the-Fly Micro-Kernel Polymerization},
  author = {Yu, Feng and Li, Guangli and Zhao, Jiacheng and Cui, Huimin and Feng, Xiaobing and Xue, Jingling},
  booktitle = {Proceedings of the 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2},
  series = {ASPLOS '24},
  volume = {2},
  pages = {797--812},
  year = {2024},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  doi = {10.1145/3620665.3640390}
}
```

## License

Non-commercial research and academic use only. See `LICENSE`.
