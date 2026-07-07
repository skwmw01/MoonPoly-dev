# Generated Runtime Artifacts

This directory contains checked-in runtime artifacts consumed by the Online
MoonPoly build.

Tracked files are part of the build contract:

- `fp16/rcc/fp16_rcc_fitted_cost.inc`: FP16 Pattern1 `g_predict`.
- `fp16/rcc/fp16_rcc_splitk_reduction_cost.inc`: FP16 Pattern3 `r_predict`.
- `fp16/rcc/fp16_rcc_pattern2_cost.inc`: FP16 Pattern2 tail-overhead model.
- `fp16/rcc/llm_decode_selector.inc`: measured LLM decode-shape selector.
- `fp16/rcc/rcc_generated_plans.inc`: curated direct-RCC selector entries.
- `fp32/rcc/fp32_rcc_fitted_cost.inc`: FP32 Pattern1 `g_predict`.
- `fp32/rcc/fp32_rcc_splitk_cost.inc`: FP32 Pattern3 `g_predict + r_predict`.
- `int8/rcc/i8_rcc_fitted_cost.inc`: INT8 Pattern1 `g_predict`.
- `int8/rcc/i8_rcc_splitk_cost.inc`: INT8 Pattern3 `g_predict + r_predict`.

Large or exploratory generated candidates belong in `artifacts/` unless they
are required by the runtime build.
