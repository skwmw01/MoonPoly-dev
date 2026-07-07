# int8/scaled_mm (SM80 Scaled INT8 GEMM with CUTLASS Sm80EVT)

该目录只保留 INT8 scaled-mm 的 C++/CUDA 实现。Python 用户入口统一在主
`moonpoly` 扩展里，不提供独立 Python 模块。

- `scaled_mm_bindings.cpp`: 主 `moonpoly` 扩展使用的 binding helper
- `scaled_mm_mikpoly_unified.cu`: SM80 `threadblock::Sm80EVT` 实现 + 配置池调度
- `scaled_mm_epilogues_mikpoly.hpp`: Epilogue Visitor Tree 定义（scale/bias fuse）
- `tests/python/test_int8_scaled_mm_correctness.py`: 正确性/性能测试

## 设计

1. Python 层只保留主入口：`moonpoly.scaled_int8_mm(a, b, a_scales, b_scales, bias, out_dtype)`。
2. Public ABI 为：`A=[M,K]`，`B=[K,N]`，`out=[M,N]`；主扩展内部完成 B packing。
3. 使用 vLLM 风格 CUTLASS C2x + Sm80EVT 模板（仅 SM80）。
4. 维护一个小型配置池（多个 threadblock/warp/stages 组合）。
5. 调度流程是 **cost-model-first**：
   - 默认使用 `select_by_analytic_cost_model(m,n,k)`。
   - 当前 Online-only 仓库保留内置 analytic selector 和 runtime 调度代码。
6. 通过函数指针表按配置 ID 调用模板实例，不使用大 `switch` 分发。

## 构建

```bash
python setup.py build_ext --inplace
```

可选指定 CUTLASS：

```bash
export CUTLASS_DIR=/path/to/cutlass
```

## Python API

INT8 scaled-mm 不进入通用 RCC GEMM selector。外部 Python 代码直接调用主
`moonpoly` 扩展：

```python
import torch
import moonpoly

A = torch.randint(-128, 127, (M, K), device="cuda", dtype=torch.int8)
B = torch.randint(-128, 127, (K, N), device="cuda", dtype=torch.int8)
A_scales = torch.ones((M, 1), device="cuda", dtype=torch.float32)
B_scales = torch.ones((1, N), device="cuda", dtype=torch.float32)

out = moonpoly.scaled_int8_mm(A, B, A_scales, B_scales, None, torch.float16)
kernel_id = moonpoly.predict_int8_scaled_mm_kernel(M, N, K)
tile_m, tile_n, tile_k, alignment = moonpoly.get_int8_scaled_mm_kernel_config(kernel_id)
```

ABI:

```text
A:        [M, K] int8 row-major
B:        [K, N] int8 row-major
A_scales: scalar or [M, 1] fp32
B_scales: scalar or [1, N] fp32
bias:     None, [N], or [1, N]
out:      [M, N] fp16/bf16
```

## 对大模型的最小上下文

建议先读这 3 个文件：

1. `scaled_mm_bindings.cpp`
2. `scaled_mm_mikpoly_unified.cu`
3. `scaled_mm_epilogues_mikpoly.hpp`
