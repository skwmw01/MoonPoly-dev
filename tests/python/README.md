# tests/python

Python 测试入口说明。

## 1) INT8 Scaled MM 正确性/性能测试（新）

测试脚本：

- `tests/python/test_int8_scaled_mm_correctness.py`

### 前置步骤

先编译主 `moonpoly` 扩展（INT8 scaled-mm 已并入主入口）：

```bash
cd /home/zhangyangyu/moonpoly_workspace/moonpoly
python setup.py build_ext --inplace
```

回到仓库根目录执行测试：

```bash
cd /home/zhangyangyu/moonpoly_workspace/moonpoly
python tests/python/test_int8_scaled_mm_correctness.py
```

### 常用参数

快速子集测试：

```bash
python tests/python/test_int8_scaled_mm_correctness.py --quick
```

详细日志：

```bash
python tests/python/test_int8_scaled_mm_correctness.py --verbose
```

只测 predictor 信息：

```bash
python tests/python/test_int8_scaled_mm_correctness.py --predictor
```

跑全部（正确性 + predictor + stress）：

```bash
python tests/python/test_int8_scaled_mm_correctness.py --all
```

## 2) 其他已有测试

- `tests/python/test_moonpoly_linear.py`
- `tests/python/test_fp16_streamk_graph.py`（新：workspace + CUDA Graph）
- `tests/python/test_fp16_splitk_runtime.py`（新：FP16 runtime split-K）

### FP16 Runtime Split-K 测试

先确保 `moonpoly` 扩展已重新编译：

```bash
cd /home/zhangyangyu/moonpoly_workspace/moonpoly
python setup.py build_ext --inplace
```

运行默认 smoke test：

```bash
CUDA_VISIBLE_DEVICES=1 python tests/python/test_fp16_splitk_runtime.py \
  --m 8 --n 1024 --k 1024 --pid 400 --split-ks 1,2,4
```

论文默认 FP16 elite kernel 池固定为 45 个：

```text
0-39,400,407,413,425,439
```

其中 `0-39` 是原始 primary FP16 RCC kernel，`400/407/413/425/439` 是从 legacy split-K family 中挑出的 5 个代表性 split-K kernel。`400-439` 仍保留为 search/debug-only，默认实验脚本不会全部遍历。

### FP16 StreamK + CUDA Graph 测试（新）

先确保 `moonpoly` 扩展已重新编译安装（包含新增 `moonpoly_ops.fp16_workspace_size` 和 `moonpoly_ops.linear_fp16_pid_ws_out`）：

```bash
cd /home/zhangyangyu/moonpoly_workspace/moonpoly
python setup.py build_ext --inplace
```

运行测试：

```bash
python tests/python/test_fp16_streamk_graph.py --m 128 --n 4096 --k 4096 --pid 40 --split-k 2
```

可调参数（默认 warmup=5, iters=20）：

```bash
python tests/python/test_fp16_streamk_graph.py --warmup 5 --iters 20
```

建议在同一 Python 环境下运行，并确保 CUDA 可用：

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## 3) E2E 图表生成（TTFT/TPOT + Breakdown）

基准与 breakdown 数据准备完成后，可直接画图并输出摘要：

```bash
python tests/python/plot_e2e_speedup.py \
  --e2e-json artifacts/e2e_ttft_tpot.json \
  --breakdown-csv artifacts/nsys_breakdown.csv \
  --out-dir artifacts/e2e_figures
```

输出内容：

- `artifacts/e2e_figures/ttft_vs_len.png`
- `artifacts/e2e_figures/tpot_vs_bs.png`
- `artifacts/e2e_figures/gpu_breakdown.png`（若提供 breakdown CSV）
- `artifacts/e2e_figures/summary.md`
