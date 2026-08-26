# MoonPoly Docker 使用与核心实验运行指南

本文档记录当前 `Moonpoly` 容器的环境、MoonPoly 编译方法、正确性测试和小规模核心 benchmark 的运行方式。

## 1. 当前环境

| 项目 | 当前配置 |
| --- | --- |
| 容器名 | `Moonpoly` |
| Docker 镜像 | `pytorch/pytorch:2.8.0-cuda12.6-cudnn9-devel` |
| 容器系统 | Ubuntu 22.04.5 LTS |
| CUDA | 12.6 |
| PyTorch | 2.8.0+cu126 |
| GPU | NVIDIA A100 80GB PCIe, SM80 |
| 主机源码目录 | `/data/mingwei/project/MoonPoly-dev` |
| 容器源码目录 | `/workspace/MoonPoly-dev` |
| CUTLASS commit | `76c96b0be35cb263debe3e3d8418b80911a544ab` |

进入现有容器：

```bash
docker start Moonpoly
docker exec -it Moonpoly bash
```

容器内默认进入项目目录。如果当前目录不正确，执行：

```bash
cd /workspace/MoonPoly-dev
```

## 2. 重新创建容器

当前 PyTorch 镜像内部是 Ubuntu 22.04，同时预装了 CUDA、cuDNN 和 PyTorch，适合直接编译 CUDA 扩展。

```bash
docker run -dit \
  --name Moonpoly \
  --gpus all \
  --ipc=host \
  --shm-size=16g \
  -v /data/mingwei/project/MoonPoly-dev:/workspace/MoonPoly-dev \
  -w /workspace/MoonPoly-dev \
  pytorch/pytorch:2.8.0-cuda12.6-cudnn9-devel \
  bash
```

安装基础工具和 Python 绘图依赖：

```bash
apt-get update
apt-get install -y git curl ca-certificates cmake ninja-build
python -m pip install matplotlib
```

## 3. 准备 CUTLASS

项目要求 CUTLASS 位于 `3rdparty/cutlass`。当前工作区已经准备完成；以下命令仅用于全新工作区。

```bash
mkdir -p 3rdparty
git clone https://gitee.com/mirrors_NVIDIA/cutlass.git 3rdparty/cutlass
cd 3rdparty/cutlass
git checkout 76c96b0be35cb263debe3e3d8418b80911a544ab
git apply ../../integrations/cutlass/cutlass_4_1_pattern2_twin_gemm.patch
cd /workspace/MoonPoly-dev
```

验证 CUTLASS commit：

```bash
git -C 3rdparty/cutlass rev-parse HEAD
```

不要对已经应用补丁的 CUTLASS 再次执行 `git apply`。

## 4. 编译

配置并编译 C++/CUDA targets：

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build -j 4
```

编译 PyTorch 扩展：

```bash
MAX_JOBS=4 python setup.py build_ext --inplace
```

成功后，项目根目录会生成类似下面的扩展文件：

```text
moonpoly.cpython-311-x86_64-linux-gnu.so
```

使用 Python 扩展前设置：

```bash
export PYTHONPATH=/workspace/MoonPoly-dev
```

直接在 Python 中使用时，应先导入 `torch`，再导入 `moonpoly`：

```python
import torch
import moonpoly
```

## 5. 正确性测试

运行 CTest 注册的四组核心测试：

```bash
CUDA_VISIBLE_DEVICES=0 ctest --test-dir build --output-on-failure
```

当前预期结果为 4/4 通过：

```text
moonpoly.fp16_rcc
moonpoly.fp16_pattern2
moonpoly.fp32_rcc
moonpoly.int8_rcc
```

运行 FP16 runtime split-K：

```bash
export PYTHONPATH=/workspace/MoonPoly-dev
CUDA_VISIBLE_DEVICES=0 python tests/python/test_fp16_splitk_runtime.py \
  --m 8 --n 1024 --k 1024 --pid 400 --split-ks 1,2,4
```

当前 `split_k=1,2,4` 均应为 `finite=True`、`allclose=True`。

运行 INT8 scaled-MM quick 测试：

```bash
CUDA_VISIBLE_DEVICES=0 python tests/python/test_int8_scaled_mm_correctness.py --quick
```

当前基线为 20 项中 16 项通过、4 项超过脚本的最大绝对/相对误差阈值。该脚本即使存在失败项也可能返回退出码 0，必须检查输出中的 `Passed` 和 `Failed` 数量。

P40/P66 StreamK 诊断命令：

```bash
CUDA_VISIBLE_DEVICES=0 python tests/python/test_fp16_p40_p66_pybind.py \
  --warmup 1 --iters 2 --mode both
```

当前 P40/P66 会产生 NaN，属于已知未解决问题；P42/P43/P44 正常。不要将这项测试记为通过。

## 6. 小规模核心 Benchmark

下面的命令运行三部分核心实验：

- predictor：分析仓库内置的 predictor 原始 CSV；
- operator：实测前三个 LLM GEMM shape；
- Pattern-3：实测 monolithic kernel 与 split-K。

```bash
export PYTHONPATH=/workspace/MoonPoly-dev
CUDA_VISIBLE_DEVICES=0 python benchmarks/tools/paper_reproduce_core.py \
  --python /opt/conda/bin/python \
  --outdir artifacts/paper_core_smoke \
  --stages predictor,operator,pattern3 \
  --max-shapes 3 \
  --pattern3-warmup 20 \
  --pattern3-iters 100 \
  --keep-going
```

主要输出：

```text
artifacts/paper_core_smoke/manifest.json
artifacts/paper_core_smoke/predictor/summary.json
artifacts/paper_core_smoke/operator/operator_raw.csv
artifacts/paper_core_smoke/operator/operator_summary.json
artifacts/paper_core_smoke/pattern3/timing_summary.json
```

`artifacts/` 默认被 Git 忽略。使用 `--max-shapes 0` 可以运行完整 operator shape sweep，但耗时会明显增加。

## 7. 论文 Table 1 算子级全量评测

论文 Table 1 使用仓库内的 1433 个 GEMM shape：1267 个 Real-World shape 和 166 个 DeepBench shape。当前可复现驱动会逐 shape 运行：

- `moonpoly_benchmark`：FP16、FP32、INT8 对 cuBLAS 的正确性和耗时；
- `moonpoly_bench_fp16_cutlass_default`：FP16 对默认 CUTLASS 的正确性和耗时。

在容器中运行（示例使用物理 GPU 1）：

```bash
cd /workspace/MoonPoly-dev
CUDA_VISIBLE_DEVICES=1 python benchmarks/tools/reproduce_operator_level.py \
  --outdir artifacts/operator_level/table1_full_shared_gpu \
  --run-note "Shared GPU run; external workloads observed"
```

驱动按 shape 写入 `runs/cublas/` 和 `runs/cutlass-fp16/` 缓存。中断后重新执行同一命令会复用已有 `result.json`；只有需要重测时才使用 `--force`。单 shape 默认超时为 900 秒，可用 `--timeout` 调整。脚本在存在任何 `incorrect`、`missing` 或 `process_error` 时返回退出码 2，但仍会保留并汇总全部已完成结果，不能因为非零退出码删除输出。

主要输出：

```text
artifacts/operator_level/table1_full_shared_gpu/manifest.json
artifacts/operator_level/table1_full_shared_gpu/operator_raw.csv
artifacts/operator_level/table1_full_shared_gpu/operator_failures.csv
artifacts/operator_level/table1_full_shared_gpu/operator_summary.json
artifacts/operator_level/table1_full_shared_gpu/operator_vs_cublas.png
artifacts/operator_level/table1_full_shared_gpu/fp16_vs_cutlass_default.png
```

`operator_summary.json` 中的 `speedup_avg` 是通过正确性检查样本的算术平均，`speedup_geomean` 是几何平均；失败或缺失样本不会进入性能平均。论文 Figure 7 没有在正文公开失败样本过滤、平均加权和 selector 版本的完整细节，因此不能仅凭平均值判断是否同一统计口径。当前仓库只提供 FP16 默认 CUTLASS 对照入口，没有完整的 FP32/INT8 默认 CUTLASS 入口；论文正文写的是 CUTLASS 4.2.0，而仓库固定的是 CUTLASS 4.1.0，因此不能据此声称已经复现论文中三种 dtype 的 CUTLASS 曲线。

## 8. 常见问题

### `import moonpoly` 找不到 `libc10.so`

先执行 `import torch`，让 PyTorch 动态库先加载，再导入 `moonpoly`。

### predictor 阶段提示缺少 `matplotlib`

```bash
python -m pip install matplotlib
```

### GitHub 无法访问

当前环境通过 Gitee 镜像获取 CUTLASS。必须核对 commit，不能直接使用镜像仓库的最新版本。

### 测试脚本退出码为 0，但输出显示失败

`test_int8_scaled_mm_correctness.py` 和部分 StreamK smoke 脚本没有完整地把数值失败传递为非零退出码。复现记录应以 `allclose`、`Passed/Failed` 和 NaN 检查为准。
