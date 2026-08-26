# MoonPoly 论文实验复现进度

更新时间：2026-08-26

## 1. 当前结论

MoonPoly 的 SM80 编译链、C++ 核心正确性测试、小规模核心 benchmark 和 Table 1 算子级全量协议已经在 NVIDIA A100 80GB 上执行。当前结果足以验证仓库可以构建并执行主要 operator 与 Pattern-3 路径，但尚不等于完整复现论文全部结果。

目前有两个需要明确保留的数值问题：

1. Table 1 全量 INT8 对照只有 1038/1433 shape 通过，300 个 shape 没有生成 INT8 结果，95 个结果未通过 cosine 阈值。
2. FP16 P40/P66 StreamK 路径持续产生 NaN，P42/P43/P44 正常；Table 1 另有 3 个 FP16 长窄 shape 出现 NaN。

因此当前状态应描述为“核心路径已跑通，部分扩展 kernel 正确性待修复”，不能描述为“全部正确性测试通过”。

## 2. 复现环境

| 项目 | 配置 |
| --- | --- |
| 容器 | `Moonpoly` |
| 系统 | Ubuntu 22.04.5 LTS |
| Docker 镜像 | `pytorch/pytorch:2.8.0-cuda12.6-cudnn9-devel` |
| GPU | NVIDIA A100 80GB PCIe |
| CUDA | 12.6 |
| PyTorch | 2.8.0+cu126 |
| CUTLASS | `76c96b0be35cb263debe3e3d8418b80911a544ab` + MoonPoly patch |
| CUDA 架构 | SM80 (`CMAKE_CUDA_ARCHITECTURES=80`) |

论文后续 vLLM 端到端实验可能要求更贴近原始环境的 PyTorch/vLLM 版本组合。当前 PyTorch 2.8 环境主要用于 operator 层编译、测试和 smoke benchmark。

## 3. 进度总览

| 工作项 | 状态 | 结果或说明 |
| --- | --- | --- |
| Ubuntu 22.04 Docker 环境 | 已完成 | 容器 `Moonpoly` 正在运行 |
| GPU/CUDA/PyTorch 可用性 | 已完成 | A100、CUDA 12.6、PyTorch CUDA 均可用 |
| CUTLASS 固定版本与补丁 | 已完成 | commit 与补丁已核对 |
| CMake/Ninja 编译 | 已完成 | 51/51 targets 成功 |
| PyTorch 扩展编译 | 已完成 | `.so` 已生成并可加载 |
| C++ 核心正确性 | 已完成 | CTest 4/4 通过 |
| FP16 runtime split-K | 已完成 | split-K 1/2/4 全部通过 |
| Python INT8 scaled-MM | 部分完成 | quick 16/20 通过 |
| FP16 StreamK P40/P66 | 未通过 | 输出包含 NaN，已完成初步定位 |
| Predictor 指标整理 | 已完成 | 使用仓库内置原始 CSV，共 12 shapes |
| Operator smoke benchmark | 已完成 | 3 shapes、3 dtype，共 9 项正确 |
| Pattern-3 split-K benchmark | 已完成 | split-K 优于 mono 与 cuBLAS |
| 完整 operator shape sweep | 已完成一轮 | 1433 shapes；cuBLAS 三 dtype + 默认 CUTLASS FP16，结果见 `artifacts/operator_level/table1_full_shared_gpu` |
| Predictor 本机重新采样 | 未开始 | 当前指标不是本机重新 profile 的数据 |
| vLLM 端到端 TTFT/TPOT | 未开始 | 需要 vLLM 0.10、模型权重和服务环境 |
| NCU/NSYS 细粒度分析 | 未开始 | 需要单独的 profiling 轮次 |
| 论文全部图表逐项对齐 | 未开始 | 需建立论文图表与脚本/数据映射 |

## 4. 编译结果

CMake 使用 Release 和 SM80 配置：

```text
cmake -S . -B build -G Ninja
-DCMAKE_BUILD_TYPE=Release
-DCMAKE_CUDA_ARCHITECTURES=80
```

结果：51/51 targets 编译成功。PyTorch 扩展也成功生成：

```text
moonpoly.cpython-311-x86_64-linux-gnu.so
```

`setup.py` 构建期间存在一条未找到 `-L/usr/local/lib -lcuda` 对应 library file 的非致命 warning，但最终扩展链接和加载成功。

## 5. 正确性结果

### 5.1 C++ CTest

2026-08-26 最后一次复核结果：

| 测试 | 结果 |
| --- | --- |
| `moonpoly.fp16_rcc` | PASS |
| `moonpoly.fp16_pattern2` | PASS |
| `moonpoly.fp32_rcc` | PASS |
| `moonpoly.int8_rcc` | PASS |

总计 4/4 通过，实际测试时间约 2.65 秒。

### 5.2 FP16 runtime split-K

测试 shape 为线性空间 `(M=8, N=1024, K=1024)`，PID 400：

| split-K | finite | allclose | max abs error |
| ---: | --- | --- | ---: |
| 1 | True | True | 0.000488 |
| 2 | True | True | 0.000122 |
| 4 | True | True | 0.000488 |

### 5.3 INT8 scaled-MM quick

结果：16/20 通过，4/20 未通过脚本阈值。

失败集中在两个较大 shape：

| Shape `(M,N,K)` | 失败配置数 | max abs error | max relative error |
| --- | ---: | --- | --- |
| `(512,1024,2048)` | 2 | 0.5、1.0 | 0.08765、0.08331 |
| `(1024,2048,4096)` | 2 | 4.0、4.0 | 1.0、0.25684 |

失败配置均来自 quick 抽样中的 FP16 输出。后续需要区分以下两种情况：

- kernel 的缩放/舍入语义与 PyTorch reference 不一致；
- 测试使用全局最大相对误差，对接近零的 reference 值过于敏感。

在完成逐元素误差分布和 vLLM/CUTLASS 对照前，不应直接放宽阈值。

### 5.4 FP16 P40/P66 StreamK

以下对照均已执行：

- eager 与 CUDA Graph；
- split-K 1 与 2；
- linear wrapper 与 RCC 入口；
- caller workspace、singleton workspace 和原生 `cudaMalloc` raw-copy 输入。

P40/P66 在这些路径中均出现 NaN；P42/P43/P44 对相同类型输入能够通过。现有证据说明问题不只是 CUDA Graph 捕获、PyTorch allocator 或外部 workspace 生命周期。

该问题不会阻塞本次 PID 25/286 的 Pattern-3 smoke benchmark，但会阻塞“全部 StreamK 路径正确”的结论。

## 6. 核心 Benchmark 结果

统一输出目录：

```text
artifacts/paper_core_smoke
```

### 6.1 Predictor

本阶段分析的是仓库已有的 `fp16_predictor_eval_raw.csv`，不是当前 A100 上重新采集的数据。

| 指标 | 结果 |
| --- | ---: |
| shapes | 12 |
| LoP mean | 1.5132% |
| LoP median | 0.0% |
| LoP P90 | 4.4101% |
| LoP max | 9.7264% |
| Top-1 hit rate | 75.0% |
| Top-3 hit rate | 91.67% |
| Top-5 hit rate | 100.0% |
| Spearman rho mean | 0.8790 |
| Kendall tau mean | 0.7309 |

### 6.2 Operator smoke benchmark

测试 shape：

```text
(4096,1,4096)
(4096,19,4096)
(4096,43,4096)
```

FP16、FP32、INT8 共 9 项均通过 benchmark 内置正确性检查，cosine similarity 均为 1.0。

| 类型 | 正确性 | speedup 几何平均 | 最小 | 最大 |
| --- | --- | ---: | ---: | ---: |
| FP16 | 3/3 | 0.8326x | 0.6675x | 0.9320x |
| FP32 | 3/3 | 0.3544x | 0.2072x | 0.5941x |
| INT8 | 3/3 | 0.6158x | 0.4818x | 0.7789x |

这里的 speedup 定义为 `cuBLAS_time / MoonPoly_time`。三个 smoke shape 上均小于 1，表示 MoonPoly 比 cuBLAS 慢；这只是前三个小规模 shape 的结果，不能外推为完整论文 shape 集合的总体结论。

### 6.3 Pattern-3

Shape：`(8,10240,5120)`；mono 使用 PID 25、split-K 使用 PID 286 和 split-K 4。

| 后端 | 平均耗时 | TFLOPS | max abs error | 正确性 |
| --- | ---: | ---: | ---: | --- |
| cuBLAS | 0.1826 ms | 4.5949 | 0.0 | PASS |
| mono | 0.2150 ms | 3.9012 | 0.25 | PASS |
| split-K | 0.1688 ms | 4.9688 | 0.25 | PASS |

派生结果：

- split-K 相对 mono：1.2737x；
- split-K 相对 cuBLAS：1.0814x；
- mono 相对 cuBLAS：0.8490x。

本轮使用 20 次预热和 100 次计时，适合作为 smoke benchmark，不替代论文级多轮统计。

### 6.4 Table 1 算子级全量评测

2026-08-26 在 `Moonpoly` 容器中完成了论文 Table 1 的全部 1433 个 shape：1267 个 Real-World、166 个 DeepBench。每个 shape 使用仓库现有 benchmark 的 5 次预热和 20 次计时；结果按 shape 缓存，原始记录共 5732 行（cuBLAS 3 dtype + 默认 CUTLASS FP16），从 03:57:19 UTC 运行到 05:11:38 UTC，约 74 分钟。

本轮 GPU 1 不是独占卡：开始时显存已使用 61856 MiB、剩余 19301 MiB，运行期间仍观察到外部负载。因此下面的耗时和 speedup 只能作为共享 GPU 上的审计性结果，不能作为干净环境下对论文 Figure 7 的严格数值复现。

| 对照 | dtype | 正确/总 shape | Real-World | DeepBench | 通过样本 speedup 平均 | 几何平均 | 峰值 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cuBLAS | FP16 | 1430/1433 | 1266/1267 | 164/166 | 0.9628x | 0.8823x | 4.6422x |
| cuBLAS | FP32 | 1433/1433 | 1267/1267 | 166/166 | 0.6217x | 0.5076x | 3.1817x |
| cuBLAS | INT8 | 1038/1433 | 888/1267 | 150/166 | 1.1598x | 1.0882x | 15.3494x |
| 默认 CUTLASS | FP16 | 1428/1433 | 1264/1267 | 164/166 | 16.3564x | 9.7427x | 219.1448x |

其中 speedup 定义为 baseline time / MoonPoly time，性能平均只对 correctness 通过的样本计算。论文 Figure 7 的报告值为：cuBLAS FP16/INT8/FP32 平均 1.27x/1.35x/1.12x、峰值 5.47x/14.03x/5.03x；默认 CUTLASS FP16 平均 3.34x。本轮结果与论文差异明显，主要应从共享 GPU 干扰、仓库 benchmark 的固定计时协议、shape 选择器/实现版本和失败样本覆盖率进一步核对，不能直接宣称图表已复现。

#### FP16 的 1.27x 差异核对

这两个数字确实差别很大，但目前不能认为它们是同一统计口径下的直接矛盾：

1. 论文 Figure 7 只给出平均值，没有在正文说明是否排除了失败/边界 shape、是否使用算术平均或其他加权方式。当前驱动使用 `gemm_trick.csv` 的全部 1433 个 shape，并对通过 correctness 的 1430 个样本做逐 shape 算术平均。
2. 当前 `moonpoly_benchmark` 的 FP16 路径是 `run_fp16_gemm -> online selector`，包含当前源码默认启用的 Pattern-3。该 selector 在极小输出 shape 上存在可观测的代价模型失配：例如 `1x1x32768` 预测 `Pattern3 pid=25 split_k=8` 成本 `0.0055 ms`，但实际在线调用约 `0.89 ms`；同一 shape 的 cuBLAS 约 `0.010 ms`。逐 PID profile 中，直接 `pid=25` 也约 `0.080 ms`，说明这不是简单的平均公式问题。
3. 全量 FP16 结果中只有 528/1430 个通过样本快于 cuBLAS，算术平均为 `0.9628x`；如果人为只保留 `speedup >= 1.05` 的 372 个样本，平均恰好约 `1.27x`，但这种筛选不能作为论文复现口径。
4. 论文实验环境明确写的是 CUTLASS 4.2.0；当前仓库固定的 vendored CUTLASS 是 4.1.0（commit `76c96b0b` 加 MoonPoly patch）。CUDA 12.6 和 A100 型号基本匹配，但 CUTLASS、源码 commit、selector 资产和 GPU 是否独占仍未完全对齐。

因此当前结论应是：`0.9628x` 是“当前仓库版本、当前默认 online selector、共享 GPU、全量 shape、通过样本算术平均”的实测值；它不能直接证明论文的 `1.27x` 错误，也不能被当作论文数值已经复现。要解释并缩小差异，需要先拿到论文对应源码/commit 和 CUTLASS 4.2.0，修复或复现论文的 selector 口径，再在独占 A100 上重跑。

正确性审计：

- FP32 cuBLAS：1433/1433 通过；
- FP16 cuBLAS：3 个 shape 的 cosine similarity 为 NaN（`qid=1134,1304,1305`），其余 1430 个通过；
- INT8 cuBLAS：95 个已生成结果但 cosine similarity 未过阈值，300 个 shape 的 INT8 行缺失，C++ benchmark 仅打印 `std::exception` 且进程仍返回 0；
- 默认 CUTLASS FP16：1428 个通过，5 个进程错误（`qid=343,391,1134,1304,1305`），其余均通过内置 cosine-difference 校验。

因此当前算子级结论是“全量协议已执行并生成可审计结果，FP32/大部分 FP16 路径正确；INT8 小/边界 shape 和少数 FP16 长窄 shape 仍有明确失败”，不是“所有算子均正确”。

## 7. 结果文件

| 内容 | 路径 |
| --- | --- |
| 统一 manifest | `artifacts/paper_core_smoke/manifest.json` |
| Predictor summary | `artifacts/paper_core_smoke/predictor/summary.json` |
| Predictor shape detail | `artifacts/paper_core_smoke/predictor/shape_detail.csv` |
| Operator raw data | `artifacts/paper_core_smoke/operator/operator_raw.csv` |
| Operator summary | `artifacts/paper_core_smoke/operator/operator_summary.json` |
| Pattern-3 timing | `artifacts/paper_core_smoke/pattern3/timing_summary.json` |
| INT8 quick detail | `test_results.json` |
| Table 1 全量 manifest | `artifacts/operator_level/table1_full_shared_gpu/manifest.json` |
| Table 1 全量 raw CSV | `artifacts/operator_level/table1_full_shared_gpu/operator_raw.csv` |
| Table 1 失败 CSV | `artifacts/operator_level/table1_full_shared_gpu/operator_failures.csv` |
| Table 1 汇总 JSON | `artifacts/operator_level/table1_full_shared_gpu/operator_summary.json` |
| Table 1 图：cuBLAS | `artifacts/operator_level/table1_full_shared_gpu/operator_vs_cublas.png` |
| Table 1 图：默认 CUTLASS FP16 | `artifacts/operator_level/table1_full_shared_gpu/fp16_vs_cutlass_default.png` |

## 8. 后续工作与工作量估计

以下估计以环境继续可用、单张 A100 可独占、模型权重可获得为前提。

| 后续工作 | 估计工作量 | 主要风险 |
| --- | --- | --- |
| 分析 INT8 4 个误差 case | 0.5-1 人日 | reference 语义与阈值判定需拆分 |
| 定位并修复 P40/P66 NaN | 1-3 人日 | 可能涉及 CUTLASS StreamK 补丁或 kernel 配置 |
| 独占 GPU 上重跑 operator sweep | 0.5 人日 + GPU 时间 | 当前共享 GPU 结果需要干净环境复核，并补充多轮方差统计 |
| 本机重新采集 predictor 数据 | 0.5-1 人日 | 需确认 PID 集合和论文采样协议 |
| 搭建 vLLM 0.10 与 MoonPoly patch | 1-2 人日 | PyTorch/CUDA ABI、模型依赖和 patch 漂移 |
| TTFT/TPOT 端到端实验 | 1-2 人日 + GPU 时间 | 模型权重、数据集和服务稳定性 |
| 论文图表逐项核对与报告 | 1-2 人日 | 需要先明确论文每张图表的原始实验协议 |

若只要求 operator 层核心结论，当前已经完成第一轮 smoke 复现。若要求论文所有端到端数据、图表和 profiling 逐项对齐，建议按约 4-8 人日安排，并单独预留 GPU 排队和模型下载时间。

## 9. 下一步建议顺序

1. 先修复或明确 P40/P66 StreamK NaN 的根因，并为失败测试补上可靠退出码。
2. 对 Table 1 INT8 的 95 个数值失败和 300 个缺失 shape 做分组审计，输出逐元素误差分布，与 CUTLASS/vLLM reference 做三方对照。
3. 对 FP16 selector 做边界 shape 的 oracle-PID/Pattern-3 ablation，并在独占 GPU、CUTLASS 4.2.0 和论文对应源码版本上重跑完整 operator sweep。
4. 固定 vLLM 0.10、PyTorch 和模型版本，再开始 TTFT/TPOT 端到端实验。
5. 最后建立论文图表到脚本、输入数据、输出文件的逐项映射表。
