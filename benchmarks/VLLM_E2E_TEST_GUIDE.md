# vLLM E2E Test Guide (cuBLAS vs MoonPoly)

本文档给出可直接复制的命令，用于测试：

- TTFT（batch size=1, prompt length 1~500）
- TPOT（prompt=128, output=256, batch size 1~16）
- cuBLAS vs MoonPoly 的端到端对比

模型路径示例：

`/home/weight/llama2/Llama-2-13b-hf`

## 1. 推荐方式：单 GPU 顺序测试

说明：同一张卡上不要同时跑两个服务，避免互相干扰。

### 1.1 跑 cuBLAS 服务

```bash
CUDA_VISIBLE_DEVICES=1 VLLM_USE_MOONPOLY=0 python -m vllm.entrypoints.openai.api_server \
  --model /home/weight/llama2/Llama-2-13b-hf \
  --dtype float16 \
  --host 127.0.0.1 --port 8008
```

新开终端做健康检查：

```bash
curl http://127.0.0.1:8008/v1/models
```

### 1.2 运行 benchmark（仅 cuBLAS）

```bash
python benchmarks/tools/benchmark_ttft_tpot_vllm.py \
  --base-url-cublas http://127.0.0.1:8008 \
  --base-url-moonpoly http://127.0.0.1:8008 \
  --model /home/weight/llama2/Llama-2-13b-hf \
  --tokenizer /home/weight/llama2/Llama-2-13b-hf \
  --out artifacts/e2e_cublas_only.json \
  --num-prompts 100 \
  --ttft-min-len 1 --ttft-max-len 500 \
  --tpot-prompt-len 128 --tpot-output-len 256 \
  --tpot-batch-sizes 1,2,4,8,16 \
  --tpot-rounds 3
```

完成后停止 cuBLAS 服务。

### 1.3 跑 MoonPoly 服务

```bash
CUDA_VISIBLE_DEVICES=1 VLLM_USE_MOONPOLY=1 python -m vllm.entrypoints.openai.api_server \
  --model /home/weight/llama2/Llama-2-13b-hf \
  --dtype float16 \
  --host 127.0.0.1 --port 8009
```

健康检查：

```bash
curl http://127.0.0.1:8009/v1/models
```

### 1.4 运行 benchmark（仅 MoonPoly）

```bash
python benchmarks/tools/benchmark_ttft_tpot_vllm.py \
  --base-url-cublas http://127.0.0.1:8009 \
  --base-url-moonpoly http://127.0.0.1:8009 \
  --model /home/weight/llama2/Llama-2-13b-hf \
  --tokenizer /home/weight/llama2/Llama-2-13b-hf \
  --out artifacts/e2e_moonpoly_only.json \
  --num-prompts 100 \
  --ttft-min-len 1 --ttft-max-len 500 \
  --tpot-prompt-len 128 --tpot-output-len 256 \
  --tpot-batch-sizes 1,2,4,8,16 \
  --tpot-rounds 3
```

### 1.5 合并结果做对比（建议）

最简单方式：也可一次同时测（见第 2 节），或者自行把两份 JSON 聚合后画图。

## 2. 可选方式：双服务同时在线（需要资源充足）

如果你确认资源足够，也可以同时起两个服务（同卡不推荐）：

```bash
CUDA_VISIBLE_DEVICES=0 VLLM_USE_MOONPOLY=0 python -m vllm.entrypoints.openai.api_server \
  --model /home/weight/llama2/Llama-2-13b-hf \
  --dtype float16 \
  --host 127.0.0.1 --port 8008
```

```bash
CUDA_VISIBLE_DEVICES=1 VLLM_USE_MOONPOLY=1 python -m vllm.entrypoints.openai.api_server \
  --model /home/weight/llama2/Llama-2-13b-hf \
  --dtype float16 \
  --host 127.0.0.1 --port 8009
```

然后：

```bash
python benchmarks/tools/benchmark_ttft_tpot_vllm.py \
  --base-url-cublas http://127.0.0.1:8008 \
  --base-url-moonpoly http://127.0.0.1:8009 \
  --model /home/weight/llama2/Llama-2-13b-hf \
  --tokenizer /home/weight/llama2/Llama-2-13b-hf \
  --out artifacts/e2e_ttft_tpot.json \
  --num-prompts 100 \
  --ttft-min-len 1 --ttft-max-len 500 \
  --tpot-prompt-len 128 --tpot-output-len 256 \
  --tpot-batch-sizes 1,2,4,8,16 \
  --tpot-rounds 3
```

## 3. 结果检查

若输出里 speedup 全是 `NaN`，优先检查：

1. benchmark 执行期间服务是否还在运行
2. 端口和 URL 是否一致
3. `curl http://127.0.0.1:8008/v1/models` / `:8009/v1/models` 是否可访问

## 4. nsys Breakdown（GEMM 占比）

nsys breakdown 属于 benchmark 分析。当前 Online-only 仓库保留分析入口，
生成结果默认放到 `artifacts/`。

## 5. 出图与摘要

```bash
python tests/python/plot_e2e_speedup.py \
  --e2e-json artifacts/e2e_ttft_tpot.json \
  --breakdown-csv artifacts/nsys_breakdown.csv \
  --out-dir artifacts/e2e_figures
```

输出：

- `artifacts/e2e_figures/ttft_vs_len.png`
- `artifacts/e2e_figures/tpot_vs_bs.png`
- `artifacts/e2e_figures/gpu_breakdown.png`
- `artifacts/e2e_figures/summary.md`
