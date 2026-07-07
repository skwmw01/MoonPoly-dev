# MoonPoly Decode Linear Profile

This document records the minimal commands for collecting decode-like linear
shape/time breakdowns from the local vLLM fork.

## Profile output

Set these environment variables before starting the vLLM server:

- `VLLM_MOONPOLY_PROFILE_LINEAR=1`
- `VLLM_MOONPOLY_PROFILE_LINEAR_MAX_M=16`
- `VLLM_MOONPOLY_PROFILE_LINEAR_OUT=/path/to/profile.json`

The profile is aggregated inside
[`vllm/model_executor/layers/linear.py`]( /home/zhangyangyu/moonpoly_workspace/moonpoly/workspace_external/moonpoly_vllm_bk/vllm/model_executor/layers/linear.py )
and written when the server process exits.

Each row stores:

- `backend`: `cublas` or `moonpoly`
- `layer_prefix`
- `m`, `n`, `k`
- `count`
- `total_ms`
- `avg_ms`
- `avg_tflops`

## Summarize a profile

```bash
python /home/zhangyangyu/moonpoly_workspace/moonpoly/benchmarks/tools/summarize_linear_profile.py \
  /tmp/llama2_13b_cublas_decode_profile.json --topk 20
```

## Model paths

- `Llama-2-13b`: `/home/weight/llama2/Llama-2-13b-hf`
- `Qwen3-32B`: `/home/zhangyangyu/.cache/huggingface/hub/models--Qwen--Qwen3-32B`

## Llama-2-13B baseline

```bash
cd /home/zhangyangyu/moonpoly_workspace/moonpoly/workspace_external/moonpoly_vllm_bk

CUDA_VISIBLE_DEVICES=1 \
VLLM_USE_MOONPOLY_LINEAR=0 \
VLLM_MOONPOLY_PROFILE_LINEAR=1 \
VLLM_MOONPOLY_PROFILE_LINEAR_MAX_M=16 \
VLLM_MOONPOLY_PROFILE_LINEAR_OUT=/tmp/llama2_13b_cublas_decode_profile.json \
python -m vllm.entrypoints.openai.api_server \
  --model /home/weight/llama2/Llama-2-13b-hf \
  --host 127.0.0.1 --port 8008 \
  --enforce-eager \
  --dtype float16 \
  --gpu-memory-utilization 0.75 \
  --max-model-len 1024 \
  --max-num-seqs 1
```

## Llama-2-13B MoonPoly

```bash
cd /home/zhangyangyu/moonpoly_workspace/moonpoly/workspace_external/moonpoly_vllm_bk

CUDA_VISIBLE_DEVICES=1 \
VLLM_USE_MOONPOLY_LINEAR=1 \
VLLM_USE_MOONPOLY_STREAMK=0 \
VLLM_MOONPOLY_PROFILE_LINEAR=1 \
VLLM_MOONPOLY_PROFILE_LINEAR_MAX_M=16 \
VLLM_MOONPOLY_PROFILE_LINEAR_OUT=/tmp/llama2_13b_moonpoly_decode_profile.json \
python -m vllm.entrypoints.openai.api_server \
  --model /home/weight/llama2/Llama-2-13b-hf \
  --host 127.0.0.1 --port 8009 \
  --enforce-eager \
  --dtype float16 \
  --gpu-memory-utilization 0.75 \
  --max-model-len 1024 \
  --max-num-seqs 1
```

## Qwen3-32B baseline

```bash
cd /home/zhangyangyu/moonpoly_workspace/moonpoly/workspace_external/moonpoly_vllm_bk

CUDA_VISIBLE_DEVICES=1 \
VLLM_USE_MOONPOLY_LINEAR=0 \
VLLM_MOONPOLY_PROFILE_LINEAR=1 \
VLLM_MOONPOLY_PROFILE_LINEAR_MAX_M=16 \
VLLM_MOONPOLY_PROFILE_LINEAR_OUT=/tmp/qwen3_32b_cublas_decode_profile.json \
python -m vllm.entrypoints.openai.api_server \
  --model /home/zhangyangyu/.cache/huggingface/hub/models--Qwen--Qwen3-32B \
  --host 127.0.0.1 --port 8018 \
  --enforce-eager \
  --dtype float16 \
  --gpu-memory-utilization 0.75 \
  --max-model-len 1024 \
  --max-num-seqs 1
```

## Qwen3-32B MoonPoly

```bash
cd /home/zhangyangyu/moonpoly_workspace/moonpoly/workspace_external/moonpoly_vllm_bk

CUDA_VISIBLE_DEVICES=1 \
VLLM_USE_MOONPOLY_LINEAR=1 \
VLLM_USE_MOONPOLY_STREAMK=0 \
VLLM_MOONPOLY_PROFILE_LINEAR=1 \
VLLM_MOONPOLY_PROFILE_LINEAR_MAX_M=16 \
VLLM_MOONPOLY_PROFILE_LINEAR_OUT=/tmp/qwen3_32b_moonpoly_decode_profile.json \
python -m vllm.entrypoints.openai.api_server \
  --model /home/zhangyangyu/.cache/huggingface/hub/models--Qwen--Qwen3-32B \
  --host 127.0.0.1 --port 8019 \
  --enforce-eager \
  --dtype float16 \
  --gpu-memory-utilization 0.75 \
  --max-model-len 1024 \
  --max-num-seqs 1
```

## Drive TTFT / TPOT

Use the local benchmark tool:

```bash
python /home/zhangyangyu/moonpoly_workspace/moonpoly/benchmarks/tools/benchmark_ttft_tpot_vllm.py \
  --base-url-cublas http://127.0.0.1:8008 \
  --base-url-moonpoly http://127.0.0.1:8009 \
  --model /home/weight/llama2/Llama-2-13b-hf \
  --tokenizer /home/weight/llama2/Llama-2-13b-hf \
  --out /tmp/llama2_ttft_tpot.json \
  --num-prompts 100 \
  --ttft-min-len 1 --ttft-max-len 500 \
  --tpot-prompt-len 128 --tpot-output-len 256 \
  --tpot-batch-sizes 1,2,4,8,16 \
  --tpot-rounds 3
```

Do the same for Qwen3 by replacing the ports and model/tokenizer paths.
