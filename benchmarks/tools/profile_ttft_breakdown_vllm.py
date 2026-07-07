#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
import os
import time
import urllib.request
from pathlib import Path

from benchmarks.tools.benchmark_ttft_tpot_vllm import AutoTokenizer, _build_prompt, _stream_completion  # type: ignore


def post_json(url: str, payload: dict | None = None) -> dict:
    data = None if payload is None else json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'}, method='POST')
    with urllib.request.urlopen(req, timeout=60) as resp:
        text = resp.read().decode('utf-8')
        return json.loads(text) if text else {}


def get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=60) as resp:
        text = resp.read().decode('utf-8')
        return json.loads(text) if text else {}


def load_trace(path: Path) -> dict:
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        return json.load(f)


def is_kernel(ev: dict) -> bool:
    cat = ev.get('cat', '')
    return 'kernel' in cat.lower()


def ev_ms(ev: dict) -> float:
    return float(ev.get('dur', 0.0)) / 1000.0


def classify(name: str) -> str:
    n = name.lower()
    if 'gemm' in n:
        return 'gemm'
    if 'act_and_mul_kernel' in n:
        return 'silu_mul'
    if 'flash_fwd' in n or 'flashattention' in n or 'flash::flash' in n:
        return 'flashattention'
    return 'other'


def profile_one(base_url: str, model: str, tokenizer_path: str, prompt_len: int, trace_dir: Path) -> dict:
    tok = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    prompt = _build_prompt(tok, prompt_len)
    trace_dir.mkdir(parents=True, exist_ok=True)
    before = {p.name for p in trace_dir.glob('*.pt.trace.json.gz')}
    post_json(base_url.rstrip('/') + '/start_profile')
    res = _stream_completion(base_url, model, prompt, 1, 300, tok)
    post_json(base_url.rstrip('/') + '/stop_profile')
    time.sleep(2.0)
    after = sorted([p for p in trace_dir.glob('*.pt.trace.json.gz') if p.name not in before], key=lambda p: p.stat().st_mtime)
    if not after:
        raise RuntimeError(f'No new trace in {trace_dir}')
    trace_path = after[-1]
    trace = load_trace(trace_path)
    events = trace.get('traceEvents', [])
    kernel_events = [ev for ev in events if is_kernel(ev)]
    totals = {'gemm': 0.0, 'silu_mul': 0.0, 'flashattention': 0.0, 'other': 0.0}
    top = []
    for ev in kernel_events:
        name = ev.get('name', '')
        ms = ev_ms(ev)
        totals[classify(name)] += ms
        top.append((ms, name))
    top.sort(reverse=True)
    kernel_total = sum(totals.values())
    return {
        'ttft_ms_estimate': res.ttft_ms,
        'trace_file': str(trace_path),
        'prompt_len': prompt_len,
        'kernel_total_ms': kernel_total,
        'gemm_ms': totals['gemm'],
        'silu_mul_ms': totals['silu_mul'],
        'flashattention_ms': totals['flashattention'],
        'other_ms': totals['other'],
        'gemm_pct': (totals['gemm'] / kernel_total * 100.0) if kernel_total else 0.0,
        'silu_mul_pct': (totals['silu_mul'] / kernel_total * 100.0) if kernel_total else 0.0,
        'flashattention_pct': (totals['flashattention'] / kernel_total * 100.0) if kernel_total else 0.0,
        'other_pct': (totals['other'] / kernel_total * 100.0) if kernel_total else 0.0,
        'top_kernels_ms': [{'name': n, 'ms': ms} for ms, n in top[:10]],
    }


def write_table(cublas: dict, system: dict, out_tex: Path) -> None:
    lines = []
    lines.append('\\begin{table}[t]')
    lines.append('\\caption{Breakdown of total CUDA kernel time for Llama TTFT (BS=1, prompt length=93).}')
    lines.append('\\centering')
    lines.append('\\small')
    lines.append('\\begin{tabular}{lcccc}')
    lines.append('\\toprule')
    lines.append('& \\multicolumn{2}{c}{cuBLAS} & \\multicolumn{2}{c}{\\system{}} \\\\')
    lines.append('\\cmidrule(lr){2-3} \\cmidrule(lr){4-5}')
    lines.append('Kernel category & Time (ms) & Percent (\\%) & Time (ms) & Percent (\\%) \\\\')
    lines.append('\\midrule')
    for key, label in [('gemm','GEMM'),('silu_mul','SiLU+Mul fusion'),('flashattention','FlashAttention'),('other','Others')]:
        lines.append(f"{label} & {cublas[key + '_ms']:.2f} & {cublas[key + '_pct']:.2f} & {system[key + '_ms']:.2f} & {system[key + '_pct']:.2f} \\\\")
    lines.append('\\midrule')
    lines.append(f"Total & {cublas['kernel_total_ms']:.2f} & 100.00 & {system['kernel_total_ms']:.2f} & 100.00 \\\\")
    lines.append('\\bottomrule')
    lines.append('\\end{tabular}')
    lines.append('\\label{tab:breakdown}')
    lines.append('\\end{table}')
    out_tex.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--base-url-cublas', required=True)
    ap.add_argument('--base-url-system', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--tokenizer', required=True)
    ap.add_argument('--prompt-len', type=int, default=93)
    ap.add_argument('--trace-dir-cublas', required=True)
    ap.add_argument('--trace-dir-system', required=True)
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cublas = profile_one(args.base_url_cublas, args.model, args.tokenizer, args.prompt_len, Path(args.trace_dir_cublas))
    system = profile_one(args.base_url_system, args.model, args.tokenizer, args.prompt_len, Path(args.trace_dir_system))
    result = {'cublas': cublas, 'system': system, 'speedup_ttft': cublas['ttft_ms_estimate'] / system['ttft_ms_estimate']}
    (out_dir/'breakdown_prompt93.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    write_table(cublas, system, out_dir/'breakdown_prompt93_table.tex')
    print(json.dumps({
        'out': str(out_dir/'breakdown_prompt93.json'),
        'speedup_ttft': result['speedup_ttft'],
        'cublas_ttft_ms': cublas['ttft_ms_estimate'],
        'system_ttft_ms': system['ttft_ms_estimate'],
        'cublas_kernel_total_ms': cublas['kernel_total_ms'],
        'system_kernel_total_ms': system['kernel_total_ms'],
    }, indent=2))

if __name__ == '__main__':
    main()
