#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from benchmarks.tools.benchmark_ttft_tpot_vllm import (  # type: ignore
    AutoTokenizer,
    _build_prompt,
    _stream_completion,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Sweep explicit prompt lengths for TTFT.')
    p.add_argument('--base-url-cublas', required=True)
    p.add_argument('--base-url-moonpoly', required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--tokenizer', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--min-len', type=int, required=True)
    p.add_argument('--max-len', type=int, required=True)
    p.add_argument('--timeout-s', type=int, default=300)
    p.add_argument('--irregular-only', action='store_true')
    p.add_argument('--must-include', default='57,93,719')
    return p.parse_args()


def summarize(values: list[float]) -> dict[str, float]:
    xs = [x for x in values if x == x]
    if not xs:
        return {'count': 0, 'mean': float('nan'), 'best': float('nan'), 'worst': float('nan')}
    return {
        'count': len(xs),
        'mean': float(sum(xs) / len(xs)),
        'best': float(max(xs)),
        'worst': float(min(xs)),
    }


def winning_bands(rows: list[dict], threshold: float = 1.0) -> list[dict]:
    wins = sorted(r['prompt_len'] for r in rows if r['speedup'] == r['speedup'] and r['speedup'] >= threshold)
    if not wins:
        return []
    bands = []
    start = prev = wins[0]
    for x in wins[1:]:
        if x == prev + 1:
            prev = x
            continue
        bands.append({'start': start, 'end': prev, 'count': prev - start + 1})
        start = prev = x
    bands.append({'start': start, 'end': prev, 'count': prev - start + 1})
    return bands


def main() -> None:
    args = parse_args()
    tokenizer = None
    if AutoTokenizer is not None:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    must_include = sorted({int(x) for x in args.must_include.split(',') if x.strip()})
    lengths = []
    for x in range(args.min_len, args.max_len + 1):
        if args.irregular_only and x % 8 == 0:
            continue
        lengths.append(x)
    for x in must_include:
        if x not in lengths and args.min_len <= x <= args.max_len:
            lengths.append(x)
    lengths = sorted(set(lengths))

    rows = []
    for prompt_len in lengths:
        prompt = _build_prompt(tokenizer, prompt_len)
        cublas = _stream_completion(args.base_url_cublas, args.model, prompt, 1, args.timeout_s, tokenizer)
        moonpoly = _stream_completion(args.base_url_moonpoly, args.model, prompt, 1, args.timeout_s, tokenizer)
        speedup = (
            cublas.ttft_ms / moonpoly.ttft_ms
            if cublas.ok and moonpoly.ok and moonpoly.ttft_ms == moonpoly.ttft_ms and moonpoly.ttft_ms > 0
            else float('nan')
        )
        rows.append({
            'prompt_len': prompt_len,
            'cublas': {'ttft_ms': cublas.ttft_ms, 'ok': cublas.ok, 'error': cublas.error},
            'moonpoly': {'ttft_ms': moonpoly.ttft_ms, 'ok': moonpoly.ok, 'error': moonpoly.error},
            'speedup': speedup,
        })

    valid = [r for r in rows if not math.isnan(r['speedup'])]
    best = max(valid, key=lambda x: x['speedup']) if valid else None
    worst = min(valid, key=lambda x: x['speedup']) if valid else None
    result = {
        'config': {
            'model': args.model,
            'tokenizer': args.tokenizer,
            'min_len': args.min_len,
            'max_len': args.max_len,
            'irregular_only': args.irregular_only,
            'must_include': must_include,
        },
        'summary': summarize([r['speedup'] for r in rows]),
        'best_case': best,
        'worst_case': worst,
        'bands_ge_1_00': winning_bands(rows, 1.00),
        'bands_ge_1_05': winning_bands(rows, 1.05),
        'rows': rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({
        'out': str(out),
        'summary': result['summary'],
        'best_case': best,
        'worst_case': worst,
        'bands_ge_1_00': result['bands_ge_1_00'][:10],
        'bands_ge_1_05': result['bands_ge_1_05'][:10],
    }, indent=2))


if __name__ == '__main__':
    main()
