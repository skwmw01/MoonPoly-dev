#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path

from benchmarks.tools.benchmark_ttft_tpot_vllm import (  # type: ignore
    AutoTokenizer,
    _build_prompt,
    _stream_completion,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark irregular-length TTFT on two vLLM backends."
    )
    p.add_argument("--base-url-cublas", required=True)
    p.add_argument("--base-url-moonpoly", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--num-samples", type=int, default=100)
    p.add_argument("--max-len", type=int, default=900)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--timeout-s", type=int, default=300)
    p.add_argument(
        "--must-include",
        default="57,719",
        help="Comma-separated prompt lengths that must be included.",
    )
    return p.parse_args()


def irregular_lengths(num_samples: int, max_len: int, seed: int,
                      must_include: list[int]) -> list[int]:
    rng = random.Random(seed)
    chosen = set(x for x in must_include if 1 <= x <= max_len and x % 8 != 0)
    while len(chosen) < num_samples:
        x = rng.randint(1, max_len)
        if x % 8 != 0:
            chosen.add(x)
    out = list(chosen)
    rng.shuffle(out)
    return out[:num_samples]


def summary(values: list[float]) -> dict[str, float]:
    xs = [x for x in values if x == x]
    if not xs:
        return {
            "count": 0,
            "mean": float("nan"),
            "p50": float("nan"),
            "p90": float("nan"),
            "best": float("nan"),
            "worst": float("nan"),
        }
    xs_sorted = sorted(xs)
    return {
        "count": len(xs),
        "mean": float(statistics.mean(xs)),
        "p50": float(xs_sorted[int(0.5 * (len(xs_sorted) - 1))]),
        "p90": float(xs_sorted[int(0.9 * (len(xs_sorted) - 1))]),
        "best": float(max(xs)),
        "worst": float(min(xs)),
    }


def main() -> None:
    args = parse_args()
    tokenizer = None
    if AutoTokenizer is not None:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    must_include = [int(x) for x in args.must_include.split(",") if x.strip()]
    lengths = irregular_lengths(args.num_samples, args.max_len, args.seed, must_include)

    rows = []
    for prompt_len in lengths:
        prompt = _build_prompt(tokenizer, prompt_len)
        cublas = _stream_completion(
            args.base_url_cublas, args.model, prompt, 1, args.timeout_s, tokenizer
        )
        moonpoly = _stream_completion(
            args.base_url_moonpoly, args.model, prompt, 1, args.timeout_s, tokenizer
        )
        speedup = (
            cublas.ttft_ms / moonpoly.ttft_ms
            if cublas.ok and moonpoly.ok and moonpoly.ttft_ms == moonpoly.ttft_ms and moonpoly.ttft_ms > 0
            else float("nan")
        )
        rows.append({
            "prompt_len": prompt_len,
            "cublas": {
                "ttft_ms": cublas.ttft_ms,
                "ok": cublas.ok,
                "error": cublas.error,
            },
            "moonpoly": {
                "ttft_ms": moonpoly.ttft_ms,
                "ok": moonpoly.ok,
                "error": moonpoly.error,
            },
            "speedup": speedup,
        })

    valid = [r for r in rows if not math.isnan(r["speedup"])]
    best = max(valid, key=lambda x: x["speedup"]) if valid else None
    worst = min(valid, key=lambda x: x["speedup"]) if valid else None

    result = {
        "config": {
            "model": args.model,
            "tokenizer": args.tokenizer,
            "num_samples": args.num_samples,
            "max_len": args.max_len,
            "seed": args.seed,
            "must_include": must_include,
        },
        "lengths": lengths,
        "summary": summary([r["speedup"] for r in rows]),
        "best_case": best,
        "worst_case": worst,
        "rows": rows,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "out": str(out),
        "summary": result["summary"],
        "best_case": best,
        "worst_case": worst,
    }, indent=2))


if __name__ == "__main__":
    main()
