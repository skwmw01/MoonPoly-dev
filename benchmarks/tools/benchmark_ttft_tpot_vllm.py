#!/usr/bin/env python3
"""
Benchmark vLLM TTFT/TPOT for cuBLAS vs MoonPoly backends.

The script assumes you already launched two vLLM servers:
- one with cuBLAS path
- one with MoonPoly path
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import random
import statistics
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib import request as urlrequest
from urllib import error as urlerror

try:
    from transformers import AutoTokenizer  # type: ignore
except Exception:
    AutoTokenizer = None


@dataclass
class StreamResult:
    prompt_tokens: int
    completion_tokens: int
    ttft_ms: float
    total_ms: float
    tpot_ms: float
    ok: bool
    error: str = ""


def _build_prompt(tokenizer: Any, target_tokens: int) -> str:
    if tokenizer is None:
        return " ".join(f"tok{i}" for i in range(target_tokens))

    unit_ids = tokenizer.encode(" hello", add_special_tokens=False)
    if not unit_ids:
        return " ".join(f"tok{i}" for i in range(target_tokens))

    ids: List[int] = []
    while len(ids) < target_tokens:
        ids.extend(unit_ids)
    ids = ids[:target_tokens]
    return tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)


def _count_tokens(tokenizer: Any, text: str) -> int:
    if tokenizer is None:
        return max(1, len(text.split()))
    return len(tokenizer.encode(text, add_special_tokens=False))


def _stream_completion(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout_s: int,
    tokenizer: Any,
) -> StreamResult:
    url = f"{base_url.rstrip('/')}/v1/completions"
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "min_tokens": max_tokens,
        "ignore_eos": True,
        "temperature": 0.0,
        "stream": True,
    }

    req_start = time.perf_counter()
    first_token_ts: Optional[float] = None
    generated_text_parts: List[str] = []
    payload_bytes = json.dumps(payload).encode("utf-8")
    req = urlrequest.Request(
        url=url,
        data=payload_bytes,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlrequest.urlopen(req, timeout=timeout_s) as resp:
            for raw_bytes in resp:
                raw = raw_bytes.decode("utf-8", errors="ignore").strip()
                if not raw:
                    continue
                if not raw.startswith("data:"):
                    continue
                data = raw[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = obj.get("choices", [])
                if not choices:
                    continue
                text_piece = choices[0].get("text", "")
                if text_piece:
                    generated_text_parts.append(text_piece)
                    if first_token_ts is None:
                        first_token_ts = time.perf_counter()
    except (urlerror.URLError, urlerror.HTTPError, TimeoutError, OSError, ValueError) as e:
        return StreamResult(
            prompt_tokens=_count_tokens(tokenizer, prompt),
            completion_tokens=0,
            ttft_ms=float("nan"),
            total_ms=float("nan"),
            tpot_ms=float("nan"),
            ok=False,
            error=str(e),
        )

    end_ts = time.perf_counter()
    generated_text = "".join(generated_text_parts)
    completion_tokens = _count_tokens(tokenizer, generated_text)
    prompt_tokens = _count_tokens(tokenizer, prompt)
    if first_token_ts is None:
        first_token_ts = end_ts

    ttft_ms = (first_token_ts - req_start) * 1000.0
    total_ms = (end_ts - req_start) * 1000.0
    if completion_tokens > 1:
        tpot_ms = ((end_ts - first_token_ts) * 1000.0) / (completion_tokens - 1)
    else:
        tpot_ms = float("nan")

    return StreamResult(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        ttft_ms=ttft_ms,
        total_ms=total_ms,
        tpot_ms=tpot_ms,
        ok=True,
    )


def _summary(values: List[float]) -> Dict[str, float]:
    xs = [x for x in values if x == x]
    if not xs:
        return {"count": 0, "mean": float("nan"), "p50": float("nan"), "p90": float("nan"), "p99": float("nan")}
    xs_sorted = sorted(xs)
    return {
        "count": len(xs),
        "mean": float(statistics.mean(xs)),
        "p50": float(xs_sorted[int(0.50 * (len(xs_sorted) - 1))]),
        "p90": float(xs_sorted[int(0.90 * (len(xs_sorted) - 1))]),
        "p99": float(xs_sorted[int(0.99 * (len(xs_sorted) - 1))]),
    }


def run_ttft(
    base_url: str,
    model: str,
    tokenizer: Any,
    num_prompts: int,
    min_len: int,
    max_len: int,
    seed: int,
    timeout_s: int,
) -> Dict[str, Any]:
    rng = random.Random(seed)
    samples: List[Dict[str, Any]] = []
    for _ in range(num_prompts):
        l = rng.randint(min_len, max_len)
        prompt = _build_prompt(tokenizer, l)
        r = _stream_completion(base_url, model, prompt, max_tokens=1, timeout_s=timeout_s, tokenizer=tokenizer)
        item = asdict(r)
        item["target_prompt_len"] = l
        samples.append(item)
    ttfts = [s["ttft_ms"] for s in samples if s["ok"]]
    return {"samples": samples, "summary": _summary(ttfts)}


def run_tpot(
    base_url: str,
    model: str,
    tokenizer: Any,
    prompt_len: int,
    output_len: int,
    batch_sizes: List[int],
    rounds: int,
    timeout_s: int,
) -> Dict[str, Any]:
    prompt = _build_prompt(tokenizer, prompt_len)
    per_bs: Dict[str, Any] = {}
    for bs in batch_sizes:
        round_samples: List[Dict[str, Any]] = []
        for _ in range(rounds):
            with concurrent.futures.ThreadPoolExecutor(max_workers=bs) as ex:
                futs = [
                    ex.submit(
                        _stream_completion,
                        base_url,
                        model,
                        prompt,
                        output_len,
                        timeout_s,
                        tokenizer,
                    )
                    for _ in range(bs)
                ]
                for fut in concurrent.futures.as_completed(futs):
                    r = fut.result()
                    round_samples.append(asdict(r))
        tpots = [x["tpot_ms"] for x in round_samples if x["ok"]]
        per_bs[str(bs)] = {"samples": round_samples, "summary": _summary(tpots)}
    return per_bs


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark vLLM TTFT/TPOT for cuBLAS vs MoonPoly.")
    p.add_argument("--base-url-cublas", required=True)
    p.add_argument("--base-url-moonpoly", required=True)
    p.add_argument("--model", required=True, help="Model name served by vLLM")
    p.add_argument("--tokenizer", default=None, help="Optional tokenizer path/name for exact token counting")
    p.add_argument("--out", default="artifacts/e2e_ttft_tpot.json")
    p.add_argument("--num-prompts", type=int, default=100)
    p.add_argument("--ttft-min-len", type=int, default=1)
    p.add_argument("--ttft-max-len", type=int, default=500)
    p.add_argument("--tpot-prompt-len", type=int, default=128)
    p.add_argument("--tpot-output-len", type=int, default=256)
    p.add_argument("--tpot-batch-sizes", default="1,2,4,8,16")
    p.add_argument("--tpot-rounds", type=int, default=3)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--timeout-s", type=int, default=600)
    args = p.parse_args()

    tokenizer = None
    if args.tokenizer:
        if AutoTokenizer is None:
            print("Warning: transformers is not installed; fallback tokenization will be used.")
        else:
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)

    batch_sizes = [int(x.strip()) for x in args.tpot_batch_sizes.split(",") if x.strip()]

    result: Dict[str, Any] = {
        "config": {
            "model": args.model,
            "num_prompts": args.num_prompts,
            "ttft_len_range": [args.ttft_min_len, args.ttft_max_len],
            "tpot_prompt_len": args.tpot_prompt_len,
            "tpot_output_len": args.tpot_output_len,
            "tpot_batch_sizes": batch_sizes,
            "tpot_rounds": args.tpot_rounds,
            "seed": args.seed,
        },
        "backends": {},
    }

    for name, base in [("cublas", args.base_url_cublas), ("moonpoly", args.base_url_moonpoly)]:
        ttft = run_ttft(
            base_url=base,
            model=args.model,
            tokenizer=tokenizer,
            num_prompts=args.num_prompts,
            min_len=args.ttft_min_len,
            max_len=args.ttft_max_len,
            seed=args.seed,
            timeout_s=args.timeout_s,
        )
        tpot = run_tpot(
            base_url=base,
            model=args.model,
            tokenizer=tokenizer,
            prompt_len=args.tpot_prompt_len,
            output_len=args.tpot_output_len,
            batch_sizes=batch_sizes,
            rounds=args.tpot_rounds,
            timeout_s=args.timeout_s,
        )
        result["backends"][name] = {"ttft": ttft, "tpot": tpot}

    # Compute speedups (>1 means MoonPoly is faster).
    speedup: Dict[str, Any] = {}
    c_ttft = result["backends"]["cublas"]["ttft"]["summary"]["mean"]
    m_ttft = result["backends"]["moonpoly"]["ttft"]["summary"]["mean"]
    speedup["ttft_mean_speedup"] = c_ttft / m_ttft if m_ttft == m_ttft and m_ttft > 0 else float("nan")

    tpot_speedups: Dict[str, float] = {}
    for bs in batch_sizes:
        c_tpot = result["backends"]["cublas"]["tpot"][str(bs)]["summary"]["mean"]
        m_tpot = result["backends"]["moonpoly"]["tpot"][str(bs)]["summary"]["mean"]
        tpot_speedups[str(bs)] = c_tpot / m_tpot if m_tpot == m_tpot and m_tpot > 0 else float("nan")
    speedup["tpot_mean_speedup_by_bs"] = tpot_speedups
    result["speedup"] = speedup

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(json.dumps({"out": str(out_path), "speedup": speedup}, indent=2))


if __name__ == "__main__":
    main()
