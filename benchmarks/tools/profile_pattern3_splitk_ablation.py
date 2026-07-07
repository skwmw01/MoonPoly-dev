#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable

import torch
import moonpoly


def bench(fn: Callable[[], torch.Tensor], warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / iters


def build_input(m: int, n: int, k: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    x = torch.randn((m, k), device="cuda", dtype=torch.float16).contiguous()
    w = torch.randn((n, k), device="cuda", dtype=torch.float16).contiguous()
    return x, w


def tflops(m: int, n: int, k: int, avg_ms: float) -> float:
    flops = 2.0 * float(m) * float(n) * float(k)
    return flops / (avg_ms / 1000.0) / 1.0e12


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark the three-way Pattern-3 split-K ablation on one GEMM shape."
    )
    p.add_argument("--m", type=int, default=8)
    p.add_argument("--n", type=int, default=10240)
    p.add_argument("--k", type=int, default=5120)
    p.add_argument("--mono-pid", type=int, default=96)
    p.add_argument("--mono-split-k", type=int, default=1)
    p.add_argument("--mono-kind", choices=["forced", "generic"], default="forced")
    p.add_argument("--splitk-pid", type=int, default=166)
    p.add_argument("--splitk", type=int, default=4)
    p.add_argument("--splitk-kind", choices=["forced", "generic"], default="forced")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument(
        "--backend",
        choices=["all", "cublas", "mono", "splitk"],
        default="all",
        help="Use a single backend for NCU tracing, or 'all' for timing summary.",
    )
    p.add_argument("--quiet-metadata", action="store_true")
    p.add_argument("--verify", action="store_true")
    p.add_argument("--out-json", default="artifacts/pattern3_splitk_ablation.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    x, w = build_input(args.m, args.n, args.k, args.seed)

    def cublas_fn() -> torch.Tensor:
        return x @ w.t()

    def mono_fn() -> torch.Tensor:
        if args.mono_kind == "generic":
            return moonpoly.linear(x, w)
        return moonpoly.linear_cpp_forced_pid(x, w, args.mono_pid, args.mono_split_k)

    def splitk_fn() -> torch.Tensor:
        if args.splitk_kind == "generic":
            return moonpoly.linear(x, w)
        return moonpoly.linear_cpp_forced_pid(x, w, args.splitk_pid, args.splitk)

    fn_map = {
        "cublas": cublas_fn,
        "mono": mono_fn,
        "splitk": splitk_fn,
    }

    if args.backend != "all":
        out = fn_map[args.backend]()
        if args.verify:
            ref = cublas_fn()
            if not torch.isfinite(out).all():
                raise RuntimeError(f"{args.backend} produced non-finite output")
            if not torch.allclose(out, ref, rtol=5e-2, atol=5e-2):
                raise RuntimeError(
                    f"{args.backend} mismatch: max_abs={(out - ref).abs().max().item()}"
                )
        torch.cuda.synchronize()
        if not args.quiet_metadata:
            print(json.dumps({
                "backend": args.backend,
                "shape": [args.m, args.n, args.k],
                "mono_kind": args.mono_kind,
                "mono_pid": args.mono_pid,
                "mono_split_k": args.mono_split_k,
                "splitk_kind": args.splitk_kind,
                "splitk_pid": args.splitk_pid,
                "splitk": args.splitk,
            }, indent=2))
        return

    ref = cublas_fn()
    rows = []
    for name, fn in [("cublas", cublas_fn), ("mono", mono_fn), ("splitk", splitk_fn)]:
        out = fn()
        if not torch.isfinite(out).all():
            raise RuntimeError(f"{name} produced non-finite output")
        max_abs = float((out - ref).abs().max().item())
        ok = bool(torch.allclose(out, ref, rtol=5e-2, atol=5e-2))
        avg_ms = bench(fn, args.warmup, args.iters)
        rows.append({
            "backend": name,
            "avg_ms": avg_ms,
            "tflops": tflops(args.m, args.n, args.k, avg_ms),
            "max_abs": max_abs,
            "ok": ok,
        })

    by_name = {row["backend"]: row for row in rows}
    summary = {
        "shape": [args.m, args.n, args.k],
        "mono_kind": args.mono_kind,
        "mono_pid": args.mono_pid,
        "mono_split_k": args.mono_split_k,
        "splitk_kind": args.splitk_kind,
        "splitk_pid": args.splitk_pid,
        "splitk": args.splitk,
        "warmup": args.warmup,
        "iters": args.iters,
        "results": rows,
        "derived": {
            "mono_speedup_vs_cublas": by_name["cublas"]["avg_ms"] / by_name["mono"]["avg_ms"],
            "splitk_speedup_vs_cublas": by_name["cublas"]["avg_ms"] / by_name["splitk"]["avg_ms"],
            "splitk_speedup_vs_mono": by_name["mono"]["avg_ms"] / by_name["splitk"]["avg_ms"],
        },
    }

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
