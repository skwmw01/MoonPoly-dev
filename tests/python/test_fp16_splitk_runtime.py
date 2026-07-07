#!/usr/bin/env python3
"""Smoke test for MoonPoly FP16 runtime split-K dispatch."""

import argparse

import torch

import moonpoly


def parse_split_ks(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=8)
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--pid", type=int, default=400)
    parser.add_argument("--split-ks", type=parse_split_ks,
                        default=parse_split_ks("1,2,4"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rtol", type=float, default=5e-2)
    parser.add_argument("--atol", type=float, default=5e-2)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    a = torch.randn((args.m, args.k), device=args.device, dtype=torch.float16) * 0.1
    b = torch.randn((args.n, args.k), device=args.device, dtype=torch.float16) * 0.1
    ref = a @ b.t()

    print(
        f"[info] linear_shape=(M={args.m}, N={args.n}, K={args.k}), "
        f"pid={args.pid}, split_ks={args.split_ks}"
    )
    for split_k in args.split_ks:
        out = moonpoly.linear_cpp_forced_pid(a, b, args.pid, split_k)
        torch.cuda.synchronize()
        finite = torch.isfinite(out).all().item()
        max_abs = (out - ref).abs().max().item()
        ok = finite and torch.allclose(out, ref, rtol=args.rtol, atol=args.atol)
        print(
            f"[check] split_k={split_k} finite={finite} "
            f"allclose={ok} max_abs={max_abs:.6f}"
        )
        if not ok:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
