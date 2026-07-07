#!/usr/bin/env python3
import argparse
import time

import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare MoonPoly StreamK pybind RCC tiles against torch matmul."
    )
    parser.add_argument("--m", type=int, default=5120, help="RCC m")
    parser.add_argument("--n", type=int, default=8, help="RCC n")
    parser.add_argument("--k", type=int, default=5120, help="RCC k")
    parser.add_argument(
        "--n-list",
        type=str,
        default="1,2,4,8",
        help="Comma-separated RCC n sweep. Ignored when --single-shape is set.",
    )
    parser.add_argument(
        "--single-shape",
        action="store_true",
        help="Run only the single RCC shape specified by --m/--n/--k.",
    )
    parser.add_argument("--split-k", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--tol-rtol", type=float, default=5e-2)
    parser.add_argument("--tol-atol", type=float, default=5e-2)
    parser.add_argument(
        "--mode",
        choices=["run", "raw-copy", "both"],
        default="both",
        help="Which pybind entrypoints to test.",
    )
    return parser.parse_args()


def parse_n_list(spec):
    values = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(int(item))
    if not values:
        raise ValueError("--n-list must contain at least one integer")
    return values


def benchmark(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / iters


def check(name, fn, ref, warmup, iters, rtol, atol):
    out = fn()
    max_abs = (out - ref).abs().max().item()
    ok = torch.allclose(out, ref, rtol=rtol, atol=atol)
    avg_ms = benchmark(fn, warmup, iters)
    print(f"{name}: allclose={ok} max_abs={max_abs:.6f} avg_ms={avg_ms:.4f}")
    return ok, max_abs, avg_ms


def run_one_shape(args, moonpoly, m, n, k):
    device = "cuda"
    dtype = torch.float16

    a = torch.randn((m, k), device=device, dtype=dtype).contiguous()
    b_col = torch.empty_strided((k, n), (1, k), device=device, dtype=dtype)
    b_col.copy_(torch.randn((k, n), device=device, dtype=dtype))
    ref = a @ b_col

    print(f"[info] rcc-shape=(m={m}, n={n}, k={k}), split_k={args.split_k}")
    print(
        f"[info] p40_workspace={moonpoly.p40_rcc_workspace_size(a, b_col, args.split_k)} "
        f"p42_workspace={moonpoly.p42_rcc_workspace_size(a, b_col, args.split_k)} "
        f"p43_workspace={moonpoly.p43_rcc_workspace_size(a, b_col, args.split_k)} "
        f"p44_workspace={moonpoly.p44_rcc_workspace_size(a, b_col, args.split_k)} "
        f"p66_workspace={moonpoly.p66_rcc_workspace_size(a, b_col, args.split_k)}"
    )

    run_tests = []
    if args.mode in ("run", "both"):
        run_tests.extend(
            [
                ("p42_run", lambda: moonpoly.p42_rcc_run(a, b_col, args.split_k)),
                ("p43_run", lambda: moonpoly.p43_rcc_run(a, b_col, args.split_k)),
                ("p44_run", lambda: moonpoly.p44_rcc_run(a, b_col, args.split_k)),
                ("p40_run", lambda: moonpoly.p40_rcc_run_singleton(a, b_col, args.split_k)),
                ("p66_run", lambda: moonpoly.p66_rcc_run(a, b_col, args.split_k)),
            ]
        )
    if args.mode in ("raw-copy", "both"):
        run_tests.extend(
            [
                ("p42_raw_host_check", lambda: moonpoly.p42_rcc_raw_copy_host_output_check(a, b_col, args.split_k, 8)),
                ("p43_raw_host_check", lambda: moonpoly.p43_rcc_raw_copy_host_output_check(a, b_col, args.split_k, 8)),
                ("p44_raw_host_check", lambda: moonpoly.p44_rcc_raw_copy_host_output_check(a, b_col, args.split_k, 8)),
                ("p40_raw_copy", lambda: moonpoly.p40_rcc_run_raw_copy(a, b_col, args.split_k)),
                ("p66_raw_copy", lambda: moonpoly.p66_rcc_run_raw_copy(a, b_col, args.split_k)),
            ]
        )

    all_ok = True
    for name, fn in run_tests:
        if name.endswith("_host_check"):
            out = fn()
            ok = int(out["nan_count"]) == 0
            print(
                f"{name}: nan_count={int(out['nan_count'])} "
                f"max_abs_head={float(out['max_abs_head']):.6f} pid={int(out['pid_tag'])}"
            )
        else:
            ok, _, _ = check(
                name, fn, ref, args.warmup, args.iters, args.tol_rtol, args.tol_atol
            )
        all_ok = all_ok and ok
    return all_ok


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    import moonpoly

    n_values = [args.n] if args.single_shape else parse_n_list(args.n_list)

    all_ok = True
    for idx, n in enumerate(n_values):
        if idx:
            print("")
        all_ok = run_one_shape(args, moonpoly, args.m, n, args.k) and all_ok

    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
