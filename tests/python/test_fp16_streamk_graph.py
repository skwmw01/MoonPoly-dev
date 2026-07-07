#!/usr/bin/env python3
import argparse
import time

import torch
import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser(
        description="FP16 StreamK workspace + CUDA Graph smoke test for moonpoly."
    )
    parser.add_argument("--m", type=int, default=128)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument(
        "--input-space",
        type=str,
        default="linear",
        choices=["linear", "rcc"],
        help=(
            "Interpret --m/--n/--k in linear space ([M,K]x[N,K]) or RCC kernel space. "
            "If rcc, provided (m,n,k) is converted to linear as (M=n, N=m, K=k)."
        ),
    )
    parser.add_argument("--pid", type=int, default=40, help="Kernel id. StreamK path is 40.")
    parser.add_argument("--split-k", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--tol-rtol", type=float, default=5e-2)
    parser.add_argument("--tol-atol", type=float, default=5e-2)
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["all", "eager", "graph"],
        help="Run eager only, graph only, or both (default: all).",
    )
    parser.add_argument(
        "--use-rcc-op",
        action="store_true",
        help="Call the direct RCC debug op instead of the linear wrapper op.",
    )
    return parser.parse_args()


def ensure_ops_available():
    import moonpoly  # noqa: F401

    needed = [
        "fp16_workspace_size",
        "linear_fp16_pid_ws_out",
    ]
    missing = [name for name in needed if not hasattr(torch.ops.moonpoly_ops, name)]
    if missing:
        raise RuntimeError(
            "moonpoly extension is missing new ops: "
            + ", ".join(missing)
            + ". Rebuild/install moonpoly extension first."
        )


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    ensure_ops_available()
    device = "cuda"
    dtype = torch.float16

    if args.input_space == "rcc":
        # MoonPoly linear path maps to RCC as (m_rcc, n_rcc, k_rcc) = (N_linear, M_linear, K_linear).
        # So invert it here for user convenience:
        #   M_linear = n_rcc, N_linear = m_rcc, K_linear = k_rcc.
        m, n, k = args.n, args.m, args.k
        print(
            f"[map] input-space=rcc: (m={args.m}, n={args.n}, k={args.k}) "
            f"-> linear(M={m}, N={n}, K={k})"
        )
    else:
        m, n, k = args.m, args.n, args.k

    x = torch.randn((m, k), device=device, dtype=dtype).contiguous()
    w = torch.randn((n, k), device=device, dtype=dtype).contiguous()
    y = torch.empty((m, n), device=device, dtype=dtype).contiguous()

    if args.use_rcc_op:
        if not hasattr(torch.ops.moonpoly_ops, "fp16_rcc_workspace_size"):
            raise RuntimeError("moonpoly extension is missing fp16_rcc_workspace_size")
        if not hasattr(torch.ops.moonpoly_ops, "fp16_rcc_pid_ws_out"):
            raise RuntimeError("moonpoly extension is missing fp16_rcc_pid_ws_out")

        a_rcc = torch.randn((args.m, args.k), device=device, dtype=dtype).contiguous()
        b_col = torch.empty_strided(
            (args.k, args.n), (1, args.k), device=device, dtype=dtype
        )
        b_col.copy_(torch.randn((args.k, args.n), device=device, dtype=dtype))
        c_col = torch.empty_strided(
            (args.m, args.n), (1, args.m), device=device, dtype=dtype
        )
        ws_bytes = int(
            torch.ops.moonpoly_ops.fp16_rcc_workspace_size(
                a_rcc, b_col, args.pid, args.split_k
            )
        )
        workspace = torch.zeros((max(ws_bytes, 1),), device=device, dtype=torch.uint8)
        print(
            f"[info] rcc-shape=(m={args.m}, n={args.n}, k={args.k}), "
            f"pid={args.pid}, split_k={args.split_k}"
        )
        print(f"[info] workspace bytes={ws_bytes}")

        ref = a_rcc @ b_col
        torch.ops.moonpoly_ops.fp16_rcc_pid_ws_out(
            a_rcc, b_col, c_col, args.pid, workspace, args.split_k
        )
        y_eval = c_col
    else:
        ws_bytes = int(
            torch.ops.moonpoly_ops.fp16_workspace_size(x, w, args.pid, args.split_k)
        )
        workspace = torch.zeros((max(ws_bytes, 1),), device=device, dtype=torch.uint8)
        print(f"[info] shape=(M={m}, N={n}, K={k}), pid={args.pid}, split_k={args.split_k}")
        print(f"[info] workspace bytes={ws_bytes}")

        ref = F.linear(x, w)
        torch.ops.moonpoly_ops.linear_fp16_pid_ws_out(
            x, w, y, args.pid, workspace, args.split_k
        )
        y_eval = y

    max_abs = (y_eval - ref).abs().max().item()
    ok = torch.allclose(y_eval, ref, rtol=args.tol_rtol, atol=args.tol_atol)
    print(f"[check] allclose={ok}, max_abs={max_abs:.6f}")

    eager_ms = None
    if args.mode in ("all", "eager"):
        # warmup
        for _ in range(args.warmup):
            workspace.zero_()
            if args.use_rcc_op:
                torch.ops.moonpoly_ops.fp16_rcc_pid_ws_out(
                    a_rcc, b_col, c_col, args.pid, workspace, args.split_k
                )
            else:
                torch.ops.moonpoly_ops.linear_fp16_pid_ws_out(
                    x, w, y, args.pid, workspace, args.split_k
                )
        torch.cuda.synchronize()

        # eager timing
        t0 = time.perf_counter()
        for _ in range(args.iters):
            workspace.zero_()
            if args.use_rcc_op:
                torch.ops.moonpoly_ops.fp16_rcc_pid_ws_out(
                    a_rcc, b_col, c_col, args.pid, workspace, args.split_k
                )
            else:
                torch.ops.moonpoly_ops.linear_fp16_pid_ws_out(
                    x, w, y, args.pid, workspace, args.split_k
                )
        torch.cuda.synchronize()
        eager_ms = (time.perf_counter() - t0) * 1000.0 / args.iters

    graph_ms = None
    if args.mode in ("all", "graph"):
        # cuda graph capture/replay
        if args.use_rcc_op:
            a_static = a_rcc.clone()
            b_static = torch.empty_strided(
                (args.k, args.n), (1, args.k), device=device, dtype=dtype
            )
            b_static.copy_(b_col)
            c_static = torch.empty_strided(
                (args.m, args.n), (1, args.m), device=device, dtype=dtype
            )
        else:
            x_static = x.clone()
            w_static = w.clone()
            y_static = torch.empty_like(y)
        g = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        with torch.cuda.graph(g):
            workspace.zero_()
            if args.use_rcc_op:
                torch.ops.moonpoly_ops.fp16_rcc_pid_ws_out(
                    a_static, b_static, c_static, args.pid, workspace, args.split_k
                )
            else:
                torch.ops.moonpoly_ops.linear_fp16_pid_ws_out(
                    x_static, w_static, y_static, args.pid, workspace, args.split_k
                )

        # graph replay timing
        for _ in range(args.warmup):
            workspace.zero_()
            g.replay()
        torch.cuda.synchronize()

        t1 = time.perf_counter()
        for _ in range(args.iters):
            workspace.zero_()
            g.replay()
        torch.cuda.synchronize()
        graph_ms = (time.perf_counter() - t1) * 1000.0 / args.iters

    perf_items = []
    if eager_ms is not None:
        perf_items.append(f"eager_avg_ms={eager_ms:.4f}")
    if graph_ms is not None:
        perf_items.append(f"graph_replay_avg_ms={graph_ms:.4f}")
    print("[perf] " + ", ".join(perf_items))


if __name__ == "__main__":
    main()
