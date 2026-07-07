#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
import statistics
import subprocess
import sys
from pathlib import Path

# Support both historical output styles:
# - "cutlass time: X ms, moonpoly time: Y ms, speed_up: Zx"
# - "cublas time: X ms, moonpoly time: Y ms, speed_up: Zx"
RESULT_RE = re.compile(
    r"(?:cutlass|cublas)\s*time:\s*([0-9]+(?:\.[0-9]+)?)\s*(?:ms)?\s*,\s*"
    r"moonpoly\s*time:\s*([0-9]+(?:\.[0-9]+)?)\s*(?:ms)?\s*,\s*"
    r"speed_up:\s*([0-9]+(?:\.[0-9]+)?)x",
    re.IGNORECASE,
)


def run_once(binary: Path, m: int, n: int, k: int, warmup: int, iters: int, pass_inner_args: bool):
    cmds = []
    if pass_inner_args:
        cmds.append([str(binary), str(m), str(n), str(k), str(warmup), str(iters)])
    cmds.append([str(binary), str(m), str(n), str(k)])

    last_output = ""
    last_rc = 0
    for cmd in cmds:
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        out = p.stdout
        last_output = out
        last_rc = p.returncode
        if p.returncode != 0:
            continue

        match = RESULT_RE.search(out)
        if match:
            return float(match.group(1)), float(match.group(2)), float(match.group(3)), out, cmd

    raise RuntimeError(
        "Run failed or output parse failed.\n"
        f"binary={binary}\n"
        f"last_return_code={last_rc}\n"
        f"last_output:\n{last_output}"
    )


def mean_std(values: list[float]) -> tuple[float, float]:
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, std


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark MoonPoly vs default CUTLASS/CUBLAS binary with warmup/iters/repeat and CSV output."
    )
    parser.add_argument("--bin", default="build/benchmarks/cpp/moonpoly_bench_fp16_cutlass_default")
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--pass-inner-loop-args", action="store_true",
                        help="Pass warmup/iters to binary as extra argv (M N K warmup iters).")
    parser.add_argument("--csv", default="benchmarks/cutlass_default_compare.csv")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    binary = Path(args.bin)
    if not binary.exists():
        print(f"ERROR: binary not found: {binary}", file=sys.stderr)
        return 1
    if args.repeat <= 0 or args.warmup < 0 or args.iters <= 0:
        print("ERROR: repeat > 0, warmup >= 0, iters > 0", file=sys.stderr)
        return 1

    ref_times: list[float] = []
    moonpoly_times: list[float] = []
    speedups: list[float] = []

    print(f"Binary : {binary}")
    print(f"Shape  : M={args.m}, N={args.n}, K={args.k}")
    print(f"Config : warmup={args.warmup}, iters={args.iters}, repeat={args.repeat}")
    print(f"CSV    : {args.csv}")

    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    for i in range(args.repeat):
        ref_ms, moonpoly_ms, speedup, raw, used_cmd = run_once(
            binary, args.m, args.n, args.k, args.warmup, args.iters, args.pass_inner_loop_args
        )
        ref_times.append(ref_ms)
        moonpoly_times.append(moonpoly_ms)
        speedups.append(speedup)

        if args.verbose:
            print(f"\n[run {i+1}/{args.repeat}] cmd: {' '.join(used_cmd)}\n{raw}")
        else:
            print(
                f"[run {i+1}/{args.repeat}] ref={ref_ms:.6f} ms, "
                f"moonpoly={moonpoly_ms:.6f} ms, speedup={speedup:.6f}x"
            )

        rows.append({
            "row_type": "run",
            "run_idx": i + 1,
            "m": args.m,
            "n": args.n,
            "k": args.k,
            "warmup": args.warmup,
            "iters": args.iters,
            "repeat": args.repeat,
            "ref_ms": f"{ref_ms:.6f}",
            "moonpoly_ms": f"{moonpoly_ms:.6f}",
            "speedup_x": f"{speedup:.6f}",
            "ref_std": "",
            "moonpoly_std": "",
            "speedup_std": "",
            "speedup_from_avg_times_x": "",
            "binary": str(binary),
        })

    ref_mean, ref_std = mean_std(ref_times)
    mp_mean, mp_std = mean_std(moonpoly_times)
    sp_mean, sp_std = mean_std(speedups)

    print("\n===== Summary =====")
    print(f"ref      avg: {ref_mean:.6f} ms (std: {ref_std:.6f})")
    print(f"moonpoly avg: {mp_mean:.6f} ms (std: {mp_std:.6f})")
    print(f"speedup  avg: {sp_mean:.6f}x (std: {sp_std:.6f})")

    ratio_from_avg = ""
    if mp_mean > 0:
        ratio_from_avg = f"{ref_mean / mp_mean:.6f}"
        print(f"speedup from avg times (ref/moonpoly): {ratio_from_avg}x")

    rows.append({
        "row_type": "summary",
        "run_idx": 0,
        "m": args.m,
        "n": args.n,
        "k": args.k,
        "warmup": args.warmup,
        "iters": args.iters,
        "repeat": args.repeat,
        "ref_ms": f"{ref_mean:.6f}",
        "moonpoly_ms": f"{mp_mean:.6f}",
        "speedup_x": f"{sp_mean:.6f}",
        "ref_std": f"{ref_std:.6f}",
        "moonpoly_std": f"{mp_std:.6f}",
        "speedup_std": f"{sp_std:.6f}",
        "speedup_from_avg_times_x": ratio_from_avg,
        "binary": str(binary),
    })

    fieldnames = [
        "row_type",
        "run_idx",
        "m",
        "n",
        "k",
        "warmup",
        "iters",
        "repeat",
        "ref_ms",
        "moonpoly_ms",
        "speedup_x",
        "ref_std",
        "moonpoly_std",
        "speedup_std",
        "speedup_from_avg_times_x",
        "binary",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"CSV written: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
