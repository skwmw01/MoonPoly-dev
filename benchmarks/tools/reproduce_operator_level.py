#!/usr/bin/env python3
"""Reproduce the Table 1 operator-level GEMM evaluation with audit-friendly output."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DTYPES = ("FP16", "FP32", "INT8")
RAW_FIELDS = (
    "shape_index",
    "qid",
    "category",
    "backend",
    "dtype",
    "m",
    "n",
    "k",
    "flops",
    "baseline_ms",
    "moonpoly_ms",
    "speedup",
    "cosine_similarity",
    "correctness_passed",
    "status",
    "error",
    "wall_seconds",
)
PAPER_RESULTS = {
    "cublas": {
        "FP16": {"speedup_avg": 1.27, "speedup_max": 5.47},
        "INT8": {"speedup_avg": 1.35, "speedup_max": 14.03},
        "FP32": {"speedup_avg": 1.12, "speedup_max": 5.03},
    },
    "cutlass-default": {
        "FP16": {"speedup_avg": 3.34},
        "INT8": {"speedup_avg": 1.63},
        "FP32": {"speedup_avg": 1.51},
    },
}
CUTLASS_RESULT_RE = re.compile(
    r"cutlass\s+time:\s*([0-9]+(?:\.[0-9]+)?)\s*ms,\s*"
    r"moonpoly\s+time:\s*([0-9]+(?:\.[0-9]+)?)\s*ms,\s*"
    r"speed_up:\s*([0-9]+(?:\.[0-9]+)?)x",
    re.IGNORECASE,
)


def repo_path(raw: str | Path) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else REPO_ROOT / path


def parse_stages(raw: str) -> list[str]:
    stages = [item.strip() for item in raw.split(",") if item.strip()]
    valid = {"cublas", "cutlass-fp16"}
    unknown = [stage for stage in stages if stage not in valid]
    if unknown:
        raise ValueError(f"Unknown stages: {', '.join(unknown)}")
    return stages


def read_shapes(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"m", "n", "k"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"Shape CSV must contain m,n,k columns: {path}")
        shapes = []
        for source_index, row in enumerate(reader, start=1):
            qid = int(row.get("qid", row.get("id", source_index - 1)))
            m, n, k = int(row["m"]), int(row["n"]), int(row["k"])
            shapes.append(
                {
                    "shape_index": source_index,
                    "qid": qid,
                    "category": "DeepBench" if qid >= 1299 else "Real-World",
                    "m": m,
                    "n": n,
                    "k": k,
                    "flops": 2 * m * n * k,
                }
            )
    return shapes


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=True), encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def last_error_line(output: str) -> str:
    candidates = []
    for line in output.splitlines():
        lowered = line.lower()
        if "error" in lowered or "fail" in lowered or "mismatch" in lowered:
            candidates.append(line.strip())
    return candidates[-1][:500] if candidates else ""


def base_record(shape: dict[str, Any], backend: str, dtype: str) -> dict[str, Any]:
    return {
        **shape,
        "backend": backend,
        "dtype": dtype,
        "baseline_ms": None,
        "moonpoly_ms": None,
        "speedup": None,
        "cosine_similarity": None,
        "correctness_passed": False,
        "status": "missing",
        "error": "",
        "wall_seconds": None,
    }


def run_process(command: list[str], cwd: Path, timeout: int) -> tuple[int, str, float]:
    started = time.perf_counter()
    try:
        proc = subprocess.run(
            command,
            cwd=str(cwd),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
        return proc.returncode, proc.stdout, time.perf_counter() - started
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or ""
        if isinstance(output, bytes):
            output = output.decode(errors="replace")
        return 124, output + f"\nTIMEOUT after {timeout}s\n", time.perf_counter() - started


def run_cublas_shape(
    shape: dict[str, Any], binary: Path, run_root: Path, timeout: int, force: bool
) -> list[dict[str, Any]]:
    name = f"qid_{shape['qid']:04d}_{shape['m']}x{shape['n']}x{shape['k']}"
    run_dir = run_root / name
    cache = run_dir / "result.json"
    if cache.exists() and not force:
        return json.loads(cache.read_text(encoding="utf-8"))["records"]

    run_dir.mkdir(parents=True, exist_ok=True)
    command = [str(binary), str(shape["m"]), str(shape["n"]), str(shape["k"])]
    returncode, output, wall_seconds = run_process(command, run_dir, timeout)
    (run_dir / "stdout.txt").write_text(output, encoding="utf-8")

    by_dtype: dict[str, dict[str, str]] = {}
    result_csv = run_dir / "moonpoly_vs_cublas_benchmark.csv"
    if result_csv.exists():
        with result_csv.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                by_dtype[row["DataType"]] = row

    records = []
    process_error = last_error_line(output)
    for dtype in DTYPES:
        record = base_record(shape, "cublas", dtype)
        record["wall_seconds"] = wall_seconds
        row = by_dtype.get(dtype)
        if row is None:
            record["status"] = "process_error" if returncode else "missing"
            record["error"] = process_error or f"No {dtype} row produced"
        else:
            passed = row["Correctness_Passed"].upper() == "TRUE"
            record.update(
                {
                    "baseline_ms": float(row["cuBLAS_time_ms"]),
                    "moonpoly_ms": float(row["Moonpoly_time_ms"]),
                    "speedup": float(row["Speedup"]),
                    "cosine_similarity": float(row["Cosine_Similarity"]),
                    "correctness_passed": passed,
                    "status": "pass" if passed else "incorrect",
                    "error": "" if passed else "Cosine similarity threshold failed",
                }
            )
        records.append(record)

    write_json(
        cache,
        {"command": command, "returncode": returncode, "records": records},
    )
    return records


def run_cutlass_shape(
    shape: dict[str, Any], binary: Path, run_root: Path, timeout: int,
    warmup: int, iters: int, force: bool,
) -> list[dict[str, Any]]:
    name = f"qid_{shape['qid']:04d}_{shape['m']}x{shape['n']}x{shape['k']}"
    run_dir = run_root / name
    cache = run_dir / "result.json"
    if cache.exists() and not force:
        return json.loads(cache.read_text(encoding="utf-8"))["records"]

    run_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(binary), str(shape["m"]), str(shape["n"]), str(shape["k"]),
        str(warmup), str(iters),
    ]
    returncode, output, wall_seconds = run_process(command, run_dir, timeout)
    (run_dir / "stdout.txt").write_text(output, encoding="utf-8")

    record = base_record(shape, "cutlass-default", "FP16")
    record["wall_seconds"] = wall_seconds
    match = CUTLASS_RESULT_RE.search(output)
    if returncode == 0 and match:
        record.update(
            {
                "baseline_ms": float(match.group(1)),
                "moonpoly_ms": float(match.group(2)),
                "speedup": float(match.group(3)),
                "correctness_passed": True,
                "status": "pass",
            }
        )
    else:
        record["status"] = "process_error" if returncode else "missing"
        record["error"] = last_error_line(output) or "CUTLASS output parse failed"

    records = [record]
    write_json(
        cache,
        {"command": command, "returncode": returncode, "records": records},
    )
    return records


def geometric_mean(values: list[float]) -> float | None:
    positive = [value for value in values if value > 0 and math.isfinite(value)]
    if not positive:
        return None
    return math.exp(sum(math.log(value) for value in positive) / len(positive))


def summarize(records: list[dict[str, Any]], expected_shapes: int) -> dict[str, Any]:
    result: dict[str, Any] = {}
    keys = sorted({(str(row["backend"]), str(row["dtype"])) for row in records})
    for backend, dtype in keys:
        selected = [
            row for row in records
            if row["backend"] == backend and row["dtype"] == dtype
        ]
        passed = [row for row in selected if row["status"] == "pass"]
        speedups = [float(row["speedup"]) for row in passed]
        by_category = {}
        for category in ("Real-World", "DeepBench"):
            category_rows = [row for row in selected if row["category"] == category]
            by_category[category] = {
                "total": len(category_rows),
                "passed": sum(row["status"] == "pass" for row in category_rows),
            }
        result[f"{backend}:{dtype}"] = {
            "expected_shapes": expected_shapes,
            "records": len(selected),
            "passed": len(passed),
            "incorrect": sum(row["status"] == "incorrect" for row in selected),
            "missing": sum(row["status"] == "missing" for row in selected),
            "process_errors": sum(row["status"] == "process_error" for row in selected),
            "speedup_avg": statistics.fmean(speedups) if speedups else None,
            "speedup_geomean": geometric_mean(speedups),
            "speedup_median": statistics.median(speedups) if speedups else None,
            "speedup_min": min(speedups) if speedups else None,
            "speedup_max": max(speedups) if speedups else None,
            "category_correctness": by_category,
            "paper_reported": PAPER_RESULTS.get(backend, {}).get(dtype),
        }
    return result


def plot_results(records: list[dict[str, Any]], outdir: Path) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    outputs = []
    cublas_rows = [row for row in records if row["backend"] == "cublas"]
    if cublas_rows:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=False)
        for axis, dtype in zip(axes, DTYPES):
            rows = [
                row for row in cublas_rows
                if row["dtype"] == dtype and row["status"] == "pass"
            ]
            x_values = [math.log10(max(1, int(row["flops"]))) for row in rows]
            y_values = [float(row["speedup"]) for row in rows]
            colors = ["#1677b8" if row["category"] == "Real-World" else "#d95f02" for row in rows]
            axis.scatter(x_values, y_values, c=colors, s=9, alpha=0.55, linewidths=0)
            axis.axhline(1.0, color="#333333", linewidth=1.0, linestyle="--")
            if y_values:
                average = statistics.fmean(y_values)
                axis.axhline(average, color="#2a8f55", linewidth=1.2)
                axis.text(0.98, 0.96, f"avg={average:.3f}x\nvalid={len(rows)}",
                          transform=axis.transAxes, ha="right", va="top")
            axis.set_title(dtype)
            axis.set_xlabel("log10(FLOPs)")
            axis.set_ylabel("MoonPoly speedup over cuBLAS")
            axis.grid(alpha=0.2)
        fig.tight_layout()
        path = outdir / "operator_vs_cublas.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        outputs.append(str(path))

    cutlass_rows = [
        row for row in records
        if row["backend"] == "cutlass-default" and row["status"] == "pass"
    ]
    if cutlass_rows:
        fig, axis = plt.subplots(figsize=(6.4, 4.2))
        x_values = [math.log10(max(1, int(row["flops"]))) for row in cutlass_rows]
        y_values = [float(row["speedup"]) for row in cutlass_rows]
        axis.scatter(x_values, y_values, color="#6a51a3", s=9, alpha=0.55, linewidths=0)
        axis.axhline(1.0, color="#333333", linewidth=1.0, linestyle="--")
        axis.axhline(statistics.fmean(y_values), color="#2a8f55", linewidth=1.2)
        axis.set_title("FP16 MoonPoly vs default CUTLASS")
        axis.set_xlabel("log10(FLOPs)")
        axis.set_ylabel("Speedup")
        axis.grid(alpha=0.2)
        fig.tight_layout()
        path = outdir / "fp16_vs_cutlass_default.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        outputs.append(str(path))
    return outputs


def gpu_snapshot() -> str:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,uuid,utilization.gpu,memory.used,memory.free",
        "--format=csv,noheader",
    ]
    try:
        return subprocess.run(
            command, check=False, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return "unavailable"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MoonPoly's Table 1 operator-level evaluation with resumable per-shape results."
    )
    parser.add_argument("--shape-csv", default="benchmarks/data/core/gemm_trick.csv")
    parser.add_argument("--outdir", default="artifacts/operator_level/full")
    parser.add_argument("--moonpoly-bin", default="build/benchmarks/cpp/moonpoly_benchmark")
    parser.add_argument(
        "--cutlass-fp16-bin",
        default="build/benchmarks/cpp/moonpoly_bench_fp16_cutlass_default",
    )
    parser.add_argument("--stages", default="cublas,cutlass-fp16")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-shapes", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--cutlass-warmup", type=int, default=5)
    parser.add_argument("--cutlass-iters", type=int, default=20)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--run-note", default="")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.start_index < 0 or args.max_shapes < 0 or args.timeout <= 0:
        raise ValueError("start-index/max-shapes must be non-negative and timeout must be positive")

    stages = parse_stages(args.stages)
    shape_csv = repo_path(args.shape_csv)
    outdir = repo_path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    all_shapes = read_shapes(shape_csv)
    shapes = all_shapes[args.start_index:]
    if args.max_shapes:
        shapes = shapes[:args.max_shapes]
    if not shapes:
        raise ValueError("No shapes selected")

    moonpoly_binary = repo_path(args.moonpoly_bin)
    cutlass_binary = repo_path(args.cutlass_fp16_bin)
    if "cublas" in stages and not moonpoly_binary.is_file():
        raise FileNotFoundError(moonpoly_binary)
    if "cutlass-fp16" in stages and not cutlass_binary.is_file():
        raise FileNotFoundError(cutlass_binary)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(REPO_ROOT),
        "shape_csv": str(shape_csv),
        "dataset_total_shapes": len(all_shapes),
        "selected_shapes": len(shapes),
        "selected_real_world": sum(s["category"] == "Real-World" for s in shapes),
        "selected_deepbench": sum(s["category"] == "DeepBench" for s in shapes),
        "stages": stages,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "gpu_before": gpu_snapshot(),
        "run_note": args.run_note,
        "limitations": [
            "The repository provides a default CUTLASS comparison binary only for FP16.",
            "C++ benchmark timing is used as provided: 5 warmups and 20 measured iterations.",
        ],
    }
    write_json(outdir / "manifest.json", manifest)

    records: list[dict[str, Any]] = []
    for stage in stages:
        print(f"\n===== stage: {stage} ({len(shapes)} shapes) =====", flush=True)
        run_root = outdir / "runs" / stage
        for position, shape in enumerate(shapes, start=1):
            if stage == "cublas":
                stage_records = run_cublas_shape(
                    shape, moonpoly_binary, run_root, args.timeout, args.force
                )
            else:
                stage_records = run_cutlass_shape(
                    shape, cutlass_binary, run_root, args.timeout,
                    args.cutlass_warmup, args.cutlass_iters, args.force,
                )
            records.extend(stage_records)
            if position == 1 or position % 10 == 0 or position == len(shapes):
                failed = sum(row["status"] != "pass" for row in records)
                print(
                    f"[{stage}] {position}/{len(shapes)} "
                    f"qid={shape['qid']} shape={shape['m']}x{shape['n']}x{shape['k']} "
                    f"non_pass_records={failed}",
                    flush=True,
                )

    summary = summarize(records, len(shapes))
    failures = [row for row in records if row["status"] != "pass"]
    write_csv(outdir / "operator_raw.csv", records, RAW_FIELDS)
    write_csv(outdir / "operator_failures.csv", failures, RAW_FIELDS)
    write_json(outdir / "operator_summary.json", summary)
    figures = plot_results(records, outdir)

    manifest.update(
        {
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "gpu_after": gpu_snapshot(),
            "outputs": {
                "raw_csv": str(outdir / "operator_raw.csv"),
                "failures_csv": str(outdir / "operator_failures.csv"),
                "summary_json": str(outdir / "operator_summary.json"),
                "figures": figures,
            },
            "record_count": len(records),
            "non_pass_record_count": len(failures),
        }
    )
    write_json(outdir / "manifest.json", manifest)
    print(f"\n[done] {outdir / 'manifest.json'}", flush=True)
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
