#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_csv_list(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def repo_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return REPO_ROOT / p


def run_cmd(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    dry_run: bool = False,
) -> subprocess.CompletedProcess[str] | None:
    shown_cwd = cwd if cwd is not None else REPO_ROOT
    print(f"[cmd cwd={shown_cwd}] {' '.join(cmd)}")
    if dry_run:
        return None
    proc = subprocess.run(
        cmd,
        cwd=str(shown_cwd),
        env=env,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if proc.returncode != 0:
        if proc.stdout:
            print(proc.stdout)
        raise subprocess.CalledProcessError(proc.returncode, cmd, output=proc.stdout)
    return proc


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True), encoding="utf-8")


def read_shapes(path: Path, max_shapes: int) -> list[tuple[int, int, int]]:
    rows = path.read_text(encoding="utf-8").splitlines()
    if not rows:
        raise ValueError(f"Empty shape file: {path}")

    first = [x.strip().lower() for x in rows[0].split(",")]
    shapes: list[tuple[int, int, int]] = []

    if {"m", "n", "k"}.issubset(set(first)):
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                shapes.append((int(row["m"]), int(row["n"]), int(row["k"])))
    else:
        for line in rows:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            parts = [x.strip() for x in line.split(",")]
            if len(parts) < 3:
                continue
            shapes.append((int(parts[0]), int(parts[1]), int(parts[2])))

    if max_shapes > 0:
        shapes = shapes[:max_shapes]
    return shapes


def geometric_mean(values: list[float]) -> float | None:
    vals = [v for v in values if v > 0.0 and math.isfinite(v)]
    if not vals:
        return None
    return math.exp(sum(math.log(v) for v in vals) / len(vals))


def summarize_operator_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_dtype: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_dtype.setdefault(str(row["DataType"]), []).append(row)

    summary: dict[str, Any] = {}
    for dtype, dtype_rows in sorted(by_dtype.items()):
        speedups = [float(r["Speedup"]) for r in dtype_rows]
        passed = [str(r["Correctness_Passed"]).upper() == "TRUE" for r in dtype_rows]
        summary[dtype] = {
            "count": len(dtype_rows),
            "correctness_passed": sum(passed),
            "correctness_total": len(passed),
            "speedup_avg": sum(speedups) / len(speedups) if speedups else None,
            "speedup_geomean": geometric_mean(speedups),
            "speedup_min": min(speedups) if speedups else None,
            "speedup_max": max(speedups) if speedups else None,
        }
    return summary


def run_operator(args: argparse.Namespace, outdir: Path, manifest: dict[str, Any]) -> None:
    stage_dir = outdir / "operator"
    stage_dir.mkdir(parents=True, exist_ok=True)
    binary = repo_path(args.operator_bin)
    shape_file = repo_path(args.shape_csv)
    shapes = read_shapes(shape_file, args.max_shapes)
    if not shapes:
        raise ValueError(f"No shapes selected from {shape_file}")

    raw_rows: list[dict[str, Any]] = []
    for idx, (m, n, k) in enumerate(shapes, start=1):
        run_dir = stage_dir / f"shape_{idx:04d}_{m}x{n}x{k}"
        run_dir.mkdir(parents=True, exist_ok=True)
        cmd = [str(binary), str(m), str(n), str(k)]
        try:
            proc = run_cmd(cmd, cwd=run_dir, dry_run=args.dry_run)
            if proc is not None:
                (run_dir / "stdout.txt").write_text(proc.stdout, encoding="utf-8")
        except subprocess.CalledProcessError as e:
            (run_dir / "stdout.txt").write_text(e.stdout or "", encoding="utf-8")
            if not args.keep_going:
                raise
            manifest["warnings"].append(f"operator shape failed: {(m, n, k)}")
            continue

        csv_path = run_dir / "moonpoly_vs_cublas_benchmark.csv"
        if args.dry_run:
            continue
        if not csv_path.exists():
            raise FileNotFoundError(f"Expected operator CSV missing: {csv_path}")
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row["shape_index"] = idx
                raw_rows.append(row)

    raw_csv = stage_dir / "operator_raw.csv"
    if raw_rows:
        fields = ["shape_index"] + [f for f in raw_rows[0].keys() if f != "shape_index"]
        with raw_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(raw_rows)
        summary = summarize_operator_rows(raw_rows)
    else:
        summary = {}

    summary_path = stage_dir / "operator_summary.json"
    write_json(summary_path, summary)
    manifest["outputs"]["operator"] = {
        "raw_csv": str(raw_csv),
        "summary_json": str(summary_path),
        "shape_csv": str(shape_file),
        "num_shapes": len(shapes),
    }


def run_predictor(args: argparse.Namespace, outdir: Path, manifest: dict[str, Any]) -> None:
    stage_dir = outdir / "predictor"
    stage_dir.mkdir(parents=True, exist_ok=True)
    raw_csv = stage_dir / "fp16_predictor_eval_raw.csv"
    if not args.run_predictor_bin:
        raw_csv = repo_path(args.predictor_raw_csv)
    cmd = [
        args.python,
        str(REPO_ROOT / "benchmarks/tools/plot_fp16_predictor_eval.py"),
        "--raw-csv",
        str(raw_csv),
        "--outdir",
        str(stage_dir),
        "--bin",
        str(repo_path(args.predictor_bin)),
        "--m-list",
        args.predictor_m_list,
        "--n-list",
        args.predictor_n_list,
        "--k-list",
        args.predictor_k_list,
        "--warmup",
        str(args.predictor_warmup),
        "--iters",
        str(args.predictor_iters),
        "--max-shapes",
        str(args.predictor_max_shapes),
        "--num-pid",
        str(args.predictor_num_pid),
    ]
    if args.run_predictor_bin:
        cmd.append("--run-bin")
    run_cmd(cmd, dry_run=args.dry_run)
    manifest["outputs"]["predictor"] = {
        "summary_json": str(stage_dir / "summary.json"),
        "detail_csv": str(stage_dir / "shape_detail.csv"),
        "raw_csv": str(raw_csv),
    }


def run_pattern3(args: argparse.Namespace, outdir: Path, manifest: dict[str, Any]) -> None:
    stage_dir = outdir / "pattern3"
    stage_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO_ROOT}:{env.get('PYTHONPATH', '')}"
    torch_lib = subprocess.run(
        [
            args.python,
            "-c",
            "import pathlib, torch; print(pathlib.Path(torch.__file__).resolve().parent / 'lib')",
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    if torch_lib.returncode == 0:
        lib_path = torch_lib.stdout.strip()
        if lib_path:
            env["LD_LIBRARY_PATH"] = f"{lib_path}:{env.get('LD_LIBRARY_PATH', '')}"
    cmd = [
        args.python,
        str(REPO_ROOT / "benchmarks/tools/profile_pattern3_splitk_ablation.py"),
        "--m",
        str(args.pattern3_m),
        "--n",
        str(args.pattern3_n),
        "--k",
        str(args.pattern3_k),
        "--mono-kind",
        args.pattern3_mono_kind,
        "--mono-pid",
        str(args.pattern3_mono_pid),
        "--mono-split-k",
        str(args.pattern3_mono_split_k),
        "--splitk-kind",
        args.pattern3_splitk_kind,
        "--splitk-pid",
        str(args.pattern3_splitk_pid),
        "--splitk",
        str(args.pattern3_split_k),
        "--warmup",
        str(args.pattern3_warmup),
        "--iters",
        str(args.pattern3_iters),
        "--out-json",
        str(stage_dir / "timing_summary.json"),
    ]
    run_cmd(cmd, env=env, dry_run=args.dry_run)
    manifest["outputs"]["pattern3"] = {
        "timing_summary": str(stage_dir / "timing_summary.json"),
        "shape": [args.pattern3_m, args.pattern3_n, args.pattern3_k],
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run the non-vLLM Online reproduction core: operator-level GEMM "
            "benchmark, FP16 predictor accuracy, and Pattern-3 timing."
        )
    )
    p.add_argument("--outdir", default="artifacts/paper_core")
    p.add_argument(
        "--stages",
        default="predictor,operator,pattern3",
        help="Comma-separated stages: predictor,operator,pattern3",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--keep-going", action="store_true")
    p.add_argument("--python", default=sys.executable,
                   help="Python interpreter used by subprocess stages.")

    p.add_argument("--operator-bin", default="build/benchmarks/cpp/moonpoly_benchmark")
    p.add_argument("--shape-csv", default="benchmarks/data/llm_gemm/llm_gemm_bench_select.csv")
    p.add_argument("--max-shapes", type=int, default=3,
                   help="0 means all shapes. Default is intentionally small.")

    p.add_argument("--predictor-bin", default="build/benchmarks/cpp/moonpoly_profile_fp16_predictor")
    p.add_argument("--predictor-raw-csv", default="benchmarks/data/fp16_predictor/fp16_predictor_eval_raw.csv")
    p.add_argument("--run-predictor-bin", action="store_true",
                   help="Regenerate raw predictor CSV by running the C++ binary.")
    p.add_argument("--predictor-m-list", default="16,64,256")
    p.add_argument("--predictor-n-list", default="1024,4096")
    p.add_argument("--predictor-k-list", default="1024,4096")
    p.add_argument("--predictor-warmup", type=int, default=5)
    p.add_argument("--predictor-iters", type=int, default=20)
    p.add_argument("--predictor-max-shapes", type=int, default=0)
    p.add_argument("--predictor-num-pid", type=int, default=41)

    p.add_argument("--pattern3-m", type=int, default=8)
    p.add_argument("--pattern3-n", type=int, default=10240)
    p.add_argument("--pattern3-k", type=int, default=5120)
    p.add_argument("--pattern3-mono-kind", choices=["forced", "generic"], default="forced")
    p.add_argument("--pattern3-mono-pid", type=int, default=25)
    p.add_argument("--pattern3-mono-split-k", type=int, default=1)
    p.add_argument("--pattern3-splitk-kind", choices=["forced", "generic"], default="forced")
    p.add_argument("--pattern3-splitk-pid", type=int, default=286)
    p.add_argument("--pattern3-split-k", type=int, default=4)
    p.add_argument("--pattern3-warmup", type=int, default=100)
    p.add_argument("--pattern3-iters", type=int, default=1000)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    outdir = repo_path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "repo_root": str(REPO_ROOT),
        "outdir": str(outdir),
        "stages": parse_csv_list(args.stages),
        "outputs": {},
        "warnings": [],
    }

    stage_handlers = {
        "operator": run_operator,
        "predictor": run_predictor,
        "pattern3": run_pattern3,
    }
    for stage in manifest["stages"]:
        if stage not in stage_handlers:
            raise ValueError(f"Unknown stage: {stage}")
        print(f"\n===== stage: {stage} =====")
        stage_handlers[stage](args, outdir, manifest)
        write_json(outdir / "manifest.json", manifest)

    write_json(outdir / "manifest.json", manifest)
    print(f"\n[done] manifest: {outdir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
