#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


@dataclass
class Row:
    m: int
    n: int
    k: int
    pid: int
    valid: bool
    runtime_ok: bool
    alignment: int
    pred_cost: float
    measured_ms: float
    predicted_pid: int


def parse_float(x: str) -> float:
    if x.lower() == "nan":
        return float("nan")
    return float(x)


def read_rows(path: Path) -> List[Row]:
    rows: List[Row] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                Row(
                    m=int(r["m"]),
                    n=int(r["n"]),
                    k=int(r["k"]),
                    pid=int(r["pid"]),
                    valid=(int(r["valid"]) == 1),
                    runtime_ok=(int(r.get("runtime_ok", "0")) == 1),
                    alignment=int(r["alignment"]),
                    pred_cost=parse_float(r["pred_cost"]),
                    measured_ms=parse_float(r["measured_ms"]),
                    predicted_pid=int(r["predicted_pid"]),
                )
            )
    return rows


def parse_int_list(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def rank_map(pairs: List[Tuple[int, float]], asc: bool = True) -> Dict[int, int]:
    pairs_sorted = sorted(pairs, key=lambda x: x[1], reverse=not asc)
    return {pid: idx + 1 for idx, (pid, _) in enumerate(pairs_sorted)}


def spearman_rho(true_rank: Dict[int, int], pred_rank: Dict[int, int]) -> float:
    pids = [pid for pid in true_rank.keys() if pid in pred_rank]
    n = len(pids)
    if n < 2:
        return float("nan")
    d2 = 0.0
    for pid in pids:
        d = true_rank[pid] - pred_rank[pid]
        d2 += d * d
    return 1.0 - (6.0 * d2) / (n * (n * n - 1))


def kendall_tau(true_rank: Dict[int, int], pred_rank: Dict[int, int]) -> float:
    pids = [pid for pid in true_rank.keys() if pid in pred_rank]
    n = len(pids)
    if n < 2:
        return float("nan")

    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            pi = pids[i]
            pj = pids[j]
            dt = true_rank[pi] - true_rank[pj]
            dp = pred_rank[pi] - pred_rank[pj]
            s = dt * dp
            if s > 0:
                concordant += 1
            elif s < 0:
                discordant += 1

    denom = concordant + discordant
    if denom == 0:
        return float("nan")
    return (concordant - discordant) / denom


def percentile(values: List[float], q: float) -> float:
    if not values:
        return float("nan")
    vals = sorted(values)
    pos = (len(vals) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def maybe_run_binary(args: argparse.Namespace) -> None:
    if not args.run_bin:
        return

    raw_csv: Path = args.raw_csv
    raw_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(args.outdir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    m_vals = parse_int_list(args.m_list)
    n_vals = parse_int_list(args.n_list)
    k_vals = parse_int_list(args.k_list)
    shapes = [(m, n, k) for m in m_vals for n in n_vals for k in k_vals]
    if args.max_shapes > 0:
        shapes = shapes[: args.max_shapes]

    header = [
        "m",
        "n",
        "k",
        "pid",
        "valid",
        "runtime_ok",
        "alignment",
        "pred_cost",
        "measured_ms",
        "predicted_pid",
    ]
    with raw_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()

        for si, (m, n, k) in enumerate(shapes, start=1):
            tmp_meta = tmp_dir / "_tmp_meta.csv"
            meta_cmd = [
                str(Path(args.bin)),
                "--csv",
                str(tmp_meta),
                "--m-list",
                str(m),
                "--n-list",
                str(n),
                "--k-list",
                str(k),
                "--warmup",
                str(args.warmup),
                "--iters",
                str(args.iters),
                "--max-shapes",
                "1",
                "--no-measure",
            ]
            if args.verbose:
                print("[meta]", " ".join(meta_cmd))

            subprocess.run(meta_cmd, check=True)
            meta_rows = read_rows(tmp_meta)
            by_pid: Dict[int, Row] = {r.pid: r for r in meta_rows}
            predicted_pid = meta_rows[0].predicted_pid if meta_rows else 0

            for pid in range(args.num_pid):
                meta = by_pid.get(pid)
                if meta is None:
                    writer.writerow(
                        {
                            "m": m,
                            "n": n,
                            "k": k,
                            "pid": pid,
                            "valid": 0,
                            "runtime_ok": 0,
                            "alignment": 1,
                            "pred_cost": "nan",
                            "measured_ms": "nan",
                            "predicted_pid": predicted_pid,
                        }
                    )
                    continue

                measured_ms = float("nan")
                runtime_ok = 0
                valid = 0

                if meta.valid:
                    tmp_meas = tmp_dir / "_tmp_meas.csv"
                    meas_cmd = [
                        str(Path(args.bin)),
                        "--csv",
                        str(tmp_meas),
                        "--m-list",
                        str(m),
                        "--n-list",
                        str(n),
                        "--k-list",
                        str(k),
                        "--warmup",
                        str(args.warmup),
                        "--iters",
                        str(args.iters),
                        "--max-shapes",
                        "1",
                        "--pid-list",
                        str(pid),
                    ]
                    if args.verbose:
                        print("[meas]", " ".join(meas_cmd))

                    p = subprocess.run(meas_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                    if p.returncode == 0:
                        meas_rows = read_rows(tmp_meas)
                        if meas_rows and meas_rows[0].runtime_ok and not math.isnan(meas_rows[0].measured_ms):
                            measured_ms = meas_rows[0].measured_ms
                            runtime_ok = 1
                            valid = 1

                writer.writerow(
                    {
                        "m": m,
                        "n": n,
                        "k": k,
                        "pid": pid,
                        "valid": valid,
                        "runtime_ok": runtime_ok,
                        "alignment": meta.alignment,
                        "pred_cost": ("nan" if math.isnan(meta.pred_cost) else f"{meta.pred_cost:.8f}"),
                        "measured_ms": ("nan" if math.isnan(measured_ms) else f"{measured_ms:.8f}"),
                        "predicted_pid": predicted_pid,
                    }
                )

            if args.verbose:
                print(f"[shape {si}/{len(shapes)}] done ({m},{n},{k})")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Plot FP16 micro-kernel predictor accuracy: LoP CDF + Top-k hit-rate, "
            "and emit summary stats."
        )
    )
    parser.add_argument("--raw-csv", default="benchmarks/data/fp16_predictor/fp16_predictor_eval_raw.csv")
    parser.add_argument("--outdir", default="artifacts/benchmarks/fp16_predictor_eval")
    parser.add_argument("--run-bin", action="store_true")
    parser.add_argument("--bin", default="build/benchmarks/cpp/moonpoly_profile_fp16_predictor")
    parser.add_argument("--m-list", default="16,64,256")
    parser.add_argument("--n-list", default="1024,4096")
    parser.add_argument("--k-list", default="1024,4096")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--max-shapes", type=int, default=0)
    parser.add_argument("--num-pid", type=int, default=41)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    args.raw_csv = Path(args.raw_csv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    maybe_run_binary(args)

    if not args.raw_csv.exists():
        raise FileNotFoundError(f"Raw CSV not found: {args.raw_csv}")

    rows = read_rows(args.raw_csv)
    grouped: Dict[Tuple[int, int, int], List[Row]] = defaultdict(list)
    for r in rows:
        grouped[(r.m, r.n, r.k)].append(r)

    lops: List[float] = []
    top1 = 0
    top3 = 0
    top5 = 0
    valid_shapes = 0
    spearmans: List[float] = []
    kendalls: List[float] = []

    detail_rows = []

    for shape, rs in sorted(grouped.items()):
        valid = [r for r in rs if r.valid and not math.isnan(r.measured_ms) and not math.isnan(r.pred_cost)]
        if not valid:
            continue

        true_rank = rank_map([(r.pid, r.measured_ms) for r in valid], asc=True)
        pred_rank = rank_map([(r.pid, r.pred_cost) for r in valid], asc=True)

        best_pid = min(valid, key=lambda r: r.measured_ms).pid
        best_ms = min(r.measured_ms for r in valid)

        pred_pid = valid[0].predicted_pid
        pred_row = next((r for r in valid if r.pid == pred_pid), None)
        if pred_row is None:
            continue

        pred_ms = pred_row.measured_ms
        lop = (pred_ms - best_ms) / best_ms if best_ms > 0 else float("nan")
        if not math.isnan(lop):
            lops.append(lop)

        tr = true_rank.get(pred_pid, 10**9)
        top1 += 1 if tr <= 1 else 0
        top3 += 1 if tr <= 3 else 0
        top5 += 1 if tr <= 5 else 0

        rho = spearman_rho(true_rank, pred_rank)
        tau = kendall_tau(true_rank, pred_rank)
        if not math.isnan(rho):
            spearmans.append(rho)
        if not math.isnan(tau):
            kendalls.append(tau)

        valid_shapes += 1
        detail_rows.append(
            {
                "m": shape[0],
                "n": shape[1],
                "k": shape[2],
                "best_pid": best_pid,
                "pred_pid": pred_pid,
                "best_ms": best_ms,
                "pred_ms": pred_ms,
                "lop": lop,
                "pred_true_rank": tr,
                "spearman_rho": rho,
                "kendall_tau": tau,
            }
        )

    if valid_shapes == 0:
        raise RuntimeError("No valid shapes found in raw CSV.")

    lop_pct = [x * 100.0 for x in lops]
    lop_sorted = sorted(lop_pct)
    cdf_y = [(i + 1) / len(lop_sorted) for i in range(len(lop_sorted))]

    topk_rates = [top1 / valid_shapes, top3 / valid_shapes, top5 / valid_shapes]

    fig_lop, ax_lop = plt.subplots(1, 1, figsize=(6, 4.5))
    ax_lop.plot(lop_sorted, cdf_y, linewidth=2)
    ax_lop.set_xlabel("LoP (%)")
    ax_lop.set_ylabel("CDF")
    ax_lop.set_title("FP16 Predictor LoP CDF")
    ax_lop.grid(True, alpha=0.3)
    fig_lop.tight_layout()
    lop_fig_path = outdir / "fp16_predictor_lop_cdf.png"
    fig_lop.savefig(lop_fig_path, dpi=180)

    ks = ["Top-1", "Top-3", "Top-5"]
    fig_topk, ax_topk = plt.subplots(1, 1, figsize=(6, 4.5))
    ax_topk.bar(ks, [x * 100.0 for x in topk_rates])
    ax_topk.set_ylim(0, 100)
    ax_topk.set_ylabel("Hit Rate (%)")
    ax_topk.set_title("FP16 Predictor Rank Accuracy")
    ax_topk.grid(True, axis="y", alpha=0.3)
    fig_topk.tight_layout()
    topk_fig_path = outdir / "fp16_predictor_topk_hit.png"
    fig_topk.savefig(topk_fig_path, dpi=180)

    summary = {
        "num_shapes": valid_shapes,
        "lop_percent": {
            "mean": statistics.mean(lop_pct) if lop_pct else float("nan"),
            "median": percentile(lop_pct, 0.5),
            "p90": percentile(lop_pct, 0.9),
            "p99": percentile(lop_pct, 0.99),
            "max": max(lop_pct) if lop_pct else float("nan"),
        },
        "topk_hit_rate": {
            "top1": topk_rates[0],
            "top3": topk_rates[1],
            "top5": topk_rates[2],
        },
        "rank_correlation": {
            "spearman_rho_mean": statistics.mean(spearmans) if spearmans else float("nan"),
            "kendall_tau_mean": statistics.mean(kendalls) if kendalls else float("nan"),
        },
        "raw_csv": str(args.raw_csv),
        "figures": {
            "lop_cdf": str(lop_fig_path),
            "topk_hit": str(topk_fig_path),
        },
    }

    with (outdir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=True)

    with (outdir / "shape_detail.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "m",
                "n",
                "k",
                "best_pid",
                "pred_pid",
                "best_ms",
                "pred_ms",
                "lop",
                "pred_true_rank",
                "spearman_rho",
                "kendall_tau",
            ],
        )
        writer.writeheader()
        writer.writerows(detail_rows)

    report = outdir / "summary.txt"
    with report.open("w", encoding="utf-8") as f:
        f.write("FP16 Micro-kernel Predictor Accuracy Summary\n")
        f.write(f"num_shapes={valid_shapes}\n")
        f.write(f"LoP median(%)={summary['lop_percent']['median']:.4f}\n")
        f.write(f"LoP p90(%)={summary['lop_percent']['p90']:.4f}\n")
        f.write(f"LoP p99(%)={summary['lop_percent']['p99']:.4f}\n")
        f.write(f"Top-1 hit={summary['topk_hit_rate']['top1']*100.0:.2f}%\n")
        f.write(f"Top-3 hit={summary['topk_hit_rate']['top3']*100.0:.2f}%\n")
        f.write(f"Top-5 hit={summary['topk_hit_rate']['top5']*100.0:.2f}%\n")
        f.write(
            f"Mean Spearman rho={summary['rank_correlation']['spearman_rho_mean']:.4f}\n"
        )
        f.write(
            f"Mean Kendall tau={summary['rank_correlation']['kendall_tau_mean']:.4f}\n"
        )

    print(f"Saved figure : {lop_fig_path}")
    print(f"Saved figure : {topk_fig_path}")
    print(f"Saved summary: {outdir / 'summary.json'}")
    print(f"Saved report : {report}")
    print(f"Saved detail : {outdir / 'shape_detail.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
