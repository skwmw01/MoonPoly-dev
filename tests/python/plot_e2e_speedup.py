#!/usr/bin/env python3
"""
Plot TTFT/TPOT E2E speedup and optional nsys breakdown.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _bin_ttft(samples: List[Dict[str, Any]], bins: int = 20) -> Tuple[List[float], List[float]]:
    ok = [s for s in samples if s.get("ok")]
    if not ok:
        return [], []
    xs = [float(s["target_prompt_len"]) for s in ok]
    ys = [float(s["ttft_ms"]) for s in ok]
    x_min, x_max = min(xs), max(xs)
    if x_max <= x_min:
        return [x_min], [sum(ys) / len(ys)]
    step = (x_max - x_min) / bins
    bx, by = [], []
    for i in range(bins):
        lo = x_min + i * step
        hi = x_min + (i + 1) * step
        vals = [y for x, y in zip(xs, ys) if lo <= x < hi]
        if vals:
            vals_sorted = sorted(vals)
            med = vals_sorted[len(vals_sorted) // 2]
            bx.append((lo + hi) * 0.5)
            by.append(med)
    return bx, by


def _plot_ttft(data: Dict[str, Any], out_png: Path) -> None:
    c_samples = data["backends"]["cublas"]["ttft"]["samples"]
    m_samples = data["backends"]["moonpoly"]["ttft"]["samples"]

    cx, cy = _bin_ttft(c_samples, bins=20)
    mx, my = _bin_ttft(m_samples, bins=20)

    plt.figure(figsize=(7, 4))
    plt.plot(cx, cy, label="cuBLAS", linewidth=2)
    plt.plot(mx, my, label="MoonPoly", linewidth=2)
    plt.xlabel("Prompt Length (tokens)")
    plt.ylabel("TTFT (ms, median in bins)")
    plt.title("TTFT vs Prompt Length")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def _plot_tpot(data: Dict[str, Any], out_png: Path) -> None:
    c_tpot = data["backends"]["cublas"]["tpot"]
    m_tpot = data["backends"]["moonpoly"]["tpot"]
    bs = sorted(int(k) for k in c_tpot.keys())
    c_mean = [float(c_tpot[str(b)]["summary"]["mean"]) for b in bs]
    m_mean = [float(m_tpot[str(b)]["summary"]["mean"]) for b in bs]

    plt.figure(figsize=(7, 4))
    plt.plot(bs, c_mean, marker="o", label="cuBLAS")
    plt.plot(bs, m_mean, marker="o", label="MoonPoly")
    plt.xlabel("Batch Size")
    plt.ylabel("TPOT (ms/token)")
    plt.title("TPOT vs Concurrency")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def _plot_breakdown(breakdown_csv: Path, out_png: Path) -> None:
    labels, gemm, attn, other = [], [], [], []
    with breakdown_csv.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            labels.append(row["label"])
            gemm.append(float(row["gemm_share"]) * 100.0)
            attn.append(float(row["attention_share"]) * 100.0)
            other.append(float(row["other_share"]) * 100.0)

    x = list(range(len(labels)))
    plt.figure(figsize=(9, 4.2))
    plt.bar(x, gemm, label="GEMM")
    plt.bar(x, attn, bottom=gemm, label="Attention")
    bottoms = [a + b for a, b in zip(gemm, attn)]
    plt.bar(x, other, bottom=bottoms, label="Other")
    plt.xticks(x, labels, rotation=30, ha="right")
    plt.ylabel("GPU Kernel Time Share (%)")
    plt.title("Kernel Breakdown from nsys")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def _write_summary(data: Dict[str, Any], summary_path: Path, breakdown_csv: Path | None) -> None:
    ttft_speed = float(data["speedup"]["ttft_mean_speedup"])
    lines = [
        "# MoonPoly E2E Summary",
        "",
        f"- TTFT mean speedup (cuBLAS / MoonPoly): {ttft_speed:.4f}x",
        "- TPOT mean speedup by batch size:",
    ]
    for k, v in data["speedup"]["tpot_mean_speedup_by_bs"].items():
        lines.append(f"  - bs={k}: {float(v):.4f}x")

    if breakdown_csv is not None and breakdown_csv.exists():
        lines.append("")
        lines.append(f"- Breakdown source: `{breakdown_csv}`")
        with breakdown_csv.open("r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                lines.append(
                    f"  - {row['label']}: GEMM={float(row['gemm_share']) * 100:.2f}%, "
                    f"Attention={float(row['attention_share']) * 100:.2f}%, "
                    f"Other={float(row['other_share']) * 100:.2f}%"
                )

    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Plot E2E TTFT/TPOT + breakdown.")
    p.add_argument("--e2e-json", required=True)
    p.add_argument("--breakdown-csv", default=None)
    p.add_argument("--out-dir", default="artifacts/e2e_figures")
    args = p.parse_args()

    e2e = _load_json(Path(args.e2e_json))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ttft_png = out_dir / "ttft_vs_len.png"
    tpot_png = out_dir / "tpot_vs_bs.png"
    breakdown_png = out_dir / "gpu_breakdown.png"
    summary_md = out_dir / "summary.md"

    _plot_ttft(e2e, ttft_png)
    _plot_tpot(e2e, tpot_png)

    breakdown_path = Path(args.breakdown_csv) if args.breakdown_csv else None
    if breakdown_path is not None and breakdown_path.exists():
        _plot_breakdown(breakdown_path, breakdown_png)

    _write_summary(e2e, summary_md, breakdown_path)
    print(f"wrote: {ttft_png}")
    print(f"wrote: {tpot_png}")
    if breakdown_path is not None and breakdown_path.exists():
        print(f"wrote: {breakdown_png}")
    print(f"wrote: {summary_md}")


if __name__ == "__main__":
    main()
