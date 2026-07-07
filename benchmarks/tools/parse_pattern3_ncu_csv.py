#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import io
import json
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Parse Nsight Compute CSV and extract dominant-kernel metrics for Pattern-3 ablation."
    )
    p.add_argument("csv_path")
    p.add_argument("--out-json", default="")
    return p.parse_args()


def to_float(text: str) -> float:
    if text is None:
        return math.nan
    text = text.strip().replace(",", "")
    if not text or text == "N/A":
        return math.nan
    return float(text)


def load_csv_rows(csv_path: str) -> list[dict[str, str]]:
    lines = Path(csv_path).read_text(encoding="utf-8").splitlines()
    header_idx = None
    for idx, line in enumerate(lines):
        if '"Kernel Name"' in line and line.lstrip().startswith('"ID"'):
            header_idx = idx
            break
    if header_idx is None:
        raise RuntimeError("Failed to locate Nsight Compute CSV header")
    payload = "\n".join(lines[header_idx:])
    reader = csv.DictReader(io.StringIO(payload))
    return [row for row in reader if row.get("Kernel Name") or row.get("Kernel Name Base")]


def is_gemm_like_kernel(kernel: str) -> bool:
    name = kernel.lower()
    positive = (
        "gemm" in name
        or "cublaslt" in name
        or "splitkreduce" in name
        or "cutlass" in name
    )
    negative = any(
        token in name
        for token in (
            "distribution_",
            "vectorized_elementwise_kernel",
            "reduce_kernel",
            "unrolled_elementwise_kernel",
            "fill_kernel",
            "copy_kernel",
        )
    )
    return positive and not negative


def summarize_wide_rows(rows: list[dict[str, str]]) -> dict[str, object]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        kernel = row.get("Kernel Name") or row.get("Kernel Name Base")
        if not kernel:
            continue
        grouped.setdefault(kernel, []).append(row)

    def row_duration(row: dict[str, str]) -> float:
        for key in ("gpu__time_duration.sum", "gpu__time_duration.avg", "gpu__time_duration"):
            value = to_float(row.get(key))
            if not math.isnan(value):
                return value
        return -1.0

    def weighted_metric(group_rows: list[dict[str, str]], key: str) -> float:
        weighted_sum = 0.0
        weight_sum = 0.0
        for row in group_rows:
            duration = row_duration(row)
            value = to_float(row.get(key))
            if duration <= 0.0 or math.isnan(value):
                continue
            weighted_sum += value * duration
            weight_sum += duration
        if weight_sum == 0.0:
            return math.nan
        return weighted_sum / weight_sum

    filtered = {
        kernel: group_rows
        for kernel, group_rows in grouped.items()
        if is_gemm_like_kernel(kernel)
    }
    if not filtered:
        filtered = grouped

    best_kernel = ""
    best_rows: list[dict[str, str]] = []
    best_total_duration = -1.0
    for kernel, group_rows in filtered.items():
        total_duration = sum(max(row_duration(row), 0.0) for row in group_rows)
        if total_duration > best_total_duration:
            best_kernel = kernel
            best_rows = group_rows
            best_total_duration = total_duration

    warps_per_subp = weighted_metric(best_rows, "smsp__warps_active.avg.per_cycle_active")
    active_warps_per_sm = warps_per_subp * 4.0 if not math.isnan(warps_per_subp) else math.nan
    occupancy_pct = active_warps_per_sm / 64.0 * 100.0 if not math.isnan(active_warps_per_sm) else math.nan

    return {
        "kernel_name": best_kernel,
        "sm_efficiency_pct": weighted_metric(
            best_rows, "sm__throughput.avg.pct_of_peak_sustained_elapsed"
        ),
        "active_warps_per_sm": active_warps_per_sm,
        "occupancy_pct": occupancy_pct,
        "elapsed_cycles_sm": weighted_metric(best_rows, "sm__cycles_elapsed.avg"),
        "dominant_kernel_duration_ns": best_total_duration,
        "candidate_kernel_names": sorted(filtered.keys()),
        "raw_metrics": best_rows[0] if best_rows else {},
    }


def summarize_tall_rows(rows: list[dict[str, str]]) -> dict[str, object]:
    duration_keys = [
        "Metric Value",
        "gpu__time_duration.sum",
    ]

    metric_rows: dict[str, dict[str, str]] = {}
    for row in rows:
        kernel = row.get("Kernel Name") or row.get("Kernel Name Base")
        metric = row.get("Metric Name")
        value = row.get("Metric Value")
        if not kernel or not metric or value is None:
            continue
        metric_rows.setdefault(kernel, {})[metric] = value

    def kernel_duration(metrics: dict[str, str]) -> float:
        for key in ("gpu__time_duration.sum", "gpu__time_duration.avg", "gpu__time_duration"):
            if key in metrics:
                return to_float(metrics[key])
        return -1.0

    best_kernel, best_metrics = max(metric_rows.items(), key=lambda kv: kernel_duration(kv[1]))

    sm_eff = to_float(best_metrics.get("sm__throughput.avg.pct_of_peak_sustained_elapsed", "nan"))
    warps_per_subp = to_float(best_metrics.get("smsp__warps_active.avg.per_cycle_active", "nan"))
    active_warps_per_sm = warps_per_subp * 4.0 if not math.isnan(warps_per_subp) else math.nan
    occupancy_pct = active_warps_per_sm / 64.0 * 100.0 if not math.isnan(active_warps_per_sm) else math.nan
    cycles_sm = to_float(best_metrics.get("sm__cycles_elapsed.avg", "nan"))
    duration_ns = kernel_duration(best_metrics)

    return {
        "kernel_name": best_kernel,
        "sm_efficiency_pct": sm_eff,
        "active_warps_per_sm": active_warps_per_sm,
        "occupancy_pct": occupancy_pct,
        "elapsed_cycles_sm": cycles_sm,
        "dominant_kernel_duration_ns": duration_ns,
        "raw_metrics": best_metrics,
    }


def main() -> None:
    args = parse_args()
    rows = load_csv_rows(args.csv_path)

    if not rows:
        raise RuntimeError("No kernel rows found in NCU CSV")

    if rows[0].get("Metric Name"):
        summary = summarize_tall_rows(rows)
    else:
        summary = summarize_wide_rows(rows)

    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
