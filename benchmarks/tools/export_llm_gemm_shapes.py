#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _group_by_shape(rows: list[dict], backend: str | None) -> list[dict]:
    grouped: dict[tuple[str, int, int, int], dict] = {}
    for row in rows:
        row_backend = str(row["backend"])
        if backend is not None and row_backend != backend:
            continue
        key = (row_backend, int(row["m"]), int(row["n"]), int(row["k"]))
        item = grouped.get(key)
        if item is None:
            item = {
                "backend": row_backend,
                "m": int(row["m"]),
                "n": int(row["n"]),
                "k": int(row["k"]),
                "count": 0,
                "total_ms": 0.0,
                "max_ms": 0.0,
                "total_tflops": 0.0,
                "num_layers": 0,
                "layers": [],
            }
            grouped[key] = item
        item["count"] += int(row["count"])
        item["total_ms"] += float(row["total_ms"])
        item["max_ms"] = max(float(item["max_ms"]), float(row.get("max_ms", 0.0)))
        item["total_tflops"] += float(row.get("total_tflops", 0.0))
        item["num_layers"] += 1
        item["layers"].append(str(row.get("layer_prefix", "<unknown>")))

    out = []
    for item in grouped.values():
        item["avg_ms"] = (item["total_ms"] / item["count"]
                          if item["count"] else 0.0)
        item["avg_tflops"] = (item["total_tflops"] / item["count"]
                              if item["count"] else 0.0)
        item["layers"] = sorted(set(item["layers"]))
        out.append(item)
    out.sort(key=lambda x: float(x["total_ms"]), reverse=True)
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description="Export grouped LLM decode GEMM shape families to benchmarks/data/llm_gemm.")
    p.add_argument("profile", help="Path to vLLM linear profile JSON")
    p.add_argument("--model-tag", required=True, help="Short model tag, e.g. llama2_13b")
    p.add_argument("--backend",
                   default=None,
                   help="Optional backend filter, e.g. cublas or moonpoly")
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--cumulative-target", type=float, default=80.0)
    p.add_argument("--out-dir",
                   default="benchmarks/data/llm_gemm",
                   help="Output directory")
    args = p.parse_args()

    profile_path = Path(args.profile)
    data = json.loads(profile_path.read_text(encoding="utf-8"))
    rows = _group_by_shape(list(data.get("rows", [])), args.backend)
    total_ms = sum(float(row["total_ms"]) for row in rows)

    cumulative = 0.0
    topk_rows = []
    until_target_rows = []
    target_closed = False
    for row in rows:
        share_pct = 100.0 * float(row["total_ms"]) / total_ms if total_ms > 0.0 else 0.0
        cumulative += share_pct
        row["share_pct"] = share_pct
        row["cumulative_share_pct"] = cumulative
        if len(topk_rows) < args.topk:
            topk_rows.append(dict(row))
        if not target_closed:
            until_target_rows.append(dict(row))
            if cumulative >= float(args.cumulative_target):
                target_closed = True

    backend_tag = args.backend or "all"
    prefix = f"{args.model_tag}_{backend_tag}"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_json = out_dir / f"{prefix}_shape_summary.json"
    topk_json = out_dir / f"{prefix}_topk.json"
    target_json = out_dir / f"{prefix}_cum_target.json"
    csv_path = out_dir / f"{prefix}_shape_summary.csv"

    summary = {
        "source_profile": str(profile_path),
        "model_tag": args.model_tag,
        "backend": args.backend,
        "total_ms": total_ms,
        "num_shapes": len(rows),
        "shapes": rows,
    }
    topk = {
        "source_profile": str(profile_path),
        "model_tag": args.model_tag,
        "backend": args.backend,
        "topk": args.topk,
        "shapes": topk_rows,
    }
    target = {
        "source_profile": str(profile_path),
        "model_tag": args.model_tag,
        "backend": args.backend,
        "cumulative_target_pct": float(args.cumulative_target),
        "selected_count": len(until_target_rows),
        "shapes": until_target_rows,
    }

    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    topk_json.write_text(json.dumps(topk, indent=2), encoding="utf-8")
    target_json.write_text(json.dumps(target, indent=2), encoding="utf-8")

    fieldnames = [
        "backend",
        "m",
        "n",
        "k",
        "count",
        "num_layers",
        "avg_ms",
        "total_ms",
        "avg_tflops",
        "share_pct",
        "cumulative_share_pct",
        "layers",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out_row = dict(row)
            out_row["layers"] = ";".join(out_row["layers"])
            writer.writerow({key: out_row[key] for key in fieldnames})

    print(f"wrote {summary_json}")
    print(f"wrote {topk_json}")
    print(f"wrote {target_json}")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
