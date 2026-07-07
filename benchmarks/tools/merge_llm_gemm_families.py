#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def merge_summaries(inputs: list[Path]) -> tuple[float, list[dict]]:
    merged: dict[tuple[int, int, int], dict] = {}
    total_score = 0.0
    for path in inputs:
        data = load_summary(path)
        model_tag = str(data.get("model_tag", path.stem))
        model_total_ms = float(data.get("total_ms", 0.0))
        for row in data.get("shapes", []):
            row_total_ms = float(row["total_ms"])
            row_share_pct = (100.0 * row_total_ms / model_total_ms
                             if model_total_ms > 0.0 else 0.0)
            total_score += row_share_pct
            key = (int(row["m"]), int(row["n"]), int(row["k"]))
            item = merged.get(key)
            if item is None:
                item = {
                    "m": int(row["m"]),
                    "n": int(row["n"]),
                    "k": int(row["k"]),
                    "count": 0,
                    "total_ms": 0.0,
                    "score_pct_sum": 0.0,
                    "num_layers": 0,
                    "models": [],
                    "sources": [],
                }
                merged[key] = item
            item["count"] += int(row.get("count", 0))
            item["total_ms"] += row_total_ms
            item["score_pct_sum"] += row_share_pct
            item["num_layers"] += int(row.get("num_layers", 0))
            item["models"].append(model_tag)
            item["sources"].append(str(path))

    rows = []
    cumulative = 0.0
    for item in merged.values():
        item["models"] = sorted(set(item["models"]))
        item["sources"] = sorted(set(item["sources"]))
        item["avg_ms"] = (item["total_ms"] / item["count"]
                          if item["count"] else 0.0)
        item["share_pct"] = (100.0 * float(item["score_pct_sum"]) / total_score
                             if total_score > 0.0 else 0.0)
        rows.append(item)
    rows.sort(key=lambda x: float(x["share_pct"]), reverse=True)
    for item in rows:
        cumulative += float(item["share_pct"])
        item["cumulative_share_pct"] = cumulative
    return total_score, rows


def main() -> None:
    p = argparse.ArgumentParser(
        description="Merge LLM GEMM shape-family summaries across models.")
    p.add_argument("inputs", nargs="+", help="Input *_shape_summary.json files")
    p.add_argument("--name", default="llm_family_merged_cublas")
    p.add_argument("--topk", type=int, default=20)
    p.add_argument("--cumulative-target", type=float, default=80.0)
    p.add_argument("--out-dir", default="benchmarks/data/llm_gemm")
    args = p.parse_args()

    input_paths = [Path(x) for x in args.inputs]
    total_score, rows = merge_summaries(input_paths)

    topk_rows = rows[:args.topk]
    cum_rows = []
    for row in rows:
        cum_rows.append(dict(row))
        if float(row["cumulative_share_pct"]) >= float(args.cumulative_target):
            break

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.name

    summary = {
        "inputs": [str(p) for p in input_paths],
        "name": prefix,
        "total_score_pct": total_score,
        "num_shapes": len(rows),
        "shapes": rows,
    }
    topk = {
        "inputs": [str(p) for p in input_paths],
        "name": prefix,
        "topk": args.topk,
        "shapes": topk_rows,
    }
    cum_target = {
        "inputs": [str(p) for p in input_paths],
        "name": prefix,
        "cumulative_target_pct": float(args.cumulative_target),
        "selected_count": len(cum_rows),
        "shapes": cum_rows,
    }

    summary_path = out_dir / f"{prefix}_shape_summary.json"
    topk_path = out_dir / f"{prefix}_topk.json"
    cum_path = out_dir / f"{prefix}_cum_target.json"
    csv_path = out_dir / f"{prefix}_shape_summary.csv"

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    topk_path.write_text(json.dumps(topk, indent=2), encoding="utf-8")
    cum_path.write_text(json.dumps(cum_target, indent=2), encoding="utf-8")

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "m",
                "n",
                "k",
                "count",
                "num_layers",
                "avg_ms",
                "total_ms",
                "share_pct",
                "cumulative_share_pct",
                "models",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "m": row["m"],
                "n": row["n"],
                "k": row["k"],
                "count": row["count"],
                "num_layers": row["num_layers"],
                "avg_ms": row["avg_ms"],
                "total_ms": row["total_ms"],
                "share_pct": row["share_pct"],
                "cumulative_share_pct": row["cumulative_share_pct"],
                "models": ";".join(row["models"]),
            })

    print(f"wrote {summary_path}")
    print(f"wrote {topk_path}")
    print(f"wrote {cum_path}")
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
