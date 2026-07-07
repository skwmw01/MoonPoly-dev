#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _group_rows(rows, group_by: str):
    grouped = {}
    for row in rows:
        if group_by == "shape":
            key = (row["backend"], row["m"], row["n"], row["k"])
        else:
            key = (row["backend"], row["layer_prefix"], row["m"], row["n"],
                   row["k"])

        item = grouped.get(key)
        if item is None:
            item = {
                "backend": row["backend"],
                "layer_prefix": row["layer_prefix"]
                if group_by == "layer" else "<merged>",
                "m": row["m"],
                "n": row["n"],
                "k": row["k"],
                "count": 0,
                "total_ms": 0.0,
                "max_ms": 0.0,
                "total_tflops": 0.0,
                "num_layers": 0,
            }
            grouped[key] = item
        item["count"] += int(row["count"])
        item["total_ms"] += float(row["total_ms"])
        item["max_ms"] = max(float(item["max_ms"]), float(row.get("max_ms", 0.0)))
        item["total_tflops"] += float(row.get("total_tflops", 0.0))
        if group_by == "shape":
            item["num_layers"] += 1

    out = []
    for item in grouped.values():
        item["avg_ms"] = (item["total_ms"] / item["count"]
                          if item["count"] else 0.0)
        item["avg_tflops"] = (item["total_tflops"] / item["count"]
                              if item["count"] else 0.0)
        out.append(item)
    out.sort(key=lambda x: float(x["total_ms"]), reverse=True)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Summarize vLLM linear profile JSON.")
    p.add_argument("profile", help="Path to profile JSON")
    p.add_argument("--topk", type=int, default=20)
    p.add_argument("--group-by",
                   choices=["shape", "layer"],
                   default="shape",
                   help="Aggregate by shape family or keep per-layer rows.")
    p.add_argument("--show-cumulative",
                   action="store_true",
                   help="Show cumulative share.")
    args = p.parse_args()

    data = json.loads(Path(args.profile).read_text(encoding="utf-8"))
    rows = _group_rows(list(data.get("rows", [])), args.group_by)
    total_ms = float(data.get("total_ms", 0.0))

    print(f"profile={args.profile}")
    print(f"group_by={args.group_by}")
    print(f"rows={len(rows)} total_ms={total_ms:.4f}")
    print("")

    cumulative = 0.0
    for row in rows[:args.topk]:
        share = 100.0 * float(row["total_ms"]) / total_ms if total_ms > 0.0 else 0.0
        cumulative += share
        print(
            f"{row['backend']:8s} "
            f"shape=({row['m']},{row['n']},{row['k']}) "
            f"count={row['count']:6d} "
            f"avg_ms={float(row['avg_ms']):8.4f} "
            f"total_ms={float(row['total_ms']):9.4f} "
            f"share={share:6.2f}% "
            + (f"cum={cumulative:6.2f}% " if args.show_cumulative else "")
            + (
                f"layers={row['num_layers']:3d}"
                if args.group_by == "shape" else f"layer={row['layer_prefix']}"
            )
        )


if __name__ == "__main__":
    main()
