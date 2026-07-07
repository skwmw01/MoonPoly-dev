#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


def short_name(name: str) -> str:
    if "ampere_fp16_s16816gemm_fp16_256x128" in name:
        return "GEMM 256x128"
    if "ampere_fp16_s16816gemm_fp16_128x256" in name:
        return "GEMM 128x256"
    if "ampere_fp16_s16816gemm_fp16_128x128" in name:
        return "GEMM 128x128"
    if "ampere_fp16_s16816gemm_fp16_128x64" in name:
        return "GEMM 128x64"
    if "ampere_fp16_s16816gemm_fp16_64x64" in name:
        return "GEMM 64x64"
    if "act_and_mul_kernel" in name:
        return "SiLU+Mul"
    if "flash_fwd_splitkv_kernel" in name:
        return "FlashAttention"
    if "fused_add_rms_norm_kernel" in name:
        return "Fused RMSNorm"
    if "rotary_embedding_kernel" in name:
        return "Rotary Emb."
    return name[:48]


def write_latex(rows, out_path: Path):
    lines = []
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\begin{tabular}{lrrrr}")
    lines.append("\\toprule")
    lines.append("Case & Top-1 (\\%) & Top-2 (\\%) & Top-3 (\\%) & Others (\\%)\\\\")
    lines.append("\\midrule")
    for row in rows:
        case = row["case"].replace("_", "\\_")
        t1 = f"{short_name(row['top1_name'])} ({row['top1_pct']:.2f})"
        t2 = f"{short_name(row['top2_name'])} ({row['top2_pct']:.2f})"
        t3 = f"{short_name(row['top3_name'])} ({row['top3_pct']:.2f})"
        others = f"{row['others_pct']:.2f}"
        lines.append(f"{case} & {t1} & {t2} & {t3} & {others}\\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\caption{Top-3 CUDA kernels and the aggregated Others category, measured as percentages of total CUDA kernel time.}")
    lines.append("\\label{tab:top3-others-breakdown}")
    lines.append("\\end{table*}")
    out_path.write_text("\n".join(lines) + "\n")


def write_markdown(rows, out_path: Path):
    lines = []
    lines.append("# Top-3 + Others Breakdown")
    lines.append("")
    lines.append("| Case | Top-1 | Top-2 | Top-3 | Others |")
    lines.append("|---|---|---|---|---:|")
    for row in rows:
        lines.append(
            f"| {row['case']} | "
            f"{short_name(row['top1_name'])} ({row['top1_pct']:.2f}%) | "
            f"{short_name(row['top2_name'])} ({row['top2_pct']:.2f}%) | "
            f"{short_name(row['top3_name'])} ({row['top3_pct']:.2f}%) | "
            f"{row['others_pct']:.2f}% |"
        )
    out_path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-json",
        default="artifacts/benchmarks/minior-revision-taco/tables/top3_others_kernel_breakdown.json",
    )
    parser.add_argument(
        "--out-dir",
        default="artifacts/benchmarks/minior-revision-taco/tables",
    )
    args = parser.parse_args()

    input_json = Path(args.input_json)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with input_json.open() as f:
        rows = json.load(f)

    write_latex(rows, out_dir / "top3_others_breakdown_table.tex")
    write_markdown(rows, out_dir / "top3_others_breakdown_table.md")


if __name__ == "__main__":
    main()
