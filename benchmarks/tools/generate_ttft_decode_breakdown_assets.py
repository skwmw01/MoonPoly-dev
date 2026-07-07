#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


def load_rows(path: Path):
    with path.open() as f:
        return json.load(f)


def write_latex_table(rows, out_path: Path):
    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\small")
    lines.append("\\begin{tabular}{lrrrr}")
    lines.append("\\toprule")
    lines.append("Case & Request (ms) & Kernel (ms) & GEMM (\\%) & Attn. (\\%)\\\\")
    lines.append("\\midrule")
    for row in rows:
        case = row["case"].replace("_", "\\_")
        lines.append(
            f"{case} & "
            f"{row['request_ms']:.2f} & "
            f"{row['kernel_total_ms']:.2f} & "
            f"{row['gemm_kernel_share'] * 100:.2f} & "
            f"{row['attention_kernel_share'] * 100:.2f}\\\\"
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\caption{TTFT and decode-oriented breakdown for Llama-2-13B and Qwen3-32B under batch size 1. Decode uses a prefix-cache-assisted approximation.}")
    lines.append("\\label{tab:ttft-decode-breakdown}")
    lines.append("\\end{table}")
    out_path.write_text("\n".join(lines) + "\n")


def write_markdown_table(rows, out_path: Path):
    lines = []
    lines.append("# TTFT/Decode Breakdown Table")
    lines.append("")
    lines.append("| Case | Request ms | Kernel ms | GEMM % | Attention % | Top kernel |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for row in rows:
        lines.append(
            f"| {row['case']} | "
            f"{row['request_ms']:.3f} | "
            f"{row['kernel_total_ms']:.3f} | "
            f"{row['gemm_kernel_share'] * 100:.2f}% | "
            f"{row['attention_kernel_share'] * 100:.2f}% | "
            f"{row['top_kernel_name']} |"
        )
    lines.append("")
    lines.append("Notes:")
    lines.append("- `*_decode_*` cases are prefix-cache-assisted decode approximations.")
    lines.append("- Percentages are computed over CUDA kernel time in the trace.")
    out_path.write_text("\n".join(lines) + "\n")


def write_plot(rows, out_path: Path):
    import matplotlib.pyplot as plt

    labels = [row["case"] for row in rows]
    gemm = [row["gemm_kernel_share"] * 100 for row in rows]
    attn = [row["attention_kernel_share"] * 100 for row in rows]
    other = [max(0.0, 100.0 - g - a) for g, a in zip(gemm, attn)]

    fig, ax = plt.subplots(figsize=(10, 4.8))
    x = range(len(labels))
    ax.bar(x, gemm, label="GEMM-family", color="#1f4e79")
    ax.bar(x, attn, bottom=gemm, label="Attention", color="#c0504d")
    ax.bar(
        x,
        other,
        bottom=[g + a for g, a in zip(gemm, attn)],
        label="Other kernels",
        color="#9bbb59",
    )
    ax.set_ylabel("Share of CUDA Kernel Time (%)")
    ax.set_ylim(0, 100)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-json",
        default="artifacts/benchmarks/minior-revision-taco/comparison/ttft_decode_breakdown_comparison.json",
    )
    parser.add_argument(
        "--out-dir",
        default="artifacts/benchmarks/minior-revision-taco",
    )
    args = parser.parse_args()

    input_json = Path(args.input_json)
    out_dir = Path(args.out_dir)
    rows = load_rows(input_json)

    tables_dir = out_dir / "tables"
    figures_dir = out_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    write_latex_table(rows, tables_dir / "ttft_decode_breakdown_table.tex")
    write_markdown_table(rows, tables_dir / "ttft_decode_breakdown_table.md")
    write_plot(rows, figures_dir / "ttft_decode_breakdown_stacked_bar.png")


if __name__ == "__main__":
    main()
