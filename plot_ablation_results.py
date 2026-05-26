import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

VARIANT_ORDER = [
    "full",
    "no_phys_stat",
    "avg_pool",
    "no_pos",
    "no_attn",
    "no_dwconv",
    "no_gate",
    "simple_patch",
]


def save_fig(fig, path_base: Path):
    fig.savefig(path_base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(path_base.with_suffix(".pdf"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def select_metric_columns(mode: str):
    mode = mode.lower()
    if mode == "raw":
        return "raw_RMSE", "raw_MAE"
    if mode == "linear":
        return "linear_RMSE", "linear_MAE"
    if mode == "ridge":
        return "ridge_RMSE", "ridge_MAE"
    raise ValueError(f"Unsupported metric mode: {mode}")


def main(args):
    summary_path = Path(args.summary_csv)
    if not summary_path.exists():
        raise FileNotFoundError(f"summary csv not found: {summary_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
    })

    df = pd.read_csv(summary_path)
    rmse_col, mae_col = select_metric_columns(args.metric_mode)

    variant_keep = VARIANT_ORDER if args.include_extra_variants else VARIANT_ORDER[:6]
    df = df[df["variant"].isin(variant_keep)].copy()
    df["variant"] = pd.Categorical(df["variant"], categories=variant_keep, ordered=True)
    df = df.sort_values("variant")

    # measured RMSE
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(df["variant"].astype(str), df[rmse_col], color="#4C78A8")
    ax.set_xlabel("Model variant")
    ax.set_ylabel("RMSE (mm h$^{-1}$)")
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=25, ha="right")
    save_fig(fig, out_dir / "ablation_measured_rmse")

    # measured MAE
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(df["variant"].astype(str), df[mae_col], color="#F58518")
    ax.set_xlabel("Model variant")
    ax.set_ylabel("MAE (mm h$^{-1}$)")
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=25, ha="right")
    save_fig(fig, out_dir / "ablation_measured_mae")

    # synthetic vs measured RMSE
    x = np.arange(len(df))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(x - w / 2, df["synthetic_RMSE"], width=w, label="Synthetic RMSE", color="#54A24B")
    ax.bar(x + w / 2, df[rmse_col], width=w, label=f"Measured RMSE ({args.metric_mode})", color="#E45756")
    ax.set_xticks(x)
    ax.set_xticklabels(df["variant"].astype(str), rotation=25, ha="right")
    ax.set_xlabel("Model variant")
    ax.set_ylabel("RMSE (mm h$^{-1}$)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    save_fig(fig, out_dir / "ablation_synthetic_vs_measured_rmse")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot ablation summary figures for paper-ready reporting.")
    parser.add_argument("--summary_csv", default="ablation_results/summary/ablation_summary_all.csv")
    parser.add_argument("--out_dir", default="ablation_results/summary")
    parser.add_argument("--metric_mode", choices=["raw", "linear", "ridge"], default="linear")
    parser.add_argument("--include_extra_variants", action="store_true")
    main(parser.parse_args())
