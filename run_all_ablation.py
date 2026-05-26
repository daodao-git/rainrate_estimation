import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

VARIANTS = [
    "full",
    "no_phys_stat",
    "avg_pool",
    "no_pos",
    "no_attn",
    "no_dwconv",
    "no_gate",
    "simple_patch",
]


def run_cmd(cmd):
    print("[cmd]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main(args):
    project_root = Path(__file__).resolve().parent
    out_dir = Path(args.out_dir)
    dsd_prior = args.dsd_prior

    rows = []
    for variant in VARIANTS:
        train_cmd = [
            sys.executable,
            str(project_root / "train_rainformer_ablation.py"),
            "--data_dir",
            args.data_dir,
            "--frequency_tag",
            args.frequency_tag,
            "--variant",
            variant,
            "--dsd_prior",
            dsd_prior,
            "--out_dir",
            args.out_dir,
            "--epochs",
            str(args.epochs),
            "--batch_size",
            str(args.batch_size),
            "--lr",
            str(args.lr),
            "--seed",
            str(args.seed),
            "--window_seconds",
            str(args.window_seconds),
            "--stride_seconds",
            str(args.stride_seconds),
            "--dt_seconds",
            str(args.dt_seconds),
            "--embed_dim",
            str(args.embed_dim),
            "--num_blocks",
            str(args.num_blocks),
            "--split_mode",
            args.split_mode,
            "--norm_mode",
            args.norm_mode,
        ]
        if args.full_checkpoint:
            train_cmd.extend(["--full_checkpoint", args.full_checkpoint])
        run_cmd(train_cmd)

        ckpt_path = out_dir / args.frequency_tag / dsd_prior / variant / "best_model.pth"
        norm_path = out_dir / args.frequency_tag / dsd_prior / "normalization_info.json"
        measured_out = out_dir / args.frequency_tag / dsd_prior / variant

        syn_metrics = json.loads((measured_out / "metrics_synthetic.json").read_text(encoding="utf-8"))
        mea_metrics = None
        if variant != "full":
            test_cmd = [
                sys.executable,
                str(project_root / "test_rainformer_ablation_LOOCV.py"),
                "--variant",
                variant,
                "--checkpoint",
                str(ckpt_path),
                "--measured_data_path",
                args.measured_data_path,
                "--out_dir",
                str(measured_out),
                "--frequency_tag",
                args.frequency_tag,
                "--dsd_prior",
                dsd_prior,
                "--embed_dim",
                str(args.embed_dim),
                "--num_blocks",
                str(args.num_blocks),
                "--normalization_info",
                str(norm_path),
                "--seed",
                str(args.seed),
            ]
            if args.seq_len is not None:
                test_cmd.extend(["--seq_len", str(args.seq_len)])
            run_cmd(test_cmd)
            mea_metrics = json.loads((measured_out / "metrics_measured.json").read_text(encoding="utf-8"))
        rows.append(
            {
                "variant": variant,
                "frequency_tag": args.frequency_tag,
                "dsd_prior": dsd_prior,
                "synthetic_RMSE": syn_metrics["rmse"],
                "synthetic_MAE": syn_metrics["mae"],
                "synthetic_Bias": syn_metrics["bias"],
                "raw_RMSE": None if mea_metrics is None else mea_metrics["raw_rmse"],
                "raw_MAE": None if mea_metrics is None else mea_metrics["raw_mae"],
                "raw_Bias": None if mea_metrics is None else mea_metrics["raw_bias"],
                "linear_RMSE": None if mea_metrics is None else mea_metrics["linear_rmse"],
                "linear_MAE": None if mea_metrics is None else mea_metrics["linear_mae"],
                "linear_Bias": None if mea_metrics is None else mea_metrics["linear_bias"],
                "ridge_RMSE": None if mea_metrics is None else mea_metrics["ridge_rmse"],
                "ridge_MAE": None if mea_metrics is None else mea_metrics["ridge_mae"],
                "ridge_Bias": None if mea_metrics is None else mea_metrics["ridge_bias"],
                "param_count": syn_metrics["param_count"],
                "checkpoint_path": syn_metrics["checkpoint_path"],
            }
        )

    summary_dir = out_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / "ablation_summary_all.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"[done] Summary saved: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run all ablation variants: train + measured LOOCV test + summary merge.")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--measured_data_path", required=True)
    parser.add_argument("--frequency_tag", required=True)
    parser.add_argument("--dsd_prior", required=True)
    parser.add_argument("--out_dir", default="ablation_results")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--window_seconds", type=float, default=20.0)
    parser.add_argument("--stride_seconds", type=float, default=10.0)
    parser.add_argument("--dt_seconds", type=float, default=0.05)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--num_blocks", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=None)
    parser.add_argument("--split_mode", choices=["default", "legacy"], default="legacy")
    parser.add_argument("--norm_mode", choices=["train", "global"], default="global")
    parser.add_argument("--full_checkpoint", default="")
    main(parser.parse_args())
