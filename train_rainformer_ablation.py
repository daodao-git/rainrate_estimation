import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from RainFormerPhys import RainFormerPhys as OriginalRainFormerPhys
from RainFormerPhys_ablation import VALID_VARIANTS, count_trainable_parameters, get_rainformer_model


class RainDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        return self.x[idx].unsqueeze(0), self.y[idx]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_rain_rate_from_name(path: Path) -> float:
    name = path.stem
    patterns = [r"(?:^|[_-])R(?:ain)?[_-]?(\d+(?:\.\d+)?)", r"(\d+(?:\.\d+)?)\s*mmph"]
    for p in patterns:
        m = re.search(p, name, flags=re.IGNORECASE)
        if m:
            return float(m.group(1))
    raise ValueError(f"Cannot parse rain rate from filename: {path.name}")


def read_power_series(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    elif suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file type: {path}")

    if "Prx_dBm" not in df.columns:
        raise KeyError(f"Column 'Prx_dBm' not found in {path}")
    return df["Prx_dBm"].to_numpy(dtype=np.float32)


def make_windows(series: np.ndarray, target: float, window_size: int, stride: int) -> Tuple[List[np.ndarray], List[float]]:
    xs, ys = [], []
    if len(series) < window_size:
        return xs, ys
    for start in range(0, len(series) - window_size + 1, stride):
        xs.append(series[start : start + window_size])
        ys.append(target)
    return xs, ys


def load_dataset_from_dir(
    data_dir: Path, window_seconds: float, stride_seconds: float, dt_seconds: float
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    if not data_dir.exists():
        raise FileNotFoundError(f"data_dir does not exist: {data_dir}")

    files = sorted([p for p in data_dir.iterdir() if p.suffix.lower() in {".xlsx", ".xls", ".csv"}])
    if not files:
        raise FileNotFoundError(f"No excel/csv files found in {data_dir}")

    window_size = int(round(window_seconds / dt_seconds))
    stride = int(round(stride_seconds / dt_seconds))
    if window_size <= 0 or stride <= 0:
        raise ValueError("window_seconds / stride_seconds must be > 0 after dt_seconds conversion")

    x_all: List[np.ndarray] = []
    y_all: List[float] = []
    for f in files:
        rain_rate = parse_rain_rate_from_name(f)
        series = read_power_series(f)
        x_list, y_list = make_windows(series, rain_rate, window_size, stride)
        x_all.extend(x_list)
        y_all.extend(y_list)

    if not x_all:
        raise RuntimeError("No windows generated. Check window_seconds/stride_seconds and input file lengths.")

    x = np.stack(x_all).astype(np.float32)
    y = np.asarray(y_all, dtype=np.float32)

    mean, std = float(x.mean()), float(x.std())
    x = (x - mean) / (std + 1e-8)
    return x, y, mean, std


def get_or_create_splits(out_root: Path, n: int, seed: int, split_mode: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    split_path = out_root / f"split_indices_seed{seed}.npz"
    if split_path.exists():
        arr = np.load(split_path)
        return arr["train"], arr["val"], arr["test"]

    idx = np.arange(n)
    if split_mode == "legacy":
        np.random.seed(seed)
        np.random.shuffle(idx)
    elif split_mode == "default":
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
    else:
        raise ValueError(f"Unsupported split_mode: {split_mode}")
    train_end = int(0.7 * n)
    val_end = int(0.85 * n)
    idx_train, idx_val, idx_test = idx[:train_end], idx[train_end:val_end], idx[val_end:]
    np.savez(split_path, train=idx_train, val=idx_val, test=idx_test)
    return idx_train, idx_val, idx_test


def eval_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    err = y_pred - y_true
    return {
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "bias": float(np.mean(err)),
    }


def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in loader:
            preds.append(model(xb.to(device)).cpu().numpy())
            trues.append(yb.numpy())
    return np.concatenate(preds), np.concatenate(trues)


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = Path(args.data_dir)
    dsd_prior = args.dsd_prior
    root = Path(args.out_dir) / args.frequency_tag / dsd_prior
    variant_dir = root / args.variant
    variant_dir.mkdir(parents=True, exist_ok=True)

    x_raw, y, global_mean, global_std = load_dataset_from_dir(
        data_dir, args.window_seconds, args.stride_seconds, args.dt_seconds
    )
    # x_raw returned by load_dataset_from_dir is globally normalized currently; invert to original scale first
    x_raw = x_raw * (global_std + 1e-8) + global_mean
    idx_train, idx_val, idx_test = get_or_create_splits(root, len(x_raw), args.seed, args.split_mode)
    split_path = root / f"split_indices_seed{args.seed}.npz"

    if args.norm_mode == "train":
        norm_mean = float(x_raw[idx_train].mean())
        norm_std = float(x_raw[idx_train].std())
    elif args.norm_mode == "global":
        norm_mean = float(x_raw.mean())
        norm_std = float(x_raw.std())
    else:
        raise ValueError(f"Unsupported norm_mode: {args.norm_mode}")
    x = (x_raw - norm_mean) / (norm_std + 1e-8)

    train_loader = DataLoader(RainDataset(x[idx_train], y[idx_train]), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(RainDataset(x[idx_val], y[idx_val]), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(RainDataset(x[idx_test], y[idx_test]), batch_size=args.batch_size, shuffle=False)

    if args.variant == "full":
        model = OriginalRainFormerPhys(in_channels=1, seq_len=x.shape[1], embed_dim=args.embed_dim, num_blocks=args.num_blocks).to(device)
    else:
        model = get_rainformer_model(args.variant, 1, x.shape[1], args.embed_dim, args.num_blocks).to(device)
    param_count = count_trainable_parameters(model)

    print(f"[run] variant={args.variant}")
    print(f"[run] frequency={args.frequency_tag}")
    print(f"[run] data_path={data_dir}")
    print(f"[run] device={device}")
    print(f"[run] parameter_count={param_count}")
    print(f"[run] output_dir={variant_dir}")
    print(f"[run] split_indices={split_path}")

    logs = []
    best_val = float("nan")
    ckpt_path = variant_dir / "best_model.pth"
    if args.variant == "full":
        if not args.full_checkpoint:
            raise ValueError("--full_checkpoint is required when variant=full")
        full_ckpt = Path(args.full_checkpoint)
        if not full_ckpt.exists():
            raise FileNotFoundError(f"full checkpoint not found: {full_ckpt}")
        state = torch.load(full_ckpt, map_location=device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=True)
        print(f"[info] full variant: skip training, load original checkpoint: {full_ckpt}")
        best_val = float("nan")
    else:
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)
        best_val = float("inf")
        best_state = None
        wait = 0
        for epoch in range(1, args.epochs + 1):
            model.train()
            train_loss_sum = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()
                train_loss_sum += loss.item() * xb.size(0)
            train_loss = train_loss_sum / len(train_loader.dataset)
            model.eval()
            val_loss_sum = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    val_loss_sum += criterion(model(xb), yb).item() * xb.size(0)
            val_loss = val_loss_sum / len(val_loader.dataset)
            scheduler.step(val_loss)
            logs.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "lr": optimizer.param_groups[0]["lr"]})
            print(f"Epoch {epoch:03d} | train={train_loss:.6f} | val={val_loss:.6f}")
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
            if wait > 8:
                break
        torch.save({"state_dict": best_state, "variant": args.variant, "seq_len": int(x.shape[1]), "norm_mean": float(norm_mean), "norm_std": float(norm_std)}, ckpt_path)
        model.load_state_dict(best_state)
        pd.DataFrame(logs).to_csv(variant_dir / "train_log.csv", index=False)

    y_pred, y_true = evaluate_model(model, test_loader, device)

    m = eval_metrics(y_true, y_pred)
    df_pred = pd.DataFrame(
        {
            "True_R_mmph": y_true,
            "Pred_R_mmph": y_pred,
            "Error_mmph": y_pred - y_true,
            "AbsError_mmph": np.abs(y_pred - y_true),
            "Variant": args.variant,
            "Frequency": args.frequency_tag,
            "DSDPrior": dsd_prior,
        }
    )
    df_pred.to_csv(variant_dir / "synthetic_test_predictions.csv", index=False, encoding="utf-8-sig")

    metrics = {
        "variant": args.variant,
        "frequency_tag": args.frequency_tag,
        "data_dir": str(data_dir),
        "rmse": m["rmse"],
        "mae": m["mae"],
        "bias": m["bias"],
        "param_count": int(param_count),
        "best_val_loss": float(best_val),
        "seed": int(args.seed),
        "checkpoint_path": str(args.full_checkpoint) if args.variant == "full" else str(ckpt_path),
        "dsd_prior": dsd_prior,
    }
    with open(variant_dir / "metrics_synthetic.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # shared normalization file for measured test
    norm_path = root / "normalization_info.json"
    with open(norm_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "mean": float(norm_mean),
                "std": float(norm_std),
                "seed": args.seed,
                "seq_len": int(x.shape[1]),
                "frequency_tag": args.frequency_tag,
                "dsd_prior": dsd_prior,
                "split_indices_path": str(split_path),
                "split_mode": args.split_mode,
                "norm_mode": args.norm_mode,
            },
            f,
            indent=2,
        )

    summary_path = Path(args.out_dir) / "summary" / "ablation_summary_synthetic.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "variant": args.variant,
        "frequency_tag": args.frequency_tag,
        "dsd_prior": dsd_prior,
        "synthetic_RMSE": m["rmse"],
        "synthetic_MAE": m["mae"],
        "synthetic_Bias": m["bias"],
        "param_count": int(param_count),
        "checkpoint_path": str(ckpt_path),
    }
    if summary_path.exists():
        df_sum = pd.read_csv(summary_path)
        df_sum = df_sum[~((df_sum["variant"] == args.variant) & (df_sum["frequency_tag"] == args.frequency_tag) & (df_sum["dsd_prior"] == dsd_prior))]
        df_sum = pd.concat([df_sum, pd.DataFrame([row])], ignore_index=True)
    else:
        df_sum = pd.DataFrame([row])
    df_sum.to_csv(summary_path, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RainFormerPhys ablation variants on synthetic rainfall dataset.")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--frequency_tag", required=True)
    parser.add_argument("--variant", required=True, choices=sorted(VALID_VARIANTS))
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
    parser.add_argument("--split_mode", choices=["default", "legacy"], default="legacy")
    parser.add_argument("--norm_mode", choices=["train", "global"], default="global")
    parser.add_argument("--full_checkpoint", default="")
    train(parser.parse_args())
