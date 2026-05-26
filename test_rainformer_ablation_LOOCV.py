import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import LeaveOneOut
from torch.utils.data import DataLoader, Dataset

from RainFormerPhys_ablation import VALID_VARIANTS, count_trainable_parameters, get_rainformer_model


class RainDataset(Dataset):
    def __init__(self, x: np.ndarray):
        self.x = torch.from_numpy(x).float()

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx: int):
        return self.x[idx].unsqueeze(0)


def try_load_mat(path: Path):
    try:
        with h5py.File(path, "r") as f:
            return {k: np.array(f[k]) for k in f.keys()}
    except Exception:
        raw = loadmat(path)
        return {k: v for k, v in raw.items() if not k.startswith("__")}


def find_key(dic: Dict, keys):
    for k in keys:
        if k in dic:
            return k
    return None


def load_mat_xy(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = try_load_mat(path)
    xk = find_key(data, ["X", "x", "inputs", "Input", "data"])
    yk = find_key(data, ["Y", "y", "labels", "Label", "target", "targets"])
    if xk is None or yk is None:
        raise KeyError(f"Missing X/Y in {path}")
    x = np.asarray(data[xk], dtype=np.float32)
    y = np.asarray(data[yk], dtype=np.float32).squeeze()
    if x.ndim != 2:
        raise ValueError(f"X must be 2D, got {x.shape}")
    if x.shape[0] == 400 and x.shape[1] != 400:
        x = x.T
    if y.ndim == 2:
        y = y.squeeze()
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"Sample mismatch: X={x.shape}, y={y.shape}")
    return x, y


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    err = y_pred - y_true
    return {
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "bias": float(np.mean(err)),
    }


def predict(model, x: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    loader = DataLoader(RainDataset(x), batch_size=batch_size, shuffle=False)
    out = []
    model.eval()
    with torch.no_grad():
        for xb in loader:
            out.append(model(xb.to(device)).cpu().numpy())
    return np.concatenate(out)


def load_checkpoint_with_variant_validation(model: torch.nn.Module, checkpoint_path: str, expected_variant: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict):
        saved_variant = ckpt.get("variant")
        if saved_variant is not None and saved_variant != expected_variant:
            raise RuntimeError(
                f"Checkpoint variant mismatch: checkpoint='{saved_variant}', requested='{expected_variant}'. "
                "Please use the matching ablation checkpoint."
            )
        state = ckpt.get("state_dict", ckpt)
    else:
        state = ckpt
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as e:
        raise RuntimeError(
            f"Checkpoint load failed for variant='{expected_variant}'. "
            f"Likely checkpoint/model variant mismatch. Details: {e}"
        )


def main(args: argparse.Namespace) -> None:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    measured_path = Path(args.measured_data_path)
    if not measured_path.exists():
        raise FileNotFoundError(f"measured_data_path not found: {measured_path}")

    norm_file = Path(args.normalization_info)
    if not norm_file.exists():
        raise FileNotFoundError(f"normalization_info not found: {norm_file}")
    norm_info = json.loads(norm_file.read_text(encoding="utf-8"))
    if "mean" not in norm_info or "std" not in norm_info:
        raise KeyError("normalization_info must include mean/std")
    if "frequency_tag" in norm_info and norm_info["frequency_tag"] != args.frequency_tag:
        raise ValueError(
            f"normalization_info frequency mismatch: file={norm_info['frequency_tag']} vs arg={args.frequency_tag}"
        )
    if "dsd_prior" in norm_info and norm_info["dsd_prior"] != args.dsd_prior:
        raise ValueError(
            f"normalization_info dsd_prior mismatch: file={norm_info['dsd_prior']} vs arg={args.dsd_prior}"
        )

    x, y = load_mat_xy(measured_path)
    x = (x - float(norm_info["mean"])) / (float(norm_info["std"]) + 1e-8)
    seq_len = args.seq_len if args.seq_len is not None else x.shape[1]

    model = get_rainformer_model(args.variant, 1, seq_len, args.embed_dim, args.num_blocks).to(device)
    param_count = count_trainable_parameters(model)

    print(f"[run] variant={args.variant}")
    print(f"[run] dsd_prior={args.dsd_prior}")
    print(f"[run] frequency={args.frequency_tag}")
    print(f"[run] data_path={measured_path}")
    print(f"[run] device={device}")
    print(f"[run] parameter_count={param_count}")
    print(f"[run] output_dir={out_dir}")

    load_checkpoint_with_variant_validation(model, args.checkpoint, args.variant, device)

    raw_preds = predict(model, x, args.batch_size, device)
    loo = LeaveOneOut()
    oof_raw = np.full_like(y, np.nan, dtype=np.float32)
    oof_lin = np.full_like(y, np.nan, dtype=np.float32)
    oof_ridge = np.full_like(y, np.nan, dtype=np.float32)

    for train_idx, val_idx in loo.split(raw_preds):
        pred_train = raw_preds[train_idx].reshape(-1, 1)
        y_train = y[train_idx]
        pred_val = raw_preds[val_idx].reshape(-1, 1)

        lin = LinearRegression().fit(pred_train, y_train)
        ridge = Ridge(alpha=1.0).fit(pred_train, y_train)

        oof_raw[val_idx] = raw_preds[val_idx]
        oof_lin[val_idx] = lin.predict(pred_val)
        oof_ridge[val_idx] = ridge.predict(pred_val)

    m_raw = metrics(y, oof_raw)
    m_lin = metrics(y, oof_lin)
    m_ridge = metrics(y, oof_ridge)

    pred_df = pd.DataFrame(
        {
            "True_R_mmph": y,
            "RawPred_R_mmph": raw_preds,
            "OOF_RawPred_R_mmph": oof_raw,
            "OOF_LinPred_R_mmph": oof_lin,
            "OOF_RidgePred_R_mmph": oof_ridge,
            "Variant": args.variant,
            "Frequency": args.frequency_tag,
            "DSDPrior": args.dsd_prior,
            "Error_mmph": oof_raw - y,
            "AbsError_mmph": np.abs(oof_raw - y),
        }
    )
    pred_df.to_csv(out_dir / "measured_predictions.csv", index=False, encoding="utf-8-sig")

    metrics_json = {
        "variant": args.variant,
        "frequency_tag": args.frequency_tag,
        "dsd_prior": args.dsd_prior,
        "raw_rmse": m_raw["rmse"],
        "raw_mae": m_raw["mae"],
        "raw_bias": m_raw["bias"],
        "linear_rmse": m_lin["rmse"],
        "linear_mae": m_lin["mae"],
        "linear_bias": m_lin["bias"],
        "ridge_rmse": m_ridge["rmse"],
        "ridge_mae": m_ridge["mae"],
        "ridge_bias": m_ridge["bias"],
        "param_count": int(param_count),
        "checkpoint_path": str(args.checkpoint),
    }
    with open(out_dir / "metrics_measured.json", "w", encoding="utf-8") as f:
        json.dump(metrics_json, f, indent=2)

    plt.figure(figsize=(5, 5))
    plt.scatter(y, oof_raw, s=12, alpha=0.7)
    xy_min, xy_max = min(y.min(), oof_raw.min()), max(y.max(), oof_raw.max())
    plt.plot([xy_min, xy_max], [xy_min, xy_max], "r--", linewidth=1)
    plt.xlabel("True rain rate (mm h$^{-1}$)")
    plt.ylabel("Predicted rain rate (mm h$^{-1}$)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "measured_scatter.png", dpi=300)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.hist(oof_raw - y, bins=30)
    plt.xlabel("Prediction error (mm h$^{-1}$)")
    plt.ylabel("Count")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "error_histogram.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ablation checkpoints on measured data with LOOCV calibration.")
    parser.add_argument("--variant", required=True, choices=sorted(VALID_VARIANTS))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--measured_data_path", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--frequency_tag", required=True)
    parser.add_argument("--dsd_prior", required=True)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--num_blocks", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=None)
    parser.add_argument("--normalization_info", required=True)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--batch_size", type=int, default=64)
    main(parser.parse_args())
