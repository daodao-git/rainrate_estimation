import os
import copy
import random
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.io import loadmat
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import LeaveOneOut
from torch.utils.data import Dataset, DataLoader



# ===================== 全局配置 =====================

@dataclass
class ExperimentConfig:
    seed: int = 2026
    batch_size: int = 64
    num_epochs: int = 35
    learning_rate: float = 1e-3
    early_stop_patience: int = 8
    ridge_alpha: float = 1.0

    # 训练集划分比例
    train_ratio: float = 0.7
    val_ratio: float = 0.15

    # 模型输出
    model_root: str = "./baseline_models"
    output_root: str = "./baseline_results"


CFG = ExperimentConfig()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FREQ_CONFIGS = [
    {
        "freq_label": "120GHz",
        "train_mat_path": "./dataset_JW_Rreg_120GHz_20s_20000samples.mat",
        "measured_mat_path": "./measured_dataset_same_format_pseudopower.mat",
    },
    {
        "freq_label": "140GHz",
        "train_mat_path": "./dataset_JW_Rreg_140GHz_20s_20000samples.mat",
        "measured_mat_path": "./measured_dataset_same_format_140_pseudopower.mat",
    },
    {
        "freq_label": "229GHz",
        "train_mat_path": "./dataset_JW_Rreg_229GHz_20s_20000samples.mat",
        "measured_mat_path": "./measured_dataset_same_format_229_pseudopower.mat",
    },
]

BASELINE_MODEL_NAMES = ["rainformer_phys", "cnn1d", "vanilla_transformer", "lstm", "tcn"]


# ===================== 工具函数 =====================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else np.nan


def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    errors = y_pred - y_true
    return {
        "RMSE": float(np.sqrt(np.mean(errors ** 2))),
        "MAE": float(np.mean(np.abs(errors))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def try_load_mat(path: str) -> Dict[str, Any]:
    try:
        with h5py.File(path, "r") as f:
            keys = list(f.keys())
            print(f"[info] h5py 读取成功: {path}")
            print(f"[info] keys={keys}")
            return {k: np.array(f[k]) for k in keys}
    except Exception as e:
        print(f"[info] h5py 读取失败，使用 loadmat: {path} | {e}")
        data = loadmat(path)
        keys = [k for k in data.keys() if not k.startswith("__")]
        print(f"[info] loadmat 读取成功 keys={keys}")
        return data


def find_first_existing_key(data_dict, candidate_keys):
    for k in candidate_keys:
        if k in data_dict:
            return k
    return None


def normalize_X_shape(X: np.ndarray) -> np.ndarray:
    X = np.array(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError(f"X 应为二维，当前 shape={X.shape}")
    if X.shape[0] == 400 and X.shape[1] != 400:
        X = X.T
    return X


def normalize_y_shape(y: np.ndarray) -> np.ndarray:
    y = np.array(y, dtype=np.float32)
    if y.ndim == 2:
        if y.shape[0] < y.shape[1]:
            y = y.T
        y = y.squeeze()
    if y.ndim != 1:
        raise ValueError(f"y 应为一维，当前 shape={y.shape}")
    return y


def load_xy_from_mat(path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    data = try_load_mat(path)

    x_key = find_first_existing_key(data, ["X", "x", "inputs", "Input", "data"])
    y_key = find_first_existing_key(data, ["Y", "y", "labels", "Label", "target", "targets"])

    if x_key is None:
        raise KeyError(f"未找到 X, path={path}")

    X = normalize_X_shape(data[x_key])
    y = normalize_y_shape(data[y_key]) if y_key is not None else None

    if y is not None and X.shape[0] != y.shape[0]:
        raise ValueError(f"样本数不匹配: X={X.shape}, y={y.shape}, path={path}")

    return X, y


def split_dataset(X: np.ndarray, y: np.ndarray, train_ratio: float, val_ratio: float):
    n = X.shape[0]
    indices = np.arange(n)
    np.random.shuffle(indices)

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    idx_train = indices[:train_end]
    idx_val = indices[train_end:val_end]
    idx_test = indices[val_end:]

    return (
        X[idx_train], y[idx_train],
        X[idx_val], y[idx_val],
        X[idx_test], y[idx_test],
    )


# ===================== Dataset =====================
class RainDataset(Dataset):
    def __init__(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        self.X = torch.from_numpy(X).float()
        self.y = None if y is None else torch.from_numpy(y).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx].unsqueeze(0)
        if self.y is None:
            return x
        return x, self.y[idx]


# ===================== 原始 RainFormerPhys 模型（内置，避免跨目录导入失败） =====================
class RainFormerHybridAttentionConvBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, conv_kernel: int = 9,
                 ffn_expansion: int = 2, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = dim * ffn_expansion

        self.dwconv = nn.Conv1d(
            dim,
            dim,
            kernel_size=conv_kernel,
            padding=conv_kernel // 2,
            groups=dim,
        )
        self.pw1 = nn.Linear(dim, hidden_dim * 2)
        self.pw2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = residual + self.dropout(attn_out)

        residual = x
        x_norm = self.norm2(x)

        conv_in = x_norm.transpose(1, 2)
        conv_out = self.dwconv(conv_in).transpose(1, 2)

        gate_in = self.pw1(conv_out)
        a, b = gate_in.chunk(2, dim=-1)
        glu_out = a * torch.sigmoid(b)
        ff_out = self.pw2(glu_out)

        return residual + self.dropout(ff_out)


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 400):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-np.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]


class AttentiveStatPooling(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(dim))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        scores = torch.einsum("blc,c->bl", tokens, self.query)
        weights = torch.softmax(scores, dim=1)
        return torch.bmm(weights.unsqueeze(1), tokens).squeeze(1)


class RainFormerPhys(nn.Module):
    """原始 RainFormerPhys：混合注意力卷积主干 + 物理统计特征融合。"""

    def __init__(self, in_channels: int, seq_len: int, embed_dim: int = 128, num_blocks: int = 4):
        super().__init__()

        self.patch_embed = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
            nn.BatchNorm1d(in_channels),
            nn.GELU(),
            nn.Conv1d(in_channels, embed_dim, kernel_size=7, padding=3),
            nn.GELU(),
        )

        self.pos_encoder = SinusoidalPositionalEncoding(embed_dim, max_len=seq_len)
        self.blocks = nn.ModuleList(
            [
                RainFormerHybridAttentionConvBlock(
                    dim=embed_dim,
                    num_heads=4,
                    conv_kernel=9,
                    ffn_expansion=2,
                    dropout=0.1,
                )
                for _ in range(num_blocks)
            ]
        )

        self.pool = AttentiveStatPooling(embed_dim)
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim + 2),
            nn.Linear(embed_dim + 2, 96),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(96, 32),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stat_mean = x.mean(dim=-1)
        stat_std = x.std(dim=-1)
        phys_feat = torch.cat([stat_mean, stat_std], dim=1)

        tokens = self.patch_embed(x).transpose(1, 2)
        tokens = self.pos_encoder(tokens)
        for block in self.blocks:
            tokens = block(tokens)

        pooled = self.pool(tokens)
        fused = torch.cat([pooled, phys_feat], dim=1)
        return self.head(fused).squeeze(-1)


# ===================== Baseline 模型 =====================
class CNN1DRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.head(self.backbone(x)).squeeze(-1)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 400):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class VanillaTransformerRegressor(nn.Module):
    def __init__(self, seq_len: int, d_model: int = 128, nhead: int = 4, num_layers: int = 4):
        super().__init__()
        self.proj = nn.Conv1d(1, d_model, kernel_size=1)
        self.pos = PositionalEncoding(d_model, max_len=seq_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        tokens = self.proj(x).transpose(1, 2)
        tokens = self.pos(tokens)
        features = self.encoder(tokens)
        pooled = features.mean(dim=1)
        return self.head(pooled).squeeze(-1)


class LSTMRegressor(nn.Module):
    def __init__(self, hidden_size: int = 128, num_layers: int = 2):
        super().__init__()
        self.proj = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1,
            bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        seq = self.proj(x).transpose(1, 2)
        out, _ = self.lstm(seq)
        pooled = out.mean(dim=1)
        return self.head(pooled).squeeze(-1)


class TemporalBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else None

    def forward(self, x):
        out = self.net(x)
        out = out[..., :x.size(-1)]
        res = x if self.downsample is None else self.downsample(x)
        return torch.relu(out + res)


class TCNRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        channels = [32, 64, 64, 128]
        layers = []
        in_ch = 1
        for i, out_ch in enumerate(channels):
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size=5, dilation=2 ** i, dropout=0.1))
            in_ch = out_ch
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(channels[-1], 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.head(self.tcn(x)).squeeze(-1)


def build_model(model_name: str, seq_len: int) -> nn.Module:
    if model_name == "rainformer_phys":
        return RainFormerPhys(in_channels=1, seq_len=seq_len)
    if model_name == "cnn1d":
        return CNN1DRegressor()
    if model_name == "vanilla_transformer":
        return VanillaTransformerRegressor(seq_len=seq_len)
    if model_name == "lstm":
        return LSTMRegressor()
    if model_name == "tcn":
        return TCNRegressor()
    raise ValueError(f"不支持的模型: {model_name}")


# ===================== 训练与预测 =====================
def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: ExperimentConfig,
) -> nn.Module:
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    best_state = None
    best_val = float("inf")
    patience = 0

    for epoch in range(cfg.num_epochs):
        model.train()
        train_loss_sum = 0.0
        for x, y in train_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * x.size(0)

        train_loss = train_loss_sum / len(train_loader.dataset)

        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(DEVICE)
                y = y.to(DEVICE)
                pred = model(x)
                val_loss_sum += criterion(pred, y).item() * x.size(0)

        val_loss = val_loss_sum / len(val_loader.dataset)
        scheduler.step(val_loss)
        print(f"Epoch {epoch + 1:03d} | train={train_loss:.6f} | val={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1

        if patience > cfg.early_stop_patience:
            print("[info] early stopping")
            break

    if best_state is None:
        raise RuntimeError("训练失败，未得到 best_state")

    model.load_state_dict(best_state)
    return model


def predict(model: nn.Module, X: np.ndarray, batch_size: int) -> np.ndarray:
    loader = DataLoader(RainDataset(X, None), batch_size=batch_size, shuffle=False)
    model.eval()
    preds = []
    with torch.no_grad():
        for x in loader:
            x = x.to(DEVICE)
            preds.append(model(x).cpu().numpy())
    return np.concatenate(preds)


def run_loocv_calibration(raw_preds: np.ndarray, y: np.ndarray, ridge_alpha: float):
    loo = LeaveOneOut()

    oof_raw = np.full_like(y, np.nan, dtype=np.float32)
    oof_lin = np.full_like(y, np.nan, dtype=np.float32)
    oof_ridge = np.full_like(y, np.nan, dtype=np.float32)

    fold_rows = []

    for fold_idx, (train_idx, val_idx) in enumerate(loo.split(raw_preds), start=1):
        pred_train = raw_preds[train_idx].reshape(-1, 1)
        y_train = y[train_idx]

        pred_val = raw_preds[val_idx].reshape(-1, 1)
        y_val = y[val_idx]

        reg_lin = LinearRegression()
        reg_lin.fit(pred_train, y_train)

        reg_ridge = Ridge(alpha=ridge_alpha)
        reg_ridge.fit(pred_train, y_train)

        y_val_pred_raw = raw_preds[val_idx]
        y_val_pred_lin = reg_lin.predict(pred_val)
        y_val_pred_ridge = reg_ridge.predict(pred_val)

        oof_raw[val_idx] = y_val_pred_raw
        oof_lin[val_idx] = y_val_pred_lin
        oof_ridge[val_idx] = y_val_pred_ridge

        fold_rows.append({
            "fold": fold_idx,
            "val_index": int(val_idx[0]),
            "true_y": float(y_val[0]),
            "raw_pred": float(y_val_pred_raw[0]),
            "lin_pred": float(y_val_pred_lin[0]),
            "lin_coef_a": float(reg_lin.coef_[0]),
            "lin_intercept_b": float(reg_lin.intercept_),
            "ridge_pred": float(y_val_pred_ridge[0]),
            "ridge_coef_a": float(reg_ridge.coef_[0]),
            "ridge_intercept_b": float(reg_ridge.intercept_),
            "ridge_alpha": float(ridge_alpha),
        })

    return oof_raw, oof_lin, oof_ridge, pd.DataFrame(fold_rows)


# ===================== 单模型单频率实验 =====================
def run_single_baseline(freq_cfg: Dict[str, str], model_name: str, cfg: ExperimentConfig):
    freq_label = freq_cfg["freq_label"]
    train_mat = freq_cfg["train_mat_path"]
    measured_mat = freq_cfg["measured_mat_path"]

    print("\n" + "=" * 90)
    print(f"[start] freq={freq_label} | model={model_name}")
    print("=" * 90)

    X_train_all, y_train_all = load_xy_from_mat(train_mat)
    if y_train_all is None:
        raise ValueError(f"训练集缺少标签: {train_mat}")

    mean_train = float(X_train_all.mean())
    std_train = float(X_train_all.std())

    X_train_all = (X_train_all - mean_train) / (std_train + 1e-8)

    X_train, y_train, X_val, y_val, X_test, y_test = split_dataset(
        X_train_all,
        y_train_all,
        train_ratio=cfg.train_ratio,
        val_ratio=cfg.val_ratio,
    )

    seq_len = X_train_all.shape[1]
    model = build_model(model_name, seq_len=seq_len).to(DEVICE)

    train_loader = DataLoader(RainDataset(X_train, y_train), batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(RainDataset(X_val, y_val), batch_size=cfg.batch_size, shuffle=False)

    model = train_model(model, train_loader, val_loader, cfg)

    model_dir = os.path.join(cfg.model_root, freq_label)
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"best_{model_name}.pth")
    torch.save(model.state_dict(), model_path)

    # synthetic test 指标
    syn_test_preds = predict(model, X_test, cfg.batch_size)
    syn_test_metrics = calc_metrics(y_test, syn_test_preds)

    # measured 迁移测试
    X_measured, y_measured = load_xy_from_mat(measured_mat)
    if y_measured is None:
        raise ValueError(f"实测集缺少标签: {measured_mat}")

    X_measured = (X_measured - mean_train) / (std_train + 1e-8)

    raw_preds = predict(model, X_measured, cfg.batch_size)
    raw_metrics = calc_metrics(y_measured, raw_preds)

    oof_raw, oof_lin, oof_ridge, df_folds = run_loocv_calibration(raw_preds, y_measured, cfg.ridge_alpha)
    oof_raw_metrics = calc_metrics(y_measured, oof_raw)
    oof_lin_metrics = calc_metrics(y_measured, oof_lin)
    oof_ridge_metrics = calc_metrics(y_measured, oof_ridge)

    output_dir = os.path.join(cfg.output_root, freq_label, model_name)
    os.makedirs(output_dir, exist_ok=True)

    df_pred = pd.DataFrame({
        "True_R_mmph": y_measured,
        "RawPred_R_mmph": raw_preds,
        "OOF_RawPred_R_mmph": oof_raw,
        "OOF_LinPred_R_mmph": oof_lin,
        "OOF_RidgePred_R_mmph": oof_ridge,
        "OOF_Raw_AbsError_mmph": np.abs(oof_raw - y_measured),
        "OOF_Lin_AbsError_mmph": np.abs(oof_lin - y_measured),
        "OOF_Ridge_AbsError_mmph": np.abs(oof_ridge - y_measured),
    })

    df_summary = pd.DataFrame([
        {"Mode": "SyntheticTest", **syn_test_metrics},
        {"Mode": "Measured_DirectRaw", **raw_metrics},
        {"Mode": "Measured_LOOCV_Raw", **oof_raw_metrics},
        {"Mode": "Measured_LOOCV_Linear", **oof_lin_metrics},
        {"Mode": "Measured_LOOCV_Ridge", **oof_ridge_metrics},
    ])

    pred_csv = os.path.join(output_dir, "predictions.csv")
    fold_csv = os.path.join(output_dir, "loocv_fold_details.csv")
    summary_csv = os.path.join(output_dir, "summary.csv")

    df_pred.to_csv(pred_csv, index=False, encoding="utf-8-sig")
    df_folds.to_csv(fold_csv, index=False, encoding="utf-8-sig")
    df_summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    print(f"[save] model={model_name} freq={freq_label} summary: {summary_csv}")

    return {
        "Frequency": freq_label,
        "Model": model_name,
        "SyntheticTest_RMSE": syn_test_metrics["RMSE"],
        "SyntheticTest_MAE": syn_test_metrics["MAE"],
        "SyntheticTest_R2": syn_test_metrics["R2"],
        "MeasuredRaw_RMSE": raw_metrics["RMSE"],
        "MeasuredRaw_MAE": raw_metrics["MAE"],
        "MeasuredRaw_R2": raw_metrics["R2"],
        "LOOCV_Raw_RMSE": oof_raw_metrics["RMSE"],
        "LOOCV_Raw_MAE": oof_raw_metrics["MAE"],
        "LOOCV_Raw_R2": oof_raw_metrics["R2"],
        "LOOCV_Linear_RMSE": oof_lin_metrics["RMSE"],
        "LOOCV_Linear_MAE": oof_lin_metrics["MAE"],
        "LOOCV_Linear_R2": oof_lin_metrics["R2"],
        "LOOCV_Ridge_RMSE": oof_ridge_metrics["RMSE"],
        "LOOCV_Ridge_MAE": oof_ridge_metrics["MAE"],
        "LOOCV_Ridge_R2": oof_ridge_metrics["R2"],
        "ModelPath": model_path,
    }


# ===================== 主流程 =====================
def main():
    set_seed(CFG.seed)
    os.makedirs(CFG.model_root, exist_ok=True)
    os.makedirs(CFG.output_root, exist_ok=True)

    all_rows: List[Dict[str, Any]] = []

    for freq_cfg in FREQ_CONFIGS:
        for model_name in BASELINE_MODEL_NAMES:
            row = run_single_baseline(freq_cfg, model_name, CFG)
            all_rows.append(row)

    df_all = pd.DataFrame(all_rows)
    all_csv = os.path.join(CFG.output_root, "all_baselines_summary.csv")
    all_xlsx = os.path.join(CFG.output_root, "all_baselines_summary.xlsx")

    df_all.to_csv(all_csv, index=False, encoding="utf-8-sig")
    df_all.to_excel(all_xlsx, index=False)

    print("\n" + "=" * 90)
    print("[done] baseline 对比实验完成")
    print(df_all.to_string(index=False))
    print(f"[save] {all_csv}")
    print(f"[save] {all_xlsx}")


if __name__ == "__main__":
    main()
