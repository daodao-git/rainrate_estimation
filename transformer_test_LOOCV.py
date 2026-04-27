import os
from typing import Tuple, Optional, Dict, Any, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import h5py
from scipy.io import loadmat
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LinearRegression, Ridge


# ===================== 全局配置 =====================
BATCH_SIZE = 64
RANDOM_SEED = 2026
RIDGE_ALPHA = 1.0

OUTPUT_ROOT = "./transformer_test_result/loocv_calibration_compare_results_multi_freq_JW"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)


# ===================== 多频率配置 =====================
# 你只需要改这里的路径
FREQ_CONFIGS = [
    {
        "freq_label": "120GHz",
        "MAT_PATH": "./measured_dataset_same_format_pseudopower.mat",
        "MODEL_PATH": "./bestmodel_transformer/best_model_JW_120_rainformer.pth",
        "TRAIN_MAT_PATH": "./dataset_JW_Rreg_120GHz_20s_20000samples.mat",
    },
    {
        "freq_label": "140GHz",
        "MAT_PATH": "./measured_dataset_same_format_140_pseudopower.mat",
        "MODEL_PATH": "./bestmodel_transformer/best_model_JW_140_rainformer.pth",
        "TRAIN_MAT_PATH": "./dataset_JW_Rreg_140GHz_20s_20000samples.mat",
    },
    {
        "freq_label": "229GHz",
        "MAT_PATH": "./measured_dataset_same_format_229_pseudopower.mat",
        "MODEL_PATH": "./bestmodel_transformer/best_model_JW_229_rainformer.pth",
        "TRAIN_MAT_PATH": "./dataset_JW_Rreg_229GHz_20s_20000samples.mat",
    },
]


# ===================== 工具函数 =====================
def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else np.nan


def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    errors = y_pred - y_true
    abs_errors = np.abs(errors)

    rmse = float(np.sqrt(np.mean(errors ** 2)))
    mae = float(np.mean(abs_errors))
    r2 = float(r2_score(y_true, y_pred))

    return {
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2
    }


def try_load_mat(path: str) -> Dict[str, Any]:
    try:
        with h5py.File(path, "r") as f:
            keys = list(f.keys())
            print(f"[info] 使用 h5py 读取成功: {path}")
            print("[info] 变量列表:", keys)
            data = {}
            for k in keys:
                data[k] = np.array(f[k])
            return data
    except Exception as e:
        print(f"[info] h5py 读取失败，尝试 loadmat: {path}")
        print("[info] 原因:", e)
        data = loadmat(path)
        keys = [k for k in data.keys() if not k.startswith("__")]
        print("[info] 使用 loadmat 读取成功，变量列表:", keys)
        return data


def find_first_existing_key(data_dict, candidate_keys):
    for k in candidate_keys:
        if k in data_dict:
            return k
    return None


def normalize_X_shape(X: np.ndarray) -> np.ndarray:
    X = np.array(X, dtype=np.float32)

    if X.ndim != 2:
        raise ValueError(f"X 维度异常，期望二维，实际为 {X.ndim} 维，shape={X.shape}")

    # 常见 MATLAB 情况：(400, N) -> (N, 400)
    if X.shape[0] == 400 and X.shape[1] != 400:
        X = X.T

    return X


def normalize_y_shape(y: np.ndarray) -> np.ndarray:
    y = np.array(y, dtype=np.float32)

    if y.ndim == 2:
        if y.shape[0] == 1 or y.shape[1] == 1:
            y = y.squeeze()
        elif y.shape[0] < y.shape[1]:
            y = y.T.squeeze()
        else:
            y = y.squeeze()

    if y.ndim != 1:
        raise ValueError(f"Y 维度异常，期望一维，实际为 {y.ndim} 维，shape={y.shape}")

    return y


def load_XY_from_mat(mat_path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    data = try_load_mat(mat_path)

    x_key = find_first_existing_key(data, ["X", "x", "inputs", "Input", "data"])
    if x_key is None:
        raise KeyError(f"未在 mat 文件中找到 X：{mat_path}")
    X = normalize_X_shape(data[x_key])

    y_key = find_first_existing_key(data, ["Y", "y", "labels", "Label", "target", "targets"])
    y = None
    if y_key is not None:
        y = normalize_y_shape(data[y_key])
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"样本数不匹配: X.shape[0]={X.shape[0]}, y.shape[0]={y.shape[0]} | file={mat_path}"
            )

    return X, y


def get_train_norm_stats(train_mat_path: str) -> Tuple[float, float]:
    X_train, _ = load_XY_from_mat(train_mat_path)

    mean_train = float(X_train.mean())
    std_train = float(X_train.std())

    print("\n[info] ===== 训练集标准化参数 =====")
    print(f"[info] TRAIN_MAT_PATH: {train_mat_path}")
    print(f"[info] X_train.shape  : {X_train.shape}")
    print(f"[info] mean_train     : {mean_train:.6f}")
    print(f"[info] std_train      : {std_train:.6f}")
    print("[info] ============================\n")

    return mean_train, std_train


def load_measured_dataset_with_train_stats(
    mat_path: str,
    mean_train: float,
    std_train: float
) -> Tuple[np.ndarray, np.ndarray]:
    X, y = load_XY_from_mat(mat_path)

    if y is None:
        raise ValueError("实测校准任务必须有 Y 标签，但当前 MAT 文件中未找到 Y。")

    print("[info] ===== 测试集原始统计 =====")
    print(f"[info] X.shape={X.shape}")
    print(f"[info] X raw min={X.min():.6f}, max={X.max():.6f}, mean={X.mean():.6f}, std={X.std():.6f}")
    print(f"[info] Y.shape={y.shape}, Y min={y.min():.6f}, Y max={y.max():.6f}")
    print("[info] =========================\n")

    X = (X - mean_train) / (std_train + 1e-8)

    print("[info] ===== 测试集标准化后统计 =====")
    print(f"[info] 使用训练集参数标准化: mean_train={mean_train:.6f}, std_train={std_train:.6f}")
    print(f"[info] X norm min={X.min():.6f}, max={X.max():.6f}, mean={X.mean():.6f}, std={X.std():.6f}")
    print("[info] =============================\n")

    return X, y


# ===================== Dataset =====================
class RainDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.from_numpy(X).float()
        self.y = None if y is None else torch.from_numpy(y).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx].unsqueeze(0)   # (1, L)
        if self.y is None:
            return x
        return x, self.y[idx]


# ===================== Transformer / RainFormerPhys 结构 =====================
class HybridAttentionConvBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, conv_kernel: int = 9,
                 ffn_expansion: int = 2, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = dim * ffn_expansion

        self.dwconv = nn.Conv1d(
            dim, dim,
            kernel_size=conv_kernel,
            padding=conv_kernel // 2,
            groups=dim
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

        conv_in = x_norm.transpose(1, 2)          # (B, C, L)
        conv_out = self.dwconv(conv_in).transpose(1, 2)  # (B, L, C)

        gate_in = self.pw1(conv_out)
        a, b = gate_in.chunk(2, dim=-1)
        glu_out = a * torch.sigmoid(b)
        ff_out = self.pw2(glu_out)

        x = residual + self.dropout(ff_out)
        return x


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 400):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-np.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L = x.size(1)
        return x + self.pe[:, :L, :]


class AttentiveStatPooling(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(dim))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        scores = torch.einsum("blc,c->bl", tokens, self.query)
        weights = torch.softmax(scores, dim=1)
        pooled = torch.bmm(weights.unsqueeze(1), tokens).squeeze(1)
        return pooled


class RainFormerPhys(nn.Module):
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

        self.blocks = nn.ModuleList([
            HybridAttentionConvBlock(
                dim=embed_dim,
                num_heads=4,
                conv_kernel=9,
                ffn_expansion=2,
                dropout=0.1,
            )
            for _ in range(num_blocks)
        ])

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

        tokens = self.patch_embed(x).transpose(1, 2)   # (B, L, C)
        tokens = self.pos_encoder(tokens)

        for block in self.blocks:
            tokens = block(tokens)

        pooled = self.pool(tokens)
        fused = torch.cat([pooled, phys_feat], dim=1)
        out = self.head(fused)
        return out.squeeze(-1)


def predict_with_model(model: nn.Module, X: np.ndarray, batch_size: int) -> np.ndarray:
    dataset = RainDataset(X, None)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    preds = []
    model.eval()

    with torch.no_grad():
        for x in loader:
            x = x.to(DEVICE)
            pred = model(x).cpu().numpy()
            preds.append(pred)

    return np.concatenate(preds)


# ===================== 单频率流程 =====================
def run_single_frequency(freq_cfg: Dict[str, str]) -> Dict[str, Any]:
    freq_label = freq_cfg["freq_label"]
    mat_path = freq_cfg["MAT_PATH"]
    model_path = freq_cfg["MODEL_PATH"]
    train_mat_path = freq_cfg["TRAIN_MAT_PATH"]

    print("\n" + "=" * 80)
    print(f"[start] 开始处理频率: {freq_label}")
    print("=" * 80)

    freq_output_dir = os.path.join(OUTPUT_ROOT, freq_label)
    os.makedirs(freq_output_dir, exist_ok=True)

    mean_train, std_train = get_train_norm_stats(train_mat_path)
    X, y = load_measured_dataset_with_train_stats(mat_path, mean_train, std_train)
    n_samples, seq_len = X.shape

    model = RainFormerPhys(in_channels=1, seq_len=seq_len).to(DEVICE)
    state_dict = torch.load(model_path, map_location=DEVICE)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if len(missing) > 0 or len(unexpected) > 0:
        print("[warning] load_state_dict 非严格匹配结果：")
        print("missing keys:", missing)
        print("unexpected keys:", unexpected)
        raise RuntimeError(f"{freq_label}: 当前模型结构与权重文件不匹配，请检查该频率对应的模型定义或权重。")

    model.eval()

    # 全样本原始预测
    raw_preds = predict_with_model(model, X, BATCH_SIZE)
    raw_metrics = calc_metrics(y, raw_preds)

    print("\n===== 预训练模型直接预测（未校准） =====")
    print("Freq :", freq_label)
    print("Model:", model_path)
    print("Train stats from:", train_mat_path)
    print("Test data:", mat_path)
    print(f"RMSE: {raw_metrics['RMSE']:.6f}")
    print(f"MAE : {raw_metrics['MAE']:.6f}")
    print(f"R2  : {raw_metrics['R2']:.6f}")

    loo = LeaveOneOut()

    oof_raw = np.full_like(y, fill_value=np.nan, dtype=np.float32)
    oof_lin = np.full_like(y, fill_value=np.nan, dtype=np.float32)
    oof_ridge = np.full_like(y, fill_value=np.nan, dtype=np.float32)

    fold_rows = []

    for fold_idx, (train_idx, val_idx) in enumerate(loo.split(raw_preds), start=1):
        pred_train = raw_preds[train_idx].reshape(-1, 1)
        y_train = y[train_idx]

        pred_val = raw_preds[val_idx].reshape(-1, 1)
        y_val = y[val_idx]

        # raw
        y_val_pred_raw = raw_preds[val_idx]

        # 线性校准
        reg_lin = LinearRegression()
        reg_lin.fit(pred_train, y_train)
        y_val_pred_lin = reg_lin.predict(pred_val)

        # Ridge 校准
        reg_ridge = Ridge(alpha=RIDGE_ALPHA)
        reg_ridge.fit(pred_train, y_train)
        y_val_pred_ridge = reg_ridge.predict(pred_val)

        oof_raw[val_idx] = y_val_pred_raw
        oof_lin[val_idx] = y_val_pred_lin
        oof_ridge[val_idx] = y_val_pred_ridge

        fold_rows.append({
            "Frequency": freq_label,
            "fold": fold_idx,
            "val_index": int(val_idx[0]),
            "true_y": float(y_val[0]),

            "raw_pred": float(y_val_pred_raw[0]),

            "lin_coef_a": float(reg_lin.coef_[0]),
            "lin_intercept_b": float(reg_lin.intercept_),
            "lin_pred": float(y_val_pred_lin[0]),

            "ridge_alpha": RIDGE_ALPHA,
            "ridge_coef_a": float(reg_ridge.coef_[0]),
            "ridge_intercept_b": float(reg_ridge.intercept_),
            "ridge_pred": float(y_val_pred_ridge[0]),
        })

        print(
            f"[{freq_label} | LOO {fold_idx:02d}/{n_samples}] "
            f"true={float(y_val[0]):.4f} | "
            f"raw={float(y_val_pred_raw[0]):.4f} | "
            f"lin={float(y_val_pred_lin[0]):.4f} | "
            f"ridge={float(y_val_pred_ridge[0]):.4f}"
        )

    # OOF 总体指标
    oof_raw_metrics = calc_metrics(y, oof_raw)
    oof_lin_metrics = calc_metrics(y, oof_lin)
    oof_ridge_metrics = calc_metrics(y, oof_ridge)

    print(f"\n===== {freq_label} | LOOCV OOF 总体结果 =====")
    print("[未校准预训练模型]")
    print(f"RMSE: {oof_raw_metrics['RMSE']:.6f}")
    print(f"MAE : {oof_raw_metrics['MAE']:.6f}")
    print(f"R2  : {oof_raw_metrics['R2']:.6f}")

    print("\n[预训练模型 + 线性校准]")
    print(f"RMSE: {oof_lin_metrics['RMSE']:.6f}")
    print(f"MAE : {oof_lin_metrics['MAE']:.6f}")
    print(f"R2  : {oof_lin_metrics['R2']:.6f}")

    print("\n[预训练模型 + Ridge校准]")
    print(f"RMSE: {oof_ridge_metrics['RMSE']:.6f}")
    print(f"MAE : {oof_ridge_metrics['MAE']:.6f}")
    print(f"R2  : {oof_ridge_metrics['R2']:.6f}")

    base_name = os.path.splitext(os.path.basename(model_path))[0]

    df_pred = pd.DataFrame({
        "Frequency": freq_label,
        "True_R_mmph": y,
        "RawPred_R_mmph": raw_preds,
        "OOF_RawPred_R_mmph": oof_raw,
        "OOF_LinPred_R_mmph": oof_lin,
        "OOF_RidgePred_R_mmph": oof_ridge,
        "OOF_Raw_AbsError_mmph": np.abs(oof_raw - y),
        "OOF_Lin_AbsError_mmph": np.abs(oof_lin - y),
        "OOF_Ridge_AbsError_mmph": np.abs(oof_ridge - y),
    })

    df_folds = pd.DataFrame(fold_rows)

    df_summary = pd.DataFrame([
        {"Frequency": freq_label, "Mode": "DirectRawFull", "RMSE": raw_metrics["RMSE"], "MAE": raw_metrics["MAE"], "R2": raw_metrics["R2"]},
        {"Frequency": freq_label, "Mode": "LOOCV_Raw", "RMSE": oof_raw_metrics["RMSE"], "MAE": oof_raw_metrics["MAE"], "R2": oof_raw_metrics["R2"]},
        {"Frequency": freq_label, "Mode": "LOOCV_Linear", "RMSE": oof_lin_metrics["RMSE"], "MAE": oof_lin_metrics["MAE"], "R2": oof_lin_metrics["R2"]},
        {"Frequency": freq_label, "Mode": "LOOCV_Ridge", "RMSE": oof_ridge_metrics["RMSE"], "MAE": oof_ridge_metrics["MAE"], "R2": oof_ridge_metrics["R2"]},
    ])

    pred_csv = os.path.join(freq_output_dir, f"loocv_predictions_{base_name}.csv")
    fold_csv = os.path.join(freq_output_dir, f"loocv_fold_details_{base_name}.csv")
    summary_csv = os.path.join(freq_output_dir, f"loocv_summary_{base_name}.csv")
    summary_txt = os.path.join(freq_output_dir, f"loocv_summary_{base_name}.txt")

    df_pred.to_csv(pred_csv, index=False, encoding="utf-8-sig")
    df_folds.to_csv(fold_csv, index=False, encoding="utf-8-sig")
    df_summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write(f"FREQ: {freq_label}\n")
        f.write(f"MODEL_PATH: {model_path}\n")
        f.write(f"TRAIN_MAT_PATH: {train_mat_path}\n")
        f.write(f"MAT_PATH: {mat_path}\n")
        f.write(f"RIDGE_ALPHA: {RIDGE_ALPHA}\n")
        f.write(f"mean_train: {mean_train}\n")
        f.write(f"std_train: {std_train}\n\n")

        f.write("[DirectRawFull]\n")
        f.write(f"RMSE: {raw_metrics['RMSE']}\n")
        f.write(f"MAE: {raw_metrics['MAE']}\n")
        f.write(f"R2: {raw_metrics['R2']}\n\n")

        f.write("[LOOCV_Raw]\n")
        f.write(f"RMSE: {oof_raw_metrics['RMSE']}\n")
        f.write(f"MAE: {oof_raw_metrics['MAE']}\n")
        f.write(f"R2: {oof_raw_metrics['R2']}\n\n")

        f.write("[LOOCV_Linear]\n")
        f.write(f"RMSE: {oof_lin_metrics['RMSE']}\n")
        f.write(f"MAE: {oof_lin_metrics['MAE']}\n")
        f.write(f"R2: {oof_lin_metrics['R2']}\n\n")

        f.write("[LOOCV_Ridge]\n")
        f.write(f"RMSE: {oof_ridge_metrics['RMSE']}\n")
        f.write(f"MAE: {oof_ridge_metrics['MAE']}\n")
        f.write(f"R2: {oof_ridge_metrics['R2']}\n")

    print(f"\n[save] {freq_label} 逐样本预测结果: {pred_csv}")
    print(f"[save] {freq_label} 每次LOO细节    : {fold_csv}")
    print(f"[save] {freq_label} 汇总结果       : {summary_csv}")
    print(f"[save] {freq_label} 文本汇总       : {summary_txt}")

    return {
        "Frequency": freq_label,
        "MeasuredSamples": int(n_samples),
        "DirectRawFull_RMSE": raw_metrics["RMSE"],
        "DirectRawFull_MAE": raw_metrics["MAE"],
        "DirectRawFull_R2": raw_metrics["R2"],
        "LOOCV_Raw_RMSE": oof_raw_metrics["RMSE"],
        "LOOCV_Raw_MAE": oof_raw_metrics["MAE"],
        "LOOCV_Raw_R2": oof_raw_metrics["R2"],
        "LOOCV_Linear_RMSE": oof_lin_metrics["RMSE"],
        "LOOCV_Linear_MAE": oof_lin_metrics["MAE"],
        "LOOCV_Linear_R2": oof_lin_metrics["R2"],
        "LOOCV_Ridge_RMSE": oof_ridge_metrics["RMSE"],
        "LOOCV_Ridge_MAE": oof_ridge_metrics["MAE"],
        "LOOCV_Ridge_R2": oof_ridge_metrics["R2"],
    }


# ===================== 主流程 =====================
def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    all_results: List[Dict[str, Any]] = []

    for freq_cfg in FREQ_CONFIGS:
        result = run_single_frequency(freq_cfg)
        all_results.append(result)

    df_all_summary = pd.DataFrame(all_results)

    multi_csv = os.path.join(OUTPUT_ROOT, "multi_frequency_loocv_summary.csv")
    multi_xlsx = os.path.join(OUTPUT_ROOT, "multi_frequency_loocv_summary.xlsx")
    multi_txt = os.path.join(OUTPUT_ROOT, "multi_frequency_loocv_summary.txt")

    df_all_summary.to_csv(multi_csv, index=False, encoding="utf-8-sig")
    df_all_summary.to_excel(multi_xlsx, index=False)

    with open(multi_txt, "w", encoding="utf-8") as f:
        f.write(df_all_summary.to_string(index=False))

    print("\n" + "=" * 80)
    print("[done] 多频率结果汇总完成")
    print(df_all_summary.to_string(index=False))
    print("=" * 80)
    print(f"[save] 多频率总汇总 CSV : {multi_csv}")
    print(f"[save] 多频率总汇总 XLSX: {multi_xlsx}")
    print(f"[save] 多频率总汇总 TXT : {multi_txt}")


if __name__ == "__main__":
    main()