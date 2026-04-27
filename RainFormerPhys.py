import os
import random
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torch.nn.functional as F
import h5py
from scipy.io import loadmat

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams.update({
    'font.size': 18,
    'axes.titlesize': 18,
    'axes.labelsize': 16,
    'xtick.labelsize': 15,
    'ytick.labelsize': 15,
    'legend.fontsize': 18,
    'figure.titlesize': 18
})


# ===================== 1. 配置区域 =====================

class Config:
    # 改成 mat 文件路径
    data_path = r"./dataset_MP_Rreg_140GHz_20s_20000samples.mat"

    # 训练相关
    batch_size = 64
    num_epochs = 30
    learning_rate = 1e-3
    early_stop_patience = 8

    # 数据集划分比例
    train_ratio = 0.7
    val_ratio = 0.15   # 剩下 0.15 为 test

    # 随机种子
    seed = 2025

    # 输出
    model_save_path = "best_model_MP_140_rainformer.pth"
    result_dir = "./results_rainformer"


cfg = Config()


# ===================== 2. 工具函数 =====================

def set_seed(seed: int = 2025):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else np.nan


def try_load_mat(path: str):
    """
    优先尝试 h5py 读取 v7.3 mat；
    若失败，则退回 scipy.io.loadmat。
    """
    try:
        with h5py.File(path, "r") as f:
            print("[info] 使用 h5py 读取成功")
            print("[info] mat 文件中的变量:", list(f.keys()))
            data = {}
            for k in f.keys():
                data[k] = np.array(f[k])
            return data
    except Exception as e:
        print("[info] h5py 读取失败，尝试 loadmat:", e)
        data = loadmat(path)
        keys = [k for k in data.keys() if not k.startswith("__")]
        print("[info] 使用 loadmat 读取成功")
        print("[info] mat 文件中的变量:", keys)
        return data


def find_first_existing_key(data_dict, candidate_keys):
    for k in candidate_keys:
        if k in data_dict:
            return k
    return None


# ===================== 3. 数据加载与预处理（改为 .mat 读取） =====================

def load_dataset_from_mat(cfg: Config) -> Tuple[np.ndarray, np.ndarray]:
    """
    直接从 mat 文件读取 X, Y
    目标:
        X.shape = (N_samples, seq_len)
        y.shape = (N_samples,)
    """
    data = try_load_mat(cfg.data_path)

    x_key = find_first_existing_key(data, ["X", "x", "inputs", "Input", "data"])
    y_key = find_first_existing_key(data, ["Y", "y", "labels", "Label", "target", "targets"])

    if x_key is None:
        raise KeyError("未在 mat 文件中找到 X")
    if y_key is None:
        raise KeyError("未在 mat 文件中找到 Y")

    X = np.array(data[x_key], dtype=np.float32)
    y = np.array(data[y_key], dtype=np.float32)

    # 参考你给的 mat 读取代码：处理 MATLAB v7.3 常见转置问题
    if X.ndim == 2 and X.shape[0] < X.shape[1]:
        X = X.T

    if y.ndim == 2:
        y = y.T if y.shape[0] < y.shape[1] else y
    y = y.squeeze()

    print(f"[info] 原始读取后形状: X={X.shape}, y={y.shape}")

    if X.ndim != 2:
        raise ValueError(f"X 维度异常，期望二维，实际为 {X.ndim} 维")
    if y.ndim != 1:
        raise ValueError(f"y 维度异常，期望一维，实际为 {y.ndim} 维")
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"样本数不匹配: X.shape[0]={X.shape[0]}, y.shape[0]={y.shape[0]}")

    # 全局标准化，保持和参考 mat 读取代码一致
    mean = X.mean()
    std = X.std()
    X = (X - mean) / (std + 1e-8)

    print(f"[info] 标准化完成 mean={mean:.4f}, std={std:.4f}")
    print(f"[info] 加载完成: 样本数={X.shape[0]}, 每个样本长度={X.shape[1]}")

    return X, y


# ===================== 4. PyTorch 数据集 =====================

class RainDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = self.X[idx].unsqueeze(0)  # (1, L)
        y = self.y[idx]
        return x, y


# ===================== 5. 你的网络结构：保持不变 =====================

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation 通道注意力模块
    """
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y


class HybridAttentionConvBlock(nn.Module):
    """
    混合注意力卷积块：这里保持结构不变
    """
    def __init__(self, dim: int, num_heads: int = 4, conv_kernel: int = 9,
                 ffn_expansion: int = 2, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads,
                                          dropout=dropout, batch_first=True)

        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = dim * ffn_expansion

        self.dwconv = nn.Conv1d(dim, dim, kernel_size=conv_kernel,
                                padding=conv_kernel // 2, groups=dim)
        self.pw1 = nn.Linear(dim, hidden_dim * 2)
        self.pw2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C)
        residual = x
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = residual + self.dropout(attn_out)

        residual = x
        x_norm = self.norm2(x)  # (B, L, C)

        # depthwise conv in temporal dimension
        conv_in = x_norm.transpose(1, 2)  # (B, C, L)
        conv_out = self.dwconv(conv_in).transpose(1, 2)  # (B, L, C)

        gate_in = self.pw1(conv_out)  # (B, L, 2*hidden)
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
        pe = pe.unsqueeze(0)  # (1, L, C)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C)
        L = x.size(1)
        return x + self.pe[:, :L, :]


class AttentiveStatPooling(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(dim))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: (B, L, C)
        scores = torch.einsum("blc,c->bl", tokens, self.query)
        weights = torch.softmax(scores, dim=1)
        pooled = torch.bmm(weights.unsqueeze(1), tokens).squeeze(1)
        return pooled


class RainFormerPhys(nn.Module):
    """混合注意力卷积主干 + 物理统计特征融合的降雨估计网络。"""

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
                HybridAttentionConvBlock(
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
        # x: (B, 1, L)，已经过标准化
        stat_mean = x.mean(dim=-1)
        stat_std = x.std(dim=-1)
        phys_feat = torch.cat([stat_mean, stat_std], dim=1)

        tokens = self.patch_embed(x).transpose(1, 2)  # (B, L, C)
        tokens = self.pos_encoder(tokens)
        for block in self.blocks:
            tokens = block(tokens)

        pooled = self.pool(tokens)
        fused = torch.cat([pooled, phys_feat], dim=1)
        out = self.head(fused)
        return out.squeeze(-1)


# ===================== 6. 数据划分 =====================

def split_dataset(X: np.ndarray, y: np.ndarray, cfg: Config):
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    train_end = int(n_samples * cfg.train_ratio)
    val_end = int(n_samples * (cfg.train_ratio + cfg.val_ratio))

    idx_train = indices[:train_end]
    idx_val = indices[train_end:val_end]
    idx_test = indices[val_end:]

    X_train, y_train = X[idx_train], y[idx_train]
    X_val, y_val = X[idx_val], y[idx_val]
    X_test, y_test = X[idx_test], y[idx_test]

    print(f"[info] 数据划分: train={len(idx_train)}, val={len(idx_val)}, test={len(idx_test)}")
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


# ===================== 7. 训练与评估流程 =====================

def train_and_evaluate(cfg: Config):
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] 使用设备: {device}")

    # ---- 1) 加载并预处理数据 ----
    X, y = load_dataset_from_mat(cfg)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_dataset(X, y, cfg)

    train_dataset = RainDataset(X_train, y_train)
    val_dataset = RainDataset(X_val, y_val)
    test_dataset = RainDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)

    # ---- 2) 定义模型与优化器 ----
    seq_len = X.shape[1]
    model = RainFormerPhys(in_channels=1, seq_len=seq_len).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    best_val = float("inf")
    best_state = None
    patience_counter = 0

    os.makedirs(cfg.result_dir, exist_ok=True)

    train_losses = []
    val_losses = []

    # ---- 3) 训练 ----
    for epoch in range(cfg.num_epochs):
        model.train()
        train_loss_sum = 0.0

        for x, yb in train_loader:
            x = x.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * x.size(0)

        train_loss = train_loss_sum / len(train_loader.dataset)
        train_losses.append(train_loss)

        # 验证
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for x, yb in val_loader:
                x = x.to(device)
                yb = yb.to(device)
                pred = model(x)
                loss = criterion(pred, yb)
                val_loss_sum += loss.item() * x.size(0)

        val_loss = val_loss_sum / len(val_loader.dataset)
        val_losses.append(val_loss)

        prev_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]["lr"]
        if new_lr < prev_lr:
            print(f"[lr] ReduceLROnPlateau: lr {prev_lr:.6g} -> {new_lr:.6g}")

        print(f"Epoch {epoch+1:03d} | Train {train_loss:.6f} | Val {val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = copy_model_state(model)
            torch.save(best_state, cfg.model_save_path)
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter > cfg.early_stop_patience:
            print("Early stopping")
            break

    # ---- 4) 测试 ----
    model.load_state_dict(torch.load(cfg.model_save_path, map_location=device))
    model.eval()

    preds = []
    targets = []

    with torch.no_grad():
        for x, yb in test_loader:
            x = x.to(device)
            pred = model(x).cpu().numpy()
            preds.append(pred)
            targets.append(yb.numpy())

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)

    errors = preds - targets
    abs_errors = np.abs(errors)

    mse = np.mean(errors ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(abs_errors)
    r2 = r2_score(targets, preds)

    print("\n===== 测试结果 =====")
    print("RMSE:", rmse)
    print("MAE :", mae)
    print("R2  :", r2)

    # ---- 5) 保存结果 ----
    df_out = pd.DataFrame({
        "True_R_mmph": targets,
        "Pred_R_mmph": preds,
        "Error_mmph": errors,
        "AbsError_mmph": abs_errors
    })

    csv_path = os.path.join(cfg.result_dir, "test_true_pred_error_rainformer.csv")
    xlsx_path = os.path.join(cfg.result_dir, "test_true_pred_error_rainformer.xlsx")
    df_out.to_csv(csv_path, index=False, encoding="utf-8-sig")
    df_out.to_excel(xlsx_path, index=False)

    print(f"[save] 测试集 True / Pred / Error 已保存到：\n  {csv_path}\n  {xlsx_path}")

    # ---- 6) 保存图 ----
    plt.figure()
    plt.scatter(targets, preds, s=10, alpha=0.5)
    m = min(targets.min(), preds.min())
    M = max(targets.max(), preds.max())
    plt.plot([m, M], [m, M], 'r--')
    plt.xlabel("True R")
    plt.ylabel("Pred R")
    plt.grid(True)
    plt.savefig(os.path.join(cfg.result_dir, "R_true_vs_pred_rainformer.png"), dpi=200, bbox_inches="tight")

    plt.figure()
    plt.hist(preds - targets, bins=40)
    plt.xlabel("Error")
    plt.ylabel("Count")
    plt.grid(True)
    plt.savefig(os.path.join(cfg.result_dir, "error_hist_rainformer.png"), dpi=200, bbox_inches="tight")


def copy_model_state(model: nn.Module):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


if __name__ == "__main__":
    train_and_evaluate(cfg)