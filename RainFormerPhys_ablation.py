import math
from typing import Literal

import torch
import torch.nn as nn


VALID_VARIANTS = {
    "full",
    "no_phys_stat",
    "avg_pool",
    "no_pos",
    "no_attn",
    "no_dwconv",
    "no_gate",
    "simple_patch",
}


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 4000):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class AttentiveStatPooling(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(dim))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        scores = torch.einsum("blc,c->bl", tokens, self.query)
        weights = torch.softmax(scores, dim=1)
        return torch.bmm(weights.unsqueeze(1), tokens).squeeze(1)


class AveragePooling(nn.Module):
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens.mean(dim=1)


class GatedDepthwiseFFN(nn.Module):
    def __init__(self, dim: int, conv_kernel: int = 9, expansion: int = 2, dropout: float = 0.1):
        super().__init__()
        hidden = dim * expansion
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=conv_kernel, padding=conv_kernel // 2, groups=dim)
        self.pw1 = nn.Linear(dim, hidden * 2)
        self.pw2 = nn.Linear(hidden, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dwconv(x.transpose(1, 2)).transpose(1, 2)
        x = self.pw1(x)
        a, b = x.chunk(2, dim=-1)
        x = a * torch.sigmoid(b)
        x = self.pw2(x)
        return self.dropout(x)


class DepthwiseNoGateFFN(nn.Module):
    def __init__(self, dim: int, conv_kernel: int = 9, expansion: int = 2, dropout: float = 0.1):
        super().__init__()
        hidden = dim * expansion
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=conv_kernel, padding=conv_kernel // 2, groups=dim)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dwconv(x.transpose(1, 2)).transpose(1, 2)
        return self.net(x)


class MLPFFN(nn.Module):
    def __init__(self, dim: int, expansion: int = 2, dropout: float = 0.1):
        super().__init__()
        hidden = dim * expansion
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HybridAttentionConvBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        conv_kernel: int = 9,
        ffn_expansion: int = 2,
        dropout: float = 0.1,
        use_attn: bool = True,
        ffn_type: Literal["gated_dwconv", "mlp", "dwconv_no_gate"] = "gated_dwconv",
    ):
        super().__init__()
        self.use_attn = use_attn
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True) if use_attn else None
        self.norm2 = nn.LayerNorm(dim)

        if ffn_type == "gated_dwconv":
            self.ffn = GatedDepthwiseFFN(dim, conv_kernel, ffn_expansion, dropout)
        elif ffn_type == "mlp":
            self.ffn = MLPFFN(dim, ffn_expansion, dropout)
        elif ffn_type == "dwconv_no_gate":
            self.ffn = DepthwiseNoGateFFN(dim, conv_kernel, ffn_expansion, dropout)
        else:
            raise ValueError(f"Unsupported ffn_type: {ffn_type}")

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_attn:
            residual = x
            attn_in = self.norm1(x)
            attn_out, _ = self.attn(attn_in, attn_in, attn_in)
            x = residual + self.dropout(attn_out)

        residual = x
        x = residual + self.ffn(self.norm2(x))
        return x


class RainFormerPhysAblation(nn.Module):
    def __init__(self, variant: str, in_channels: int, seq_len: int, embed_dim: int = 128, num_blocks: int = 4):
        super().__init__()
        if variant not in VALID_VARIANTS:
            raise ValueError(f"Unknown variant '{variant}', expected one of: {sorted(VALID_VARIANTS)}")
        self.variant = variant

        if variant == "simple_patch":
            self.patch_embed = nn.Sequential(
                nn.Conv1d(in_channels, embed_dim, kernel_size=7, padding=3),
                nn.GELU(),
            )
        else:
            self.patch_embed = nn.Sequential(
                nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels),
                nn.BatchNorm1d(in_channels),
                nn.GELU(),
                nn.Conv1d(in_channels, embed_dim, kernel_size=7, padding=3),
                nn.GELU(),
            )

        self.pos_encoder = nn.Identity() if variant == "no_pos" else SinusoidalPositionalEncoding(embed_dim, seq_len)

        use_attn = variant != "no_attn"
        if variant == "no_dwconv":
            ffn_type = "mlp"
        elif variant == "no_gate":
            ffn_type = "dwconv_no_gate"
        else:
            ffn_type = "gated_dwconv"

        self.blocks = nn.ModuleList(
            [
                HybridAttentionConvBlock(
                    dim=embed_dim,
                    num_heads=4,
                    conv_kernel=9,
                    ffn_expansion=2,
                    dropout=0.1,
                    use_attn=use_attn,
                    ffn_type=ffn_type,
                )
                for _ in range(num_blocks)
            ]
        )

        self.pool = AveragePooling() if variant == "avg_pool" else AttentiveStatPooling(embed_dim)

        head_in_dim = embed_dim if variant == "no_phys_stat" else embed_dim + 2
        self.use_phys_stat = variant != "no_phys_stat"
        self.head = nn.Sequential(
            nn.LayerNorm(head_in_dim),
            nn.Linear(head_in_dim, 96),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(96, 32),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_embed(x).transpose(1, 2)
        tokens = self.pos_encoder(tokens)
        for block in self.blocks:
            tokens = block(tokens)
        pooled = self.pool(tokens)

        if self.use_phys_stat:
            stat_mean = x.mean(dim=-1)
            stat_std = x.std(dim=-1)
            pooled = torch.cat([pooled, stat_mean, stat_std], dim=1)

        return self.head(pooled).squeeze(-1)


def get_rainformer_model(variant: str, in_channels: int, seq_len: int, embed_dim: int = 128, num_blocks: int = 4) -> nn.Module:
    return RainFormerPhysAblation(
        variant=variant,
        in_channels=in_channels,
        seq_len=seq_len,
        embed_dim=embed_dim,
        num_blocks=num_blocks,
    )


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
