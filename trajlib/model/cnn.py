import torch
import torch.nn as nn


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model: int, hidden_size: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden_size)
        self.fc2 = nn.Linear(hidden_size, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

# utils: 把输入 mask 扩展到与 x 同形状
def _expand_mask(mask: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    将 mask 扩展 / 转换成与 x (B, L, D) 同形状。
    支持形状：
      1) (B, L)                 —— 序列掩码
      2) (B, D)                 —— 特征维掩码
      3) (B, L, D)              —— 已对齐
      4) (B, L, L) & L == x.size(1)
    """
    B, L, D = x.shape
    # ① (B, L)  —— token-mask
    if mask.dim() == 2 and mask.size(0) == B and mask.size(1) == L:
        return mask.unsqueeze(-1).expand(B, L, D)
    # ② (B, D)  —— channel-mask
    if mask.dim() == 2 and mask.size(0) == B and mask.size(1) == D:
        return mask.unsqueeze(1).expand(B, L, D)
    # ③ (B, L, D) —— 已经对齐
    if mask.dim() == 3 and mask.shape == (B, L, D):
        return mask
    # ④ (B, L, L) —— Transformer padding mask/broadcast mask
    if mask.dim() == 3 and mask.size(0) == B and mask.size(1) == mask.size(2) == L:
        # 取第 0 列即可（所有列相同）；也可用 mask.any(dim=-1)
        seq_mask = mask[..., 0]                      # (B, L)
        return seq_mask.unsqueeze(-1).expand(B, L, D)
    # — 其它未支持形状 —
    raise ValueError(f"Mask shape {mask.shape} incompatible with x {x.shape}")


class ConvEncoderLayer(nn.Module):
    """
    单层 CNN 编码器 = Conv1d + 残差 + LN + FFN + 残差 + LN
    """
    def __init__(self, config: dict):
        super().__init__()
        d_model   = config["d_model"]
        k_size    = config.get("kernel_size", 3)
        dilation  = config.get("dilation", 1)
        padding   = (k_size // 2) * dilation   # 保持长度不变
        dropout_p = config["dropout"]

        self.conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=k_size,
            padding=padding,
            dilation=dilation,
            groups=config.get("groups", 1),
            bias=True
        )
        self.bn       = nn.BatchNorm1d(d_model)
        self.relu     = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_p)

        self.ffn       = PositionWiseFeedForward(d_model, config["hidden_size"])
        self.dropout2  = nn.Dropout(dropout_p)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        """
        x:    (B, L, D)
        mask: (B, L)|(B, D)|(B, L, D) —— 1=keep, 0=pad
        """
        if mask is not None:
            full_mask = _expand_mask(mask, x)
            x = x.masked_fill(full_mask == 0, 0.0)

        # --- 时序卷积 ---
        y = x.transpose(1, 2)               # (B, D, L)
        y = self.conv(y)
        y = self.bn(y)
        y = self.relu(y)
        y = y.transpose(1, 2)               # (B, L, D)

        if mask is not None:
            full_mask = _expand_mask(mask, y)
            y = y.masked_fill(full_mask == 0, 0.0)

        x = self.norm1(x + self.dropout1(y))  # Residual-1

        # --- FFN ---
        z = self.ffn(x)
        x = self.norm2(x + self.dropout2(z))  # Residual-2

        if mask is not None:
            full_mask = _expand_mask(mask, x)
            x = x.masked_fill(full_mask == 0, 0.0)

        return x


class CNNEncoder(nn.Module):
    """
    堆叠多层 ConvEncoderLayer。
    """
    def __init__(self, config: dict):
        super().__init__()
        self.layers = nn.ModuleList(
            [ConvEncoderLayer(config) for _ in range(config["num_layers"])]
        )

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x


class CNN(nn.Module):
    def __init__(self, config):
        super(CNN, self).__init__()
        self.encoder = CNNEncoder(config)

    def forward(self, x, mask=None):
        """
        x   : (B, T, d_model)
        mask: (B, T)  1=有效，0=padding
        """
        return self.encoder(x, mask)
