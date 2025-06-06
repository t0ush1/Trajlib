import torch
import torch.nn as nn


class PositionWiseFeedForward(nn.Module):
    """
    FFN: Linear → ReLU → Linear  (隐藏维为 hidden_size)
    """
    def __init__(self, d_model: int, hidden_size: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden_size)
        self.fc2 = nn.Linear(hidden_size, d_model)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


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


class MLPEncoderLayer(nn.Module):
    """
    单层 MLP 编码器
    = Pre-Norm → FFN → Dropout → Residual → LayerNorm
    """
    def __init__(self, config: dict):
        super().__init__()
        self.norm  = nn.LayerNorm(config["d_model"])
        self.ffn   = PositionWiseFeedForward(config["d_model"],
                                             config["hidden_size"])
        self.dropout = nn.Dropout(config["dropout"])

    def forward(self, x, mask=None):
        if mask is not None:                        # ① Padding 置零
            full_mask = _expand_mask(mask, x)
            x = x.masked_fill(full_mask == 0, 0.0)

        y = self.dropout(self.ffn(self.norm(x)))   # Pre-Norm → FFN
        x = x + y                                  # Residual

        if mask is not None:                        # ② 再次置零
            full_mask = _expand_mask(mask, x)
            x = x.masked_fill(full_mask == 0, 0.0)

        return x


class MLPEncoder(nn.Module):
    """
    堆叠多个 MLPEncoderLayer
    """
    def __init__(self, config: dict):
        super().__init__()
        self.layers = nn.ModuleList(
            [MLPEncoderLayer(config) for _ in range(config["num_layers"])]
        )

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x
        

class MLP(nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__()
        self.encoder = MLPEncoder(config)

    def forward(self, x, mask=None):
        """
        x   : (B, T, d_model)
        mask: (B, T)  1=有效，0=padding
        """
        return self.encoder(x, mask)
