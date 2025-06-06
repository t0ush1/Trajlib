import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

def _to_token_mask(mask: torch.Tensor, seq_len: int) -> torch.Tensor:
    """
    把任意合理形状的 mask 变成 (B, L) 形式：
      1) (B, L)               —— 直接返回
      2) (B, L, D)            —— 任一特征为有效即有效：mask.any(-1)
      3) (B, L, L)            —— Transformer 方阵：取第 0 列
    返回 bool 类型 (B, L)
    """
    if mask.dim() == 2:                          # (B,L)
        return mask.bool()
    if mask.dim() == 3:
        B, L1, L2 = mask.shape
        # (B,L,D)               -> any over D
        if L2 != L1:                             
            return mask.any(dim=2)
        # (B,L,L) 方阵           -> 取第 0 列
        if L1 == L2:
            return mask[:, :, 0].bool()
    raise ValueError(f"Unsupported mask shape {mask.shape}")


class LSTMEncoder(nn.Module):
    """
    LSTM → Drop → Residual → Norm
    config 需要包含:
        - 'd_model'      : 每个时间步的输入维度
        - 'hidden_size'  : LSTM 隐藏层维度
        - 'num_layers'   : LSTM 堆叠层数
        - 'bidirectional': bool, 是否双向
        - 'dropout'      : LSTM 层间 dropout（num_layers>1 时才生效）
    """
    def __init__(self, config: dict):
        super().__init__()
        self.d_model = config["d_model"]
        self.hidden_size = config["hidden_size"]
        self.num_layers = config.get("num_layers", 1)
        self.bidirectional = config.get("bidirectional", False)

        self.lstm = nn.LSTM(
            input_size=self.d_model,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=config.get("dropout", 0.0) if self.num_layers > 1 else 0.0, 
            bidirectional=self.bidirectional,
        )

        out_dim = self.hidden_size * (2 if self.bidirectional else 1)
        self.proj = nn.Linear(out_dim, self.d_model) if out_dim != self.d_model else nn.Identity()
        self.norm = nn.LayerNorm(self.d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        """
        x    : (B, T, d_model)
        mask : (B, T)  —— 1 表示有效，0 表示 padding；若 None 则视为定长
        return:
            output : (B, T, hidden_size * num_directions)
        """
        if mask is not None:
            token_mask = _to_token_mask(mask, x.size(1))      # (B,L) bool
            lengths = token_mask.sum(dim=1)                   # (B,)
            # 避免 0-length，至少设 1
            lengths = lengths.clamp(min=1).cpu()
            packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
            packed_out, _ = self.lstm(packed)
            output, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=x.size(1))
            # 把 padding 位置清零，便于后续池化
            output = output.masked_fill(token_mask.unsqueeze(-1) == 0, 0.0)
        else:
            output, _ = self.lstm(x)

        output = self.proj(output)         # 稳定训练
        return self.norm(output+x)


class LSTM(nn.Module):
    def __init__(self, config):
        super(LSTM, self).__init__()
        self.encoder = LSTMEncoder(config)

    def forward(self, x, mask=None):
        """
        x   : (B, T, d_model)
        mask: (B, T)  1=有效，0=padding
        """
        return self.encoder(x, mask)
