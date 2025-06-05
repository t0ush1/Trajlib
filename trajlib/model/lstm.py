import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTMEncoder(nn.Module):
    """
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

        # 为了对齐 Transformer 的层归一化，这里给输出做一次 LayerNorm
        self.layernorm = nn.LayerNorm(
            self.hidden_size * (2 if self.bidirectional else 1)
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        """
        x    : (B, T, d_model)
        mask : (B, T)  —— 1 表示有效，0 表示 padding；若 None 则视为定长
        return:
            output : (B, T, hidden_size * num_directions)
        """
        if mask is not None:
            lengths = mask.sum(dim=1).cpu()
            packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
            packed_out, _ = self.lstm(packed)
            output, _ = pad_packed_sequence(packed_out, batch_first=True)
        else:
            output, _ = self.lstm(x)

        output = self.layernorm(output)         # 稳定训练
        return output

        # lengths = mask.sum(dim=1).cpu()
        # packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        # _, (h_n, _) = self.lstm(packed)
        # return h_n[-1]  # (B, H)

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
