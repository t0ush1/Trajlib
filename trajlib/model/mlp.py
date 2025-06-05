import torch
import torch.nn as nn

class MLPEncoder(nn.Module):
    """
    config 需要包含:
        - 'd_model'      : 每个时间步的输入维度
        - 'hidden_size'  : MLP 输出特征维度（每一层的隐藏单元数）
        - 'num_layers'   : MLP 层数
        - 'dropout'      : dropout 概率
    """
    def __init__(self, config: dict):
        super(MLPEncoder, self).__init__()
        self.d_model = config["d_model"]
        self.hidden_size = config["hidden_size"]
        self.num_layers = config.get("num_layers", 2)
        self.dropout = config.get("dropout", 0.0)

        # 多层感知机
        self.mlp_layers = nn.ModuleList()
        in_features = self.d_model
        for _ in range(self.num_layers):
            self.mlp_layers.append(nn.Linear(in_features, self.hidden_size))
            self.mlp_layers.append(nn.ReLU())
            self.mlp_layers.append(nn.Dropout(self.dropout))  # Dropout 层
            in_features = self.hidden_size
        
        # 输出层
        self.fc_out = nn.Linear(self.hidden_size, 1)  # 用于回归任务（预测）

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        """
        x    : (B, T, d_model)
        mask : (B, T)  —— 1 表示有效，0 表示 padding；若 None 则视为定长
        return:
            output : (B, T, 1)
        """
        B, T, D = x.shape
        # 将数据转化为 (B * T, D) 适配 MLP 输入
        x = x.view(B * T, D)  # (B * T, D)
        
        # 通过多层感知机处理
        for layer in self.mlp_layers:
            x = layer(x)
        
        # 恢复回 (B, T, hidden_size) 形状
        x = x.view(B, T, self.hidden_size)  # (B, T, hidden_size)

        # 通过输出层预测每个时间步的值
        output = self.fc_out(x)  # (B, T, 1)
        return output


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
