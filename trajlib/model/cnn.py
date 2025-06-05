import torch
import torch.nn as nn

class CNNEncoder(nn.Module):
    """
    config 需要包含:
        - 'd_model'      : 每个时间步的输入维度
        - 'hidden_size'  : CNN 输出特征维度（对应卷积后的通道数）
        - 'kernel_size'  : 卷积核大小
        - 'num_layers'   : 卷积层堆叠数
        - 'dropout'      : dropout 概率
    """
    def __init__(self, config: dict):
        super().__init__()
        self.d_model = config["d_model"]
        self.hidden_size = config["hidden_size"]
        self.kernel_size = config["kernel_size"]
        self.num_layers = config.get("num_layers", 2)
        self.dropout = config.get("dropout", 0.0)

        # 多层卷积
        self.conv_layers = nn.ModuleList()
        in_channels = self.d_model
        for _ in range(self.num_layers):
            self.conv_layers.append(nn.Conv1d(in_channels, self.hidden_size, kernel_size=self.kernel_size, padding=self.kernel_size//2))
            in_channels = self.hidden_size
        
        # Dropout层
        self.dropout_layer = nn.Dropout(self.dropout)
        
        # 用于输出预测的全连接层
        self.fc = nn.Linear(self.hidden_size, 1)  # 用于回归任务

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        """
        x    : (B, T, d_model)
        mask : (B, T)  —— 1 表示有效，0 表示 padding；若 None 则视为定长
        return:
            output : (B, T, 1)
        """
        # x 的形状是 (B, T, d_model)，需要转为 (B, d_model, T) 来匹配 Conv1d 的输入要求
        x = x.permute(0, 2, 1)  # 变为 (B, d_model, T)

        for conv in self.conv_layers:
            x = conv(x)
            x = torch.relu(x)
            x = self.dropout_layer(x)  # Dropout

        # 将输出卷积层后的特征从 (B, hidden_size, T) 转回 (B, T, hidden_size)
        x = x.permute(0, 2, 1)  # 变为 (B, T, hidden_size)

        # 使用全连接层进行最终的预测
        output = self.fc(x)  # (B, T, 1) 预测每个时间步的值（例如轨迹的下一个坐标）
        
        return output


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
