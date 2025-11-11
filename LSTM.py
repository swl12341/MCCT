import torch
import torch.nn as nn
from torch import Tensor


class LSTMClassifier(nn.Module):
    def __init__(self, input_dim: int, seq_len: int, num_classes: int, hidden_dim: int, num_layers: int,
                 dropout: float):
        super(LSTMClassifier, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # LSTM 层
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)

        # 全连接层
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        # LSTM 处理时序数据
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)  # 初始隐藏状态
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)  # 初始细胞状态

        out, _ = self.lstm(x, (h0, c0))  # out: [batch_size, seq_len, hidden_dim]

        # 取最后一个时间步的输出
        out = out[:, -1, :]  # [batch_size, hidden_dim]

        # 全连接层输出
        out = self.fc(out)  # [batch_size, num_classes]

        return out
