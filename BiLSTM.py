import torch
import torch.nn as nn
from torch import Tensor
class BiLSTMClassifier(nn.Module):
    def __init__(self, input_dim: int, seq_len: int, num_classes: int, hidden_dim: int, num_layers: int,
                 dropout: float):
        super(BiLSTMClassifier, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # 双向 LSTM 层
        self.bilstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout, bidirectional=True)

        # 全连接层
        self.fc = nn.Linear(hidden_dim * 2, num_classes)  # 双向 LSTM 输出维度是 hidden_dim * 2

    def forward(self, x: Tensor) -> Tensor:
        # LSTM 处理时序数据
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).to(x.device)  # 初始隐藏状态，双向需要 * 2
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_dim).to(x.device)  # 初始细胞状态

        out, _ = self.bilstm(x, (h0, c0))  # out: [batch_size, seq_len, hidden_dim * 2]

        # 取最后一个时间步的输出
        out = out[:, -1, :]  # [batch_size, hidden_dim * 2]

        # 全连接层输出
        out = self.fc(out)  # [batch_size, num_classes]

        return out
