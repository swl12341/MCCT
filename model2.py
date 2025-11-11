# model2.py

import math
import torch
import torch.nn as nn
from torch import Tensor
from config import *

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: Tensor) -> Tensor:
        length = x.size(1)
        return x + self.pe[:, :length, :]

class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Conv1d(channels, channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels // reduction, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        w = self.pool(x)
        w = self.fc(w)
        return x * w

class MultiChannelCNN(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        base_ch = d_model // 2
        rem = d_model - base_ch * 2

        def make_branch(out_ch: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv1d(3, 32, kernel_size=3, padding=1),
                nn.BatchNorm1d(32),
                nn.ReLU(inplace=True),
                nn.Conv1d(32, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True)
            )

        self.acc_branch = make_branch(base_ch)
        self.gyr_branch = make_branch(base_ch)
        self.se = SEBlock(d_model)

    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 1)
        a = self.acc_branch(x[:, 0:3, :])
        g = self.gyr_branch(x[:, 3:6, :])
        fused = torch.cat([a, g], dim=1)
        return self.se(fused)

class CNNTransformerClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        num_classes: int,
        d_model: int,
        nhead: int,
        num_layers: int = 3,
        dropout: float = 0.1,
        dim_feedforward: float = 256,
    ) -> None:
        super().__init__()
        assert input_dim, "input_dim 必须为9"
        self.feature_extractor = MultiChannelCNN(d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len + 1)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=int(dim_feedforward),
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, 1)  # 输出一个logit用于sigmoid

    def forward(self, x: Tensor) -> Tensor:
        B, T, _ = x.size()
        features = self.feature_extractor(x)
        features = features.permute(0, 2, 1)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls_tokens, features], dim=1)
        tokens = self.pos_encoder(tokens)
        enc_out = self.transformer_encoder(tokens)
        cls_out = enc_out[:, 0, :]
        out = self.norm(cls_out)
        return self.fc(out).squeeze(1)  # 输出为[B]


class LSTMClassifier(nn.Module):
    def __init__(self, input_dim: int, seq_len: int, num_classes: int, hidden_dim: int, num_layers: int, dropout: float,
                 *args, **kwargs):
        super(LSTMClassifier, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # LSTM 层
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)

        # 输出层，单一得分用于二分类
        self.fc = nn.Linear(hidden_dim, 1)  # 输出一个单一得分用于二分类

    def forward(self, x: Tensor) -> Tensor:
        # LSTM 处理时序数据
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)  # 初始隐藏状态
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)  # 初始细胞状态

        out, _ = self.lstm(x, (h0, c0))  # out: [batch_size, seq_len, hidden_dim]

        # 取最后一个时间步的输出
        out = out[:, -1, :]  # [batch_size, hidden_dim]

        # 全连接层输出
        out = self.fc(out).squeeze(1)  # 输出为单一得分 [batch_size]，用于二分类

        return out


class BiLSTMClassifier(nn.Module):
    def __init__(self, input_dim: int, seq_len: int, num_classes: int, hidden_dim: int, num_layers: int,
                 dropout: float, *args, **kwargs):
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










