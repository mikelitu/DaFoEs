import torch
import torch.nn as nn
from typing import List
import torch.nn.functional as F


class RecurrencyBlock(nn.Module):
    def __init__(self, embed_dim: int = 512, hidden_size: int = 12, num_blocks: int = 2) -> None:
        super(RecurrencyBlock, self).__init__()
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.num_blocks = num_blocks

        self.lstm1 = nn.LSTM(input_size=embed_dim, hidden_size=hidden_size, num_layers=num_blocks, batch_first=True, dropout=0.)
        self.lstm2 = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_blocks, batch_first=True, dropout=0.)
        self.fc = nn.Linear(hidden_size, 3)

    def forward(self, features: torch.Tensor, robot_state: torch.Tensor) -> torch.Tensor:
        batch_size = features.shape[0]

        if robot_state is not None:
            rs_size = robot_state.shape[-1]
            padding_dim = (512 - rs_size - 1)
            padded_state = F.pad(robot_state, (1, padding_dim), 'constant', 0)
            x = torch.cat([features, padded_state], dim=1)
        
        x = x.reshape(batch_size, -1, self.embed_dim)
        h_0 = torch.autograd.Variable(torch.randn(self.num_blocks, batch_size, self.hidden_size).float().cuda())
        c_0 = torch.autograd.Variable(torch.randn(self.num_blocks, batch_size, self.hidden_size).float().cuda())
        x, (h_n, c_n) = self.lstm1(x, (h_0, c_0))
        x, _ = self.lstm2(x, (h_n, c_n))
        x = x[:, -1, :]
        x = self.fc(x)

        return x

        
            