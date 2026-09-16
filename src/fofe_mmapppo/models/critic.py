"""Centralized MLP critic used by the fixed-vector MAPPO baseline."""
from __future__ import annotations

import torch
from torch import nn


def _init(layer: nn.Linear, gain: float):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0.0)
    return layer


class MLPCritic(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            _init(nn.Linear(input_dim, hidden_dim), gain=nn.init.calculate_gain("tanh")),
            nn.Tanh(),
            _init(nn.Linear(hidden_dim, hidden_dim), gain=nn.init.calculate_gain("tanh")),
            nn.Tanh(),
            _init(nn.Linear(hidden_dim, 1), gain=1.0),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).squeeze(-1)
