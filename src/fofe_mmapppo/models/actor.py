"""Categorical MLP actor used by the fixed-vector MAPPO baseline."""
from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Categorical


def _init(layer: nn.Linear, gain: float):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0.0)
    return layer


class MLPActor(nn.Module):
    def __init__(self, input_dim: int, action_dim: int = 7, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            _init(nn.Linear(input_dim, hidden_dim), gain=nn.init.calculate_gain("tanh")),
            nn.Tanh(),
            _init(nn.Linear(hidden_dim, hidden_dim), gain=nn.init.calculate_gain("tanh")),
            nn.Tanh(),
            _init(nn.Linear(hidden_dim, action_dim), gain=0.01),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    def distribution(self, obs: torch.Tensor) -> Categorical:
        return Categorical(logits=self(obs))

    def act(self, obs: torch.Tensor, deterministic: bool = False):
        dist = self.distribution(obs)
        action = torch.argmax(dist.logits, dim=-1) if deterministic else dist.sample()
        return action, dist.log_prob(action), dist.entropy()

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        dist = self.distribution(obs)
        return dist.log_prob(actions), dist.entropy()
