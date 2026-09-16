"""On-policy rollout storage and GAE for MAPPO."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
import torch


@dataclass
class RolloutBuffer:
    n_agents: int
    obs: List[np.ndarray] = field(default_factory=list)
    states: List[np.ndarray] = field(default_factory=list)
    actions: List[np.ndarray] = field(default_factory=list)
    log_probs: List[np.ndarray] = field(default_factory=list)
    rewards: List[np.ndarray] = field(default_factory=list)
    dones: List[np.ndarray] = field(default_factory=list)
    values: List[np.ndarray] = field(default_factory=list)
    active: List[np.ndarray] = field(default_factory=list)

    def add(self, obs, states, actions, log_probs, rewards, dones, values, active):
        self.obs.append(np.asarray(obs, dtype=np.float32))
        self.states.append(np.asarray(states, dtype=np.float32))
        self.actions.append(np.asarray(actions, dtype=np.int64))
        self.log_probs.append(np.asarray(log_probs, dtype=np.float32))
        self.rewards.append(np.asarray(rewards, dtype=np.float32))
        self.dones.append(np.asarray(dones, dtype=np.float32))
        self.values.append(np.asarray(values, dtype=np.float32))
        self.active.append(np.asarray(active, dtype=np.float32))

    def __len__(self):
        return len(self.rewards)

    def as_arrays(self):
        return {
            "obs": np.asarray(self.obs, dtype=np.float32),
            "states": np.asarray(self.states, dtype=np.float32),
            "actions": np.asarray(self.actions, dtype=np.int64),
            "log_probs": np.asarray(self.log_probs, dtype=np.float32),
            "rewards": np.asarray(self.rewards, dtype=np.float32),
            "dones": np.asarray(self.dones, dtype=np.float32),
            "values": np.asarray(self.values, dtype=np.float32),
            "active": np.asarray(self.active, dtype=np.float32),
        }

    def compute_gae(self, gamma: float, gae_lambda: float, last_values=None):
        """Compute generalized advantage estimation independently per agent.

        ``dones[t, i]`` describes the transition stored at t: it is one when
        agent i is terminal after receiving ``rewards[t, i]``.  This includes
        both episode termination and an individual UAV being destroyed.
        """
        data = self.as_arrays()
        rewards, values, dones = data["rewards"], data["values"], data["dones"]
        T, N = rewards.shape
        last_values = np.zeros(N, dtype=np.float32) if last_values is None else np.asarray(last_values, dtype=np.float32)

        advantages = np.zeros_like(rewards, dtype=np.float32)
        gae = np.zeros(N, dtype=np.float32)
        next_values = last_values
        for t in reversed(range(T)):
            nonterminal = 1.0 - dones[t]
            delta = rewards[t] + gamma * next_values * nonterminal - values[t]
            gae = delta + gamma * gae_lambda * nonterminal * gae
            advantages[t] = gae
            next_values = values[t]

        returns = advantages + values
        data["advantages"] = advantages
        data["returns"] = returns
        return data

    @staticmethod
    def to_torch(data, device):
        out = {}
        for key, value in data.items():
            tensor = torch.as_tensor(value, device=device)
            out[key] = tensor.long() if key == "actions" else tensor.float()
        return out
