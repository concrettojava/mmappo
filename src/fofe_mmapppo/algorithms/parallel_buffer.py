"""Rollout storage for multiple independent environments.

Shape convention:
    time x env x agent x feature

GAE is computed independently for every (environment, agent) trajectory so
parallel rollout collection does not leak returns across environment boundaries.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
import torch


@dataclass
class ParallelRolloutBuffer:
    n_envs: int
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

    def compute_gae(self, gamma: float, gae_lambda: float):
        data = self.as_arrays()
        rewards = data["rewards"]
        values = data["values"]
        dones = data["dones"]
        active = data["active"]
        if rewards.ndim != 3:
            raise ValueError(f"Expected rewards [T,E,N], got {rewards.shape}")

        T, E, N = rewards.shape
        advantages = np.zeros_like(rewards, dtype=np.float32)
        gae = np.zeros((E, N), dtype=np.float32)
        next_values = np.zeros((E, N), dtype=np.float32)
        next_dones = np.ones((E, N), dtype=np.float32)

        for t in reversed(range(T)):
            nonterminal = 1.0 - next_dones
            delta = rewards[t] + gamma * next_values * nonterminal - values[t]
            gae = delta + gamma * gae_lambda * nonterminal * gae
            # Entries after an environment has already terminated are padding.
            gae *= active[t]
            advantages[t] = gae
            next_values = values[t]
            next_dones = dones[t]

        data["advantages"] = advantages
        data["returns"] = advantages + values
        return data

    @staticmethod
    def to_torch(data, device):
        out = {}
        for key, value in data.items():
            tensor = torch.as_tensor(value, device=device)
            out[key] = tensor.long() if key == "actions" else tensor.float()
        return out
