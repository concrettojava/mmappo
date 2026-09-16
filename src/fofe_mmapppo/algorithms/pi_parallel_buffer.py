"""Sequence-preserving rollout storage for PI-Net.

PI-Net carries a persistent per-entity belief, so actor observations cannot be
flattened across time before policy replay.  This buffer keeps the complete
``time x environment x agent`` structure and stores the structured actor input
needed to deterministically reconstruct the recurrent policy state.

The current Phase-8 trainer contract is intentionally strict: every batch of
parallel environments starts at an episode boundary, therefore PI belief state
is zero at t=0 for every environment.  Padding after termination is represented
by ``active=0`` and must not advance actor belief state during replay.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
import torch


@dataclass
class PIParallelRolloutBuffer:
    n_envs: int
    n_agents: int
    self_features: List[np.ndarray] = field(default_factory=list)
    entity_features: List[np.ndarray] = field(default_factory=list)
    evidence_mask: List[np.ndarray] = field(default_factory=list)
    evidence_meta: List[np.ndarray] = field(default_factory=list)
    states: List[np.ndarray] = field(default_factory=list)
    actions: List[np.ndarray] = field(default_factory=list)
    log_probs: List[np.ndarray] = field(default_factory=list)
    rewards: List[np.ndarray] = field(default_factory=list)
    dones: List[np.ndarray] = field(default_factory=list)
    values: List[np.ndarray] = field(default_factory=list)
    active: List[np.ndarray] = field(default_factory=list)

    def add(
        self,
        self_features,
        entity_features,
        evidence_mask,
        evidence_meta,
        states,
        actions,
        log_probs,
        rewards,
        dones,
        values,
        active,
    ) -> None:
        self.self_features.append(np.asarray(self_features, dtype=np.float32))
        self.entity_features.append(np.asarray(entity_features, dtype=np.float32))
        self.evidence_mask.append(np.asarray(evidence_mask, dtype=np.float32))
        self.evidence_meta.append(np.asarray(evidence_meta, dtype=np.float32))
        self.states.append(np.asarray(states, dtype=np.float32))
        self.actions.append(np.asarray(actions, dtype=np.int64))
        self.log_probs.append(np.asarray(log_probs, dtype=np.float32))
        self.rewards.append(np.asarray(rewards, dtype=np.float32))
        self.dones.append(np.asarray(dones, dtype=np.float32))
        self.values.append(np.asarray(values, dtype=np.float32))
        self.active.append(np.asarray(active, dtype=np.float32))

    def __len__(self) -> int:
        return len(self.rewards)

    def as_arrays(self) -> dict[str, np.ndarray]:
        return {
            "self_features": np.asarray(self.self_features, dtype=np.float32),
            "entity_features": np.asarray(self.entity_features, dtype=np.float32),
            "evidence_mask": np.asarray(self.evidence_mask, dtype=np.float32),
            "evidence_meta": np.asarray(self.evidence_meta, dtype=np.float32),
            "states": np.asarray(self.states, dtype=np.float32),
            "actions": np.asarray(self.actions, dtype=np.int64),
            "log_probs": np.asarray(self.log_probs, dtype=np.float32),
            "rewards": np.asarray(self.rewards, dtype=np.float32),
            "dones": np.asarray(self.dones, dtype=np.float32),
            "values": np.asarray(self.values, dtype=np.float32),
            "active": np.asarray(self.active, dtype=np.float32),
        }

    def compute_gae(self, gamma: float, gae_lambda: float) -> dict[str, np.ndarray]:
        """Compute GAE independently for each environment/agent trajectory."""
        data = self.as_arrays()
        rewards = data["rewards"]
        values = data["values"]
        dones = data["dones"]
        active = data["active"]
        if rewards.ndim != 3:
            raise ValueError(f"Expected rewards [T,E,N], got {rewards.shape}")

        T, E, N = rewards.shape
        if E != self.n_envs or N != self.n_agents:
            raise ValueError(
                f"buffer metadata says E={self.n_envs},N={self.n_agents} but data is {rewards.shape}"
            )
        advantages = np.zeros_like(rewards, dtype=np.float32)
        gae = np.zeros((E, N), dtype=np.float32)
        next_values = np.zeros((E, N), dtype=np.float32)

        for t in reversed(range(T)):
            nonterminal = 1.0 - dones[t]
            delta = rewards[t] + gamma * next_values * nonterminal - values[t]
            gae = delta + gamma * gae_lambda * nonterminal * gae
            gae *= active[t]
            advantages[t] = gae
            next_values = values[t]

        data["advantages"] = advantages
        data["returns"] = advantages + values
        return data

    @staticmethod
    def to_torch(data: dict[str, np.ndarray], device: torch.device | str) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for key, value in data.items():
            tensor = torch.as_tensor(value, device=device)
            out[key] = tensor.long() if key == "actions" else tensor.float()
        return out
