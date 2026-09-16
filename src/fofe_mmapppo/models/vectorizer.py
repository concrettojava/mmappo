"""Fixed-size vectorization for the traditional MAPPO baseline.

The paper states that the MAPPO/MADDPG baselines replace flexible observation
with a fixed-size vectorized observation, but it does not publish the exact
padding/normalization recipe.  This module makes that implementation choice
explicit and isolated so it can later be swapped without changing the
environment or FOFE path.

Rules used here:
- deterministic ordering by object index;
- zero-padding to the known scenario maxima;
- one presence bit per padded record so a real all-zero pose is distinguishable
  from padding;
- positions/ranges normalized by battlefield size;
- angles normalized by pi;
- UAV type represented as a one-hot vector.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np


TYPE_ORDER = ("Stk", "Rec", "Com")


@dataclass(frozen=True)
class FixedVectorizer:
    world_size: float = 4000.0
    n_uavs: int = 8
    n_targets: int = 4
    n_threats: int = 3

    @property
    def uav_dim(self) -> int:
        # presence + normalized id + type(3) + alive + dual pose(8)
        return 14

    @property
    def target_dim(self) -> int:
        # presence + alive + dual pose(8)
        return 10

    @property
    def threat_dim(self) -> int:
        # presence + normalized threat radius + dual pose(8)
        return 10

    @property
    def observation_dim(self) -> int:
        return (
            self.uav_dim
            + (self.n_uavs - 1) * self.uav_dim
            + self.n_targets * self.target_dim
            + self.n_threats * self.threat_dim
        )

    @property
    def state_dim(self) -> int:
        # Mirrors the current Eq.(9) structured state schema exactly:
        # uavs / neighbors / targets / threats.
        return (
            self.n_uavs * self.uav_dim
            + (self.n_uavs - 1) * self.uav_dim
            + self.n_targets * self.target_dim
            + self.n_threats * self.threat_dim
        )

    def _pose(self, pose: Mapping[str, Any]) -> list[float]:
        geo = pose["geo"]
        body = pose["body"]
        scale = float(self.world_size)
        return [
            float(geo["x"]) / scale,
            float(geo["y"]) / scale,
            float(geo["yaw"]) / np.pi,
            float(body["x"]) / scale,
            float(body["y"]) / scale,
            float(body["yaw"]) / np.pi,
            float(body["range"]) / scale,
            float(body["bearing"]) / np.pi,
        ]

    def _uav(self, record: Mapping[str, Any]) -> np.ndarray:
        idx_scale = max(1, self.n_uavs - 1)
        type_one_hot = [1.0 if record["type"] == t else 0.0 for t in TYPE_ORDER]
        values = [
            1.0,
            float(record["idx"]) / idx_scale,
            *type_one_hot,
            1.0 if record["alive"] else 0.0,
            *self._pose(record["pose"]),
        ]
        return np.asarray(values, dtype=np.float32)

    def _target(self, record: Mapping[str, Any]) -> np.ndarray:
        values = [
            1.0,
            1.0 if record["alive"] else 0.0,
            *self._pose(record["pose"]),
        ]
        return np.asarray(values, dtype=np.float32)

    def _threat(self, record: Mapping[str, Any]) -> np.ndarray:
        values = [
            1.0,
            float(record["range"]) / float(self.world_size),
            *self._pose(record["pose"]),
        ]
        return np.asarray(values, dtype=np.float32)

    @staticmethod
    def _pad(records: Iterable[Mapping[str, Any]], count: int, encode, dim: int) -> np.ndarray:
        ordered = sorted(records, key=lambda x: int(x.get("idx", 0)))
        out = np.zeros((count, dim), dtype=np.float32)
        for row, record in enumerate(ordered[:count]):
            out[row] = encode(record)
        return out.reshape(-1)

    def observation(self, obs: Mapping[str, Any] | None) -> np.ndarray:
        if obs is None:
            return np.zeros(self.observation_dim, dtype=np.float32)

        chunks = [
            self._uav(obs["self"]),
            self._pad(obs["neighbors"], self.n_uavs - 1, self._uav, self.uav_dim),
            self._pad(obs["targets"], self.n_targets, self._target, self.target_dim),
            self._pad(obs["threats"], self.n_threats, self._threat, self.threat_dim),
        ]
        vector = np.concatenate(chunks).astype(np.float32, copy=False)
        if vector.shape != (self.observation_dim,):
            raise RuntimeError(f"unexpected observation shape {vector.shape}")
        return vector

    def state(self, state: Mapping[str, Any]) -> np.ndarray:
        chunks = [
            self._pad(state["uavs"], self.n_uavs, self._uav, self.uav_dim),
            self._pad(state["neighbors"], self.n_uavs - 1, self._uav, self.uav_dim),
            self._pad(state["targets"], self.n_targets, self._target, self.target_dim),
            self._pad(state["threats"], self.n_threats, self._threat, self.threat_dim),
        ]
        vector = np.concatenate(chunks).astype(np.float32, copy=False)
        if vector.shape != (self.state_dim,):
            raise RuntimeError(f"unexpected state shape {vector.shape}")
        return vector

    def batch_observations(self, observations: Mapping[int, Mapping[str, Any] | None]) -> np.ndarray:
        return np.stack([self.observation(observations[i]) for i in range(self.n_uavs)])

    def batch_states(self, states: Mapping[int, Mapping[str, Any]]) -> np.ndarray:
        return np.stack([self.state(states[i]) for i in range(self.n_uavs)])
