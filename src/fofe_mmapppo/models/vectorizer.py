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
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


TYPE_ORDER = ("Stk", "Rec", "Com")
TYPE_TO_INDEX = {name: i for i, name in enumerate(TYPE_ORDER)}
OBSERVATION_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class FixedVectorizer:
    world_size: float = 4000.0
    n_uavs: int = 8
    n_targets: int = 4
    n_threats: int = 3

    @property
    def uav_dim(self) -> int:
        return 14

    @property
    def self_dim(self) -> int:
        """Own-UAV record size: regular UAV state plus two local qualities."""
        return self.uav_dim + 2

    @property
    def target_dim(self) -> int:
        return 10

    @property
    def threat_dim(self) -> int:
        return 10

    @property
    def observation_dim(self) -> int:
        return (
            self.self_dim
            + (self.n_uavs - 1) * self.uav_dim
            + self.n_targets * self.target_dim
            + self.n_threats * self.threat_dim
        )

    @property
    def state_dim(self) -> int:
        return (
            self.n_uavs * self.uav_dim
            + (self.n_uavs - 1) * self.uav_dim
            + self.n_targets * self.target_dim
            + self.n_threats * self.threat_dim
        )

    @staticmethod
    def assert_checkpoint_compatible(checkpoint: Mapping[str, Any]) -> None:
        """Reject checkpoints trained before the self-quality schema change."""
        version = checkpoint.get("observation_schema_version")
        if version != OBSERVATION_SCHEMA_VERSION:
            raise ValueError(
                "checkpoint uses the legacy 182-dimensional observation schema; "
                "it cannot be resumed or evaluated with the 184-dimensional "
                "comm_quality/recon_quality schema. Retrain from scratch."
            )

    def _write_pose(self, out: np.ndarray, offset: int, pose: Mapping[str, Any]) -> None:
        geo = pose["geo"]
        body = pose["body"]
        scale = float(self.world_size)
        out[offset + 0] = float(geo["x"]) / scale
        out[offset + 1] = float(geo["y"]) / scale
        out[offset + 2] = float(geo["yaw"]) / np.pi
        out[offset + 3] = float(body["x"]) / scale
        out[offset + 4] = float(body["y"]) / scale
        out[offset + 5] = float(body["yaw"]) / np.pi
        out[offset + 6] = float(body["range"]) / scale
        out[offset + 7] = float(body["bearing"]) / np.pi

    def _write_uav(self, out: np.ndarray, offset: int, record: Mapping[str, Any]) -> None:
        out[offset] = 1.0
        out[offset + 1] = float(record["idx"]) / max(1, self.n_uavs - 1)
        type_idx = TYPE_TO_INDEX.get(record["type"])
        if type_idx is not None:
            out[offset + 2 + type_idx] = 1.0
        out[offset + 5] = 1.0 if record["alive"] else 0.0
        self._write_pose(out, offset + 6, record["pose"])

    def _write_self(self, out: np.ndarray, offset: int, record: Mapping[str, Any]) -> None:
        self._write_uav(out, offset, record)
        out[offset + self.uav_dim] = float(record["comm_quality"])
        out[offset + self.uav_dim + 1] = float(record["recon_quality"])

    def _write_target(self, out: np.ndarray, offset: int, record: Mapping[str, Any]) -> None:
        out[offset] = 1.0
        out[offset + 1] = 1.0 if record["alive"] else 0.0
        self._write_pose(out, offset + 2, record["pose"])

    def _write_threat(self, out: np.ndarray, offset: int, record: Mapping[str, Any]) -> None:
        out[offset] = 1.0
        out[offset + 1] = float(record["range"]) / float(self.world_size)
        self._write_pose(out, offset + 2, record["pose"])

    def _write_records(self, out, offset, records, count, dim, writer) -> None:
        # Scenario records are already almost always in idx order. Avoid sorting
        # when possible, but retain deterministic semantics for arbitrary input.
        ordered = records
        if len(records) > 1:
            last = -1
            in_order = True
            for record in records:
                idx = int(record.get("idx", 0))
                if idx < last:
                    in_order = False
                    break
                last = idx
            if not in_order:
                ordered = sorted(records, key=lambda x: int(x.get("idx", 0)))
        for row, record in enumerate(ordered[:count]):
            writer(out, offset + row * dim, record)

    def _pose(self, pose: Mapping[str, Any]) -> list[float]:
        out = np.zeros(8, dtype=np.float32)
        self._write_pose(out, 0, pose)
        return out.tolist()

    def _uav(self, record: Mapping[str, Any]) -> np.ndarray:
        out = np.zeros(self.uav_dim, dtype=np.float32)
        self._write_uav(out, 0, record)
        return out

    def _self(self, record: Mapping[str, Any]) -> np.ndarray:
        out = np.zeros(self.self_dim, dtype=np.float32)
        self._write_self(out, 0, record)
        return out

    def _target(self, record: Mapping[str, Any]) -> np.ndarray:
        out = np.zeros(self.target_dim, dtype=np.float32)
        self._write_target(out, 0, record)
        return out

    def _threat(self, record: Mapping[str, Any]) -> np.ndarray:
        out = np.zeros(self.threat_dim, dtype=np.float32)
        self._write_threat(out, 0, record)
        return out

    @staticmethod
    def _pad(records: Iterable[Mapping[str, Any]], count: int, encode, dim: int) -> np.ndarray:
        ordered = sorted(records, key=lambda x: int(x.get("idx", 0)))
        out = np.zeros((count, dim), dtype=np.float32)
        for row, record in enumerate(ordered[:count]):
            out[row] = encode(record)
        return out.reshape(-1)

    def observation_into(self, obs: Mapping[str, Any] | None, out: np.ndarray) -> None:
        out.fill(0.0)
        if obs is None:
            return
        p = 0
        self._write_self(out, p, obs["self"])
        p += self.self_dim
        self._write_records(out, p, obs["neighbors"], self.n_uavs - 1, self.uav_dim, self._write_uav)
        p += (self.n_uavs - 1) * self.uav_dim
        self._write_records(out, p, obs["targets"], self.n_targets, self.target_dim, self._write_target)
        p += self.n_targets * self.target_dim
        self._write_records(out, p, obs["threats"], self.n_threats, self.threat_dim, self._write_threat)

    def state_into(self, state: Mapping[str, Any], out: np.ndarray) -> None:
        out.fill(0.0)
        p = 0
        self._write_records(out, p, state["uavs"], self.n_uavs, self.uav_dim, self._write_uav)
        p += self.n_uavs * self.uav_dim
        self._write_records(out, p, state["neighbors"], self.n_uavs - 1, self.uav_dim, self._write_uav)
        p += (self.n_uavs - 1) * self.uav_dim
        self._write_records(out, p, state["targets"], self.n_targets, self.target_dim, self._write_target)
        p += self.n_targets * self.target_dim
        self._write_records(out, p, state["threats"], self.n_threats, self.threat_dim, self._write_threat)

    def observation(self, obs: Mapping[str, Any] | None) -> np.ndarray:
        out = np.zeros(self.observation_dim, dtype=np.float32)
        self.observation_into(obs, out)
        return out

    def state(self, state: Mapping[str, Any]) -> np.ndarray:
        out = np.zeros(self.state_dim, dtype=np.float32)
        self.state_into(state, out)
        return out

    def batch_observations(self, observations: Mapping[int, Mapping[str, Any] | None]) -> np.ndarray:
        out = np.zeros((self.n_uavs, self.observation_dim), dtype=np.float32)
        for i in range(self.n_uavs):
            self.observation_into(observations[i], out[i])
        return out

    def batch_states(self, states: Mapping[int, Mapping[str, Any]]) -> np.ndarray:
        out = np.zeros((self.n_uavs, self.state_dim), dtype=np.float32)
        for i in range(self.n_uavs):
            self.state_into(states[i], out[i])
        return out

    def parallel_batch(
        self,
        observations: Sequence[Mapping[int, Mapping[str, Any] | None]],
        states: Sequence[Mapping[int, Mapping[str, Any]]],
        finished: np.ndarray | Sequence[bool] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Encode many environments into preallocated contiguous arrays.

        This removes repeated np.stack/np.concatenate allocations from the hot
        rollout path while preserving exactly the same fixed-vector semantics.
        """
        E = len(observations)
        obs_batch = np.zeros((E, self.n_uavs, self.observation_dim), dtype=np.float32)
        state_batch = np.zeros((E, self.n_uavs, self.state_dim), dtype=np.float32)
        active = np.zeros((E, self.n_uavs), dtype=np.float32)
        if finished is None:
            finished = np.zeros(E, dtype=bool)
        for e in range(E):
            if finished[e]:
                continue
            env_obs = observations[e]
            env_states = states[e]
            for i in range(self.n_uavs):
                obs_i = env_obs[i]
                if obs_i is not None:
                    active[e, i] = 1.0
                    self.observation_into(obs_i, obs_batch[e, i])
                self.state_into(env_states[i], state_batch[e, i])
        return obs_batch, state_batch, active
