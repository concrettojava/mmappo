"""Direct environment-to-array encoding for high-throughput MAPPO training.

This module reproduces ``FixedVectorizer`` semantics without first materializing
Eq. (8)/(9) nested dict/list pose structures. The structured observation/state
path remains the reference/debug representation; this fast path exists only to
remove Python object construction from the training hot loop.
"""
from __future__ import annotations

import math

import numpy as np

from .vectorizer import TYPE_TO_INDEX, FixedVectorizer


class DirectFixedVectorizer:
    """Encode a ``CooperativeUAVEnv`` directly into fixed float32 arrays."""

    def __init__(self, base: FixedVectorizer):
        self.base = base
        self.world_size = float(base.world_size)
        self.n_uavs = int(base.n_uavs)
        self.n_targets = int(base.n_targets)
        self.n_threats = int(base.n_threats)
        self.observation_dim = int(base.observation_dim)
        self.state_dim = int(base.state_dim)
        self.uav_dim = int(base.uav_dim)
        self.target_dim = int(base.target_dim)
        self.threat_dim = int(base.threat_dim)

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi

    def _write_pose(self, out, offset, observer, obj, yaw=None):
        obj_yaw = float(getattr(obj, "yaw", 0.0) if yaw is None else yaw)
        ox = float(observer.x)
        oy = float(observer.y)
        psi = float(observer.yaw)
        x = float(obj.x)
        y = float(obj.y)
        dx = x - ox
        dy = y - oy
        sin_psi = math.sin(psi)
        cos_psi = math.cos(psi)
        forward = dx * sin_psi + dy * cos_psi
        right = dx * cos_psi - dy * sin_psi
        scale = self.world_size

        out[offset + 0] = x / scale
        out[offset + 1] = y / scale
        out[offset + 2] = obj_yaw / math.pi
        out[offset + 3] = forward / scale
        out[offset + 4] = right / scale
        out[offset + 5] = self._wrap_angle(obj_yaw - psi) / math.pi
        out[offset + 6] = math.hypot(dx, dy) / scale
        out[offset + 7] = math.atan2(right, forward) / math.pi

    def _write_uav(self, out, offset, observer, uav):
        out[offset] = 1.0
        out[offset + 1] = float(uav.idx) / max(1, self.n_uavs - 1)
        type_idx = TYPE_TO_INDEX.get(uav.uav_type)
        if type_idx is not None:
            out[offset + 2 + type_idx] = 1.0
        out[offset + 5] = 1.0 if uav.alive else 0.0
        self._write_pose(out, offset + 6, observer, uav)

    def _write_target(self, out, offset, observer, target):
        out[offset] = 1.0
        out[offset + 1] = 1.0 if target.alive else 0.0
        self._write_pose(out, offset + 2, observer, target)

    def _write_threat(self, out, offset, observer, threat):
        out[offset] = 1.0
        out[offset + 1] = float(threat.radius) / self.world_size
        self._write_pose(out, offset + 2, observer, threat, yaw=0.0)

    def _record_cache(self, observer, uavs, targets, threats):
        """Encode each observer/object pair once and reuse the record blocks.

        The fixed local observation and centralized state duplicate many of the
        same objects.  Previously every copy recomputed sin/cos, body-frame
        coordinates, range and bearing.  Cache the final float32 record blocks
        for this observer, then assemble observation/state by slice copies.
        """
        uav_records = np.zeros((self.n_uavs, self.uav_dim), dtype=np.float32)
        target_records = np.zeros((self.n_targets, self.target_dim), dtype=np.float32)
        threat_records = np.zeros((self.n_threats, self.threat_dim), dtype=np.float32)

        for u in uavs[: self.n_uavs]:
            self._write_uav(uav_records[u.idx], 0, observer, u)
        for t in targets[: self.n_targets]:
            self._write_target(target_records[t.idx], 0, observer, t)
        for th in threats[: self.n_threats]:
            self._write_threat(threat_records[th.idx], 0, observer, th)
        return uav_records, target_records, threat_records

    def encode_env(self, env):
        """Return ``obs, state, active`` with exactly FixedVectorizer shapes."""
        obs = np.zeros((self.n_uavs, self.observation_dim), dtype=np.float32)
        state = np.zeros((self.n_uavs, self.state_dim), dtype=np.float32)
        active = np.zeros(self.n_uavs, dtype=np.float32)

        # One communication/detection pass per environment, not once per agent.
        graph = env.communication_graph()
        components = env.communication_components(graph=graph)
        shared = env.shared_detection(graph=graph)

        uavs = env.uavs
        targets = env.targets
        threats = env.threats

        for observer_idx in range(self.n_uavs):
            observer = uavs[observer_idx]
            uav_records, target_records, threat_records = self._record_cache(
                observer, uavs, targets, threats
            )

            # Local observation: same channel layout/padding as FixedVectorizer.
            if observer.alive:
                active[observer_idx] = 1.0
                p = 0
                obs[observer_idx, p:p + self.uav_dim] = uav_records[observer_idx]
                p += self.uav_dim

                subgroup = components.get(observer_idx, {observer_idx})
                row = 0
                for idx in sorted(subgroup - {observer_idx}):
                    if uavs[idx].alive and row < self.n_uavs - 1:
                        start = p + row * self.uav_dim
                        obs[observer_idx, start:start + self.uav_dim] = uav_records[idx]
                        row += 1
                p += (self.n_uavs - 1) * self.uav_dim

                visible_targets, visible_threats = shared.get(observer_idx, (set(), set()))
                row = 0
                for idx in sorted(visible_targets):
                    if targets[idx].alive and row < self.n_targets:
                        start = p + row * self.target_dim
                        obs[observer_idx, start:start + self.target_dim] = target_records[idx]
                        row += 1
                p += self.n_targets * self.target_dim

                for row, idx in enumerate(sorted(visible_threats)[: self.n_threats]):
                    start = p + row * self.threat_dim
                    obs[observer_idx, start:start + self.threat_dim] = threat_records[idx]

            # Centralized state: exact current FixedVectorizer state schema.
            p = 0
            state[observer_idx, p:p + self.n_uavs * self.uav_dim] = uav_records.reshape(-1)
            p += self.n_uavs * self.uav_dim

            row = 0
            for idx in range(self.n_uavs):
                if idx == observer_idx:
                    continue
                start = p + row * self.uav_dim
                state[observer_idx, start:start + self.uav_dim] = uav_records[idx]
                row += 1
            p += (self.n_uavs - 1) * self.uav_dim

            state[observer_idx, p:p + self.n_targets * self.target_dim] = target_records.reshape(-1)
            p += self.n_targets * self.target_dim
            state[observer_idx, p:p + self.n_threats * self.threat_dim] = threat_records.reshape(-1)

        return obs, state, active

    def encode_parallel(self, envs, finished=None):
        E = len(envs)
        obs = np.zeros((E, self.n_uavs, self.observation_dim), dtype=np.float32)
        state = np.zeros((E, self.n_uavs, self.state_dim), dtype=np.float32)
        active = np.zeros((E, self.n_uavs), dtype=np.float32)
        if finished is None:
            finished = np.zeros(E, dtype=bool)
        for e, env in enumerate(envs):
            if finished[e]:
                continue
            obs[e], state[e], active[e] = self.encode_env(env)
        return obs, state, active
