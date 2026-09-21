"""Identity-aligned structured observations for PI-Net.

The fixed-vector MAPPO baseline intentionally packs only the records that are
currently available.  That representation is appropriate for a feed-forward
baseline, but it cannot support a persistent per-entity belief because target
and threat identity is lost when records disappear/reappear.

``EntityTensorizer`` keeps stable slots:

- 7 teammate slots, keyed by global UAV id (excluding the observer)
- 4 target slots, keyed by target id
- 3 threat slots, keyed by threat id

No simulator-private information is added.  The tensorizer only rearranges the
structured local observation already available to the decentralized actor.
Future stale-information scenarios may attach optional ``age`` and ``source``
metadata to records; the current reference/contested scenarios simply emit
fresh evidence with unknown source.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from ..envs.entities import TYPE_PARAMS
from .vectorizer import TYPE_TO_INDEX


SELF_DIM = 16
ENTITY_DIM = 20
EVIDENCE_META_DIM = 4

ENTITY_TEAMMATE = 0
ENTITY_TARGET = 1
ENTITY_THREAT = 2

SOURCE_DIRECT = 0
SOURCE_RELAYED = 1
SOURCE_MEMORY = 2


@dataclass(frozen=True)
class EntityBatch:
    """Numpy tensors consumed by the PI actor.

    Shapes are documented for the parallel case ``[E, N, ...]`` where ``E``
    is the number of environments and ``N`` the number of controlled UAVs.
    ``single``/``agent`` helpers simply remove leading dimensions.
    """

    self_features: np.ndarray
    entity_features: np.ndarray
    evidence_mask: np.ndarray
    evidence_meta: np.ndarray
    active: np.ndarray


@dataclass(frozen=True)
class EntityTensorizer:
    world_size: float = 4000.0
    n_uavs: int = 8
    n_targets: int = 4
    n_threats: int = 3
    max_age: float = 200.0

    @property
    def n_entities(self) -> int:
        return (self.n_uavs - 1) + self.n_targets + self.n_threats

    @property
    def teammate_offset(self) -> int:
        return 0

    @property
    def target_offset(self) -> int:
        return self.n_uavs - 1

    @property
    def threat_offset(self) -> int:
        return self.target_offset + self.n_targets

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

    def _self(self, record: Mapping[str, Any]) -> np.ndarray:
        out = np.zeros(SELF_DIM, dtype=np.float32)
        out[0] = 1.0
        out[1] = float(record["idx"]) / max(1, self.n_uavs - 1)
        type_idx = TYPE_TO_INDEX.get(str(record["type"]))
        if type_idx is not None:
            out[2 + type_idx] = 1.0
        out[5] = 1.0 if record.get("alive", True) else 0.0
        self._write_pose(out, 6, record["pose"])
        out[14] = float(record["comm_quality"])
        out[15] = float(record["recon_quality"])
        return out

    def _teammate_slot(self, observer_idx: int, other_idx: int) -> int:
        if other_idx == observer_idx:
            raise ValueError("observer cannot occupy a teammate slot")
        # IDs below the observer keep their index; IDs above shift left by one.
        return other_idx if other_idx < observer_idx else other_idx - 1

    def _entity_slot(self, observer_idx: int, category: int, idx: int) -> int:
        if category == ENTITY_TEAMMATE:
            if not 0 <= idx < self.n_uavs:
                raise IndexError(f"UAV idx {idx} out of range")
            return self.teammate_offset + self._teammate_slot(observer_idx, idx)
        if category == ENTITY_TARGET:
            if not 0 <= idx < self.n_targets:
                raise IndexError(f"target idx {idx} out of range")
            return self.target_offset + idx
        if category == ENTITY_THREAT:
            if not 0 <= idx < self.n_threats:
                raise IndexError(f"threat idx {idx} out of range")
            return self.threat_offset + idx
        raise ValueError(f"unknown entity category {category}")

    @staticmethod
    def _source_one_hot(record: Mapping[str, Any]) -> tuple[float, float, float]:
        source = str(record.get("source", "unknown")).lower()
        if source == "direct":
            return 1.0, 0.0, 0.0
        if source in {"relay", "relayed", "shared"}:
            return 0.0, 1.0, 0.0
        if source in {"memory", "stale"}:
            return 0.0, 0.0, 1.0
        return 0.0, 0.0, 0.0

    def _write_entity(
        self,
        out: np.ndarray,
        meta: np.ndarray,
        record: Mapping[str, Any],
        category: int,
    ) -> None:
        """Encode only decentralized evidence available in ``record``."""
        out[0] = 1.0
        out[1 + category] = 1.0

        idx = int(record["idx"])
        if category == ENTITY_TEAMMATE:
            type_name = str(record["type"])
            type_idx = TYPE_TO_INDEX.get(type_name)
            if type_idx is not None:
                out[4 + type_idx] = 1.0
            out[7] = 1.0 if record.get("alive", True) else 0.0
            out[8] = idx / max(1, self.n_uavs - 1)
            params = TYPE_PARAMS[type_name]
            out[18] = float(params["speed"]) / 50.0
            out[19] = float(params["max_turn_deg_s"]) / 40.0
        elif category == ENTITY_TARGET:
            out[7] = 1.0 if record.get("alive", True) else 0.0
            out[8] = idx / max(1, self.n_targets - 1)
            out[18] = 8.0 / 50.0
            # Scenario-2 targets can add up to roughly 10 deg/s evasive steering.
            out[19] = 10.0 / 40.0
        else:
            out[7] = 1.0
            out[8] = idx / max(1, self.n_threats - 1)
            out[17] = float(record.get("range", 0.0)) / float(self.world_size)
            out[18] = 0.0
            out[19] = 0.0

        pose = record["pose"]
        geo = pose["geo"]
        body = pose["body"]
        scale = float(self.world_size)
        out[9] = float(geo["x"]) / scale
        out[10] = float(geo["y"]) / scale
        out[11] = float(geo["yaw"]) / np.pi
        out[12] = float(body["x"]) / scale
        out[13] = float(body["y"]) / scale
        out[14] = float(body["yaw"]) / np.pi
        out[15] = float(body["range"]) / scale
        out[16] = float(body["bearing"]) / np.pi

        age = float(record.get("age", 0.0))
        meta[0] = np.clip(age / max(self.max_age, 1.0), 0.0, 1.0)
        meta[1:4] = self._source_one_hot(record)

    def agent(self, obs: Mapping[str, Any] | None, observer_idx: int) -> EntityBatch:
        self_features = np.zeros((SELF_DIM,), dtype=np.float32)
        entities = np.zeros((self.n_entities, ENTITY_DIM), dtype=np.float32)
        mask = np.zeros((self.n_entities,), dtype=np.float32)
        meta = np.zeros((self.n_entities, EVIDENCE_META_DIM), dtype=np.float32)
        active = np.array(0.0, dtype=np.float32)
        if obs is None:
            return EntityBatch(self_features, entities, mask, meta, active)

        active[...] = 1.0
        self_features[:] = self._self(obs["self"])
        groups = (
            (obs.get("neighbors", ()), ENTITY_TEAMMATE),
            (obs.get("targets", ()), ENTITY_TARGET),
            (obs.get("threats", ()), ENTITY_THREAT),
        )
        for records, category in groups:
            for record in records:
                idx = int(record["idx"])
                slot = self._entity_slot(observer_idx, category, idx)
                self._write_entity(entities[slot], meta[slot], record, category)
                mask[slot] = 1.0
        return EntityBatch(self_features, entities, mask, meta, active)

    def environment(self, observations: Mapping[int, Mapping[str, Any] | None]) -> EntityBatch:
        self_features = np.zeros((self.n_uavs, SELF_DIM), dtype=np.float32)
        entities = np.zeros((self.n_uavs, self.n_entities, ENTITY_DIM), dtype=np.float32)
        mask = np.zeros((self.n_uavs, self.n_entities), dtype=np.float32)
        meta = np.zeros(
            (self.n_uavs, self.n_entities, EVIDENCE_META_DIM), dtype=np.float32
        )
        active = np.zeros((self.n_uavs,), dtype=np.float32)
        for observer_idx in range(self.n_uavs):
            encoded = self.agent(observations[observer_idx], observer_idx)
            self_features[observer_idx] = encoded.self_features
            entities[observer_idx] = encoded.entity_features
            mask[observer_idx] = encoded.evidence_mask
            meta[observer_idx] = encoded.evidence_meta
            active[observer_idx] = encoded.active
        return EntityBatch(self_features, entities, mask, meta, active)

    def parallel(
        self,
        observations: Sequence[Mapping[int, Mapping[str, Any] | None]],
        finished: np.ndarray | Sequence[bool] | None = None,
    ) -> EntityBatch:
        n_envs = len(observations)
        self_features = np.zeros((n_envs, self.n_uavs, SELF_DIM), dtype=np.float32)
        entities = np.zeros(
            (n_envs, self.n_uavs, self.n_entities, ENTITY_DIM), dtype=np.float32
        )
        mask = np.zeros((n_envs, self.n_uavs, self.n_entities), dtype=np.float32)
        meta = np.zeros(
            (n_envs, self.n_uavs, self.n_entities, EVIDENCE_META_DIM), dtype=np.float32
        )
        active = np.zeros((n_envs, self.n_uavs), dtype=np.float32)
        if finished is None:
            finished = np.zeros((n_envs,), dtype=bool)
        for e in range(n_envs):
            if bool(finished[e]):
                continue
            encoded = self.environment(observations[e])
            self_features[e] = encoded.self_features
            entities[e] = encoded.entity_features
            mask[e] = encoded.evidence_mask
            meta[e] = encoded.evidence_meta
            active[e] = encoded.active
        return EntityBatch(self_features, entities, mask, meta, active)
