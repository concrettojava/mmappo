"""Paper-style flexible local observations (Eq. 8).

Each living RSUAV receives four information channels:

- ``self``: its own state
- ``neighbors``: other RSUAVs reachable through the current communication
  topology (multi-hop subgroup, excluding self)
- ``targets``: mobile targets directly observed by any subgroup member
- ``threats``: threats currently observed or remembered by any subgroup member

The paper states that states containing position and heading angle are expressed
in both geo coordinates and the observing RSUAV's body coordinate frame.  This
module preserves the structured, variable-length representation; FOFE tensor
encoding is intentionally left for the model layer.
"""
from __future__ import annotations

from typing import Any, Dict

from .coordinates import dual_pose


def _uav_record(observer, uav) -> Dict[str, Any]:
    return {
        "idx": int(uav.idx),
        "type": str(uav.uav_type),
        "alive": bool(uav.alive),
        "pose": dual_pose(observer, uav),
    }


def _self_record(env, observer) -> Dict[str, Any]:
    """Return the observer-only record, including local equipment health.

    The jammer field remains simulator-private. These values describe only the
    observing UAV's local communication and reconnaissance equipment; they are
    deliberately not attached to teammate records.
    """
    record = _uav_record(observer, observer)
    record["comm_quality"] = float(env.communication_factor(observer))
    record["recon_quality"] = float(env.reconnaissance_factor(observer))
    return record


def _target_record(observer, target) -> Dict[str, Any]:
    return {
        "idx": int(target.idx),
        "alive": bool(target.alive),
        "pose": dual_pose(observer, target),
    }


def _threat_record(observer, threat) -> Dict[str, Any]:
    # Threatened areas are fixed; no physical heading is defined in the paper.
    # Use yaw=0 only to keep the pose schema uniform.  The relevant threat
    # orientation-independent state is its range and position.
    return {
        "idx": int(threat.idx),
        "range": float(threat.radius),
        "pose": dual_pose(observer, threat, yaw=0.0),
    }


def _observation_from_shared(env, observer_idx: int, components, shared):
    """Materialize one observation from already-computed topology/detections."""
    observer = env.uavs[observer_idx]
    if not observer.alive:
        return None

    subgroup = components.get(observer_idx, {observer_idx})
    visible_targets, visible_threats = shared[observer_idx]
    return {
        "self": _self_record(env, observer),
        "neighbors": [
            _uav_record(observer, env.uavs[idx])
            for idx in sorted(subgroup - {observer_idx})
            if env.uavs[idx].alive
        ],
        "targets": [
            _target_record(observer, env.targets[idx])
            for idx in sorted(visible_targets)
            if env.targets[idx].alive
        ],
        "threats": [
            _threat_record(observer, env.threats[idx])
            for idx in sorted(visible_threats)
        ],
    }


def get_observation(env, observer_idx: int):
    """Return Eq. (8)-style local flexible observation for one RSUAV.

    This single-agent helper preserves the public/debug API.  The all-agent hot
    path below computes topology and shared detections once for the entire
    environment step.
    """
    observer = env.uavs[observer_idx]
    if not observer.alive:
        return None
    graph = env.communication_graph()
    components = env.communication_components(graph=graph)
    shared = env.shared_detection(graph=graph)
    return _observation_from_shared(env, observer_idx, components, shared)


def get_observations(env):
    """Return local flexible observations for all RSUAVs.

    Communication topology and direct/shared detections are properties of the
    environment step, not of the observer.  Older code rebuilt them separately
    for every UAV (two communication-graph builds per observer).  Compute them
    once here and reuse the result while only the coordinate-frame
    materialization remains observer-specific.
    """
    graph = env.communication_graph()
    components = env.communication_components(graph=graph)
    shared = env.shared_detection(graph=graph)
    return {
        u.idx: _observation_from_shared(env, u.idx, components, shared)
        for u in env.uavs
    }
