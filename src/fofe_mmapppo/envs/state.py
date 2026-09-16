"""Centralized training state in the same flexible schema as Eq. (8).

Eq. (9) describes the global state from each RSUAV's viewpoint using the same
information types as the local observation, but with observability constraints
removed.  Thus every agent-specific critic state contains all RSUAVs, all
mobile targets and all threatened areas, represented relative to that agent.
"""
from __future__ import annotations

from .observation import _target_record, _threat_record, _uav_record


def get_agent_state(env, observer_idx: int):
    observer = env.uavs[observer_idx]

    return {
        # x^I: state information of all RSUAVs.  Keep self first, followed by
        # the remaining UAVs, for a deterministic structured representation.
        "uavs": [
            _uav_record(observer, env.uavs[idx])
            for idx in [observer_idx] + [u.idx for u in env.uavs if u.idx != observer_idx]
        ],
        # x^N, x^M and x^T eliminate local observability constraints.  The
        # explicit channels below preserve all objects for centralized training.
        "neighbors": [
            _uav_record(observer, u)
            for u in env.uavs
            if u.idx != observer_idx
        ],
        "targets": [
            _target_record(observer, target)
            for target in env.targets
        ],
        "threats": [
            _threat_record(observer, threat)
            for threat in env.threats
        ],
    }


def get_global_states(env):
    """Return one full-information state for each RSUAV viewpoint."""
    return {u.idx: get_agent_state(env, u.idx) for u in env.uavs}
