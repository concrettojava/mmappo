"""Batch geometry cache for the parallel trainer.

The reference environment remains object-based.  This helper batches only the
post-movement geometry shared by many reward/combat calculations across several
environments, then installs per-environment distance matrices that
``CooperativeUAVEnv._dist`` can reuse.
"""
from __future__ import annotations

import numpy as np


def install_geometry_cache(envs, active_mask=None) -> None:
    """Install post-movement pairwise distance caches for active environments.

    The expected scenario sizes are fixed (8 UAVs, 4 targets, 3 threats), but
    the implementation derives sizes from each environment and validates that
    all active environments agree.  Inactive environments are ignored.
    """
    if active_mask is None:
        active_indices = list(range(len(envs)))
    else:
        active_indices = [i for i, active in enumerate(active_mask) if active]
    if not active_indices:
        return

    first = envs[active_indices[0]]
    n_uavs = len(first.uavs)
    n_targets = len(first.targets)
    n_threats = len(first.threats)
    e_count = len(active_indices)

    uav_xy = np.empty((e_count, n_uavs, 2), dtype=np.float64)
    target_xy = np.empty((e_count, n_targets, 2), dtype=np.float64)
    threat_xy = np.empty((e_count, n_threats, 2), dtype=np.float64)

    for row, env_index in enumerate(active_indices):
        env = envs[env_index]
        if (len(env.uavs), len(env.targets), len(env.threats)) != (
            n_uavs, n_targets, n_threats
        ):
            raise ValueError("all batched environments must have matching entity counts")
        for i, obj in enumerate(env.uavs):
            uav_xy[row, i, 0] = obj.x
            uav_xy[row, i, 1] = obj.y
        for i, obj in enumerate(env.targets):
            target_xy[row, i, 0] = obj.x
            target_xy[row, i, 1] = obj.y
        for i, obj in enumerate(env.threats):
            threat_xy[row, i, 0] = obj.x
            threat_xy[row, i, 1] = obj.y

    def distances(a, b):
        delta = a[:, :, None, :] - b[:, None, :, :]
        return np.sqrt(np.sum(delta * delta, axis=-1))

    uu = distances(uav_xy, uav_xy)
    ut = distances(uav_xy, target_xy)
    uh = distances(uav_xy, threat_xy)

    for row, env_index in enumerate(active_indices):
        envs[env_index]._geometry_cache = {
            "uav_uav": uu[row],
            "uav_target": ut[row],
            "uav_threat": uh[row],
        }
