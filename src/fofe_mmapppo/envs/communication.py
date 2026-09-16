"""Communication subgroups, local detection and shared knowledge."""
from __future__ import annotations

from typing import Dict, Set

import numpy as np

from .entities import UAV


def _positions(objects) -> np.ndarray:
    """Return object xy coordinates as a compact float64 matrix."""
    if not objects:
        return np.empty((0, 2), dtype=np.float64)
    return np.fromiter(
        (coord for obj in objects for coord in (obj.x, obj.y)),
        dtype=np.float64,
        count=2 * len(objects),
    ).reshape(len(objects), 2)


def _distance_matrix(a, b=None) -> np.ndarray:
    """Pairwise Euclidean distances without Python-level nested loops."""
    axy = _positions(a)
    bxy = axy if b is None else _positions(b)
    if axy.size == 0 or bxy.size == 0:
        return np.empty((len(a), len(a) if b is None else len(b)), dtype=np.float64)
    delta = axy[:, None, :] - bxy[None, :, :]
    return np.sqrt(np.einsum("...k,...k->...", delta, delta))


def communication_graph(env) -> Dict[int, Set[int]]:
    alive = [u for u in env.uavs if u.alive]
    graph = {u.idx: set() for u in alive}
    if len(alive) < 2:
        return graph

    distances = _distance_matrix(alive)
    comm_ranges = np.asarray([u.p["comm_range"] for u in alive], dtype=np.float64)
    thresholds = np.maximum(comm_ranges[:, None], comm_ranges[None, :])
    linked = distances < thresholds
    np.fill_diagonal(linked, False)

    rows, cols = np.nonzero(np.triu(linked, k=1))
    for i, j in zip(rows.tolist(), cols.tolist()):
        a = alive[i].idx
        b = alive[j].idx
        graph[a].add(b)
        graph[b].add(a)
    return graph


def communication_components(env, graph=None) -> Dict[int, Set[int]]:
    graph = env.communication_graph() if graph is None else graph
    comps = {}
    visited = set()
    for node in graph:
        if node in visited:
            continue
        stack = [node]
        comp = set()
        while stack:
            n = stack.pop()
            if n in comp:
                continue
            comp.add(n)
            visited.add(n)
            stack.extend(graph[n] - comp)
        for n in comp:
            comps[n] = set(comp)
    return comps


def _direct_detected_targets(env, uav: UAV) -> Set[int]:
    alive_targets = [t for t in env.targets if t.alive]
    if not alive_targets:
        return set()
    xy = _positions(alive_targets)
    dx = xy[:, 0] - uav.x
    dy = xy[:, 1] - uav.y
    found = np.nonzero(dx * dx + dy * dy < float(uav.p["recon_range"]) ** 2)[0]
    return {alive_targets[i].idx for i in found.tolist()}


def _direct_detected_threats(env, uav: UAV) -> Set[int]:
    if not env.threats:
        return set()
    xy = _positions(env.threats)
    dx = xy[:, 0] - uav.x
    dy = xy[:, 1] - uav.y
    found_idx = np.nonzero(dx * dx + dy * dy < float(uav.p["recon_range"]) ** 2)[0]
    found = {env.threats[i].idx for i in found_idx.tolist()}
    uav.threat_memory |= found
    return found


def shared_detection(
    env,
    *,
    graph=None,
    direct_targets=None,
    direct_threats=None,
):
    """Return component-shared detections.

    Callers that already computed communication/detection data may pass it in;
    this avoids repeating the same geometry work during reward construction.
    """
    graph = env.communication_graph() if graph is None else graph
    comps = communication_components(env, graph=graph)

    if direct_targets is None or direct_threats is None:
        direct_targets = {} if direct_targets is None else direct_targets
        direct_threats = {} if direct_threats is None else direct_threats
        for u in env.uavs:
            if not u.alive:
                continue
            if u.idx not in direct_targets:
                direct_targets[u.idx] = env._direct_detected_targets(u)
            if u.idx not in direct_threats:
                direct_threats[u.idx] = env._direct_detected_threats(u)

    uav_by_id = {u.idx: u for u in env.uavs}
    result = {}
    for u in env.uavs:
        if not u.alive:
            continue
        comp = comps.get(u.idx, {u.idx})
        visible_targets = set().union(*(direct_targets.get(v, set()) for v in comp))
        visible_threats = set().union(*(
            (direct_threats.get(v, set()) | uav_by_id[v].threat_memory)
            for v in comp
        ))
        result[u.idx] = (visible_targets, visible_threats)
    return result
