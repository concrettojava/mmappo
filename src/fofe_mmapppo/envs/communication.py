"""Communication subgroups, local detection and shared knowledge."""
from __future__ import annotations

from typing import Dict, Set

from .entities import UAV


def communication_graph(env) -> Dict[int, Set[int]]:
    """Build the direct communication graph.

    In the reference scenario this is the reproduced range rule. In the
    contested scenario each endpoint's effective communication range is reduced
    by the local jammer field; a link must fit inside the better endpoint's
    degraded radio range, preserving the original heterogeneous-range logic.
    """
    alive = [u for u in env.uavs if u.alive]
    graph = {u.idx: set() for u in alive}
    for i, a in enumerate(alive):
        for b in alive[i + 1:]:
            dx = a.x - b.x
            dy = a.y - b.y
            range_a = float(a.p["comm_range"]) * env.communication_factor(a)
            range_b = float(b.p["comm_range"]) * env.communication_factor(b)
            limit = max(range_a, range_b)
            if dx * dx + dy * dy < limit * limit:
                graph[a.idx].add(b.idx)
                graph[b.idx].add(a.idx)
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
    effective_range = float(uav.p["recon_range"]) * env.reconnaissance_factor(uav)
    limit2 = effective_range ** 2
    ux, uy = uav.x, uav.y
    return {
        t.idx
        for t in env.targets
        if t.alive and (ux - t.x) ** 2 + (uy - t.y) ** 2 < limit2
    }


def _direct_detected_threats(env, uav: UAV) -> Set[int]:
    effective_range = float(uav.p["recon_range"]) * env.reconnaissance_factor(uav)
    limit2 = effective_range ** 2
    ux, uy = uav.x, uav.y
    found = {
        th.idx
        for th in env.threats
        if (ux - th.x) ** 2 + (uy - th.y) ** 2 < limit2
    }
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
