"""Communication subgroups, local detection and shared knowledge."""
from typing import Dict, Set
from .entities import UAV

def communication_graph(env) -> Dict[int, Set[int]]:
    alive = [u for u in env.uavs if u.alive]
    graph = {u.idx: set() for u in alive}
    for i, a in enumerate(alive):
        for b in alive[i+1:]:
            # Paper Eq.(1): direct communication if distance is within
            # max(comm_range_i, comm_range_j).
            if env._dist(a, b) < max(a.p["comm_range"], b.p["comm_range"]):
                graph[a.idx].add(b.idx)
                graph[b.idx].add(a.idx)
    return graph


def communication_components(env) -> Dict[int, Set[int]]:
    graph = env.communication_graph()
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
    return {t.idx for t in env.targets
            if t.alive and env._dist(uav, t) < uav.p["recon_range"]}


def _direct_detected_threats(env, uav: UAV) -> Set[int]:
    found = {th.idx for th in env.threats
             if env._dist(uav, th) < uav.p["recon_range"]}
    uav.threat_memory |= found
    return found


def shared_detection(env):
    comps = env.communication_components()
    direct_targets = {}
    direct_threats = {}
    for u in env.uavs:
        if u.alive:
            direct_targets[u.idx] = env._direct_detected_targets(u)
            direct_threats[u.idx] = env._direct_detected_threats(u)

    result = {}
    for u in env.uavs:
        if not u.alive:
            continue
        comp = comps.get(u.idx, {u.idx})
        visible_targets = set().union(*(direct_targets.get(v, set()) for v in comp))
        visible_threats = set().union(*(
            (direct_threats.get(v, set()) |
             next(x for x in env.uavs if x.idx == v).threat_memory)
            for v in comp
        ))
        result[u.idx] = (visible_targets, visible_threats)
    return result
