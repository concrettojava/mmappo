"""Original structured observations and centralized state (not yet training tensors)."""

def get_observations(env):
    shared = env.shared_detection()
    comps = env.communication_components()
    observations = {}

    for u in env.uavs:
        if not u.alive:
            observations[u.idx] = None
            continue

        comp = comps.get(u.idx, {u.idx})
        neighbor_ids = sorted(comp - {u.idx})
        visible_targets, visible_threats = shared[u.idx]

        observations[u.idx] = {
            "self": {
                "idx": u.idx, "type": u.uav_type, "alive": u.alive,
                "x": u.x, "y": u.y, "yaw": u.yaw,
            },
            "neighbors": [
                {
                    "idx": v, "type": env.uavs[v].uav_type,
                    "alive": env.uavs[v].alive,
                    "x": env.uavs[v].x, "y": env.uavs[v].y,
                    "yaw": env.uavs[v].yaw,
                } for v in neighbor_ids
            ],
            "targets": [
                {
                    "idx": tid, "alive": env.targets[tid].alive,
                    "x": env.targets[tid].x, "y": env.targets[tid].y,
                    "yaw": env.targets[tid].yaw,
                } for tid in sorted(visible_targets)
            ],
            "threats": [
                {
                    "idx": kid, "radius": env.threats[kid].radius,
                    "x": env.threats[kid].x, "y": env.threats[kid].y,
                } for kid in sorted(visible_threats)
            ],
        }
    return observations


def get_global_state(env):
    return {
        "uavs": [vars(u).copy() for u in env.uavs],
        "targets": [vars(t).copy() for t in env.targets],
        "threats": [vars(th).copy() for th in env.threats],
    }
