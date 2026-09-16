"""Automatic strikes, collision losses and probabilistic threat damage."""
import math
from .entities import UAV, Target

def _bearing_error(env, uav: UAV, target: Target) -> float:
    dx, dy = target.x - uav.x, target.y - uav.y
    if env.paper_equation_yaw:
        bearing = math.atan2(dy, dx)
    else:
        # yaw=0 North
        bearing = math.atan2(dx, dy)
    return env._wrap_angle(bearing - uav.yaw)


def _automatic_strikes(env, shared):
    destroyed = []
    for u in env.uavs:
        if not u.alive:
            continue
        visible_targets, _ = shared.get(u.idx, (set(), set()))
        for tid in sorted(visible_targets):
            t = env.targets[tid]
            if not t.alive:
                continue
            d = env._dist(u, t)
            angle = abs(env._bearing_error(u, t))
            if (d < u.p["strike_range"] and
                angle < math.radians(u.p["strike_angle_deg"]) / 2.0):
                t.alive = False
                destroyed.append((u.idx, tid))
    return destroyed


def _apply_collisions(env):
    dead = set()
    alive = [u for u in env.uavs if u.alive]
    for i, a in enumerate(alive):
        for b in alive[i+1:]:
            if env._dist(a, b) < min(a.p["collision_range"], b.p["collision_range"]):
                dead.add(a.idx)
                dead.add(b.idx)
    for idx in dead:
        env.uavs[idx].alive = False
    return dead


def _apply_threat_damage(env):
    dead = set()
    for u in env.uavs:
        if not u.alive:
            continue
        for th in env.threats:
            rho = env._dist(u, th)
            if rho >= th.radius:
                continue
            # Paper Eq.(7), with DeltaT = dt.
            base = 1.0 - (1.0 - rho / th.radius) ** env.threat_eta
            p_shot = 1.0 - base ** env.dt
            if env.rng.random() < p_shot:
                u.alive = False
                dead.add(u.idx)
                break
    return dead
