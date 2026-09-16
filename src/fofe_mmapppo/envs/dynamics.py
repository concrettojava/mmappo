"""Movement and geometry, preserving the original coordinate conventions."""
import math
import numpy as np
from .entities import ACTION_VALUES


def _dist(a, b) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def _wrap_angle(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def _advance_xy(env, x, y, yaw, speed):
    if env.paper_equation_yaw:
        # Literal Eq.(5)/(6)
        x += speed * math.cos(yaw) * env.dt
        y += speed * math.sin(yaw) * env.dt
    else:
        # Match Table 2/Fig.7 convention: yaw=0 points North.
        x += speed * math.sin(yaw) * env.dt
        y += speed * math.cos(yaw) * env.dt
    return x, y


def _update_uavs(env, action_indices):
    for uav, aidx in zip(env.uavs, action_indices):
        if not uav.alive:
            continue
        u = float(ACTION_VALUES[int(aidx)])
        uav.last_action_u = u
        max_turn = math.radians(uav.p["max_turn_deg_s"])
        omega = max_turn * u
        uav.yaw = env._wrap_angle(uav.yaw + omega * env.dt)
        uav.x, uav.y = env._advance_xy(uav.x, uav.y, uav.yaw, uav.p["speed"])


def _away_yaw(env, target, uav):
    dx = target.x - uav.x
    dy = target.y - uav.y
    if env.paper_equation_yaw:
        return math.atan2(dy, dx)
    return math.atan2(dx, dy)


def _update_targets(env):
    for t in env.targets:
        if not t.alive:
            continue

        alpha = float(env.rng.normal(0.0, env.target_turn_accel_std))
        t.omega += alpha * env.dt

        if env.contested:
            alive_uavs = [u for u in env.uavs if u.alive]
            if alive_uavs:
                nearest = min(alive_uavs, key=lambda u: (u.x - t.x) ** 2 + (u.y - t.y) ** 2)
                d = math.hypot(nearest.x - t.x, nearest.y - t.y)
                if d < env.evasive_trigger_range:
                    desired = _away_yaw(env, t, nearest)
                    error = env._wrap_angle(desired - t.yaw)
                    max_turn = env.evasive_turn_rate * env.dt
                    # Blend stochastic motion with a bounded reactive evasive turn.
                    t.yaw = env._wrap_angle(t.yaw + float(np.clip(error, -max_turn, max_turn)))
                    t.omega *= 0.65

        t.yaw = env._wrap_angle(t.yaw + t.omega * env.dt)
        nx, ny = env._advance_xy(t.x, t.y, t.yaw, env.target_speed)

        # Paper only states targets remain in mission area.
        # Reflect at boundaries as a minimally invasive implementation.
        if nx < 0 or nx > env.world_size:
            t.yaw = env._wrap_angle(-t.yaw)
        if ny < 0 or ny > env.world_size:
            t.yaw = env._wrap_angle(math.pi - t.yaw)
        t.x, t.y = env._advance_xy(t.x, t.y, t.yaw, env.target_speed)
        t.x = float(np.clip(t.x, 0.0, env.world_size))
        t.y = float(np.clip(t.y, 0.0, env.world_size))
