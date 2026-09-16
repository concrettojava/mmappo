import math
import numpy as np
from .entities import UAV, Target, Threat

# Paper Table 2
INITIAL_UAVS = [
    # id, x, y, type
    (0, 1700.0, 200.0, "Stk"),
    (1, 1900.0, 200.0, "Stk"),
    (2, 2100.0, 200.0, "Stk"),
    (3, 2300.0, 200.0, "Stk"),
    (4, 1800.0, 300.0, "Rec"),
    (5, 2200.0, 300.0, "Rec"),
    (6, 1800.0, 100.0, "Com"),
    (7, 2200.0, 100.0, "Com"),
]

def reset_scene(env, seed: int | None = None):
    if seed is not None:
        env.seed = seed
        env.rng = np.random.default_rng(seed)

    env.step_count = 0
    env.uavs = [UAV(idx=i, x=x, y=y, uav_type=t, yaw=0.0)
                 for i, x, y, t in INITIAL_UAVS]

    # Paper: targets and threat areas are randomly distributed in battlefield.
    # Use a margin only to avoid degenerate spawning on the exact boundary.
    margin = 250.0
    env.targets = []
    for j in range(4):
        x = float(env.rng.uniform(margin, env.world_size - margin))
        y = float(env.rng.uniform(1200.0, env.world_size - margin))
        yaw = float(env.rng.uniform(-math.pi, math.pi))
        env.targets.append(Target(j, x, y, yaw))

    env.threats = []
    for k in range(3):
        x = float(env.rng.uniform(margin, env.world_size - margin))
        y = float(env.rng.uniform(700.0, env.world_size - margin))
        radius = float(env.rng.uniform(120.0, 200.0))
        env.threats.append(Threat(k, x, y, radius))
