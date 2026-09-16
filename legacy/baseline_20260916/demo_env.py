
import numpy as np
import matplotlib.pyplot as plt
from uav_env import CooperativeUAVEnv

env = CooperativeUAVEnv(seed=7)
obs, state = env.reset()

print("Initial UAVs:")
for u in env.uavs:
    print(u.idx, u.uav_type, (u.x, u.y), "yaw=", u.yaw)

print("\nTargets:")
for t in env.targets:
    print(t.idx, round(t.x, 1), round(t.y, 1), round(t.yaw, 3))

print("\nThreats:")
for th in env.threats:
    print(th.idx, round(th.x, 1), round(th.y, 1), round(th.radius, 1))

# Straight flight demo: action index 3 -> steering u=0.
for _ in range(10):
    actions = np.full(8, 3, dtype=int)
    obs, state, done, info = env.step(actions)
    if done:
        break

print("\nAfter 10 steps:", info)

ax = env.render(show_comm=True, show_ranges=True)
plt.tight_layout()
plt.savefig("scene_demo.png", dpi=180)
print("Saved scene_demo.png")
