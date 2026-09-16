"""Capture deterministic behavior without changing the original implementation."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from fofe_mmapppo.envs import CooperativeUAVEnv
from fofe_mmapppo.visualization.live_viewer import LiveViewer


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, set):
        return sorted(plain(v) for v in value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def rollout(seed, random_actions):
    env = CooperativeUAVEnv(seed=seed)
    obs, state = env.reset()
    records = [{"observation": plain(obs), "state": plain(state)}]
    action_rng = np.random.default_rng(20260916)
    for _ in range(env.max_steps):
        actions = action_rng.integers(0, 7, len(env.uavs)) if random_actions else np.full(len(env.uavs), 3)
        obs, state, done, info = env.step(actions)
        records.append(plain(dict(actions=actions, observation=obs, state=state, done=done, info=info)))
        if done:
            break
    return records


def main():
    baseline = ROOT / "outputs/baseline_20260916"
    saved = json.loads((baseline / "trajectories.json").read_text())
    for seed in (0, 7, 42):
        for random_actions in (False, True):
            name = f"seed{seed}_{'random' if random_actions else 'straight'}"
            assert rollout(seed, random_actions) == saved[name], name
    viewer = LiveViewer(seed=7)
    viewer.paused = True
    states = [plain(viewer.env.get_global_state())]
    out = ROOT / "outputs/checks"
    out.mkdir(parents=True, exist_ok=True)
    for step in range(51):
        if step in (0, 50):
            viewer.renderer.draw()
            viewer.fig.canvas.draw()
            name = "v5_initial.png" if step == 0 else "v5_step050.png"
            viewer.fig.savefig(out / name, dpi=150)
            assert np.array_equal(plt.imread(out / name), plt.imread(baseline / name)), name
        if step < 50:
            viewer._advance_scene_only()
            states.append(plain(viewer.env.get_global_state()))
    assert states == saved["v5_seed7"], "preview states"
    from types import SimpleNamespace
    viewer._on_key(SimpleNamespace(key=" "))
    assert not viewer.paused
    viewer._on_key(SimpleNamespace(key="r"))
    assert viewer.env.step_count == 0
    assert all(len(track) == 1 for track in viewer.renderer.uav_tracks.values())
    viewer._on_key(SimpleNamespace(key="q"))
    assert not plt.fignum_exists(viewer.fig.number)
    print("PASS: 6 full environment trajectories, 50 preview steps, 2 pixel-identical images, keyboard handlers")


if __name__ == "__main__":
    main()
