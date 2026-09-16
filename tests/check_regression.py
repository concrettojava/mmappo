"""Regression checks for the frozen stage-1 physics and visualization baseline.

The Dec-POMDP observation/state API intentionally changes in phase 2, so this
regression no longer compares those public structures.  It compares the raw
entity world state, termination/info values, preview trajectory and rendered
figures instead.  Rule-level observation/state behavior is covered separately
by ``tests/test_observation_state.py``.
"""
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


def legacy_world_state(env):
    """Raw entity snapshot matching the frozen stage-1 get_global_state()."""
    return plain({
        "uavs": [vars(u).copy() for u in env.uavs],
        "targets": [vars(t).copy() for t in env.targets],
        "threats": [vars(th).copy() for th in env.threats],
    })


def check_rollout_against_saved(seed, random_actions, saved_records):
    env = CooperativeUAVEnv(seed=seed)
    env.reset()
    assert legacy_world_state(env) == saved_records[0]["state"]

    action_rng = np.random.default_rng(20260916)
    for step in range(env.max_steps):
        actions = (action_rng.integers(0, 7, len(env.uavs))
                   if random_actions else np.full(len(env.uavs), 3))
        _obs, _state, done, info = env.step(actions)
        expected = saved_records[step + 1]
        assert plain(actions) == expected["actions"]
        assert legacy_world_state(env) == expected["state"]
        assert bool(done) == expected["done"]
        assert plain(info) == expected["info"]
        if done:
            break


def main():
    baseline = ROOT / "outputs/baseline_20260916"
    saved = json.loads((baseline / "trajectories.json").read_text())

    for seed in (0, 7, 42):
        for random_actions in (False, True):
            name = f"seed{seed}_{'random' if random_actions else 'straight'}"
            check_rollout_against_saved(seed, random_actions, saved[name])

    viewer = LiveViewer(seed=7)
    viewer.paused = True
    states = [legacy_world_state(viewer.env)]
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
            states.append(legacy_world_state(viewer.env))

    assert states == saved["v5_seed7"], "preview states"

    from types import SimpleNamespace
    viewer._on_key(SimpleNamespace(key=" "))
    assert not viewer.paused
    viewer._on_key(SimpleNamespace(key="r"))
    assert viewer.env.step_count == 0
    assert all(len(track) == 1 for track in viewer.renderer.uav_tracks.values())
    viewer._on_key(SimpleNamespace(key="q"))
    assert not plt.fignum_exists(viewer.fig.number)

    print("PASS: stage-1 physics/preview regression remains unchanged")


if __name__ == "__main__":
    main()
