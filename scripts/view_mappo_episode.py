"""Interactively view one trained MAPPO episode.

The viewer uses actor-only deterministic inference. Matplotlib's normal toolbar
remains available, so any displayed frame can be saved manually.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from fofe_mmapppo.envs import CooperativeUAVEnv
from fofe_mmapppo.evaluation import load_fixed_mappo_checkpoint, policy_actions_batch
from fofe_mmapppo.visualization.renderer import SceneRenderer


class MAPPOEpisodeViewer:
    def __init__(self, checkpoint: Path, seed: int, device: str, interval_ms: int, stochastic: bool):
        self.seed = int(seed)
        self.device = device
        self.interval_ms = int(interval_ms)
        self.stochastic = bool(stochastic)
        self.learner, self.vectorizer, _ = load_fixed_mappo_checkpoint(checkpoint, device)
        self.env = CooperativeUAVEnv(seed=self.seed)
        self.obs, _ = self.env.reset(seed=self.seed)
        self.renderer = SceneRenderer(self.env)
        self.fig = self.renderer.fig
        self.paused = False
        self.done = False
        self.total_return = np.zeros(self.learner.n_agents, dtype=np.float32)
        self.final_info = None

        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self.anim = FuncAnimation(
            self.fig,
            self._animate,
            interval=self.interval_ms,
            blit=False,
            cache_frame_data=False,
            init_func=self.renderer.draw,
        )

    def _reset(self):
        self.obs, _ = self.env.reset(seed=self.seed)
        self.renderer.reset_tracks()
        self.paused = False
        self.done = False
        self.total_return.fill(0.0)
        self.final_info = None

    def _on_key(self, event):
        if event.key == " ":
            self.paused = not self.paused
        elif event.key in ("r", "R"):
            self._reset()
        elif event.key in ("q", "Q", "escape"):
            plt.close(self.fig)

    def _policy_action(self):
        obs_vec = self.vectorizer.batch_observations(self.obs)
        active = np.asarray(
            [self.obs[i] is not None for i in range(self.learner.n_agents)],
            dtype=np.float32,
        )
        return policy_actions_batch(
            self.learner,
            obs_vec[None, ...],
            active[None, ...],
            deterministic=not self.stochastic,
        )[0]

    def _step(self):
        actions = self._policy_action()
        self.obs, _, rewards, done, info = self.env.step(actions)
        self.total_return += np.fromiter(
            (rewards[i] for i in range(self.learner.n_agents)),
            dtype=np.float32,
            count=self.learner.n_agents,
        )
        self.renderer.record_tracks()
        if done:
            self.done = True
            self.paused = True
            self.final_info = info
            print(
                "episode finished: "
                f"steps={info['step']} "
                f"completion={info['completion_ratio']:.3f} "
                f"survival={info['survival_ratio']:.3f} "
                f"mean_agent_return={self.total_return.mean():.3f}"
            )

    def _animate(self, _):
        if not self.paused and not self.done:
            self._step()
        self.renderer.draw()
        if self.done and self.final_info is not None:
            self.renderer.ax.set_title(
                f"Episode finished | completion={self.final_info['completion_ratio']:.3f} "
                f"survival={self.final_info['survival_ratio']:.3f} "
                f"steps={self.final_info['step']}",
                fontsize=9,
            )
        return []

    def show(self):
        self.renderer.draw()
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="View one trained MAPPO episode")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--interval", type=int, default=80, help="milliseconds between frames")
    parser.add_argument("--stochastic", action="store_true", help="sample policy actions instead of argmax")
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("interval must be positive")

    viewer = MAPPOEpisodeViewer(
        args.checkpoint,
        seed=args.seed,
        device=args.device,
        interval_ms=args.interval,
        stochastic=args.stochastic,
    )
    viewer.show()


if __name__ == "__main__":
    main()
