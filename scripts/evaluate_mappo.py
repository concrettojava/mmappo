"""Evaluate a trained fixed-vector MAPPO checkpoint."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from fofe_mmapppo.algorithms import MAPPO, MAPPOConfig
from fofe_mmapppo.envs import CooperativeUAVEnv
from fofe_mmapppo.models import FixedVectorizer


def main():
    parser = argparse.ArgumentParser(description="Evaluate fixed-vector MAPPO baseline")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--stochastic", action="store_true", help="sample actions instead of argmax")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    vec_cfg = checkpoint.get("vectorizer", {})
    vectorizer = FixedVectorizer(**vec_cfg) if vec_cfg else FixedVectorizer()
    config = MAPPOConfig(**checkpoint.get("config", {}))
    learner = MAPPO(
        vectorizer.observation_dim,
        vectorizer.state_dim,
        n_agents=int(checkpoint.get("n_agents", 8)),
        action_dim=int(checkpoint.get("action_dim", 7)),
        config=config,
        device=args.device,
    )
    learner.load_checkpoint(checkpoint)

    completion, survival, steps, returns = [], [], [], []
    for episode in range(args.episodes):
        env = CooperativeUAVEnv(seed=args.seed + episode)
        obs, states = env.reset()
        total = np.zeros(8, dtype=np.float32)
        while True:
            obs_vec = vectorizer.batch_observations(obs)
            state_vec = vectorizer.batch_states(states)
            active = np.asarray([obs[i] is not None for i in range(8)], dtype=np.float32)
            actions, _, _ = learner.act(
                obs_vec, state_vec, active, deterministic=not args.stochastic
            )
            obs, states, reward, done, info = env.step(actions)
            total += np.asarray([reward[i] for i in range(8)], dtype=np.float32)
            if done:
                break

        completion.append(info["completion_ratio"])
        survival.append(info["survival_ratio"])
        steps.append(info["step"])
        returns.append(total.mean())

    def stat(values):
        arr = np.asarray(values, dtype=np.float64)
        return float(arr.mean()), float(arr.std())

    print(f"episodes: {args.episodes}")
    for name, values in [
        ("completion_ratio", completion),
        ("survival_ratio", survival),
        ("completion_time_steps", steps),
        ("mean_agent_return", returns),
    ]:
        mean, std = stat(values)
        print(f"{name}: {mean:.4f} ± {std:.4f}")


if __name__ == "__main__":
    main()
