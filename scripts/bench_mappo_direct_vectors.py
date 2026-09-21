"""Throughput benchmark for direct environment-to-fixed-vector MAPPO rollout."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from fofe_mmapppo.algorithms import MAPPO, MAPPOConfig, ParallelRolloutBuffer
from fofe_mmapppo.envs import CooperativeUAVEnv
from fofe_mmapppo.models import DirectFixedVectorizer, FixedVectorizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--resume", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--num-envs", type=int, default=64)
    p.add_argument("--minibatch-size", type=int, default=1024)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    checkpoint = torch.load(args.resume, map_location=args.device, weights_only=False)
    FixedVectorizer.assert_checkpoint_compatible(checkpoint)
    start_episode = int(checkpoint.get("episode", 0))
    cfg = MAPPOConfig(**checkpoint.get("config", {}))
    cfg.minibatch_size = args.minibatch_size
    base = FixedVectorizer(**checkpoint["vectorizer"])
    direct = DirectFixedVectorizer(base)
    learner = MAPPO(base.observation_dim, base.state_dim, 8, 7, cfg, args.device)
    learner.load_checkpoint(checkpoint, load_optimizers=True)
    torch.set_float32_matmul_precision("high")

    processed = 0
    wall_start = time.perf_counter()
    while processed < args.episodes:
        E = min(args.num_envs, args.episodes - processed)
        episode_ids = np.arange(start_episode + processed + 1, start_episode + processed + E + 1)
        envs = [CooperativeUAVEnv(seed=args.seed + int(ep) - 1) for ep in episode_ids]

        obs_vec = np.zeros((E, 8, base.observation_dim), dtype=np.float32)
        state_vec = np.zeros((E, 8, base.state_dim), dtype=np.float32)
        active = np.zeros((E, 8), dtype=np.float32)
        for e, (env, ep) in enumerate(zip(envs, episode_ids)):
            obs_vec[e], state_vec[e], active[e] = env.reset_vectors(
                direct, seed=args.seed + int(ep) - 1
            )

        finished = np.zeros(E, dtype=bool)
        buffer = ParallelRolloutBuffer(n_envs=E, n_agents=8)

        while not finished.all():
            actions, log_probs, values = learner.act_batch(obs_vec, state_vec, active)
            rewards = np.zeros((E, 8), dtype=np.float32)
            dones = np.ones((E, 8), dtype=np.float32)
            next_obs = np.zeros_like(obs_vec)
            next_state = np.zeros_like(state_vec)
            next_active = np.zeros_like(active)

            for e, env in enumerate(envs):
                if finished[e]:
                    continue
                o, s, a, r, done, _ = env.step_vectors(actions[e], direct)
                next_obs[e] = o
                next_state[e] = s
                next_active[e] = a
                rewards[e] = np.fromiter((r[i] for i in range(8)), dtype=np.float32, count=8)
                dones[e] = np.asarray(
                    [1.0 if done or a[i] == 0.0 else 0.0 for i in range(8)],
                    dtype=np.float32,
                )
                finished[e] = done

            buffer.add(
                obs_vec, state_vec, actions, log_probs,
                rewards, dones, values, active,
            )
            obs_vec, state_vec, active = next_obs, next_state, next_active

        learner.update_parallel(buffer)
        processed += E
        elapsed = time.perf_counter() - wall_start
        print(
            f"directvec {processed}/{args.episodes}  "
            f"{processed/elapsed:.3f} ep/s  {elapsed/processed:.3f}s/ep"
        )


if __name__ == "__main__":
    main()
