"""Throughput benchmark for cross-environment geometry batching."""
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
from fofe_mmapppo.envs.batch_geometry import install_geometry_cache
from fofe_mmapppo.models import FixedVectorizer


def encode_batch(vectorizer, observations, states, finished):
    E = len(observations)
    obs = np.zeros((E, 8, vectorizer.observation_dim), dtype=np.float32)
    state = np.zeros((E, 8, vectorizer.state_dim), dtype=np.float32)
    active = np.zeros((E, 8), dtype=np.float32)
    for e in range(E):
        if finished[e]:
            continue
        obs[e] = vectorizer.batch_observations(observations[e])
        state[e] = vectorizer.batch_states(states[e])
        active[e] = np.asarray([observations[e][i] is not None for i in range(8)], dtype=np.float32)
    return obs, state, active


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--resume", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=200, help="number of episodes to benchmark")
    p.add_argument("--num-envs", type=int, default=64)
    p.add_argument("--minibatch-size", type=int, default=1024)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    checkpoint = torch.load(args.resume, map_location=args.device, weights_only=False)
    start_episode = int(checkpoint.get("episode", 0))
    cfg = MAPPOConfig(**checkpoint.get("config", {}))
    cfg.minibatch_size = args.minibatch_size
    vectorizer = FixedVectorizer(**checkpoint["vectorizer"])
    learner = MAPPO(vectorizer.observation_dim, vectorizer.state_dim, 8, 7, cfg, args.device)
    learner.load_checkpoint(checkpoint, load_optimizers=True)
    torch.set_float32_matmul_precision("high")

    processed = 0
    wall_start = time.perf_counter()
    while processed < args.episodes:
        E = min(args.num_envs, args.episodes - processed)
        episode_ids = np.arange(start_episode + processed + 1, start_episode + processed + E + 1)
        envs = [CooperativeUAVEnv(seed=args.seed + int(ep) - 1) for ep in episode_ids]
        resets = [env.reset(seed=args.seed + int(ep) - 1) for env, ep in zip(envs, episode_ids)]
        observations = [x[0] for x in resets]
        states = [x[1] for x in resets]
        finished = np.zeros(E, dtype=bool)
        buffer = ParallelRolloutBuffer(n_envs=E, n_agents=8)

        while not finished.all():
            obs_vec, state_vec, active = encode_batch(vectorizer, observations, states, finished)
            actions, log_probs, values = learner.act_batch(obs_vec, state_vec, active)
            rewards = np.zeros((E, 8), dtype=np.float32)
            dones = np.ones((E, 8), dtype=np.float32)
            next_obs = list(observations)
            next_states = list(states)

            contexts = [None] * E
            for e, env in enumerate(envs):
                if not finished[e]:
                    contexts[e] = env.prepare_step(actions[e])
            install_geometry_cache(envs, active_mask=~finished)

            for e, env in enumerate(envs):
                if finished[e]:
                    continue
                o, s, r, done, _ = env.finish_step(contexts[e])
                rewards[e] = np.fromiter((r[i] for i in range(8)), dtype=np.float32, count=8)
                dones[e] = np.fromiter((done or o[i] is None for i in range(8)), dtype=np.float32, count=8)
                next_obs[e], next_states[e] = o, s
                finished[e] = done

            buffer.add(obs_vec, state_vec, actions, log_probs, rewards, dones, values, active)
            observations, states = next_obs, next_states

        learner.update_parallel(buffer)
        processed += E
        elapsed = time.perf_counter() - wall_start
        print(f"batchgeom {processed}/{args.episodes}  {processed/elapsed:.3f} ep/s  {elapsed/processed:.3f}s/ep")


if __name__ == "__main__":
    main()
