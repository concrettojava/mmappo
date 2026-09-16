"""Profile the direct-vector parallel MAPPO training hot path.

This diagnostic mirrors the current high-throughput training path:
- scalar CooperativeUAVEnv dynamics/reward resolution;
- direct environment-to-fixed-vector encoding;
- batched MAPPO inference and PPO updates.

It reports a CUDA-synchronized wall-clock breakdown and a cProfile ranking of
``step_vectors`` so the next optimization target is chosen from measured cost.
No checkpoints or metrics are written.
"""
from __future__ import annotations

import argparse
import cProfile
import io
from pathlib import Path
import pstats
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from fofe_mmapppo.algorithms import MAPPO, MAPPOConfig, ParallelRolloutBuffer
from fofe_mmapppo.envs import CooperativeUAVEnv
from fofe_mmapppo.models import DirectFixedVectorizer, FixedVectorizer


def choose_device(name: str) -> str:
    if name != "auto":
        return name
    return "cuda" if torch.cuda.is_available() else "cpu"


def sync_cuda(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def timed_rollout(learner, base_vectorizer, direct, envs, episode_ids, seed, device):
    N = base_vectorizer.n_uavs
    E = len(envs)

    reset_t0 = time.perf_counter()
    obs_vec = np.zeros((E, N, base_vectorizer.observation_dim), dtype=np.float32)
    state_vec = np.zeros((E, N, base_vectorizer.state_dim), dtype=np.float32)
    active = np.zeros((E, N), dtype=np.float32)
    for e, (env, ep) in enumerate(zip(envs, episode_ids)):
        obs_vec[e], state_vec[e], active[e] = env.reset_vectors(
            direct, seed=seed + int(ep) - 1
        )
    t_reset = time.perf_counter() - reset_t0

    finished = np.zeros(E, dtype=bool)
    buffer = ParallelRolloutBuffer(n_envs=E, n_agents=N)
    episode_returns = np.zeros((E, N), dtype=np.float32)
    final_infos = [None] * E

    times = {
        "inference": 0.0,
        "env_step_vectors": 0.0,
        "reward_pack": 0.0,
        "buffer_add": 0.0,
    }
    steps = 0

    while not bool(finished.all()):
        sync_cuda(device)
        t0 = time.perf_counter()
        actions, log_probs, values = learner.act_batch(obs_vec, state_vec, active)
        sync_cuda(device)
        times["inference"] += time.perf_counter() - t0

        reward_batch = np.zeros((E, N), dtype=np.float32)
        done_batch = np.ones((E, N), dtype=np.float32)
        next_obs = np.zeros_like(obs_vec)
        next_state = np.zeros_like(state_vec)
        next_active = np.zeros_like(active)
        step_results = [None] * E

        t0 = time.perf_counter()
        for e, env in enumerate(envs):
            if finished[e]:
                continue
            step_results[e] = env.step_vectors(actions[e], direct)
        times["env_step_vectors"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        for e, result in enumerate(step_results):
            if result is None:
                continue
            o, s, a, rewards, done, info = result
            next_obs[e] = o
            next_state[e] = s
            next_active[e] = a
            reward_batch[e] = np.fromiter(
                (rewards[i] for i in range(N)), dtype=np.float32, count=N
            )
            done_batch[e] = np.fromiter(
                (done or a[i] == 0.0 for i in range(N)),
                dtype=np.float32,
                count=N,
            )
            episode_returns[e] += reward_batch[e]
            if done:
                finished[e] = True
                final_infos[e] = info
        times["reward_pack"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        buffer.add(
            obs_vec, state_vec, actions, log_probs,
            reward_batch, done_batch, values, active,
        )
        times["buffer_add"] += time.perf_counter() - t0
        obs_vec, state_vec, active = next_obs, next_state, next_active
        steps += 1

    sync_cuda(device)
    t0 = time.perf_counter()
    losses = learner.update_parallel(buffer)
    sync_cuda(device)
    t_update = time.perf_counter() - t0

    return t_reset, times, t_update, steps, losses, episode_returns, final_infos


def profile_step_vectors(base_vectorizer, direct, envs, episode_ids, seed, learner, device, max_steps):
    E = len(envs)
    N = base_vectorizer.n_uavs
    obs_vec = np.zeros((E, N, base_vectorizer.observation_dim), dtype=np.float32)
    state_vec = np.zeros((E, N, base_vectorizer.state_dim), dtype=np.float32)
    active = np.zeros((E, N), dtype=np.float32)
    for e, (env, ep) in enumerate(zip(envs, episode_ids)):
        obs_vec[e], state_vec[e], active[e] = env.reset_vectors(
            direct, seed=seed + int(ep) - 1
        )

    finished = np.zeros(E, dtype=bool)
    profiler = cProfile.Profile()
    steps = 0
    while not bool(finished.all()) and steps < max_steps:
        actions, _, _ = learner.act_batch(obs_vec, state_vec, active)
        next_obs = np.zeros_like(obs_vec)
        next_state = np.zeros_like(state_vec)
        next_active = np.zeros_like(active)

        profiler.enable()
        for e, env in enumerate(envs):
            if finished[e]:
                continue
            o, s, a, _, done, _ = env.step_vectors(actions[e], direct)
            next_obs[e] = o
            next_state[e] = s
            next_active[e] = a
            if done:
                finished[e] = True
        profiler.disable()

        obs_vec, state_vec, active = next_obs, next_state, next_active
        steps += 1
    return profiler, steps


def print_breakdown(t_reset, times, t_update):
    measured = t_reset + sum(times.values()) + t_update
    rows = [("reset_vectors", t_reset), *times.items(), ("ppo_update", t_update)]
    rows.sort(key=lambda x: x[1], reverse=True)
    print("\n=== WALL-CLOCK BREAKDOWN: DIRECT-VECTOR PATH (CUDA-synchronized) ===")
    print(f"{'category':<20} {'seconds':>10} {'share':>9}")
    for name, value in rows:
        print(f"{name:<20} {value:10.3f} {100.0 * value / max(measured, 1e-12):8.2f}%")
    print(f"{'measured_total':<20} {measured:10.3f} {100.0:8.2f}%")


def main():
    parser = argparse.ArgumentParser(description="Profile direct-vector parallel MAPPO")
    parser.add_argument("--resume", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--minibatch-size", type=int, default=1024)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--profile-envs", type=int, default=16)
    parser.add_argument("--profile-steps", type=int, default=80)
    parser.add_argument("--top", type=int, default=30)
    args = parser.parse_args()

    if not args.resume.exists():
        parser.error(f"checkpoint not found: {args.resume}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")

    device = choose_device(args.device)
    checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
    start_episode = int(checkpoint.get("episode", 0))
    cfg = MAPPOConfig(**checkpoint.get("config", {}))
    cfg.minibatch_size = args.minibatch_size

    probe = CooperativeUAVEnv(seed=args.seed)
    base = FixedVectorizer(
        world_size=probe.world_size,
        n_uavs=8,
        n_targets=4,
        n_threats=3,
    )
    direct = DirectFixedVectorizer(base)
    learner = MAPPO(
        base.observation_dim,
        base.state_dim,
        n_agents=8,
        action_dim=7,
        config=cfg,
        device=device,
    )
    restored = learner.load_checkpoint(checkpoint, load_optimizers=True)
    print(
        f"device={device} checkpoint_ep={start_episode} num_envs={args.num_envs} "
        f"optimizer={'restored' if restored else 'restarted'}"
    )

    episode_ids = np.arange(start_episode + 1, start_episode + args.num_envs + 1, dtype=np.int64)
    envs = [CooperativeUAVEnv(seed=args.seed + int(ep) - 1) for ep in episode_ids]

    wall0 = time.perf_counter()
    t_reset, times, t_update, steps, losses, _, _ = timed_rollout(
        learner, base, direct, envs, episode_ids, args.seed, device
    )
    wall_total = time.perf_counter() - wall0
    print_breakdown(t_reset, times, t_update)
    print(
        f"\nfull_batch_wall={wall_total:.3f}s  envs={args.num_envs}  "
        f"effective={args.num_envs / max(wall_total, 1e-12):.3f} ep/s  "
        f"rollout_steps={steps}  actor={losses['actor_loss']:.4f}  "
        f"critic={losses['critic_loss']:.4f}"
    )

    detail_envs_n = min(args.profile_envs, args.num_envs)
    detail_ids = episode_ids[:detail_envs_n]
    detail_envs = [CooperativeUAVEnv(seed=args.seed + int(ep) - 1) for ep in detail_ids]
    profiler, prof_steps = profile_step_vectors(
        base, direct, detail_envs, detail_ids, args.seed, learner, device, args.profile_steps
    )
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats("cumulative")
    stats.print_stats(args.top)
    print(f"\n=== cProfile: env.step_vectors, {detail_envs_n} envs x {prof_steps} steps ===")
    print(stream.getvalue())


if __name__ == "__main__":
    main()
