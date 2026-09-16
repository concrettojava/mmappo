"""High-throughput MAPPO training with multiple independent environments.

This is an engineering acceleration path.  The paper does not explicitly state
that its 32 random environments are stepped as a vectorized training batch.
The single-environment trainer remains the strict reference implementation.

Acceleration comes from:
1. collecting several independent episodes before each PPO update;
2. batching one agent's Actor/Critic inference across all environments;
3. using much larger PPO mini-batches on GPU;
4. enabling TF32-friendly float32 matmul precision on supported NVIDIA GPUs;
5. reducing Python/file-I/O overhead in the training loop.

Environment transitions themselves remain ordinary CooperativeUAVEnv.step()
calls, so environment/reward semantics are unchanged.  Since multiple episodes
are combined into one PPO update, optimization dynamics are not identical to
the single-environment reference trainer.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from fofe_mmapppo.algorithms import MAPPO, MAPPOConfig, ParallelRolloutBuffer
from fofe_mmapppo.envs import CooperativeUAVEnv
from fofe_mmapppo.models import FixedVectorizer


def choose_device(name: str) -> str:
    if name != "auto":
        return name
    return "cuda" if torch.cuda.is_available() else "cpu"


def encode_batch(vectorizer, observations, states, finished):
    E = len(observations)
    N = 8
    obs_batch = np.zeros((E, N, vectorizer.observation_dim), dtype=np.float32)
    state_batch = np.zeros((E, N, vectorizer.state_dim), dtype=np.float32)
    active = np.zeros((E, N), dtype=np.float32)
    for e in range(E):
        if finished[e]:
            continue
        obs_batch[e] = vectorizer.batch_observations(observations[e])
        state_batch[e] = vectorizer.batch_states(states[e])
        active[e] = np.asarray(
            [observations[e][i] is not None for i in range(N)], dtype=np.float32
        )
    return obs_batch, state_batch, active


def save_checkpoint(path, learner, vectorizer, episode, num_envs):
    torch.save(
        {
            **learner.checkpoint(include_optimizers=True),
            "episode": int(episode),
            "parallel_num_envs": int(num_envs),
            "vectorizer": {
                "world_size": vectorizer.world_size,
                "n_uavs": vectorizer.n_uavs,
                "n_targets": vectorizer.n_targets,
                "n_threats": vectorizer.n_threats,
            },
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser(description="High-throughput fixed-vector MAPPO")
    parser.add_argument("--episodes", type=int, default=5000, help="final global episode number")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--minibatch-size", type=int, default=1024)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "mappo_parallel")
    args = parser.parse_args()

    if args.episodes <= 0 or args.num_envs <= 0 or args.minibatch_size <= 0:
        parser.error("episodes, num-envs and minibatch-size must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    # On Ampere+ this permits TF32 internally for eligible float32 GEMMs.
    torch.set_float32_matmul_precision("high")

    device = choose_device(args.device)
    probe = CooperativeUAVEnv(seed=args.seed)
    vectorizer = FixedVectorizer(
        world_size=probe.world_size,
        n_uavs=8,
        n_targets=4,
        n_threats=3,
    )

    start_episode = 0
    checkpoint = None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        start_episode = int(checkpoint.get("episode", 0))
        cfg = MAPPOConfig(**checkpoint.get("config", {}))
        cfg.minibatch_size = args.minibatch_size
    else:
        cfg = MAPPOConfig(minibatch_size=args.minibatch_size)

    learner = MAPPO(
        vectorizer.observation_dim,
        vectorizer.state_dim,
        n_agents=8,
        action_dim=7,
        config=cfg,
        device=device,
    )
    if checkpoint is not None:
        restored_opt = learner.load_checkpoint(checkpoint, load_optimizers=True)
        status = "restored" if restored_opt else "restarted (old checkpoint has no optimizer state)"
        print(f"resume={args.resume} episode={start_episode} optimizer={status}")

    args.output.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output / "metrics.jsonl"
    print(
        f"device={device} num_envs={args.num_envs} minibatch={cfg.minibatch_size} "
        f"obs_dim={vectorizer.observation_dim} state_dim={vectorizer.state_dim}"
    )

    recent_returns = []
    processed = start_episode
    run_start = time.perf_counter()
    last_log_episode = processed

    while processed < args.episodes:
        # Do not cross a requested checkpoint boundary; this keeps filenames at
        # exact 500/1000/... episode numbers even when num_envs is not a divisor.
        next_save = ((processed // args.save_every) + 1) * args.save_every
        batch_envs = min(
            args.num_envs,
            args.episodes - processed,
            max(1, next_save - processed),
        )
        episode_ids = np.arange(processed + 1, processed + batch_envs + 1, dtype=np.int64)
        envs = [CooperativeUAVEnv(seed=args.seed + int(ep) - 1) for ep in episode_ids]
        reset_results = [
            env.reset(seed=args.seed + int(ep) - 1)
            for env, ep in zip(envs, episode_ids)
        ]
        observations = [item[0] for item in reset_results]
        states = [item[1] for item in reset_results]

        finished = np.zeros(batch_envs, dtype=bool)
        episode_returns = np.zeros((batch_envs, 8), dtype=np.float32)
        final_infos = [None] * batch_envs
        buffer = ParallelRolloutBuffer(n_envs=batch_envs, n_agents=8)

        while not bool(finished.all()):
            obs_vec, state_vec, active = encode_batch(vectorizer, observations, states, finished)
            actions, log_probs, values = learner.act_batch(obs_vec, state_vec, active)

            reward_batch = np.zeros((batch_envs, 8), dtype=np.float32)
            done_batch = np.ones((batch_envs, 8), dtype=np.float32)
            next_observations = list(observations)
            next_states = list(states)

            for e, env in enumerate(envs):
                if finished[e]:
                    continue
                next_obs, next_state, rewards, done, info = env.step(actions[e])
                reward_batch[e] = np.fromiter(
                    (rewards[i] for i in range(8)), dtype=np.float32, count=8
                )
                done_batch[e] = np.fromiter(
                    (done or next_obs[i] is None for i in range(8)),
                    dtype=np.float32,
                    count=8,
                )
                episode_returns[e] += reward_batch[e]
                next_observations[e] = next_obs
                next_states[e] = next_state
                if done:
                    finished[e] = True
                    final_infos[e] = info

            buffer.add(
                obs_vec, state_vec, actions, log_probs,
                reward_batch, done_batch, values, active,
            )
            observations, states = next_observations, next_states

        losses = learner.update_parallel(buffer)
        now = time.perf_counter()

        records = []
        for e, episode_id in enumerate(episode_ids.tolist()):
            info = final_infos[e]
            mean_return = float(episode_returns[e].mean())
            recent_returns.append(mean_return)
            if len(recent_returns) > 100:
                recent_returns.pop(0)
            records.append({
                "episode": int(episode_id),
                "mean_return": mean_return,
                "steps": int(info["step"]),
                "completion_ratio": float(info["completion_ratio"]),
                "survival_ratio": float(info["survival_ratio"]),
                "num_envs": int(batch_envs),
                **losses,
            })
        with metrics_path.open("a", encoding="utf-8") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in records)

        previous = processed
        processed += batch_envs

        if processed == args.episodes or processed - last_log_episode >= args.log_every:
            elapsed = now - run_start
            eps_per_sec = (processed - start_episode) / max(elapsed, 1e-9)
            print(
                f"ep={processed:6d} avg100={np.mean(recent_returns):9.3f} "
                f"batch_completion={np.mean([i['completion_ratio'] for i in final_infos]):.3f} "
                f"batch_survival={np.mean([i['survival_ratio'] for i in final_infos]):.3f} "
                f"actor={losses['actor_loss']:.4f} critic={losses['critic_loss']:.4f} "
                f"speed={eps_per_sec:.3f} ep/s"
            )
            last_log_episode = processed

        if processed % args.save_every == 0 or processed == args.episodes:
            save_checkpoint(
                args.output / f"checkpoint_{processed:06d}.pt",
                learner, vectorizer, processed, args.num_envs,
            )


if __name__ == "__main__":
    main()
