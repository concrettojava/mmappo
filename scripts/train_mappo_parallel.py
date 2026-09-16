"""High-throughput MAPPO training with multiple independent environments.

This is an engineering acceleration path. The paper does not explicitly state
that its 32 random environments are stepped as a vectorized training batch.
The single-environment trainer remains the strict reference implementation.

The parallel trainer uses a direct environment-to-fixed-vector rollout path so
training does not materialize the flexible Eq. (8)/(9) dict/list observation
objects and then immediately encode them back into arrays. The direct encoder is
covered by parity tests against the structured reference representation.
"""
from __future__ import annotations

import argparse
import atexit
from datetime import datetime, timedelta
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from fofe_mmapppo.algorithms import MAPPO, MAPPOConfig, ParallelRolloutBuffer
from fofe_mmapppo.envs import CooperativeUAVEnv
from fofe_mmapppo.models import DirectFixedVectorizer, FixedVectorizer
from fofe_mmapppo.terminal_status import FixedStatusHeader


def choose_device(name: str) -> str:
    if name != "auto":
        return name
    return "cuda" if torch.cuda.is_available() else "cpu"


def format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def status_text(processed, total, speed, sec_per_ep, elapsed):
    remaining = max(0, total - processed)
    eta_seconds = remaining / speed if speed > 0 else 0.0
    finish = datetime.now() + timedelta(seconds=eta_seconds)
    percent = 100.0 * processed / max(total, 1)
    return (
        f"MAPPO  {processed}/{total}  {percent:6.2f}%  |  "
        f"{speed:.3f} ep/s  {sec_per_ep:.2f}s/ep  |  "
        f"elapsed {format_duration(elapsed)}  |  ETA {format_duration(eta_seconds)}  |  "
        f"finish≈{finish:%H:%M:%S}"
    )


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
    parser.add_argument(
        "--no-tensorboard", action="store_true", help="disable TensorBoard scalar logging"
    )
    args = parser.parse_args()

    if args.episodes <= 0 or args.num_envs <= 0 or args.minibatch_size <= 0:
        parser.error("episodes, num-envs and minibatch-size must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")

    device = choose_device(args.device)
    probe = CooperativeUAVEnv(seed=args.seed)
    vectorizer = FixedVectorizer(
        world_size=probe.world_size,
        n_uavs=8,
        n_targets=4,
        n_threats=3,
    )
    direct_vectorizer = DirectFixedVectorizer(vectorizer)

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
    writer = None
    if not args.no_tensorboard:
        writer = SummaryWriter(log_dir=str(args.output / "tensorboard"))
        atexit.register(writer.close)

    print(
        f"device={device} num_envs={args.num_envs} minibatch={cfg.minibatch_size} "
        f"obs_dim={vectorizer.observation_dim} state_dim={vectorizer.state_dim} "
        f"rollout=direct-vector"
    )
    if writer is not None:
        print(f"tensorboard={args.output / 'tensorboard'}")

    recent_returns = []
    processed = start_episode
    run_start = time.perf_counter()
    last_log_episode = processed
    header = FixedStatusHeader()
    header.start("MAPPO starting...")

    while processed < args.episodes:
        next_save = ((processed // args.save_every) + 1) * args.save_every
        batch_envs = min(
            args.num_envs,
            args.episodes - processed,
            max(1, next_save - processed),
        )
        episode_ids = np.arange(processed + 1, processed + batch_envs + 1, dtype=np.int64)
        envs = [CooperativeUAVEnv(seed=args.seed + int(ep) - 1) for ep in episode_ids]

        obs_vec = np.zeros(
            (batch_envs, 8, vectorizer.observation_dim), dtype=np.float32
        )
        state_vec = np.zeros(
            (batch_envs, 8, vectorizer.state_dim), dtype=np.float32
        )
        active = np.zeros((batch_envs, 8), dtype=np.float32)
        for e, (env, episode_id) in enumerate(zip(envs, episode_ids)):
            obs_vec[e], state_vec[e], active[e] = env.reset_vectors(
                direct_vectorizer, seed=args.seed + int(episode_id) - 1
            )

        finished = np.zeros(batch_envs, dtype=bool)
        episode_returns = np.zeros((batch_envs, 8), dtype=np.float32)
        final_infos = [None] * batch_envs
        buffer = ParallelRolloutBuffer(n_envs=batch_envs, n_agents=8)

        while not bool(finished.all()):
            actions, log_probs, values = learner.act_batch(obs_vec, state_vec, active)

            reward_batch = np.zeros((batch_envs, 8), dtype=np.float32)
            done_batch = np.ones((batch_envs, 8), dtype=np.float32)
            next_obs = np.zeros_like(obs_vec)
            next_state = np.zeros_like(state_vec)
            next_active = np.zeros_like(active)

            for e, env in enumerate(envs):
                if finished[e]:
                    continue
                o, s, a, rewards, done, info = env.step_vectors(
                    actions[e], direct_vectorizer
                )
                next_obs[e] = o
                next_state[e] = s
                next_active[e] = a
                reward_batch[e] = np.fromiter(
                    (rewards[i] for i in range(8)), dtype=np.float32, count=8
                )
                done_batch[e] = np.asarray(
                    [1.0 if done or a[i] == 0.0 else 0.0 for i in range(8)],
                    dtype=np.float32,
                )
                episode_returns[e] += reward_batch[e]
                if done:
                    finished[e] = True
                    final_infos[e] = info

            buffer.add(
                obs_vec,
                state_vec,
                actions,
                log_probs,
                reward_batch,
                done_batch,
                values,
                active,
            )
            obs_vec, state_vec, active = next_obs, next_state, next_active

        losses = learner.update_parallel(buffer)
        now = time.perf_counter()

        records = []
        for e, episode_id in enumerate(episode_ids.tolist()):
            info = final_infos[e]
            mean_return = float(episode_returns[e].mean())
            recent_returns.append(mean_return)
            if len(recent_returns) > 100:
                recent_returns.pop(0)
            avg100 = float(np.mean(recent_returns))
            record = {
                "episode": int(episode_id),
                "mean_return": mean_return,
                "steps": int(info["step"]),
                "completion_ratio": float(info["completion_ratio"]),
                "survival_ratio": float(info["survival_ratio"]),
                "num_envs": int(batch_envs),
                **losses,
            }
            records.append(record)
            if writer is not None:
                writer.add_scalar("train/mean_return", mean_return, episode_id)
                writer.add_scalar("train/avg100_return", avg100, episode_id)
                writer.add_scalar(
                    "task/completion_ratio", record["completion_ratio"], episode_id
                )
                writer.add_scalar(
                    "task/survival_ratio", record["survival_ratio"], episode_id
                )
                writer.add_scalar("task/episode_steps", record["steps"], episode_id)

        with metrics_path.open("a", encoding="utf-8") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in records)

        processed += batch_envs
        elapsed = max(now - run_start, 1e-9)
        trained = processed - start_episode
        speed = trained / elapsed
        sec_per_ep = elapsed / max(trained, 1)
        header.update(status_text(processed, args.episodes, speed, sec_per_ep, elapsed))

        if writer is not None:
            writer.add_scalar("loss/actor", losses["actor_loss"], processed)
            writer.add_scalar("loss/critic", losses["critic_loss"], processed)
            writer.add_scalar("performance/episodes_per_second", speed, processed)
            writer.add_scalar("performance/seconds_per_episode", sec_per_ep, processed)
            writer.add_scalar("performance/active_num_envs", batch_envs, processed)

        if processed == args.episodes or processed - last_log_episode >= args.log_every:
            last = records[-1]
            batch_completion = float(
                np.mean([i["completion_ratio"] for i in final_infos])
            )
            batch_survival = float(np.mean([i["survival_ratio"] for i in final_infos]))
            header.log(
                f"ep={processed:6d} return={last['mean_return']:9.3f} "
                f"avg100={np.mean(recent_returns):9.3f} steps={last['steps']:3d} "
                f"completion={last['completion_ratio']:.3f} "
                f"survival={last['survival_ratio']:.3f} batch_c={batch_completion:.3f} "
                f"batch_s={batch_survival:.3f} actor={losses['actor_loss']:.4f} "
                f"critic={losses['critic_loss']:.4f}"
            )
            if writer is not None:
                writer.add_scalar(
                    "task/batch_completion_ratio", batch_completion, processed
                )
                writer.add_scalar("task/batch_survival_ratio", batch_survival, processed)
                writer.flush()
            if not header.enabled:
                header.plain_status_if_needed()
            last_log_episode = processed

        if processed % args.save_every == 0 or processed == args.episodes:
            save_checkpoint(
                args.output / f"checkpoint_{processed:06d}.pt",
                learner,
                vectorizer,
                processed,
                args.num_envs,
            )

    header.close()
    if writer is not None:
        writer.flush()
        writer.close()


if __name__ == "__main__":
    main()
