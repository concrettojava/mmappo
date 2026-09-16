"""Train the fixed-vector MAPPO baseline.

Paper-matched defaults where published:
- episode horizon: 200
- gamma: 0.95
- GAE lambda: 0.95
- PPO clip epsilon: 0.1
- PPO epochs: 15
- actor/critic learning rate: 4e-5

The paper does not publish the baseline's exact vector padding/normalization,
MLP hidden width, entropy coefficient, value coefficient or gradient clipping;
those choices are explicit in ``MAPPOConfig`` and can be changed independently.
"""
from __future__ import annotations

import argparse
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

from fofe_mmapppo.algorithms import MAPPO, MAPPOConfig, RolloutBuffer
from fofe_mmapppo.envs import CooperativeUAVEnv
from fofe_mmapppo.models import FixedVectorizer


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


def progress_lines(
    episode: int,
    total: int,
    mean_return: float,
    avg100: float,
    record: dict,
    losses: dict,
    run_start: float,
    last_log_time: float,
    last_log_episode: int,
    start_episode: int,
) -> tuple[str, str]:
    now = time.perf_counter()
    trained = episode - start_episode + 1
    elapsed = max(now - run_start, 1e-9)
    speed = trained / elapsed
    sec_per_ep = elapsed / trained
    window_eps = max(1, episode - last_log_episode)
    window_time = now - last_log_time
    remaining = max(0, total - episode)
    eta_seconds = remaining / speed if speed > 0 else float("inf")
    finish = datetime.now() + timedelta(seconds=eta_seconds if np.isfinite(eta_seconds) else 0)
    percent = 100.0 * episode / total

    line1 = (
        f"[{episode:5d}/{total:<5d} {percent:6.2f}%]  "
        f"return={mean_return:9.3f}  avg100={avg100:9.3f}  steps={record['steps']:3d}  "
        f"completion={record['completion_ratio']:.3f}  survival={record['survival_ratio']:.3f}  "
        f"actor={losses['actor_loss']:.4f}  critic={losses['critic_loss']:.4f}"
    )
    line2 = (
        f"    speed={speed:.3f} ep/s ({sec_per_ep:.2f}s/ep)  |  "
        f"last{window_eps}={format_duration(window_time)}  |  "
        f"elapsed={format_duration(elapsed)}  |  ETA={format_duration(eta_seconds)}  |  "
        f"finish≈{finish:%H:%M:%S}"
    )
    return line1, line2


def main():
    parser = argparse.ArgumentParser(description="Train fixed-vector MAPPO baseline")
    parser.add_argument("--episodes", type=int, default=32000,
                        help="target total episode count; with --resume, continue until this episode")
    parser.add_argument("--resume", type=Path,
                        help="resume from checkpoint; old checkpoints restore model weights, newer ones also restore Adam state")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto", help="auto/cpu/cuda/cuda:0")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--minibatch-size", type=int, default=64)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "mappo")
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error("episodes must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = choose_device(args.device)
    env = CooperativeUAVEnv(seed=args.seed)

    resume_checkpoint = None
    start_episode = 1
    if args.resume is not None:
        if not args.resume.exists():
            parser.error(f"resume checkpoint not found: {args.resume}")
        resume_checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        completed_episode = int(resume_checkpoint.get("episode", 0))
        if args.episodes <= completed_episode:
            parser.error(
                f"--episodes is the target total episode count and must be > checkpoint episode "
                f"({completed_episode})"
            )
        start_episode = completed_episode + 1

    if resume_checkpoint is not None and resume_checkpoint.get("vectorizer"):
        vectorizer = FixedVectorizer(**resume_checkpoint["vectorizer"])
    else:
        vectorizer = FixedVectorizer(
            world_size=env.world_size,
            n_uavs=len(env.uavs) or 8,
            n_targets=4,
            n_threats=3,
        )

    if resume_checkpoint is not None and resume_checkpoint.get("config"):
        config = MAPPOConfig(**resume_checkpoint["config"])
    else:
        config = MAPPOConfig(hidden_dim=args.hidden_dim, minibatch_size=args.minibatch_size)

    n_agents = int(resume_checkpoint.get("n_agents", 8)) if resume_checkpoint else 8
    action_dim = int(resume_checkpoint.get("action_dim", 7)) if resume_checkpoint else 7
    learner = MAPPO(
        vectorizer.observation_dim,
        vectorizer.state_dim,
        n_agents=n_agents,
        action_dim=action_dim,
        config=config,
        device=device,
    )

    if resume_checkpoint is not None:
        restored_optimizer = learner.load_checkpoint(resume_checkpoint, load_optimizers=True)
        print(
            f"resumed={args.resume} episode={start_episode - 1} "
            f"optimizer_state={'restored' if restored_optimizer else 'not available; Adam restarted'}"
        )

    args.output.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output / "metrics.jsonl"
    print(f"device={device} obs_dim={vectorizer.observation_dim} state_dim={vectorizer.state_dim}")

    recent_returns = []
    run_start = time.perf_counter()
    last_log_time = run_start
    last_log_episode = start_episode - 1

    for episode in range(start_episode, args.episodes + 1):
        obs, states = env.reset(seed=args.seed + episode - 1)
        buffer = RolloutBuffer(n_agents=n_agents)
        episode_returns = np.zeros(n_agents, dtype=np.float32)
        final_info = None

        while True:
            obs_vec = vectorizer.batch_observations(obs)
            state_vec = vectorizer.batch_states(states)
            active = np.asarray([obs[i] is not None for i in range(n_agents)], dtype=np.float32)
            actions, log_probs, values = learner.act(obs_vec, state_vec, active)

            next_obs, next_states, rewards, done, info = env.step(actions)
            reward_vec = np.asarray([rewards[i] for i in range(n_agents)], dtype=np.float32)
            agent_dones = np.asarray(
                [done or next_obs[i] is None for i in range(n_agents)], dtype=np.float32
            )

            buffer.add(
                obs_vec,
                state_vec,
                actions,
                log_probs,
                reward_vec,
                agent_dones,
                values,
                active,
            )
            episode_returns += reward_vec
            obs, states = next_obs, next_states
            final_info = info
            if done:
                break

        losses = learner.update(buffer)
        mean_return = float(episode_returns.mean())
        recent_returns.append(mean_return)
        if len(recent_returns) > 100:
            recent_returns.pop(0)

        record = {
            "episode": episode,
            "mean_return": mean_return,
            "steps": int(final_info["step"]),
            "completion_ratio": float(final_info["completion_ratio"]),
            "survival_ratio": float(final_info["survival_ratio"]),
            **losses,
        }
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        if episode == start_episode or episode % args.log_every == 0:
            line1, line2 = progress_lines(
                episode,
                args.episodes,
                mean_return,
                float(np.mean(recent_returns)),
                record,
                losses,
                run_start,
                last_log_time,
                last_log_episode,
                start_episode,
            )
            print(line1)
            print(line2)
            last_log_time = time.perf_counter()
            last_log_episode = episode

        if episode % args.save_every == 0 or episode == args.episodes:
            torch.save(
                {
                    **learner.checkpoint(include_optimizers=True),
                    "episode": episode,
                    "vectorizer": {
                        "world_size": vectorizer.world_size,
                        "n_uavs": vectorizer.n_uavs,
                        "n_targets": vectorizer.n_targets,
                        "n_threats": vectorizer.n_threats,
                    },
                },
                args.output / f"checkpoint_{episode:06d}.pt",
            )


if __name__ == "__main__":
    main()
