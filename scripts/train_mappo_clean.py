"""Clean MAPPO trainer for the contested UAV environment.

One training round does exactly this:
1. run `num_envs` independent environments until each finishes one episode;
2. collect those complete trajectories;
3. perform one MAPPO/PPO update.

The main progress counter is therefore *rounds/updates*, not episodes. With
`num_envs=16`, one round performs one policy update using 16 collected
episodes. The log also reports `data_ep` so the amount of sampled experience is
explicit without pretending that 16 parallel environments are 16 update rounds.
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

from fofe_mmapppo.algorithms import MAPPO, MAPPOConfig, ParallelRolloutBuffer
from fofe_mmapppo.envs import CooperativeUAVEnv
from fofe_mmapppo.models import DirectFixedVectorizer, FixedVectorizer


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


def save_checkpoint(
    path: Path,
    learner: MAPPO,
    vectorizer: FixedVectorizer,
    round_idx: int,
    num_envs: int,
    scenario: str,
    reward_profile: str,
) -> None:
    torch.save(
        {
            **learner.checkpoint(include_optimizers=True),
            "observation_schema_version": 2,
            "round": int(round_idx),
            "episodes_collected": int(round_idx * num_envs),
            "num_envs": int(num_envs),
            "scenario": str(scenario),
            "reward_profile": str(reward_profile),
            "vectorizer": {
                "world_size": vectorizer.world_size,
                "n_uavs": vectorizer.n_uavs,
                "n_targets": vectorizer.n_targets,
                "n_threats": vectorizer.n_threats,
            },
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean MAPPO training: one round = num_envs episodes + one PPO update"
    )
    parser.add_argument("--rounds", type=int, default=5000)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto", help="auto/cpu/cuda/cuda:0")
    parser.add_argument("--scenario", choices=("reference", "contested"), default="contested")
    parser.add_argument("--reward-profile", choices=("paper", "task_aligned"), default="paper")
    parser.add_argument("--minibatch-size", type=int, default=1024)
    parser.add_argument("--ppo-epochs", type=int, default=15)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "info_mappo_clean",
    )
    args = parser.parse_args()

    if args.rounds <= 0:
        parser.error("--rounds must be positive")
    if args.num_envs <= 0:
        parser.error("--num-envs must be positive")
    if args.minibatch_size <= 0 or args.ppo_epochs <= 0:
        parser.error("--minibatch-size and --ppo-epochs must be positive")
    if args.log_every <= 0 or args.save_every <= 0:
        parser.error("--log-every and --save-every must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")

    device = choose_device(args.device)

    probe = CooperativeUAVEnv(
        seed=args.seed,
        scenario=args.scenario,
        reward_profile=args.reward_profile,
    )
    vectorizer = FixedVectorizer(
        world_size=probe.world_size,
        n_uavs=8,
        n_targets=4,
        n_threats=3,
    )
    direct_vectorizer = DirectFixedVectorizer(vectorizer)

    checkpoint = None
    start_round = 0
    if args.resume is not None:
        if not args.resume.exists():
            parser.error(f"checkpoint not found: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        try:
            FixedVectorizer.assert_checkpoint_compatible(checkpoint)
        except ValueError as exc:
            parser.error(str(exc))

        start_round = int(checkpoint.get("round", 0))
        ckpt_envs = int(checkpoint.get("num_envs", args.num_envs))
        ckpt_scenario = str(checkpoint.get("scenario", args.scenario))
        ckpt_reward = str(checkpoint.get("reward_profile", args.reward_profile))
        if ckpt_envs != args.num_envs:
            parser.error(
                f"checkpoint num_envs={ckpt_envs} does not match --num-envs={args.num_envs}"
            )
        if ckpt_scenario != args.scenario:
            parser.error(
                f"checkpoint scenario={ckpt_scenario!r} does not match --scenario={args.scenario!r}"
            )
        if ckpt_reward != args.reward_profile:
            parser.error(
                f"checkpoint reward_profile={ckpt_reward!r} does not match "
                f"--reward-profile={args.reward_profile!r}"
            )
        if args.rounds <= start_round:
            parser.error(
                f"--rounds is the final target round and must exceed checkpoint round {start_round}"
            )
        cfg = MAPPOConfig(**checkpoint["config"])
    else:
        cfg = MAPPOConfig(
            minibatch_size=args.minibatch_size,
            ppo_epochs=args.ppo_epochs,
        )

    learner = MAPPO(
        vectorizer.observation_dim,
        vectorizer.state_dim,
        n_agents=8,
        action_dim=7,
        config=cfg,
        device=device,
    )
    if checkpoint is not None:
        restored = learner.load_checkpoint(checkpoint, load_optimizers=True)
        print(
            f"resume={args.resume} round={start_round} "
            f"optimizer={'restored' if restored else 'restarted'}"
        )

    args.output.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output / "metrics.jsonl"

    print(
        f"device={device} scenario={args.scenario} reward={args.reward_profile} "
        f"num_envs={args.num_envs} obs_dim={vectorizer.observation_dim} "
        f"state_dim={vectorizer.state_dim}"
    )
    print(
        "progress_unit=round/update; "
        f"1 round = {args.num_envs} collected episodes + 1 PPO update"
    )

    recent_round_returns: list[float] = []
    run_start = time.perf_counter()
    rounds_this_run = 0

    for round_idx in range(start_round + 1, args.rounds + 1):
        round_start = time.perf_counter()

        # Each environment contributes exactly one complete episode in this round.
        envs = []
        obs_vec = np.zeros(
            (args.num_envs, 8, vectorizer.observation_dim), dtype=np.float32
        )
        state_vec = np.zeros(
            (args.num_envs, 8, vectorizer.state_dim), dtype=np.float32
        )
        active = np.zeros((args.num_envs, 8), dtype=np.float32)

        for e in range(args.num_envs):
            episode_serial = (round_idx - 1) * args.num_envs + e
            episode_seed = args.seed + episode_serial
            env = CooperativeUAVEnv(
                seed=episode_seed,
                scenario=args.scenario,
                reward_profile=args.reward_profile,
            )
            envs.append(env)
            obs_vec[e], state_vec[e], active[e] = env.reset_vectors(
                direct_vectorizer, seed=episode_seed
            )

        finished = np.zeros(args.num_envs, dtype=bool)
        episode_returns = np.zeros((args.num_envs, 8), dtype=np.float32)
        final_infos = [None] * args.num_envs
        buffer = ParallelRolloutBuffer(n_envs=args.num_envs, n_agents=8)

        while not bool(finished.all()):
            actions, log_probs, values = learner.act_batch(
                obs_vec, state_vec, active
            )

            reward_batch = np.zeros((args.num_envs, 8), dtype=np.float32)
            done_batch = np.ones((args.num_envs, 8), dtype=np.float32)
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
                    (rewards[i] for i in range(8)),
                    dtype=np.float32,
                    count=8,
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

        per_episode_returns = episode_returns.mean(axis=1)
        round_return = float(per_episode_returns.mean())
        recent_round_returns.append(round_return)
        if len(recent_round_returns) > 100:
            recent_round_returns.pop(0)
        avg100_return = float(np.mean(recent_round_returns))

        completion = float(
            np.mean([info["completion_ratio"] for info in final_infos])
        )
        survival = float(
            np.mean([info["survival_ratio"] for info in final_infos])
        )
        mean_steps = float(np.mean([info["step"] for info in final_infos]))

        round_time = time.perf_counter() - round_start
        rounds_this_run += 1
        elapsed = time.perf_counter() - run_start
        avg_sec_per_round = elapsed / max(rounds_this_run, 1)
        remaining_rounds = args.rounds - round_idx
        eta_seconds = remaining_rounds * avg_sec_per_round
        finish_time = datetime.now() + timedelta(seconds=eta_seconds)

        record = {
            "round": round_idx,
            "episodes_collected": round_idx * args.num_envs,
            "num_envs": args.num_envs,
            "scenario": args.scenario,
            "reward_profile": args.reward_profile,
            "round_return": round_return,
            "avg100_round_return": avg100_return,
            "completion_ratio": completion,
            "survival_ratio": survival,
            "mean_steps": mean_steps,
            "actor_loss": float(losses["actor_loss"]),
            "critic_loss": float(losses["critic_loss"]),
            "entropy": float(losses["entropy"]),
            "round_time_seconds": float(round_time),
            "elapsed_seconds_this_run": float(elapsed),
            "eta_seconds": float(eta_seconds),
        }
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        if round_idx % args.log_every == 0 or round_idx == args.rounds:
            percent = 100.0 * round_idx / args.rounds
            print(
                f"[{round_idx:5d}/{args.rounds:<5d} {percent:6.2f}%] "
                f"round_return={round_return:9.3f} "
                f"avg100={avg100_return:9.3f} "
                f"completion={completion:.3f} survival={survival:.3f} "
                f"actor={losses['actor_loss']:.4f} critic={losses['critic_loss']:.4f}"
            )
            print(
                f"    {round_time:.2f}s/round | "
                f"elapsed={format_duration(elapsed)} | "
                f"ETA={format_duration(eta_seconds)} | "
                f"finish≈{finish_time:%Y-%m-%d %H:%M:%S} | "
                f"data_ep={round_idx * args.num_envs}"
            )

        if round_idx % args.save_every == 0 or round_idx == args.rounds:
            save_checkpoint(
                args.output / f"checkpoint_round_{round_idx:06d}.pt",
                learner,
                vectorizer,
                round_idx,
                args.num_envs,
                args.scenario,
                args.reward_profile,
            )


if __name__ == "__main__":
    main()
