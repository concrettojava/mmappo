"""Evaluate MAPPO checkpoints and draw training/evaluation curves.

Outputs two 2x2 figures:
1) training_curves.(png|pdf): statistics recorded during PPO training.
2) evaluation_curves.(png|pdf): independent deterministic evaluation of checkpoints.

Metric convention for "Average Cumulative Reward":
    For one episode, sum raw per-step rewards separately for every UAV,
    then average those episode sums across UAVs. For a batch of evaluation
    episodes, average again across episodes.

This is an undiscounted episodic return averaged across agents/environments.
It is intentionally NOT PPO's discounted/GAE training return.

Example:
    python scripts/evaluate_checkpoints_and_plot.py \
        --run-dir outputs/mappo_11000 \
        --eval-episodes 32 \
        --device cuda
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib.pyplot as plt
import numpy as np
import torch

from fofe_mmapppo.algorithms import MAPPO, MAPPOConfig
from fofe_mmapppo.envs import CooperativeUAVEnv
from fofe_mmapppo.models import DirectFixedVectorizer, FixedVectorizer


CHECKPOINT_RE = re.compile(r"checkpoint_round_(\d+)\.pt$")


def choose_device(name: str) -> str:
    if name != "auto":
        return name
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_no}: {exc}") from exc
    if not rows:
        raise ValueError(f"no records found in {path}")
    return rows


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if window <= 1:
        return values.copy()
    out = np.empty_like(values)
    csum = np.cumsum(np.insert(values, 0, 0.0))
    for i in range(len(values)):
        start = max(0, i + 1 - window)
        out[i] = (csum[i + 1] - csum[start]) / (i + 1 - start)
    return out


def rolling_std(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if window <= 1:
        return np.zeros_like(values)
    out = np.empty_like(values)
    for i in range(len(values)):
        start = max(0, i + 1 - window)
        segment = values[start : i + 1]
        out[i] = float(np.std(segment, ddof=0))
    return out


def discover_checkpoints(run_dir: Path, max_round: int | None) -> list[tuple[int, Path]]:
    found: list[tuple[int, Path]] = []
    for path in run_dir.glob("checkpoint_round_*.pt"):
        match = CHECKPOINT_RE.match(path.name)
        if not match:
            continue
        round_idx = int(match.group(1))
        if max_round is None or round_idx <= max_round:
            found.append((round_idx, path))
    found.sort(key=lambda item: item[0])
    if not found:
        raise FileNotFoundError(f"no checkpoint_round_*.pt found under {run_dir}")
    return found


def episode_batch_eval(
    learner: MAPPO,
    vectorizer: FixedVectorizer,
    direct_vectorizer: DirectFixedVectorizer,
    *,
    scenario: str,
    reward_profile: str,
    seeds: list[int],
    deterministic: bool,
) -> dict[str, np.ndarray]:
    """Evaluate one checkpoint on a fixed batch of independent environments."""
    n_envs = len(seeds)
    n_agents = learner.n_agents

    envs: list[CooperativeUAVEnv] = []
    obs_vec = np.zeros((n_envs, n_agents, vectorizer.observation_dim), dtype=np.float32)
    state_vec = np.zeros((n_envs, n_agents, vectorizer.state_dim), dtype=np.float32)
    active = np.zeros((n_envs, n_agents), dtype=np.float32)

    for e, seed in enumerate(seeds):
        env = CooperativeUAVEnv(
            seed=seed,
            scenario=scenario,
            reward_profile=reward_profile,
        )
        envs.append(env)
        obs_vec[e], state_vec[e], active[e] = env.reset_vectors(
            direct_vectorizer, seed=seed
        )

    finished = np.zeros(n_envs, dtype=bool)
    episode_agent_rewards = np.zeros((n_envs, n_agents), dtype=np.float64)
    final_infos: list[dict | None] = [None] * n_envs

    while not bool(finished.all()):
        actions, _, _ = learner.act_batch(
            obs_vec,
            state_vec,
            active,
            deterministic=deterministic,
        )

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
            episode_agent_rewards[e] += np.fromiter(
                (rewards[i] for i in range(n_agents)),
                dtype=np.float64,
                count=n_agents,
            )

            if done:
                finished[e] = True
                final_infos[e] = info

        obs_vec, state_vec, active = next_obs, next_state, next_active

    if any(info is None for info in final_infos):
        raise RuntimeError("evaluation ended with missing final episode info")

    infos = [info for info in final_infos if info is not None]

    # Average cumulative reward in the same scale as training's round_return:
    # sum raw rewards over the episode for each UAV, then mean over UAVs.
    cumulative_reward = episode_agent_rewards.mean(axis=1)
    completion_ratio = np.asarray(
        [float(info["completion_ratio"]) for info in infos], dtype=np.float64
    )
    survival_ratio = np.asarray(
        [float(info["survival_ratio"]) for info in infos], dtype=np.float64
    )
    completion_time = np.asarray(
        [float(info["step"]) * float(envs[i].dt) for i, info in enumerate(infos)],
        dtype=np.float64,
    )

    return {
        "cumulative_reward": cumulative_reward,
        "completion_ratio": completion_ratio,
        "survival_ratio": survival_ratio,
        "completion_time": completion_time,
    }


def summarize(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    return float(np.mean(values)), float(np.std(values, ddof=0))


def evaluate_checkpoints(
    checkpoints: list[tuple[int, Path]],
    *,
    device: str,
    eval_episodes: int,
    eval_seed: int,
    deterministic: bool,
    output_path: Path,
) -> list[dict]:
    first_checkpoint = torch.load(
        checkpoints[0][1], map_location=device, weights_only=False
    )
    FixedVectorizer.assert_checkpoint_compatible(first_checkpoint)

    scenario = str(first_checkpoint.get("scenario", "contested"))
    reward_profile = str(first_checkpoint.get("reward_profile", "paper"))
    vec_meta = first_checkpoint.get("vectorizer", {})

    probe = CooperativeUAVEnv(
        seed=eval_seed,
        scenario=scenario,
        reward_profile=reward_profile,
    )
    vectorizer = FixedVectorizer(
        world_size=float(vec_meta.get("world_size", probe.world_size)),
        n_uavs=int(vec_meta.get("n_uavs", 8)),
        n_targets=int(vec_meta.get("n_targets", 4)),
        n_threats=int(vec_meta.get("n_threats", 3)),
    )
    direct_vectorizer = DirectFixedVectorizer(vectorizer)

    cfg = MAPPOConfig(**first_checkpoint["config"])
    learner = MAPPO(
        vectorizer.observation_dim,
        vectorizer.state_dim,
        n_agents=int(first_checkpoint.get("n_agents", 8)),
        action_dim=int(first_checkpoint.get("action_dim", 7)),
        config=cfg,
        device=device,
    )
    learner.actors.eval()
    learner.critics.eval()

    # Use exactly the same independent evaluation worlds for every checkpoint.
    seeds = [eval_seed + i for i in range(eval_episodes)]

    rows: list[dict] = []
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    for pos, (round_idx, checkpoint_path) in enumerate(checkpoints, 1):
        t0 = time.perf_counter()
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        FixedVectorizer.assert_checkpoint_compatible(checkpoint)

        if str(checkpoint.get("scenario", scenario)) != scenario:
            raise ValueError(f"scenario mismatch in {checkpoint_path}")
        if str(checkpoint.get("reward_profile", reward_profile)) != reward_profile:
            raise ValueError(f"reward profile mismatch in {checkpoint_path}")
        if checkpoint["config"] != first_checkpoint["config"]:
            raise ValueError(f"MAPPO config mismatch in {checkpoint_path}")

        learner.load_checkpoint(checkpoint, load_optimizers=False)
        learner.actors.eval()
        learner.critics.eval()

        values = episode_batch_eval(
            learner,
            vectorizer,
            direct_vectorizer,
            scenario=scenario,
            reward_profile=reward_profile,
            seeds=seeds,
            deterministic=deterministic,
        )

        reward_mean, reward_std = summarize(values["cumulative_reward"])
        completion_mean, completion_std = summarize(values["completion_ratio"])
        survival_mean, survival_std = summarize(values["survival_ratio"])
        time_mean, time_std = summarize(values["completion_time"])

        row = {
            "round": round_idx,
            "checkpoint": checkpoint_path.name,
            "eval_episodes": eval_episodes,
            "eval_seed_start": eval_seed,
            "deterministic": deterministic,
            "scenario": scenario,
            "reward_profile": reward_profile,
            "average_cumulative_reward": reward_mean,
            "average_cumulative_reward_std": reward_std,
            "completion_ratio": completion_mean,
            "completion_ratio_std": completion_std,
            "survival_ratio": survival_mean,
            "survival_ratio_std": survival_std,
            "completion_time_seconds": time_mean,
            "completion_time_seconds_std": time_std,
        }
        rows.append(row)

        with output_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

        dt = time.perf_counter() - t0
        print(
            f"[{pos:02d}/{len(checkpoints):02d}] round={round_idx:6d} "
            f"reward={reward_mean:9.3f} completion={completion_mean:.3f} "
            f"survival={survival_mean:.3f} time={time_mean:.2f}s "
            f"eval_wall={dt:.2f}s"
        )

    return rows


def style_axes(ax) -> None:
    ax.grid(True, alpha=0.30)
    ax.spines["top"].set_alpha(0.55)
    ax.spines["right"].set_alpha(0.55)


def plot_training_curves(
    metrics_path: Path,
    output_dir: Path,
    smoothing_window: int,
) -> None:
    rows = load_jsonl(metrics_path)
    rounds = np.asarray([row["round"] for row in rows], dtype=np.float64)
    reward = np.asarray([row["round_return"] for row in rows], dtype=np.float64)
    completion = 100.0 * np.asarray(
        [row["completion_ratio"] for row in rows], dtype=np.float64
    )
    survival = 100.0 * np.asarray(
        [row["survival_ratio"] for row in rows], dtype=np.float64
    )
    steps = np.asarray([row["mean_steps"] for row in rows], dtype=np.float64)

    metrics = [
        (
            reward,
            "Average Cumulative Reward",
            "(a) Average Cumulative Reward",
        ),
        (
            completion,
            "Average Completion Ratio/%",
            "(b) Completion Ratio",
        ),
        (
            survival,
            "Average Survival Ratio/%",
            "(c) Survival Ratio",
        ),
        (
            steps,
            "Average Completion Time/s",
            "(d) Completion Time",
        ),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0))
    for ax, (values, ylabel, caption) in zip(axes.flat, metrics):
        smooth = rolling_mean(values, smoothing_window)
        spread = rolling_std(values, smoothing_window)

        ax.plot(rounds, values, linewidth=0.7, alpha=0.16, label="Raw training")
        ax.plot(
            rounds,
            smooth,
            linewidth=1.6,
            label=f"{smoothing_window}-round mean",
        )
        ax.fill_between(
            rounds,
            smooth - spread,
            smooth + spread,
            alpha=0.12,
            linewidth=0.0,
            label=f"{smoothing_window}-round std",
        )
        ax.set_xlabel("Round")
        ax.set_ylabel(ylabel)
        ax.set_title(caption, y=-0.23, fontsize=10)
        style_axes(ax)

    axes[0, 0].legend(loc="best", fontsize=8)
    fig.suptitle("MAPPO Training Curves", fontsize=13)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97), h_pad=2.0, w_pad=1.8)

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "training_curves.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "training_curves.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_evaluation_curves(rows: list[dict], output_dir: Path) -> None:
    rounds = np.asarray([row["round"] for row in rows], dtype=np.float64)

    metrics = [
        (
            "average_cumulative_reward",
            "average_cumulative_reward_std",
            "Average Cumulative Reward",
            "(a) Average Cumulative Reward",
            1.0,
        ),
        (
            "completion_ratio",
            "completion_ratio_std",
            "Average Completion Ratio/%",
            "(b) Completion Ratio",
            100.0,
        ),
        (
            "survival_ratio",
            "survival_ratio_std",
            "Average Survival Ratio/%",
            "(c) Survival Ratio",
            100.0,
        ),
        (
            "completion_time_seconds",
            "completion_time_seconds_std",
            "Average Completion Time/s",
            "(d) Completion Time",
            1.0,
        ),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0))
    for ax, (mean_key, std_key, ylabel, caption, scale) in zip(axes.flat, metrics):
        mean = scale * np.asarray([row[mean_key] for row in rows], dtype=np.float64)
        std = scale * np.asarray([row[std_key] for row in rows], dtype=np.float64)

        ax.plot(rounds, mean, linewidth=1.7, marker="o", markersize=3.0)
        ax.fill_between(
            rounds,
            mean - std,
            mean + std,
            alpha=0.16,
            linewidth=0.0,
            label="±1 std across evaluation episodes",
        )
        ax.set_xlabel("Training Round")
        ax.set_ylabel(ylabel)
        ax.set_title(caption, y=-0.23, fontsize=10)
        style_axes(ax)

    axes[0, 0].legend(loc="best", fontsize=8)
    fig.suptitle("MAPPO Independent Checkpoint Evaluation", fontsize=13)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97), h_pad=2.0, w_pad=1.8)

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "evaluation_curves.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "evaluation_curves.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate saved MAPPO checkpoints and draw paper-style curves."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Training output directory containing metrics.jsonl and checkpoints.",
    )
    parser.add_argument("--eval-episodes", type=int, default=32)
    parser.add_argument(
        "--eval-seed",
        type=int,
        default=100_000,
        help="First seed of a fixed evaluation set reused for every checkpoint.",
    )
    parser.add_argument("--device", default="auto", help="auto/cpu/cuda/cuda:0")
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample policy actions during evaluation instead of argmax actions.",
    )
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=100,
        help="Rolling window used only for the training figure.",
    )
    parser.add_argument(
        "--max-round",
        type=int,
        default=None,
        help="Optionally ignore checkpoints after this round.",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=None,
        help="Defaults to <run-dir>/plots.",
    )
    parser.add_argument(
        "--evaluation-file",
        type=Path,
        default=None,
        help="Defaults to <run-dir>/checkpoint_evaluation.jsonl.",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.exists():
        parser.error(f"metrics file not found: {metrics_path}")
    if args.eval_episodes <= 0:
        parser.error("--eval-episodes must be positive")
    if args.smoothing_window <= 0:
        parser.error("--smoothing-window must be positive")

    plots_dir = (
        args.plots_dir.resolve()
        if args.plots_dir is not None
        else run_dir / "plots"
    )
    evaluation_file = (
        args.evaluation_file.resolve()
        if args.evaluation_file is not None
        else run_dir / "checkpoint_evaluation.jsonl"
    )
    device = choose_device(args.device)

    checkpoints = discover_checkpoints(run_dir, args.max_round)
    print(
        f"run_dir={run_dir}\n"
        f"device={device}\n"
        f"checkpoints={len(checkpoints)} "
        f"({checkpoints[0][0]}..{checkpoints[-1][0]})\n"
        f"eval_episodes/checkpoint={args.eval_episodes}\n"
        f"evaluation_policy={'stochastic' if args.stochastic else 'deterministic'}"
    )

    # Figure 1: training-rollout statistics already recorded in metrics.jsonl.
    plot_training_curves(
        metrics_path,
        plots_dir,
        smoothing_window=args.smoothing_window,
    )
    print(f"wrote {plots_dir / 'training_curves.png'}")

    # Figure 2: independent evaluation of every saved checkpoint.
    rows = evaluate_checkpoints(
        checkpoints,
        device=device,
        eval_episodes=args.eval_episodes,
        eval_seed=args.eval_seed,
        deterministic=not args.stochastic,
        output_path=evaluation_file,
    )
    plot_evaluation_curves(rows, plots_dir)

    print(f"wrote {evaluation_file}")
    print(f"wrote {plots_dir / 'evaluation_curves.png'}")
    print(f"wrote {plots_dir / 'training_curves.pdf'}")
    print(f"wrote {plots_dir / 'evaluation_curves.pdf'}")


if __name__ == "__main__":
    main()
