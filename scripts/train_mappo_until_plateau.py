"""Train fixed-vector MAPPO with periodic deterministic validation and plateau stopping.

This controller deliberately leaves the high-throughput trainer unchanged. It
runs training in chunks, evaluates actors only after each chunk, and resumes
from the produced checkpoint until either a validation plateau or the maximum
episode budget is reached.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fofe_mmapppo.evaluation import evaluate_fixed_mappo, load_fixed_mappo_checkpoint


HIGHER_IS_BETTER = {
    "completion_ratio": True,
    "survival_ratio": True,
    "mean_agent_return": True,
    "completion_time_steps": False,
}


def checkpoint_path(output: Path, episode: int) -> Path:
    return output / f"checkpoint_{episode:06d}.pt"


def run_training_chunk(args, target_episode: int, resume: Path | None) -> Path:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "train_mappo_parallel.py"),
        "--episodes",
        str(target_episode),
        "--num-envs",
        str(args.num_envs),
        "--minibatch-size",
        str(args.minibatch_size),
        "--device",
        args.device,
        "--log-every",
        str(args.log_every),
        "--save-every",
        str(args.save_every or args.eval_every),
        "--seed",
        str(args.seed),
        "--output",
        str(args.output),
    ]
    if resume is not None:
        cmd += ["--resume", str(resume)]
    if args.no_tensorboard:
        cmd.append("--no-tensorboard")

    print(f"\n=== TRAIN to episode {target_episode} ===")
    subprocess.run(cmd, check=True)
    produced = checkpoint_path(args.output, target_episode)
    if not produced.exists():
        raise FileNotFoundError(f"expected checkpoint not found: {produced}")
    return produced


def evaluate_checkpoint(args, checkpoint: Path, episode: int) -> dict:
    learner, vectorizer, _ = load_fixed_mappo_checkpoint(checkpoint, args.device)
    stats = evaluate_fixed_mappo(
        learner,
        vectorizer,
        episodes=args.eval_episodes,
        seed=args.eval_seed,
        deterministic=True,
        batch_size=args.eval_batch_size,
    )
    print(f"\n=== EVAL episode {episode} ({args.eval_episodes} deterministic episodes) ===")
    for name, values in stats.items():
        print(f"{name}: {values['mean']:.4f} ± {values['std']:.4f}")
    return stats


def is_significant_improvement(current: float, best: float, delta: float, higher: bool) -> bool:
    if higher:
        return current >= best + delta
    return current <= best - delta


def main():
    parser = argparse.ArgumentParser(
        description="MAPPO training with periodic actor-only validation and plateau stopping"
    )
    parser.add_argument("--episodes", type=int, default=200000, help="maximum global episode number")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--minibatch-size", type=int, default=1024)
    parser.add_argument("--log-every", type=int, default=128)
    parser.add_argument("--save-every", type=int, default=None,
                        help="training checkpoint interval; defaults to eval-every")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "mappo_until_plateau")
    parser.add_argument("--no-tensorboard", action="store_true")

    parser.add_argument("--eval-every", type=int, default=5120,
                        help="training episodes between deterministic validations")
    parser.add_argument("--eval-episodes", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--eval-seed", type=int, default=10000)

    parser.add_argument(
        "--early-stop-metric",
        choices=tuple(HIGHER_IS_BETTER),
        default="completion_ratio",
    )
    parser.add_argument("--early-stop-min-episodes", type=int, default=20480,
                        help="do not count plateau patience before this global episode")
    parser.add_argument("--early-stop-patience", type=int, default=4,
                        help="stop after this many validation rounds without significant improvement")
    parser.add_argument("--early-stop-min-delta", type=float, default=0.02,
                        help="minimum significant improvement; 0.02 means two percentage points for ratios")
    parser.add_argument("--no-early-stop", action="store_true")
    args = parser.parse_args()

    positives = {
        "episodes": args.episodes,
        "num-envs": args.num_envs,
        "minibatch-size": args.minibatch_size,
        "eval-every": args.eval_every,
        "eval-episodes": args.eval_episodes,
        "eval-batch-size": args.eval_batch_size,
        "early-stop-patience": args.early_stop_patience,
    }
    for name, value in positives.items():
        if value <= 0:
            parser.error(f"{name} must be positive")
    if args.save_every is not None and args.save_every <= 0:
        parser.error("save-every must be positive")
    if args.early_stop_min_delta < 0:
        parser.error("early-stop-min-delta must be non-negative")

    args.output.mkdir(parents=True, exist_ok=True)
    eval_log = args.output / "eval_metrics.jsonl"

    start_episode = 0
    current_checkpoint = None
    if args.resume is not None:
        import torch
        cp = torch.load(args.resume, map_location="cpu", weights_only=False)
        start_episode = int(cp.get("episode", 0))
        current_checkpoint = args.resume
        print(f"controller resume={args.resume} episode={start_episode}")
    if start_episode >= args.episodes:
        parser.error("resume checkpoint is already at or beyond --episodes")

    processed = start_episode
    best = None
    best_episode = None
    stale = 0
    higher = HIGHER_IS_BETTER[args.early_stop_metric]

    while processed < args.episodes:
        target = min(processed + args.eval_every, args.episodes)
        current_checkpoint = run_training_chunk(args, target, current_checkpoint)
        processed = target

        stats = evaluate_checkpoint(args, current_checkpoint, processed)
        current = float(stats[args.early_stop_metric]["mean"])
        eligible = processed >= args.early_stop_min_episodes
        improved = False

        if eligible:
            if best is None:
                improved = True
            else:
                improved = is_significant_improvement(
                    current, best, args.early_stop_min_delta, higher
                )
            if improved:
                best = current
                best_episode = processed
                stale = 0
                shutil.copy2(current_checkpoint, args.output / "checkpoint_best.pt")
            else:
                stale += 1

        record = {
            "episode": processed,
            "eval_episodes": args.eval_episodes,
            "eval_seed": args.eval_seed,
            "metric": args.early_stop_metric,
            "metric_value": current,
            "eligible_for_early_stop": eligible,
            "significant_improvement": improved,
            "best_value": best,
            "best_episode": best_episode,
            "stale_evals": stale,
            "stats": stats,
        }
        with eval_log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        if eligible:
            print(
                f"plateau metric={args.early_stop_metric} current={current:.4f} "
                f"best={best:.4f} best_ep={best_episode} stale={stale}/{args.early_stop_patience}"
            )

        if (
            not args.no_early_stop
            and eligible
            and best is not None
            and stale >= args.early_stop_patience
        ):
            shutil.copy2(current_checkpoint, args.output / "checkpoint_stopped.pt")
            print(
                f"EARLY STOP at episode {processed}: no significant "
                f"{args.early_stop_metric} improvement for {stale} validations."
            )
            break

    print(f"controller finished at episode {processed}")
    if best is not None:
        print(f"best {args.early_stop_metric}={best:.4f} at episode {best_episode}")


if __name__ == "__main__":
    main()
