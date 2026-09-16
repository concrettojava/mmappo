"""Evaluate a trained fixed-vector MAPPO checkpoint."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fofe_mmapppo.evaluation import evaluate_fixed_mappo, load_fixed_mappo_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Evaluate fixed-vector MAPPO baseline")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--stochastic", action="store_true", help="sample actions instead of argmax")
    args = parser.parse_args()

    learner, vectorizer, _ = load_fixed_mappo_checkpoint(args.checkpoint, args.device)
    stats = evaluate_fixed_mappo(
        learner,
        vectorizer,
        episodes=args.episodes,
        seed=args.seed,
        deterministic=not args.stochastic,
        batch_size=args.batch_size,
    )

    print(f"episodes: {args.episodes}")
    for name, values in stats.items():
        print(f"{name}: {values['mean']:.4f} ± {values['std']:.4f}")


if __name__ == "__main__":
    main()
