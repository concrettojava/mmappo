"""Fixed-seed evaluation for MAPPO and PI-MAPPO checkpoints.

Every checkpoint is evaluated on exactly the same environment seeds.  The
default policy is deterministic (argmax), which makes checkpoint comparisons
paired and reproducible.

Outputs
-------
summary.json
    Aggregate metrics and 95% confidence intervals for each checkpoint.
episodes.csv
    One row per checkpoint x test episode for later paired/failure analysis.

Mean return is recorded for diagnostics, but it should only be compared
directly between checkpoints using the same reward profile.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from fofe_mmapppo.algorithms import PIMAPPO, PIMAPPOConfig
from fofe_mmapppo.envs import CooperativeUAVEnv
from fofe_mmapppo.evaluation import load_fixed_mappo_checkpoint, policy_actions_batch
from fofe_mmapppo.models import (
    DirectFixedVectorizer,
    EntityTensorizer,
    FixedVectorizer,
    PIActorConfig,
)


def choose_device(name: str) -> str:
    if name != "auto":
        return name
    return "cuda" if torch.cuda.is_available() else "cpu"


def metric(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "mean": None, "std": None, "ci95": None}
    arr = np.asarray(values, dtype=np.float64)
    std = float(arr.std())
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": std,
        "ci95": float(1.96 * std / math.sqrt(arr.size)),
    }


def safe_label(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_") or "checkpoint"


def checkpoint_kind(checkpoint: dict[str, Any]) -> str:
    if checkpoint.get("algorithm") == "pi_mappo_v1" or "actor_config" in checkpoint:
        return "pi_mappo"
    return "mappo"


def load_pi_checkpoint(
    path: Path,
    device: str,
    compile_actors: bool,
) -> tuple[PIMAPPO, FixedVectorizer, EntityTensorizer, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    fixed_cfg = checkpoint.get("fixed_vectorizer", {})
    fixed = FixedVectorizer(**fixed_cfg) if fixed_cfg else FixedVectorizer()

    tensor_cfg = checkpoint.get("entity_tensorizer", {})
    tensorizer = EntityTensorizer(**tensor_cfg) if tensor_cfg else EntityTensorizer()

    config = PIMAPPOConfig(**checkpoint.get("config", {}))
    # These options are training-only concerns.  Evaluation only needs actor
    # weights and recurrent belief state.
    config.batch_actor_updates = False
    config.compile_actors = bool(compile_actors)

    actor_config = PIActorConfig(**checkpoint.get("actor_config", {}))
    learner = PIMAPPO(
        fixed.state_dim,
        n_agents=int(checkpoint.get("n_agents", 8)),
        action_dim=int(checkpoint.get("action_dim", 7)),
        config=config,
        actor_config=actor_config,
        device=device,
    )
    learner.load_checkpoint(checkpoint, load_optimizers=False)
    learner.actors.eval()
    learner.critics.eval()
    return learner, fixed, tensorizer, checkpoint


def make_episode_row(
    *,
    checkpoint_path: Path,
    algorithm: str,
    checkpoint_episode: int,
    scenario: str,
    reward_profile: str,
    test_index: int,
    test_seed: int,
    info: dict[str, Any],
    mean_return: float,
) -> dict[str, Any]:
    targets_destroyed = 4 - int(info["alive_targets"])
    full_success = int(info["alive_targets"]) == 0
    return {
        "checkpoint": str(checkpoint_path),
        "algorithm": algorithm,
        "checkpoint_episode": checkpoint_episode,
        "scenario": scenario,
        "reward_profile": reward_profile,
        "test_index": int(test_index),
        "seed": int(test_seed),
        "completion_ratio": float(info["completion_ratio"]),
        "full_success": int(full_success),
        "survival_ratio": float(info["survival_ratio"]),
        "steps": int(info["step"]),
        "success_completion_steps": int(info["step"]) if full_success else "",
        "targets_destroyed": int(targets_destroyed),
        "alive_targets": int(info["alive_targets"]),
        "alive_uavs": int(info["alive_uavs"]),
        "uavs_destroyed": 8 - int(info["alive_uavs"]),
        "mean_agent_return": float(mean_return),
    }


def evaluate_mappo(
    path: Path,
    *,
    episodes: int,
    seed: int,
    batch_size: int,
    scenario: str,
    device: str,
    deterministic: bool,
    max_steps_override: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    learner, vectorizer, checkpoint = load_fixed_mappo_checkpoint(path, device)
    direct = DirectFixedVectorizer(vectorizer)

    checkpoint_episode = int(checkpoint.get("episode", -1))
    reward_profile = str(checkpoint.get("reward_profile", "paper"))
    max_steps = int(
        max_steps_override
        if max_steps_override is not None
        else checkpoint.get("max_steps", 200)
    )

    rows: list[dict[str, Any]] = []
    for start in range(0, episodes, batch_size):
        count = min(batch_size, episodes - start)
        seeds = [seed + start + e for e in range(count)]
        envs = [
            CooperativeUAVEnv(
                seed=s,
                scenario=scenario,
                max_steps=max_steps,
                reward_profile=reward_profile,
            )
            for s in seeds
        ]

        obs = np.zeros(
            (count, learner.n_agents, vectorizer.observation_dim), dtype=np.float32
        )
        active = np.zeros((count, learner.n_agents), dtype=np.float32)
        for e, (env, test_seed) in enumerate(zip(envs, seeds)):
            obs[e], _, active[e] = env.reset_vectors(direct, seed=test_seed)

        finished = np.zeros(count, dtype=bool)
        returns = np.zeros((count, learner.n_agents), dtype=np.float32)
        infos: list[dict[str, Any] | None] = [None] * count

        while not bool(finished.all()):
            actions = policy_actions_batch(
                learner, obs, active, deterministic=deterministic
            )
            next_obs = np.zeros_like(obs)
            next_active = np.zeros_like(active)

            for e, env in enumerate(envs):
                if finished[e]:
                    continue
                o, _, a, rewards, done, info = env.step_vectors(actions[e], direct)
                next_obs[e], next_active[e] = o, a
                returns[e] += np.fromiter(
                    (rewards[i] for i in range(learner.n_agents)),
                    dtype=np.float32,
                    count=learner.n_agents,
                )
                if done:
                    finished[e] = True
                    infos[e] = info

            obs, active = next_obs, next_active

        for e, info in enumerate(infos):
            if info is None:
                raise RuntimeError(
                    f"MAPPO test environment {start + e} ended without final info"
                )
            rows.append(
                make_episode_row(
                    checkpoint_path=path,
                    algorithm="mappo",
                    checkpoint_episode=checkpoint_episode,
                    scenario=scenario,
                    reward_profile=reward_profile,
                    test_index=start + e,
                    test_seed=seeds[e],
                    info=info,
                    mean_return=float(returns[e].mean()),
                )
            )

        print(
            f"  MAPPO checkpoint_ep={checkpoint_episode}: "
            f"{start + count}/{episodes}",
            flush=True,
        )

    return rows, {
        "algorithm": "mappo",
        "checkpoint_episode": checkpoint_episode,
        "reward_profile": reward_profile,
        "max_steps": max_steps,
    }


def evaluate_pi(
    path: Path,
    *,
    episodes: int,
    seed: int,
    batch_size: int,
    scenario: str,
    device: str,
    deterministic: bool,
    max_steps_override: int | None,
    compile_actors: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    learner, fixed, tensorizer, checkpoint = load_pi_checkpoint(
        path, device, compile_actors
    )

    checkpoint_episode = int(checkpoint.get("episode", -1))
    reward_profile = str(checkpoint.get("reward_profile", "paper"))
    max_steps = int(
        max_steps_override
        if max_steps_override is not None
        else checkpoint.get("max_steps", 200)
    )

    rows: list[dict[str, Any]] = []
    for start in range(0, episodes, batch_size):
        count = min(batch_size, episodes - start)
        seeds = [seed + start + e for e in range(count)]
        envs = [
            CooperativeUAVEnv(
                seed=s,
                scenario=scenario,
                max_steps=max_steps,
                reward_profile=reward_profile,
            )
            for s in seeds
        ]

        observations = []
        for env, test_seed in zip(envs, seeds):
            obs, _ = env.reset(seed=test_seed)
            observations.append(obs)

        finished = np.zeros(count, dtype=bool)
        returns = np.zeros((count, learner.n_agents), dtype=np.float32)
        infos: list[dict[str, Any] | None] = [None] * count
        beliefs = learner.initial_belief_states(count)

        # PI actor actions are independent of critic outputs. PIMAPPO.act_batch
        # currently also evaluates critics, so correctly-shaped zeros avoid
        # constructing centralized states solely for unused evaluation values.
        zero_states = np.zeros(
            (count, learner.n_agents, fixed.state_dim), dtype=np.float32
        )

        while not bool(finished.all()):
            structured = tensorizer.parallel(observations, finished=finished)
            actions, _, _, beliefs = learner.act_batch(
                structured.self_features,
                structured.entity_features,
                structured.evidence_mask,
                structured.evidence_meta,
                zero_states,
                structured.active.astype(np.float32, copy=False),
                belief_states=beliefs,
                deterministic=deterministic,
            )

            next_observations = list(observations)
            for e, env in enumerate(envs):
                if finished[e]:
                    continue
                next_obs, _, rewards, done, info = env.step(actions[e])
                returns[e] += np.fromiter(
                    (rewards[i] for i in range(learner.n_agents)),
                    dtype=np.float32,
                    count=learner.n_agents,
                )
                next_observations[e] = next_obs
                if done:
                    finished[e] = True
                    infos[e] = info
            observations = next_observations

        for e, info in enumerate(infos):
            if info is None:
                raise RuntimeError(
                    f"PI test environment {start + e} ended without final info"
                )
            rows.append(
                make_episode_row(
                    checkpoint_path=path,
                    algorithm="pi_mappo",
                    checkpoint_episode=checkpoint_episode,
                    scenario=scenario,
                    reward_profile=reward_profile,
                    test_index=start + e,
                    test_seed=seeds[e],
                    info=info,
                    mean_return=float(returns[e].mean()),
                )
            )

        print(
            f"  PI-MAPPO checkpoint_ep={checkpoint_episode}: "
            f"{start + count}/{episodes}",
            flush=True,
        )

    return rows, {
        "algorithm": "pi_mappo",
        "checkpoint_episode": checkpoint_episode,
        "reward_profile": reward_profile,
        "max_steps": max_steps,
        "compile_actors": bool(compile_actors),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    successes = [r for r in rows if int(r["full_success"]) == 1]
    return {
        "episodes": len(rows),
        "completion_ratio": metric([float(r["completion_ratio"]) for r in rows]),
        "full_success_rate": metric([float(r["full_success"]) for r in rows]),
        "survival_ratio": metric([float(r["survival_ratio"]) for r in rows]),
        "steps": metric([float(r["steps"]) for r in rows]),
        "success_completion_steps": metric(
            [float(r["steps"]) for r in successes]
        ),
        "targets_destroyed": metric(
            [float(r["targets_destroyed"]) for r in rows]
        ),
        "mean_agent_return": metric(
            [float(r["mean_agent_return"]) for r in rows]
        ),
        "targets_destroyed_distribution": {
            str(k): sum(int(r["targets_destroyed"]) == k for r in rows)
            for k in range(5)
        },
    }


def fmt(stats: dict[str, Any]) -> str:
    if stats["n"] == 0:
        return "n=0"
    return (
        f"{stats['mean']:.4f} +/- {stats['ci95']:.4f} "
        f"(95% CI, std={stats['std']:.4f}, n={stats['n']})"
    )


def print_summary(
    label: str,
    metadata: dict[str, Any],
    summary: dict[str, Any],
) -> None:
    print()
    print("=" * 78)
    print(
        f"{label} | {metadata['algorithm']} | checkpoint_ep="
        f"{metadata['checkpoint_episode']} | reward={metadata['reward_profile']}"
    )
    print(f"completion_ratio:         {fmt(summary['completion_ratio'])}")
    print(f"full_success_rate:        {fmt(summary['full_success_rate'])}")
    print(f"survival_ratio:           {fmt(summary['survival_ratio'])}")
    print(f"steps:                    {fmt(summary['steps'])}")
    print(
        f"success_completion_steps: {fmt(summary['success_completion_steps'])}"
    )
    print(f"targets_destroyed:        {fmt(summary['targets_destroyed'])}")
    print(f"mean_agent_return:        {fmt(summary['mean_agent_return'])}")
    print(
        "targets_destroyed_distribution: "
        + ", ".join(
            f"{k}={v}"
            for k, v in summary["targets_destroyed_distribution"].items()
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate MAPPO and PI-MAPPO checkpoints on identical fixed seeds"
        )
    )
    parser.add_argument(
        "checkpoints",
        nargs="+",
        type=Path,
        help="one or more MAPPO / PI-MAPPO checkpoint .pt files",
    )
    parser.add_argument("--episodes", type=int, default=128)
    parser.add_argument("--seed", type=int, default=50000)
    parser.add_argument(
        "--scenario",
        choices=("reference", "contested"),
        default="contested",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="parallel test environments per batch; seeds stay unchanged",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="sample actions; default is deterministic argmax",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="override checkpoint max_steps (normally leave unset)",
    )
    parser.add_argument(
        "--compile-pi-actors",
        action="store_true",
        help="enable torch.compile for PI actor inference",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "result directory; default "
            "outputs/fixed_seed_eval_<scenario>_seed<seed>_n<N>"
        ),
    )
    args = parser.parse_args()

    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.max_steps is not None and args.max_steps <= 0:
        parser.error("--max-steps must be positive")

    device = choose_device(args.device)
    output = args.output or (
        ROOT
        / "outputs"
        / f"fixed_seed_eval_{args.scenario}_seed{args.seed}_n{args.episodes}"
    )
    output.mkdir(parents=True, exist_ok=True)

    print(
        f"fixed-seed evaluation: scenario={args.scenario} "
        f"episodes={args.episodes} "
        f"seeds={args.seed}..{args.seed + args.episodes - 1} "
        f"policy={'stochastic' if args.stochastic else 'deterministic'} "
        f"device={device}"
    )

    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    for path in args.checkpoints:
        if not path.exists():
            raise FileNotFoundError(path)

        raw = torch.load(path, map_location="cpu", weights_only=False)
        kind = checkpoint_kind(raw)
        del raw

        print()
        print(f"evaluating {path} ({kind})")
        if kind == "pi_mappo":
            rows, metadata = evaluate_pi(
                path,
                episodes=args.episodes,
                seed=args.seed,
                batch_size=args.batch_size,
                scenario=args.scenario,
                device=device,
                deterministic=not args.stochastic,
                max_steps_override=args.max_steps,
                compile_actors=args.compile_pi_actors,
            )
        else:
            rows, metadata = evaluate_mappo(
                path,
                episodes=args.episodes,
                seed=args.seed,
                batch_size=args.batch_size,
                scenario=args.scenario,
                device=device,
                deterministic=not args.stochastic,
                max_steps_override=args.max_steps,
            )

        summary = summarize(rows)
        label = (
            f"{metadata['algorithm']}_ep{metadata['checkpoint_episode']:06d}_"
            f"{safe_label(path.stem)}"
        )
        print_summary(label, metadata, summary)

        all_rows.extend(rows)
        summaries.append(
            {
                "label": label,
                "checkpoint": str(path),
                **metadata,
                "scenario": args.scenario,
                "seed_start": args.seed,
                "seed_end": args.seed + args.episodes - 1,
                "deterministic": not args.stochastic,
                "summary": summary,
            }
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    csv_path = output / "episodes.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    summary_path = output / "summary.json"
    payload = {
        "protocol": {
            "scenario": args.scenario,
            "episodes": args.episodes,
            "seed_start": args.seed,
            "seed_end": args.seed + args.episodes - 1,
            "deterministic": not args.stochastic,
            "batch_size": args.batch_size,
            "device": device,
            "return_comparison_note": (
                "Compare mean_agent_return only within the same reward_profile. "
                "Task metrics are the primary cross-profile comparison."
            ),
        },
        "checkpoints": summaries,
    }
    summary_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print()
    print("=" * 78)
    print(f"saved summary:  {summary_path}")
    print(f"saved episodes: {csv_path}")
    print(
        "NOTE: mean_agent_return is directly comparable only between "
        "checkpoints using the same reward_profile."
    )


if __name__ == "__main__":
    main()
