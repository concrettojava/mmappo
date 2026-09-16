"""Torch profiler for the correctness-first PI-MAPPO training path.

This script runs one real parallel rollout batch and profiles three labelled
regions separately inside one trace:

- ``PI/collect``: environment stepping + actor inference + tensorization
- ``PI/replay_diagnostics``: unchanged-policy recurrent replay check
- ``PI/update_parallel``: recurrent PPO update / backward

It complements cProfile: cProfile tells us which Python functions dominate,
while torch.profiler shows CPU operators, CUDA kernels, launch fragmentation,
and memory behavior.
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
from torch.profiler import ProfilerActivity, profile, record_function

from fofe_mmapppo.algorithms import PIParallelRolloutBuffer, PIMAPPO, PIMAPPOConfig
from fofe_mmapppo.envs import CooperativeUAVEnv
from fofe_mmapppo.models import EntityTensorizer, FixedVectorizer, PIActorConfig


def choose_device(name: str) -> str:
    if name != "auto":
        return name
    return "cuda" if torch.cuda.is_available() else "cpu"


def sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _profiler_table(prof, sort_by: str, row_limit: int) -> str:
    try:
        return prof.key_averages().table(sort_by=sort_by, row_limit=row_limit)
    except Exception as exc:
        return f"failed to render profiler table sort_by={sort_by!r}: {exc}\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Torch-profiler probe for PI-MAPPO")
    parser.add_argument("--scenario", choices=("reference", "contested"), default="contested")
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--ppo-epochs", type=int, default=1)
    parser.add_argument("--sequence-env-minibatch-size", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--belief-dim", type=int, default=64)
    parser.add_argument("--context-dim", type=int, default=32)
    parser.add_argument("--relation-dim", type=int, default=64)
    parser.add_argument("--rows", type=int, default=80)
    parser.add_argument("--skip-replay-check", action="store_true")
    parser.add_argument("--profile-memory", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "pi_torch_profile")
    args = parser.parse_args()

    for name in (
        "num_envs",
        "max_steps",
        "ppo_epochs",
        "sequence_env_minibatch_size",
        "horizon",
        "belief_dim",
        "context_dim",
        "relation_dim",
        "rows",
    ):
        if getattr(args, name.replace("_", "-"), None) is not None:
            pass
    if min(
        args.num_envs,
        args.max_steps,
        args.ppo_epochs,
        args.sequence_env_minibatch_size,
        args.horizon,
        args.belief_dim,
        args.context_dim,
        args.relation_dim,
        args.rows,
    ) <= 0:
        parser.error("all size/count arguments must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")

    device = torch.device(choose_device(args.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda requested but CUDA is unavailable")

    fixed = FixedVectorizer()
    tensorizer = EntityTensorizer(max_age=float(args.max_steps))
    actor_cfg = PIActorConfig(
        n_hypotheses=3,
        horizon=args.horizon,
        belief_dim=args.belief_dim,
        context_dim=args.context_dim,
        relation_dim=args.relation_dim,
        max_age=float(args.max_steps),
    )
    ppo_cfg = PIMAPPOConfig(
        ppo_epochs=args.ppo_epochs,
        sequence_env_minibatch_size=args.sequence_env_minibatch_size,
    )
    learner = PIMAPPO(
        fixed.state_dim,
        n_agents=8,
        action_dim=7,
        config=ppo_cfg,
        actor_config=actor_cfg,
        device=device,
    )

    envs = [
        CooperativeUAVEnv(
            seed=args.seed + e,
            scenario=args.scenario,
            max_steps=args.max_steps,
        )
        for e in range(args.num_envs)
    ]
    observations = []
    global_states = []
    for e, env in enumerate(envs):
        obs, states = env.reset(seed=args.seed + e)
        observations.append(obs)
        global_states.append(states)

    finished = np.zeros(args.num_envs, dtype=bool)
    beliefs = learner.initial_belief_states(args.num_envs)
    buffer = PIParallelRolloutBuffer(n_envs=args.num_envs, n_agents=8)

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    args.output.mkdir(parents=True, exist_ok=True)
    trace_path = args.output / "trace.json"
    cpu_table_path = args.output / "cpu_self_time.txt"
    cuda_table_path = args.output / "cuda_self_time.txt"
    device_total_path = args.output / "device_total_time.txt"
    summary_path = args.output / "summary.json"

    wall = {}
    sync_if_cuda(device)
    profile_start = time.perf_counter()
    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=args.profile_memory,
        with_stack=False,
    ) as prof:
        sync_if_cuda(device)
        t0 = time.perf_counter()
        with record_function("PI/collect"):
            while not bool(finished.all()):
                structured = tensorizer.parallel(observations, finished=finished)
                state_vectors = np.zeros(
                    (args.num_envs, 8, fixed.state_dim), dtype=np.float32
                )
                for e in range(args.num_envs):
                    if not finished[e]:
                        state_vectors[e] = fixed.batch_states(global_states[e])

                active = structured.active.astype(np.float32, copy=False)
                actions, log_probs, values, beliefs = learner.act_batch(
                    structured.self_features,
                    structured.entity_features,
                    structured.evidence_mask,
                    structured.evidence_meta,
                    state_vectors,
                    active,
                    belief_states=beliefs,
                    deterministic=False,
                )

                reward_batch = np.zeros((args.num_envs, 8), dtype=np.float32)
                done_batch = np.ones((args.num_envs, 8), dtype=np.float32)
                next_observations = list(observations)
                next_states = list(global_states)
                for e, env in enumerate(envs):
                    if finished[e]:
                        continue
                    next_obs, next_state, rewards, done, _ = env.step(actions[e])
                    reward_batch[e] = np.fromiter(
                        (rewards[i] for i in range(8)), dtype=np.float32, count=8
                    )
                    done_batch[e] = np.asarray(
                        [1.0 if done or next_obs[i] is None else 0.0 for i in range(8)],
                        dtype=np.float32,
                    )
                    next_observations[e] = next_obs
                    next_states[e] = next_state
                    if done:
                        finished[e] = True

                buffer.add(
                    structured.self_features,
                    structured.entity_features,
                    structured.evidence_mask,
                    structured.evidence_meta,
                    state_vectors,
                    actions,
                    log_probs,
                    reward_batch,
                    done_batch,
                    values,
                    active,
                )
                observations = next_observations
                global_states = next_states
        sync_if_cuda(device)
        wall["collect_seconds"] = time.perf_counter() - t0

        if not args.skip_replay_check:
            t0 = time.perf_counter()
            with record_function("PI/replay_diagnostics"):
                replay = learner.replay_diagnostics(buffer)
            sync_if_cuda(device)
            wall["replay_seconds"] = time.perf_counter() - t0
            wall.update({f"replay_{k}": float(v) for k, v in replay.items()})

        t0 = time.perf_counter()
        with record_function("PI/update_parallel"):
            metrics = learner.update_parallel(buffer)
        sync_if_cuda(device)
        wall["update_seconds"] = time.perf_counter() - t0
        wall.update({f"metric_{k}": float(v) for k, v in metrics.items()})

    sync_if_cuda(device)
    wall["profile_wall_seconds"] = time.perf_counter() - profile_start

    prof.export_chrome_trace(str(trace_path))
    cpu_table_path.write_text(
        _profiler_table(prof, "self_cpu_time_total", args.rows), encoding="utf-8"
    )

    if device.type == "cuda":
        cuda_text = _profiler_table(prof, "self_cuda_time_total", args.rows)
        if cuda_text.startswith("failed to render"):
            cuda_text = _profiler_table(prof, "self_device_time_total", args.rows)
        cuda_table_path.write_text(cuda_text, encoding="utf-8")

        device_text = _profiler_table(prof, "cuda_time_total", args.rows)
        if device_text.startswith("failed to render"):
            device_text = _profiler_table(prof, "device_time_total", args.rows)
        device_total_path.write_text(device_text, encoding="utf-8")
    else:
        cuda_table_path.write_text("CUDA profiling was not enabled.\n", encoding="utf-8")
        device_total_path.write_text("CUDA profiling was not enabled.\n", encoding="utf-8")

    summary = {
        "device": str(device),
        "scenario": args.scenario,
        "num_envs": args.num_envs,
        "max_steps": args.max_steps,
        "ppo_epochs": args.ppo_epochs,
        "sequence_env_minibatch_size": args.sequence_env_minibatch_size,
        "actor_config": {
            "horizon": args.horizon,
            "belief_dim": args.belief_dim,
            "context_dim": args.context_dim,
            "relation_dim": args.relation_dim,
        },
        **wall,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"trace={trace_path}")
    print(f"cpu_table={cpu_table_path}")
    print(f"cuda_table={cuda_table_path}")
    print(f"device_total_table={device_total_path}")
    print("Open trace.json in chrome://tracing or Perfetto for the timeline view.")


if __name__ == "__main__":
    main()
