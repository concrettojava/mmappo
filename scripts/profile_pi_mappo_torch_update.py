"""Low-memory torch profiler for the PI-MAPPO update path.

The rollout is collected *outside* torch.profiler so profiler event storage does
not grow with every environment/actor inference call. Only ``update_parallel``
is profiled. This is intentionally a diagnostic script, not a training entrypoint.
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


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def table(prof, keys: tuple[str, ...], rows: int) -> str:
    last = None
    for key in keys:
        try:
            return prof.key_averages().table(sort_by=key, row_limit=rows)
        except Exception as exc:
            last = exc
    return f"failed to render profiler table: {last}\n"


def collect_rollout(args, learner, fixed, tensorizer):
    envs = [
        CooperativeUAVEnv(seed=args.seed + e, scenario=args.scenario, max_steps=args.max_steps)
        for e in range(args.num_envs)
    ]
    observations, global_states = [], []
    for e, env in enumerate(envs):
        obs, state = env.reset(seed=args.seed + e)
        observations.append(obs)
        global_states.append(state)

    finished = np.zeros(args.num_envs, dtype=bool)
    beliefs = learner.initial_belief_states(args.num_envs)
    buffer = PIParallelRolloutBuffer(n_envs=args.num_envs, n_agents=8)

    while not bool(finished.all()):
        structured = tensorizer.parallel(observations, finished=finished)
        state_vectors = np.zeros((args.num_envs, 8, fixed.state_dim), dtype=np.float32)
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
            reward_batch[e] = np.fromiter((rewards[i] for i in range(8)), dtype=np.float32, count=8)
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

    return buffer


def main() -> None:
    p = argparse.ArgumentParser(description="Low-memory CUDA profiler for PI-MAPPO update")
    p.add_argument("--scenario", choices=("reference", "contested"), default="contested")
    p.add_argument("--num-envs", type=int, default=2)
    p.add_argument("--max-steps", type=int, default=32)
    p.add_argument("--seed", type=int, default=1701)
    p.add_argument("--device", default="auto")
    p.add_argument("--ppo-epochs", type=int, default=1)
    p.add_argument("--sequence-env-minibatch-size", type=int, default=2)
    p.add_argument("--horizon", type=int, default=8)
    p.add_argument("--belief-dim", type=int, default=64)
    p.add_argument("--context-dim", type=int, default=32)
    p.add_argument("--relation-dim", type=int, default=64)
    p.add_argument("--rows", type=int, default=60)
    p.add_argument("--record-shapes", action="store_true")
    p.add_argument("--profile-memory", action="store_true")
    p.add_argument("--output", type=Path, default=ROOT / "outputs" / "pi_torch_profile_update")
    args = p.parse_args()

    if min(args.num_envs, args.max_steps, args.ppo_epochs, args.sequence_env_minibatch_size,
           args.horizon, args.belief_dim, args.context_dim, args.relation_dim, args.rows) <= 0:
        p.error("all size/count arguments must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")

    device = torch.device(choose_device(args.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        p.error("--device cuda requested but CUDA is unavailable")

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
    cfg = PIMAPPOConfig(
        ppo_epochs=args.ppo_epochs,
        sequence_env_minibatch_size=args.sequence_env_minibatch_size,
    )
    learner = PIMAPPO(
        fixed.state_dim,
        n_agents=8,
        action_dim=7,
        config=cfg,
        actor_config=actor_cfg,
        device=device,
    )

    sync(device)
    t0 = time.perf_counter()
    buffer = collect_rollout(args, learner, fixed, tensorizer)
    sync(device)
    collect_seconds = time.perf_counter() - t0

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    args.output.mkdir(parents=True, exist_ok=True)
    trace_path = args.output / "trace_update.json"
    summary_path = args.output / "summary.json"

    sync(device)
    t0 = time.perf_counter()
    with profile(
        activities=activities,
        record_shapes=args.record_shapes,
        profile_memory=args.profile_memory,
        with_stack=False,
    ) as prof:
        with record_function("PI/update_parallel"):
            metrics = learner.update_parallel(buffer)
        sync(device)
    update_seconds = time.perf_counter() - t0

    # Export only after the profiler has stopped; this keeps the profiled section clean.
    prof.export_chrome_trace(str(trace_path))
    (args.output / "cpu_self_time.txt").write_text(
        table(prof, ("self_cpu_time_total",), args.rows), encoding="utf-8"
    )
    if device.type == "cuda":
        (args.output / "cuda_self_time.txt").write_text(
            table(prof, ("self_cuda_time_total", "self_device_time_total"), args.rows),
            encoding="utf-8",
        )
        (args.output / "device_total_time.txt").write_text(
            table(prof, ("cuda_time_total", "device_time_total"), args.rows),
            encoding="utf-8",
        )

    summary = {
        "device": str(device),
        "scenario": args.scenario,
        "num_envs": args.num_envs,
        "max_steps": args.max_steps,
        "ppo_epochs": args.ppo_epochs,
        "sequence_env_minibatch_size": args.sequence_env_minibatch_size,
        "record_shapes": bool(args.record_shapes),
        "profile_memory": bool(args.profile_memory),
        "collect_seconds_unprofiled": collect_seconds,
        "update_seconds_profiled": update_seconds,
        **{f"metric_{k}": float(v) for k, v in metrics.items()},
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"trace={trace_path}")
    print(f"cuda_table={args.output / 'cuda_self_time.txt'}")
    print(f"device_total_table={args.output / 'device_total_time.txt'}")


if __name__ == "__main__":
    main()
