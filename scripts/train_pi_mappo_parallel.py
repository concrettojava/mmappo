"""Correctness-first parallel trainer for PI-MAPPO.

Unlike the feed-forward MAPPO trainer, PI-Net must preserve complete temporal
sequences because the actor carries a persistent per-entity belief.  This
trainer therefore collects complete episodes for each parallel environment and
passes the time-major batch to :class:`PIMAPPO` without timestep shuffling.

Phase-8 priorities are deliberate:

1. exact recurrent-policy replay;
2. finite/stable PPO updates;
3. reproducible checkpoints and diagnostics;
4. only then throughput optimisation.

The actor receives only decentralized structured observations through
``EntityTensorizer``.  The centralized critic keeps using the same fixed global
state representation as the MAPPO baseline.
"""
from __future__ import annotations

import argparse
import atexit
from datetime import datetime, timedelta
import json
import math
from pathlib import Path
import random
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from fofe_mmapppo.algorithms import PIParallelRolloutBuffer, PIMAPPO, PIMAPPOConfig
from fofe_mmapppo.envs import CooperativeUAVEnv
from fofe_mmapppo.gpu_monitor import GPUPhaseMonitor, GPUPhaseStats
from fofe_mmapppo.models import EntityTensorizer, FixedVectorizer, PIActorConfig
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


def status_text(processed: int, total: int, speed: float, sec_per_ep: float, elapsed: float) -> str:
    remaining = max(0, total - processed)
    eta_seconds = remaining / speed if speed > 0 else 0.0
    finish = datetime.now() + timedelta(seconds=eta_seconds)
    percent = 100.0 * processed / max(total, 1)
    return (
        f"PI-MAPPO  {processed}/{total}  {percent:6.2f}%  |  "
        f"{speed:.3f} ep/s  {sec_per_ep:.2f}s/ep  |  "
        f"elapsed {format_duration(elapsed)}  |  ETA {format_duration(eta_seconds)}  |  "
        f"finish≈{finish:%H:%M:%S}"
    )


def parameter_count(module: torch.nn.Module) -> int:
    return sum(int(p.numel()) for p in module.parameters() if p.requires_grad)


def start_tensorboard(log_dir: Path, host: str, requested_port: int):
    """Start a local TensorBoard for this run and return its process and URL."""
    port = requested_port
    for candidate in range(requested_port, requested_port + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            try:
                probe.bind((host, candidate))
            except OSError:
                continue
        port = candidate
        break
    else:
        raise RuntimeError(
            f"no free TensorBoard port in {requested_port}..{requested_port + 19}"
        )

    server_log = (log_dir.parent / "tensorboard_server.log").open("a", encoding="utf-8")
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "tensorboard.main",
            "--logdir",
            str(log_dir),
            "--host",
            host,
            "--port",
            str(port),
        ],
        stdin=subprocess.DEVNULL,
        stdout=server_log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )

    def stop_tensorboard() -> None:
        if process.poll() is None:
            process.terminate()
        server_log.close()

    atexit.register(stop_tensorboard)
    display_host = "localhost" if host in ("127.0.0.1", "0.0.0.0") else host
    return process, f"http://{display_host}:{port}"


def assert_finite_array(name: str, value: np.ndarray) -> None:
    if not np.isfinite(value).all():
        bad = int(np.size(value) - np.isfinite(value).sum())
        raise FloatingPointError(f"non-finite values in {name}: {bad}")


def assert_finite_buffer(buffer: PIParallelRolloutBuffer) -> None:
    data = buffer.as_arrays()
    for key, value in data.items():
        if key == "actions":
            if not ((value >= 0) & (value < 7)).all():
                raise ValueError("PI rollout contains invalid action index")
            continue
        assert_finite_array(key, value)


def save_checkpoint(
    path: Path,
    learner: PIMAPPO,
    fixed: FixedVectorizer,
    tensorizer: EntityTensorizer,
    episode: int,
    num_envs: int,
    scenario: str,
    max_steps: int,
    reward_profile: str,
) -> None:
    torch.save(
        {
            **learner.checkpoint(include_optimizers=True),
            "episode": int(episode),
            "parallel_num_envs": int(num_envs),
            "scenario": str(scenario),
            "max_steps": int(max_steps),
            "reward_profile": str(reward_profile),
            "fixed_vectorizer": {
                "world_size": fixed.world_size,
                "n_uavs": fixed.n_uavs,
                "n_targets": fixed.n_targets,
                "n_threats": fixed.n_threats,
            },
            "entity_tensorizer": {
                "world_size": tensorizer.world_size,
                "n_uavs": tensorizer.n_uavs,
                "n_targets": tensorizer.n_targets,
                "n_threats": tensorizer.n_threats,
                "max_age": tensorizer.max_age,
            },
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Sequence-preserving parallel PI-MAPPO trainer")
    parser.add_argument("--actor-batch-size", type=int, default=8,
                        help="number of independent actors per update group")
    parser.add_argument("--actor-saved-steps", type=int, default=0,
                        help="retain this many final steps per TBPTT chunk to reduce recomputation")
    parser.add_argument("--batch-actor-updates", action=argparse.BooleanOptionalAction, default=None,
                        help="batch independent actor updates (default: enabled on CUDA)")
    parser.add_argument("--episodes", type=int, default=64, help="final global episode number")
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--warm-start-actors-only",
        action="store_true",
        help="load actor weights from --resume while resetting critics and all optimizers",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cuda-memory-fraction", type=float, default=None,
                        help="optional PyTorch allocation ceiling as a fraction of GPU memory")
    parser.add_argument("--scenario", choices=("reference", "contested"), default="contested")
    parser.add_argument(
        "--reward-profile",
        choices=("paper", "task_aligned"),
        default="paper",
    )
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--ppo-epochs", type=int, default=1)
    parser.add_argument(
        "--sequence-env-minibatch-size",
        type=int,
        default=8,
        help="environment sequences per PPO minibatch; 8 is the throughput default",
    )
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--actor-learning-rate", type=float, default=None)
    parser.add_argument("--critic-learning-rate", type=float, default=None)
    parser.add_argument("--entropy-coef", type=float, default=None)
    parser.add_argument("--critic-hidden-dim", type=int, default=256)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--belief-dim", type=int, default=64)
    parser.add_argument("--context-dim", type=int, default=32)
    parser.add_argument("--relation-dim", type=int, default=64)
    parser.add_argument(
        "--replay-check-every",
        type=int,
        default=1,
        help="check unchanged-policy replay every N rollout batches; 0 disables",
    )
    parser.add_argument(
        "--gpu-monitor-interval",
        type=float,
        default=1.0,
        help="nvidia-smi sampling interval in seconds (minimum 0.2)",
    )
    parser.add_argument(
        "--disable-gpu-monitor",
        action="store_true",
        help="disable phased nvidia-smi/PyTorch CUDA telemetry",
    )
    parser.add_argument("--log-every", type=int, default=4)
    parser.add_argument("--save-every", type=int, default=32)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "pi_mappo_smoke")
    parser.add_argument("--no-tensorboard", action="store_true")
    parser.add_argument(
        "--launch-tensorboard",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="start a local TensorBoard while training (default: enabled)",
    )
    parser.add_argument("--tensorboard-host", default="127.0.0.1")
    parser.add_argument("--tensorboard-port", type=int, default=6006)
    parser.add_argument("--phase-progress", action="store_true",
                        help="print collection/update phase starts, including cold compilation")
    args = parser.parse_args()
    if args.cuda_memory_fraction is not None and not 0 < args.cuda_memory_fraction <= 1:
        parser.error("cuda-memory-fraction must be in (0, 1]")
    if args.warm_start_actors_only and args.resume is None:
        parser.error("--warm-start-actors-only requires --resume")
    for name in ("learning_rate", "actor_learning_rate", "critic_learning_rate"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"{name.replace('_', '-')} must be positive")
    if args.entropy_coef is not None and args.entropy_coef < 0:
        parser.error("entropy-coef must be non-negative")

    positive = {
        "episodes": args.episodes,
        "num-envs": args.num_envs,
        "max-steps": args.max_steps,
        "ppo-epochs": args.ppo_epochs,
        "sequence-env-minibatch-size": args.sequence_env_minibatch_size,
        "horizon": args.horizon,
        "belief-dim": args.belief_dim,
        "context-dim": args.context_dim,
        "relation-dim": args.relation_dim,
        "gpu-monitor-interval": args.gpu_monitor_interval,
    }
    for name, value in positive.items():
        if value <= 0:
            parser.error(f"{name} must be positive")
    if args.replay_check_every < 0:
        parser.error("replay-check-every must be non-negative")
    if not 1 <= args.tensorboard_port <= 65535:
        parser.error("tensorboard-port must be in [1, 65535]")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")

    device = choose_device(args.device)
    if args.cuda_memory_fraction is not None and torch.device(device).type == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.cuda_memory_fraction, device)
    probe = CooperativeUAVEnv(
        seed=args.seed,
        scenario=args.scenario,
        max_steps=args.max_steps,
        reward_profile=args.reward_profile,
    )
    fixed = FixedVectorizer(
        world_size=probe.world_size,
        n_uavs=8,
        n_targets=4,
        n_threats=3,
    )
    tensorizer = EntityTensorizer(
        world_size=probe.world_size,
        n_uavs=8,
        n_targets=4,
        n_threats=3,
        max_age=float(args.max_steps),
    )

    start_episode = 0
    checkpoint = None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        start_episode = int(checkpoint.get("episode", 0))
        checkpoint_scenario = str(checkpoint.get("scenario", "contested"))
        if checkpoint_scenario != args.scenario:
            parser.error(
                f"checkpoint scenario={checkpoint_scenario!r} does not match "
                f"--scenario={args.scenario!r}"
            )
        checkpoint_steps = int(checkpoint.get("max_steps", args.max_steps))
        if checkpoint_steps != args.max_steps:
            parser.error(
                f"checkpoint max_steps={checkpoint_steps} does not match --max-steps={args.max_steps}"
            )
        checkpoint_reward_profile = str(checkpoint.get("reward_profile", "paper"))
        if not args.warm_start_actors_only and checkpoint_reward_profile != args.reward_profile:
            parser.error(
                f"checkpoint reward_profile={checkpoint_reward_profile!r} does not match "
                f"--reward-profile={args.reward_profile!r}; use --warm-start-actors-only "
                "to intentionally change the objective"
            )
        cfg = PIMAPPOConfig(**checkpoint.get("config", {}))
        actor_cfg = PIActorConfig(**checkpoint.get("actor_config", {}))
        # Runtime batching/epoch count may be changed safely on resume; actor
        # architecture must remain exactly checkpoint-compatible.
        cfg.ppo_epochs = args.ppo_epochs
        cfg.sequence_env_minibatch_size = args.sequence_env_minibatch_size
        if args.learning_rate is not None:
            cfg.learning_rate = args.learning_rate
        if args.actor_learning_rate is not None:
            cfg.actor_learning_rate = args.actor_learning_rate
        if args.critic_learning_rate is not None:
            cfg.critic_learning_rate = args.critic_learning_rate
        if args.entropy_coef is not None:
            cfg.entropy_coef = args.entropy_coef
    else:
        learning_rate = 4e-5 if args.learning_rate is None else args.learning_rate
        entropy_coef = 0.01 if args.entropy_coef is None else args.entropy_coef
        cfg = PIMAPPOConfig(
            ppo_epochs=args.ppo_epochs,
            learning_rate=learning_rate,
            actor_learning_rate=args.actor_learning_rate,
            critic_learning_rate=args.critic_learning_rate,
            hidden_dim=args.critic_hidden_dim,
            entropy_coef=entropy_coef,
            sequence_env_minibatch_size=args.sequence_env_minibatch_size,
        )
        actor_cfg = PIActorConfig(
            horizon=args.horizon,
            belief_dim=args.belief_dim,
            context_dim=args.context_dim,
            relation_dim=args.relation_dim,
            max_age=float(args.max_steps),
        )

    cfg.batch_actor_updates = (torch.device(device).type == "cuda" if args.batch_actor_updates is None
                               else args.batch_actor_updates)
    cfg.actor_batch_size = args.actor_batch_size
    cfg.actor_saved_steps = args.actor_saved_steps

    print(f"actor_update={'batched' if cfg.batch_actor_updates else 'sequential'} actor_batch_size={cfg.actor_batch_size}")
    learner = PIMAPPO(
        fixed.state_dim,
        n_agents=8,
        action_dim=7,
        config=cfg,
        actor_config=actor_cfg,
        device=device,
    )
    if checkpoint is not None:
        if args.warm_start_actors_only:
            learner.actors.load_state_dict(checkpoint["actors"])
            status = "actors restored; critics and optimizers reset"
        else:
            restored_opt = learner.load_checkpoint(checkpoint, load_optimizers=True)
            status = "restored" if restored_opt else "restarted (checkpoint has no optimizer state)"
        print(f"resume={args.resume} episode={start_episode} optimizer={status}")

    gpu_monitor = GPUPhaseMonitor(device, interval_seconds=args.gpu_monitor_interval)
    if args.disable_gpu_monitor:
        gpu_monitor.enabled = False

    args.output.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output / "metrics.jsonl"
    writer = None
    tensorboard_url = None
    if not args.no_tensorboard:
        tensorboard_dir = args.output / "tensorboard"
        writer = SummaryWriter(log_dir=str(tensorboard_dir))
        atexit.register(writer.close)
        if args.launch_tensorboard:
            _, tensorboard_url = start_tensorboard(
                tensorboard_dir, args.tensorboard_host, args.tensorboard_port
            )

    actor_params = parameter_count(learner.actors[0])
    critic_params = parameter_count(learner.critics[0])
    print(
        f"device={device} scenario={args.scenario} num_envs={args.num_envs} "
        f"max_steps={args.max_steps} ppo_epochs={cfg.ppo_epochs} "
        f"seq_env_mb={cfg.sequence_env_minibatch_size} horizon={actor_cfg.horizon} "
        f"actor_params/agent={actor_params:,} critic_params/agent={critic_params:,}"
    )
    print(
        "trainer=full-episode sequence replay; timestep shuffle=disabled; "
        "critic=baseline centralized MLP"
    )
    print(
        f"gpu_monitor={'on' if gpu_monitor.enabled else 'off'} "
        f"interval={gpu_monitor.interval_seconds:.1f}s backend=nvidia-smi+torch.cuda"
    )
    if writer is not None:
        print(f"tensorboard={args.output / 'tensorboard'}")
        if tensorboard_url is not None:
            print(f"tensorboard_url={tensorboard_url}")

    recent_returns: list[float] = []
    processed = start_episode
    run_start = time.perf_counter()
    last_log_episode = processed
    batch_index = 0
    header = FixedStatusHeader()
    header.start("PI-MAPPO starting...")

    while processed < args.episodes:
        next_save = ((processed // args.save_every) + 1) * args.save_every
        batch_envs = min(
            args.num_envs,
            args.episodes - processed,
            max(1, next_save - processed),
        )
        episode_ids = np.arange(processed + 1, processed + batch_envs + 1, dtype=np.int64)
        envs = [
            CooperativeUAVEnv(
                seed=args.seed + int(ep) - 1,
                scenario=args.scenario,
                max_steps=args.max_steps,
                reward_profile=args.reward_profile,
            )
            for ep in episode_ids
        ]
        observations = []
        global_states = []
        for env, episode_id in zip(envs, episode_ids):
            obs, states = env.reset(seed=args.seed + int(episode_id) - 1)
            observations.append(obs)
            global_states.append(states)

        finished = np.zeros(batch_envs, dtype=bool)
        episode_returns = np.zeros((batch_envs, 8), dtype=np.float32)
        final_infos: list[dict | None] = [None] * batch_envs
        beliefs = learner.initial_belief_states(batch_envs)
        buffer = PIParallelRolloutBuffer(n_envs=batch_envs, n_agents=8)

        if args.phase_progress:
            print(f"phase=collect environments={batch_envs}", flush=True)
        collect_start = time.perf_counter()
        with gpu_monitor.measure("collect") as collect_gpu:
            while not bool(finished.all()):
                structured = tensorizer.parallel(observations, finished=finished)
                state_vectors = np.zeros((batch_envs, 8, fixed.state_dim), dtype=np.float32)
                for e in range(batch_envs):
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

                reward_batch = np.zeros((batch_envs, 8), dtype=np.float32)
                done_batch = np.ones((batch_envs, 8), dtype=np.float32)
                next_observations = list(observations)
                next_states = list(global_states)

                for e, env in enumerate(envs):
                    if finished[e]:
                        continue
                    next_obs, next_state, rewards, done, info = env.step(actions[e])
                    reward_batch[e] = np.fromiter(
                        (rewards[i] for i in range(8)), dtype=np.float32, count=8
                    )
                    done_batch[e] = np.asarray(
                        [1.0 if done or next_obs[i] is None else 0.0 for i in range(8)],
                        dtype=np.float32,
                    )
                    episode_returns[e] += reward_batch[e]
                    next_observations[e] = next_obs
                    next_states[e] = next_state
                    if done:
                        finished[e] = True
                        final_infos[e] = info

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

        collect_seconds = time.perf_counter() - collect_start
        collect_gpu_stats = collect_gpu.stats
        assert_finite_buffer(buffer)
        batch_index += 1

        replay_diag = None
        replay_seconds = 0.0
        replay_gpu_stats = GPUPhaseStats()
        if args.replay_check_every and batch_index % args.replay_check_every == 0:
            replay_start = time.perf_counter()
            with gpu_monitor.measure("replay") as replay_gpu:
                replay_diag = learner.replay_diagnostics(buffer)
            replay_seconds = time.perf_counter() - replay_start
            replay_gpu_stats = replay_gpu.stats
            if replay_diag["max_abs_ratio_error"] > 1e-4:
                raise RuntimeError(
                    "unchanged-policy replay drifted before PPO update: "
                    f"{replay_diag}"
                )

        if args.phase_progress:
            print(f"phase=update environments={batch_envs} steps={len(buffer)} "
                  f"collect_seconds={collect_seconds:.2f}", flush=True)
        update_start = time.perf_counter()
        with gpu_monitor.measure("update") as update_gpu:
            losses = learner.update_parallel(buffer)
        update_seconds = time.perf_counter() - update_start
        update_gpu_stats = update_gpu.stats
        for key, value in losses.items():
            if not math.isfinite(value):
                raise FloatingPointError(f"non-finite training metric {key}={value}")

        now = time.perf_counter()
        phase_metrics = {
            "replay_seconds": float(replay_seconds),
            **collect_gpu_stats.as_dict("collect"),
            **replay_gpu_stats.as_dict("replay"),
            **update_gpu_stats.as_dict("update"),
        }
        records = []
        for e, episode_id in enumerate(episode_ids.tolist()):
            info = final_infos[e]
            if info is None:
                raise RuntimeError(f"environment {e} ended without final info")
            mean_return = float(episode_returns[e].mean())
            recent_returns.append(mean_return)
            if len(recent_returns) > 100:
                recent_returns.pop(0)
            avg100 = float(np.mean(recent_returns))
            record = {
                "episode": int(episode_id),
                "scenario": args.scenario,
                "reward_profile": args.reward_profile,
                "mean_return": mean_return,
                "steps": int(info["step"]),
                "completion_ratio": float(info["completion_ratio"]),
                "survival_ratio": float(info["survival_ratio"]),
                "num_envs": int(batch_envs),
                "rollout_steps": int(len(buffer)),
                "collect_seconds": float(collect_seconds),
                "update_seconds": float(update_seconds),
                **phase_metrics,
                **losses,
            }
            if replay_diag is not None:
                record.update({f"replay_{k}": float(v) for k, v in replay_diag.items()})
            records.append(record)
            if writer is not None:
                writer.add_scalar("episode/reward", mean_return, episode_id)
                writer.add_scalar(
                    "episode/completion_ratio", record["completion_ratio"], episode_id
                )
                writer.add_scalar(
                    "episode/survival_ratio", record["survival_ratio"], episode_id
                )
                writer.add_scalar("episode/steps", record["steps"], episode_id)

        with metrics_path.open("a", encoding="utf-8") as f:
            f.writelines(json.dumps(record, ensure_ascii=False) + "\n" for record in records)

        processed += batch_envs
        elapsed = max(now - run_start, 1e-9)
        trained = processed - start_episode
        speed = trained / elapsed
        sec_per_ep = elapsed / max(trained, 1)
        header.update(status_text(processed, args.episodes, speed, sec_per_ep, elapsed))

        if writer is not None:
            writer.add_scalar("loss/actor", losses["actor_loss"], processed)
            writer.add_scalar("loss/critic", losses["critic_loss"], processed)

        if processed == args.episodes or processed - last_log_episode >= args.log_every:
            last = records[-1]
            batch_completion = float(np.mean([info["completion_ratio"] for info in final_infos]))
            batch_survival = float(np.mean([info["survival_ratio"] for info in final_infos]))
            replay_text = ""
            if replay_diag is not None:
                replay_text = f" replay_err={replay_diag['max_abs_ratio_error']:.2e}"
            gpu_text = ""
            if gpu_monitor.enabled:
                gpu_text = (
                    f" gpu[c/r/u]={collect_gpu_stats.util_avg_pct:.0f}/"
                    f"{replay_gpu_stats.util_avg_pct:.0f}/"
                    f"{update_gpu_stats.util_avg_pct:.0f}%"
                    f" vram_peak={max(collect_gpu_stats.vram_peak_mib, replay_gpu_stats.vram_peak_mib, update_gpu_stats.vram_peak_mib):.0f}MiB"
                    f" torch_peak={max(collect_gpu_stats.torch_peak_allocated_mib, replay_gpu_stats.torch_peak_allocated_mib, update_gpu_stats.torch_peak_allocated_mib):.0f}MiB"
                )
            header.log(
                f"ep={processed:5d} return={last['mean_return']:9.3f} "
                f"avg100={np.mean(recent_returns):9.3f} steps={last['steps']:3d} "
                f"completion={last['completion_ratio']:.3f} survival={last['survival_ratio']:.3f} "
                f"batch_c={batch_completion:.3f} batch_s={batch_survival:.3f} "
                f"actor={losses['actor_loss']:.4f} critic={losses['critic_loss']:.4f} "
                f"entropy={losses['entropy']:.4f} ratio={losses['ratio_mean']:.4f} "
                f"clip={losses['clip_fraction']:.3f} collect={collect_seconds:.1f}s "
                f"replay={replay_seconds:.1f}s update={update_seconds:.1f}s"
                f"{replay_text}{gpu_text}"
            )
            if writer is not None:
                writer.flush()
            if not header.enabled:
                header.plain_status_if_needed()
            last_log_episode = processed

        if processed % args.save_every == 0 or processed == args.episodes:
            save_checkpoint(
                args.output / f"checkpoint_{processed:06d}.pt",
                learner,
                fixed,
                tensorizer,
                processed,
                args.num_envs,
                args.scenario,
                args.max_steps,
                args.reward_profile,
            )

    header.close()
    if writer is not None:
        writer.flush()
        writer.close()


if __name__ == "__main__":
    main()
