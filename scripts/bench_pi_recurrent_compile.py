"""Benchmark eager/default/reduce-overhead on a real PI recurrent unroll.

The single-step PIActor micro-benchmark showed ~10x speedup from torch.compile,
but the real PPO path replays a complete recurrent sequence and backpropagates
through the belief state.  This benchmark reproduces that pattern for one actor:

    T timesteps x B environments -> unroll_pi_actor -> PPO-like loss -> backward

`--mode all` runs eager, torch.compile(default), and
`torch.compile(reduce-overhead)` in isolated subprocesses.  Each compiled mode
gets its own TorchInductor cache so compile-warmup timings are comparable and do
not accidentally reuse artifacts from a previous run.

No profiler or Chrome trace is used; memory use stays small.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="PI recurrent torch.compile benchmark")
    p.add_argument("--mode", choices=("all", "eager", "default", "reduce-overhead"), default="all")
    p.add_argument("--device", default="cuda")
    p.add_argument("--timesteps", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--iters", type=int, default=3)
    p.add_argument("--seed", type=int, default=1701)
    p.add_argument("--output", type=Path, default=ROOT / "outputs" / "pi_recurrent_compile_bench")
    p.add_argument(
        "--reuse-cache",
        action="store_true",
        help="reuse the normal TorchInductor cache instead of a mode-specific cold cache",
    )
    p.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    return p


def child_command(args: argparse.Namespace, mode: str, result_path: Path) -> list[str]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child",
        "--mode", mode,
        "--device", args.device,
        "--timesteps", str(args.timesteps),
        "--batch-size", str(args.batch_size),
        "--warmup", str(args.warmup),
        "--iters", str(args.iters),
        "--seed", str(args.seed),
        "--output", str(result_path),
    ]
    if args.reuse_cache:
        cmd.append("--reuse-cache")
    return cmd


def run_parent(args: argparse.Namespace) -> None:
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    modes = ("eager", "default", "reduce-overhead")
    results = {}
    for mode in modes:
        result_path = output_dir / f"{mode}.json"
        env = os.environ.copy()
        if mode != "eager" and not args.reuse_cache:
            cache_dir = output_dir / f"inductor_cache_{mode}"
            cache_dir.mkdir(parents=True, exist_ok=True)
            env["TORCHINDUCTOR_CACHE_DIR"] = str(cache_dir.resolve())
        print(f"\n=== recurrent benchmark: {mode} ===", flush=True)
        t0 = time.perf_counter()
        completed = subprocess.run(child_command(args, mode, result_path), env=env)
        wall = time.perf_counter() - t0
        if completed.returncode != 0:
            raise SystemExit(f"mode {mode!r} failed with exit code {completed.returncode}")
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        payload["subprocess_wall_seconds"] = wall
        results[mode] = payload

    eager_ms = results["eager"]["steady_ms_per_iter"]
    for mode in ("default", "reduce-overhead"):
        results[mode]["speedup_vs_eager"] = eager_ms / results[mode]["steady_ms_per_iter"]

    summary = {
        "timesteps": args.timesteps,
        "batch_size": args.batch_size,
        "warmup": args.warmup,
        "iters": args.iters,
        "fresh_inductor_cache": not args.reuse_cache,
        "results": results,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n=== summary ===")
    print(json.dumps(summary, indent=2))
    print(f"output={summary_path}")


def run_child(args: argparse.Namespace) -> None:
    # Import torch only after the parent has had a chance to set an isolated
    # TORCHINDUCTOR_CACHE_DIR for this subprocess.
    sys.path.insert(0, str(ROOT / "src"))
    import torch
    from fofe_mmapppo.algorithms.pi_sequence import unroll_pi_actor
    from fofe_mmapppo.models import PIActor, PIActorConfig
    from fofe_mmapppo.models.entity_tensorizer import ENTITY_DIM, EVIDENCE_META_DIM, SELF_DIM

    if min(args.timesteps, args.batch_size, args.warmup, args.iters) <= 0:
        raise SystemExit("timesteps, batch-size, warmup and iters must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")

    T, B = args.timesteps, args.batch_size
    cfg = PIActorConfig()

    def sync() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def make_sequence():
        # Semantically valid EntityTensorizer-like values.  Small deterministic
        # time variation prevents the benchmark from being an identical-input
        # special case while preserving fixed shapes throughout the unroll.
        sf = torch.zeros(T, B, SELF_DIM, device=device)
        sf[..., 0] = 1.0
        sf[..., 2] = 1.0  # strike UAV subtype
        sf[..., 5] = 1.0  # alive
        base_x = torch.linspace(0.20, 0.24, T, device=device)[:, None]
        base_y = torch.linspace(0.10, 0.18, T, device=device)[:, None]
        sf[..., 6] = base_x
        sf[..., 7] = base_y
        sf[..., 8] = 0.10
        sf[..., 9] = base_x
        sf[..., 10] = base_y
        sf[..., 11] = 0.10

        ef = torch.zeros(T, B, cfg.n_entities, ENTITY_DIM, device=device)
        em = torch.ones(T, B, cfg.n_entities, device=device)
        meta = torch.zeros(T, B, cfg.n_entities, EVIDENCE_META_DIM, device=device)
        meta[..., 1] = 1.0  # direct/fresh source
        ef[..., 0] = 1.0  # present
        ef[..., 7] = 1.0  # alive

        # Stable entity categories: 7 teammates, 4 targets, 3 threats.
        ef[:, :, :7, 1] = 1.0
        ef[:, :, 7:11, 2] = 1.0
        ef[:, :, 11:, 3] = 1.0
        ef[:, :, :7, 4] = 1.0  # teammate subtype, only relevant to teammates

        ent_ids = torch.arange(cfg.n_entities, device=device, dtype=torch.float32)
        ef[..., 8] = (ent_ids / max(1, cfg.n_entities - 1))[None, None, :]
        x0 = 0.25 + 0.025 * ent_ids
        y0 = 0.30 + 0.015 * ent_ids
        drift = torch.linspace(0.0, 0.03, T, device=device)[:, None, None]
        ef[..., 9] = x0[None, None, :] + drift
        ef[..., 10] = y0[None, None, :] + 0.5 * drift
        ef[..., 11] = 0.05
        ef[..., 12] = ef[..., 9] - sf[..., 6, None]
        ef[..., 13] = ef[..., 10] - sf[..., 7, None]
        ef[..., 14] = 0.05
        ef[..., 15] = torch.sqrt(ef[..., 12].square() + ef[..., 13].square())
        ef[..., 16] = 0.0
        ef[:, :, :7, 18] = 40.0 / 50.0
        ef[:, :, :7, 19] = 36.0 / 40.0
        ef[:, :, 7:11, 18] = 8.0 / 50.0
        ef[:, :, 7:11, 19] = 10.0 / 40.0
        ef[:, :, 11:, 18] = 0.0
        ef[:, :, 11:, 19] = 0.0

        actions = (torch.arange(T, device=device)[:, None] + torch.arange(B, device=device)[None, :]) % 7
        active = torch.ones(T, B, device=device, dtype=torch.bool)
        # Fixed zero-mean-ish advantages for a PPO-shaped objective.
        adv = torch.linspace(-1.0, 1.0, T * B, device=device).reshape(T, B)
        adv = (adv - adv.mean()) / adv.std(unbiased=False).clamp_min(1e-8)
        return sf, ef, em, meta, actions.long(), active, adv

    inputs = make_sequence()

    def make_actor() -> PIActor:
        torch.manual_seed(args.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)
        return PIActor(cfg).to(device).train()

    def run_once(actor: PIActor, executor, *, keep_outputs: bool = False):
        actor.zero_grad(set_to_none=True)
        sf, ef, em, meta, actions, active, adv = inputs
        out = unroll_pi_actor(
            executor,
            sf,
            ef,
            em,
            meta,
            initial_state=actor.initial_state(B, device=device),
            actions=actions,
            active=active,
        )
        old_lp = out.log_probs.detach()
        ratio = torch.exp(out.log_probs - old_lp)
        clipped = torch.clamp(ratio, 0.9, 1.1)
        surr = torch.minimum(ratio * adv, clipped * adv)
        loss = -surr.mean() - 0.01 * out.entropy.mean()
        loss.backward()
        sync()
        if not keep_outputs:
            return float(loss.detach().cpu())
        grads = [p.grad.detach().clone() for p in actor.parameters() if p.grad is not None]
        state_tensors = [v.detach().clone() for v in out.final_state.__dict__.values()]
        return {
            "loss": float(loss.detach().cpu()),
            "logits": out.logits.detach().clone(),
            "states": state_tensors,
            "grads": grads,
        }

    actor = make_actor()
    if args.mode == "eager":
        executor = actor
        compile_create_seconds = 0.0
    else:
        t0 = time.perf_counter()
        executor = torch.compile(actor, mode=args.mode, fullgraph=False)
        compile_create_seconds = time.perf_counter() - t0

    # Warmup includes the first recurrent forward/backward and therefore captures
    # the expensive AOTAutograd/TorchInductor compilation that matters in training.
    sync()
    t0 = time.perf_counter()
    for _ in range(args.warmup):
        run_once(actor, executor)
    sync()
    warmup_seconds = time.perf_counter() - t0

    sync()
    t0 = time.perf_counter()
    losses = []
    for _ in range(args.iters):
        losses.append(run_once(actor, executor))
    sync()
    steady_seconds = time.perf_counter() - t0

    result = {
        "mode": args.mode,
        "device": str(device),
        "timesteps": T,
        "batch_size": B,
        "warmup": args.warmup,
        "iters": args.iters,
        "compile_create_seconds": compile_create_seconds,
        "warmup_seconds": warmup_seconds,
        "steady_seconds": steady_seconds,
        "steady_ms_per_iter": 1000.0 * steady_seconds / args.iters,
        "steady_sequences_per_second": args.iters / steady_seconds,
        "loss": losses[-1],
    }

    # Recurrent numerical check against a separate eager actor with identical
    # initialization.  This is intentionally outside the timing region.
    if args.mode != "eager":
        eager_actor = make_actor()
        eager_ref = run_once(eager_actor, eager_actor, keep_outputs=True)
        actor.zero_grad(set_to_none=True)
        compiled_ref = run_once(actor, executor, keep_outputs=True)
        result["max_logits_abs_diff"] = float((eager_ref["logits"] - compiled_ref["logits"]).abs().max().item())
        result["max_state_abs_diff"] = max(
            float((a - b).abs().max().item())
            for a, b in zip(eager_ref["states"], compiled_ref["states"])
        )
        result["max_grad_abs_diff"] = max(
            float((a - b).abs().max().item())
            for a, b in zip(eager_ref["grads"], compiled_ref["grads"])
        )

    if device.type == "cuda":
        result["cuda_peak_allocated_mib"] = torch.cuda.max_memory_allocated(device) / 2**20
        result["cuda_peak_reserved_mib"] = torch.cuda.max_memory_reserved(device) / 2**20

    out_path = args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


def main() -> None:
    args = build_parser().parse_args()
    if args.child:
        if args.mode == "all":
            raise SystemExit("--child requires a concrete mode")
        run_child(args)
    else:
        if args.mode == "all":
            run_parent(args)
        else:
            # A direct single-mode invocation is useful for quick experiments.
            # It runs in-process and therefore uses the current cache environment.
            args.child = True
            run_child(args)


if __name__ == "__main__":
    main()
