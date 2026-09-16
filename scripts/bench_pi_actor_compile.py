"""Benchmark eager vs torch.compile for one PIActor without changing model semantics.

The profiler showed many tiny CUDA kernels.  This probe answers a narrower
question: can TorchInductor reduce Python/launch overhead for the existing
PIActor before we make structural changes?

No trace is exported and no profiler is enabled, so memory use stays small.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from fofe_mmapppo.models import PIActor, PIActorConfig
from fofe_mmapppo.models.entity_tensorizer import ENTITY_DIM, EVIDENCE_META_DIM, SELF_DIM


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def make_inputs(actor: PIActor, batch_size: int, device: torch.device):
    c = actor.config
    g = torch.Generator(device=device)
    g.manual_seed(1701)
    self_features = torch.randn(batch_size, SELF_DIM, generator=g, device=device)
    entity_features = torch.randn(
        batch_size, c.n_entities, ENTITY_DIM, generator=g, device=device
    )
    evidence_mask = torch.ones(batch_size, c.n_entities, device=device)
    evidence_meta = torch.zeros(
        batch_size, c.n_entities, EVIDENCE_META_DIM, device=device
    )
    state = actor.initial_state(batch_size, device=device)
    return self_features, entity_features, evidence_mask, evidence_meta, state


def one_step(module, inputs, backward: bool):
    logits, next_state, aux = module(*inputs)
    if backward:
        loss = logits.square().mean()
        loss.backward()
    # Touch outputs so the call cannot be trivially elided.
    checksum = logits.detach().sum() + next_state.latent.detach().sum() * 0.0
    return float(checksum.item()), logits.detach(), next_state


def warmup(module, actor: PIActor, inputs, n: int, backward: bool, device: torch.device):
    for _ in range(n):
        actor.zero_grad(set_to_none=True)
        one_step(module, inputs, backward)
    sync(device)


def timed(module, actor: PIActor, inputs, n: int, backward: bool, device: torch.device):
    sync(device)
    t0 = time.perf_counter()
    checksum = 0.0
    for _ in range(n):
        actor.zero_grad(set_to_none=True)
        value, _, _ = one_step(module, inputs, backward)
        checksum += value
    sync(device)
    elapsed = time.perf_counter() - t0
    return {
        "seconds": elapsed,
        "ms_per_iter": 1000.0 * elapsed / n,
        "iters_per_second": n / elapsed,
        "checksum": checksum,
    }


def max_state_diff(a, b) -> float:
    diff = 0.0
    for name in a.__dict__:
        x = getattr(a, name)
        y = getattr(b, name)
        diff = max(diff, float((x - y).abs().max().item()))
    return diff


def main() -> None:
    p = argparse.ArgumentParser(description="Eager vs torch.compile PIActor benchmark")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument(
        "--mode",
        choices=("default", "reduce-overhead", "max-autotune"),
        default="reduce-overhead",
    )
    p.add_argument("--forward-only", action="store_true")
    p.add_argument("--output", type=Path, default=ROOT / "outputs" / "pi_actor_compile_bench.json")
    args = p.parse_args()

    if args.batch_size <= 0 or args.warmup <= 0 or args.iters <= 0:
        p.error("batch-size, warmup and iters must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        p.error("CUDA requested but unavailable")

    torch.manual_seed(1701)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(1701)
    torch.set_float32_matmul_precision("high")

    actor = PIActor(PIActorConfig()).to(device)
    actor.train()
    inputs = make_inputs(actor, args.batch_size, device)
    backward = not args.forward_only

    warmup(actor, actor, inputs, args.warmup, backward, device)
    eager = timed(actor, actor, inputs, args.iters, backward, device)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    result = {
        "device": str(device),
        "batch_size": args.batch_size,
        "warmup": args.warmup,
        "iters": args.iters,
        "backward": backward,
        "mode": args.mode,
        "eager": eager,
    }

    try:
        compiled = torch.compile(actor, mode=args.mode, fullgraph=False)
        # First calls include compilation and are deliberately excluded.
        t0 = time.perf_counter()
        warmup(compiled, actor, inputs, args.warmup, backward, device)
        result["compile_warmup_seconds"] = time.perf_counter() - t0

        actor.zero_grad(set_to_none=True)
        with torch.no_grad():
            eager_logits, eager_state, _ = actor(*inputs)
            compiled_logits, compiled_state, _ = compiled(*inputs)
        sync(device)
        result["max_logits_abs_diff"] = float(
            (eager_logits - compiled_logits).abs().max().item()
        )
        result["max_state_abs_diff"] = max_state_diff(eager_state, compiled_state)

        compiled_stats = timed(compiled, actor, inputs, args.iters, backward, device)
        result["compiled"] = compiled_stats
        result["speedup"] = eager["ms_per_iter"] / compiled_stats["ms_per_iter"]
        if device.type == "cuda":
            result["cuda_peak_allocated_mib"] = torch.cuda.max_memory_allocated(device) / 2**20
            result["cuda_peak_reserved_mib"] = torch.cuda.max_memory_reserved(device) / 2**20
    except Exception as exc:
        result["compile_error"] = f"{type(exc).__name__}: {exc}"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
