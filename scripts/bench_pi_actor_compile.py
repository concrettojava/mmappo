"""Benchmark eager vs torch.compile for one PIActor without changing model semantics.

The profiler showed many tiny CUDA kernels. This probe asks whether TorchInductor
can reduce Python/kernel-launch overhead for the existing PIActor before any
structural model changes.

Inputs are deliberately *semantically valid* EntityTensorizer-style features.
Using unconstrained Gaussian entity features is invalid here because fields such
as ``alive`` enter aggregation weights and can make a denominator clamp amplify
otherwise tiny floating-point differences into huge logits.

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
    """Create bounded, semantically valid decentralized actor inputs.

    Layout follows ``EntityTensorizer``. In particular, presence/alive/category
    fields are binary, pose fields are normalized, and speed/turn fields are in
    their documented normalized ranges.
    """
    c = actor.config
    g = torch.Generator(device=device)
    g.manual_seed(1701)

    self_features = torch.zeros(batch_size, SELF_DIM, device=device)
    self_features[:, 0] = 1.0  # present
    if batch_size > 1:
        self_features[:, 1] = torch.linspace(0.0, 1.0, batch_size, device=device)
    self_features[:, 2] = 1.0  # Stk subtype one-hot
    self_features[:, 5] = 1.0  # alive
    self_features[:, 6:8] = 0.1 + 0.8 * torch.rand(
        batch_size, 2, generator=g, device=device
    )
    self_features[:, 8] = 2.0 * torch.rand(batch_size, generator=g, device=device) - 1.0
    self_features[:, 9:11] = 0.2 * (
        2.0 * torch.rand(batch_size, 2, generator=g, device=device) - 1.0
    )
    self_features[:, 11] = 2.0 * torch.rand(batch_size, generator=g, device=device) - 1.0
    self_features[:, 12] = 0.2 * torch.rand(batch_size, generator=g, device=device)
    self_features[:, 13] = 2.0 * torch.rand(batch_size, generator=g, device=device) - 1.0

    entity_features = torch.zeros(
        batch_size, c.n_entities, ENTITY_DIM, device=device
    )
    entity_features[:, :, 0] = 1.0  # present
    # Stable entity categories: 7 teammates, 4 targets, 3 threats.
    entity_features[:, : c.n_teammates, 1] = 1.0
    entity_features[:, c.n_teammates : c.n_teammates + c.n_targets, 2] = 1.0
    entity_features[:, c.n_teammates + c.n_targets :, 3] = 1.0
    entity_features[:, : c.n_teammates, 4] = 1.0  # teammate Stk subtype
    entity_features[:, :, 7] = 1.0  # alive must be binary/non-negative

    ids = torch.arange(c.n_entities, device=device, dtype=torch.float32)
    entity_features[:, :, 8] = (ids / max(1, c.n_entities - 1))[None, :]
    entity_features[:, :, 9:11] = 0.05 + 0.90 * torch.rand(
        batch_size, c.n_entities, 2, generator=g, device=device
    )
    entity_features[:, :, 11] = (
        2.0 * torch.rand(batch_size, c.n_entities, generator=g, device=device) - 1.0
    )
    entity_features[:, :, 12:14] = 0.25 * (
        2.0 * torch.rand(batch_size, c.n_entities, 2, generator=g, device=device) - 1.0
    )
    entity_features[:, :, 14] = (
        2.0 * torch.rand(batch_size, c.n_entities, generator=g, device=device) - 1.0
    )
    entity_features[:, :, 15] = 0.25 * torch.rand(
        batch_size, c.n_entities, generator=g, device=device
    )
    entity_features[:, :, 16] = (
        2.0 * torch.rand(batch_size, c.n_entities, generator=g, device=device) - 1.0
    )

    # Physical fields by category, matching the tensorizer conventions.
    entity_features[:, : c.n_teammates, 18] = 40.0 / 50.0
    entity_features[:, : c.n_teammates, 19] = 36.0 / 40.0
    t0 = c.n_teammates
    t1 = t0 + c.n_targets
    entity_features[:, t0:t1, 18] = 8.0 / 50.0
    entity_features[:, t0:t1, 19] = 10.0 / 40.0
    entity_features[:, t1:, 17] = 0.05
    entity_features[:, t1:, 18:20] = 0.0

    evidence_mask = torch.ones(batch_size, c.n_entities, device=device)
    evidence_meta = torch.zeros(
        batch_size, c.n_entities, EVIDENCE_META_DIM, device=device
    )
    evidence_meta[:, :, 1] = 1.0  # direct/fresh source
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


def capture_grads(actor: PIActor) -> dict[str, torch.Tensor]:
    return {
        name: p.grad.detach().clone()
        for name, p in actor.named_parameters()
        if p.grad is not None
    }


def max_grad_diff(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> float:
    names = set(a) | set(b)
    if not names:
        return 0.0
    diff = 0.0
    for name in names:
        if name not in a or name not in b:
            return float("inf")
        diff = max(diff, float((a[name] - b[name]).abs().max().item()))
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
        max_logit_diff = float((eager_logits - compiled_logits).abs().max().item())
        max_logit_mag = float(eager_logits.abs().max().item())
        result["max_logits_abs_diff"] = max_logit_diff
        result["max_logits_abs_value"] = max_logit_mag
        result["max_logits_rel_diff"] = max_logit_diff / max(1.0, max_logit_mag)
        result["max_state_abs_diff"] = max_state_diff(eager_state, compiled_state)

        if backward:
            actor.zero_grad(set_to_none=True)
            eager_logits_g, _, _ = actor(*inputs)
            eager_logits_g.square().mean().backward()
            eager_grads = capture_grads(actor)
            actor.zero_grad(set_to_none=True)
            compiled_logits_g, _, _ = compiled(*inputs)
            compiled_logits_g.square().mean().backward()
            compiled_grads = capture_grads(actor)
            sync(device)
            result["max_grad_abs_diff"] = max_grad_diff(eager_grads, compiled_grads)

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
