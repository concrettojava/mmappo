"""Low-memory CUDA micro-profiler for one PIActor forward/backward.

This probe intentionally does *not* profile a whole rollout or recurrent PPO
update.  Full torch.profiler traces are enormous for PI-Net because the current
implementation launches many small operators; on a 16 GiB WSL instance that
can exhaust host RAM during profiler post-processing.

Instead we profile exactly one PIActor forward/backward after a few unprofiled
warm-up iterations.  Selected PIActor methods are wrapped dynamically with
``record_function`` ranges so the operator table can attribute aggregate device
time to the main architectural stages without modifying the model itself.

No Chrome trace is exported, no tensor shapes/stacks/memory timeline are
recorded, and only key-averaged text tables plus a small JSON summary are
written.
"""
from __future__ import annotations

import argparse
import functools
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from fofe_mmapppo.models.entity_tensorizer import ENTITY_DIM, EVIDENCE_META_DIM, SELF_DIM
from fofe_mmapppo.models.pi_actor import PIActor, PIActorConfig


RANGED_METHODS = (
    "update_belief",
    "_predict_current",
    "_correct_with_evidence",
    "_physical_future",
    "_action_future",
    "_information_future",
    "_interaction_lattice",
    "_type_apply",
)


def choose_device(name: str) -> str:
    if name != "auto":
        return name
    return "cuda" if torch.cuda.is_available() else "cpu"


def sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def install_record_ranges(actor: PIActor) -> None:
    """Wrap selected bound methods with profiler ranges on this instance only."""

    for name in RANGED_METHODS:
        original = getattr(actor, name)

        @functools.wraps(original)
        def wrapped(*args, __original=original, __name=name, **kwargs):
            with record_function(f"PIActor/{__name}"):
                return __original(*args, **kwargs)

        setattr(actor, name, wrapped)


def profiler_table(prof, preferred: tuple[str, ...], rows: int) -> tuple[str, str]:
    errors: list[str] = []
    for key in preferred:
        try:
            return key, prof.key_averages().table(sort_by=key, row_limit=rows)
        except Exception as exc:  # compatibility across PyTorch profiler versions
            errors.append(f"{key}: {exc}")
    return "unavailable", "failed to render profiler table\n" + "\n".join(errors) + "\n"


def make_inputs(batch_size: int, config: PIActorConfig, device: torch.device):
    """Create finite decentralized actor inputs with realistic normalized ranges."""

    gen = torch.Generator(device=device)
    gen.manual_seed(1701)

    self_features = torch.rand(
        batch_size, SELF_DIM, generator=gen, device=device, dtype=torch.float32
    )
    entity_features = torch.rand(
        batch_size,
        config.n_entities,
        ENTITY_DIM,
        generator=gen,
        device=device,
        dtype=torch.float32,
    )

    # Evidence presence is mixed so both correction and persistence paths are active.
    evidence_mask = (
        torch.rand(
            batch_size,
            config.n_entities,
            generator=gen,
            device=device,
            dtype=torch.float32,
        )
        > 0.25
    ).to(torch.float32)

    evidence_meta = torch.zeros(
        batch_size,
        config.n_entities,
        EVIDENCE_META_DIM,
        device=device,
        dtype=torch.float32,
    )
    # age / source-like metadata remain finite and normalized; exact semantics do
    # not affect the operator topology this micro-profile is intended to measure.
    evidence_meta[..., 0] = 0.1
    if EVIDENCE_META_DIM > 1:
        evidence_meta[..., 1] = 1.0

    return self_features, entity_features, evidence_mask, evidence_meta


def run_forward_backward(
    actor: PIActor,
    inputs,
    device: torch.device,
) -> float:
    actor.zero_grad(set_to_none=True)
    logits, _, _ = actor(*inputs, state=None)
    # Every policy logit participates.  This propagates gradients through the
    # task and information branches that feed the final policy logits.
    loss = logits.square().mean()
    loss.backward()
    sync_if_cuda(device)
    return float(loss.detach().cpu())


def main() -> None:
    parser = argparse.ArgumentParser(description="Low-memory PIActor CUDA micro-profiler")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--rows", type=int, default=40)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--belief-dim", type=int, default=64)
    parser.add_argument("--context-dim", type=int, default=32)
    parser.add_argument("--relation-dim", type=int, default=64)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "pi_actor_micro_profile",
    )
    args = parser.parse_args()

    if min(
        args.batch_size,
        args.rows,
        args.horizon,
        args.belief_dim,
        args.context_dim,
        args.relation_dim,
    ) <= 0:
        parser.error("batch/model sizes and rows must be positive")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")

    device = torch.device(choose_device(args.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda requested but CUDA is unavailable")

    torch.manual_seed(1701)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(1701)
    torch.set_float32_matmul_precision("high")

    config = PIActorConfig(
        n_hypotheses=3,
        horizon=args.horizon,
        belief_dim=args.belief_dim,
        context_dim=args.context_dim,
        relation_dim=args.relation_dim,
        max_age=200.0,
    )
    actor = PIActor(config).to(device)
    actor.train()
    inputs = make_inputs(args.batch_size, config, device)

    # Warm up CUDA kernels and allocator outside the profiler.  These iterations
    # deliberately use backward too, so the measured iteration is not paying
    # one-time autograd/kernel initialization costs.
    for _ in range(args.warmup):
        run_forward_backward(actor, inputs, device)

    install_record_ranges(actor)

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    sync_if_cuda(device)
    t0 = time.perf_counter()
    with profile(
        activities=activities,
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as prof:
        with record_function("PIActor/forward_backward"):
            loss = run_forward_backward(actor, inputs, device)
    sync_if_cuda(device)
    profiled_wall = time.perf_counter() - t0

    # Deliberately do not call export_chrome_trace(): the purpose of this probe
    # is operator/range attribution without multi-gigabyte event serialization.
    cpu_key, cpu_text = profiler_table(
        prof, ("self_cpu_time_total",), args.rows
    )
    device_key, device_text = profiler_table(
        prof,
        ("self_cuda_time_total", "self_device_time_total"),
        args.rows,
    )
    total_key, total_text = profiler_table(
        prof,
        ("cuda_time_total", "device_time_total"),
        args.rows,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "cpu_self_time.txt").write_text(cpu_text, encoding="utf-8")
    (args.output / "device_self_time.txt").write_text(device_text, encoding="utf-8")
    (args.output / "device_total_time.txt").write_text(total_text, encoding="utf-8")

    summary = {
        "device": str(device),
        "batch_size": args.batch_size,
        "warmup": args.warmup,
        "loss": loss,
        "profiled_wall_seconds": profiled_wall,
        "sort_keys": {
            "cpu": cpu_key,
            "device_self": device_key,
            "device_total": total_key,
        },
        "actor_config": {
            "n_hypotheses": 3,
            "horizon": args.horizon,
            "belief_dim": args.belief_dim,
            "context_dim": args.context_dim,
            "relation_dim": args.relation_dim,
        },
        "trace_exported": False,
    }
    if device.type == "cuda":
        summary["cuda_peak_allocated_mib"] = torch.cuda.max_memory_allocated(device) / (1024**2)
        summary["cuda_peak_reserved_mib"] = torch.cuda.max_memory_reserved(device) / (1024**2)

    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(json.dumps(summary, indent=2))
    print("\n=== device self time ===")
    print(device_text)
    print("\n=== device total time ===")
    print(total_text)
    print(f"\noutput={args.output}")
    print("No Chrome trace was generated by design.")


if __name__ == "__main__":
    main()
