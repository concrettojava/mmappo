from __future__ import annotations

import argparse
import copy
import time

import torch
from torch.func import functional_call, stack_module_state, vmap

from fofe_mmapppo.models import PIActor, PIActorConfig, PIBeliefState


STATE_FIELDS = (
    "latent",
    "mode_logits",
    "position",
    "yaw",
    "speed",
    "turn_limit",
    "age",
    "known",
    "alive",
    "info_context",
)


def _stack_states(states):
    return tuple(torch.stack([getattr(s, name) for s in states], dim=0) for name in STATE_FIELDS)


def _state_from_tensors(values):
    return PIBeliefState(**dict(zip(STATE_FIELDS, values)))


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time_iters(fn, iters, device):
    _sync(device)
    t0 = time.perf_counter()
    loss = None
    for _ in range(iters):
        loss = fn()
    _sync(device)
    return (time.perf_counter() - t0) / iters, loss


def main():
    p = argparse.ArgumentParser(description="Benchmark sequential vs vmapped execution of independent PI actors.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--agents", type=int, default=8)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--backward", action="store_true")
    p.add_argument("--compile-sequential", action="store_true")
    p.add_argument("--compile-vmap", action="store_true")
    args = p.parse_args()

    torch.manual_seed(7)
    device = torch.device(args.device)
    cfg = PIActorConfig()
    actors = [PIActor(cfg).to(device) for _ in range(args.agents)]

    # Keep the exact independent-actor semantics. stack_module_state creates an
    # actor dimension over otherwise independent parameters; vmap only changes
    # execution layout, not parameter sharing.
    params, buffers = stack_module_state(actors)
    base = copy.deepcopy(actors[0]).to("meta")

    A, B, E = args.agents, args.batch, cfg.n_entities
    sf = torch.randn(A, B, 14, device=device)
    ef = torch.randn(A, B, E, 20, device=device)
    em = torch.ones(A, B, E, device=device)
    meta = torch.zeros(A, B, E, 4, device=device)

    # Semantically benign values avoid meaningless invalid-scale behavior.
    sf[..., 2:5] = 0.0
    sf[..., 2] = 1.0
    sf[..., 6:8] = torch.rand(A, B, 2, device=device)
    sf[..., 8] = 0.0
    ef[..., 7] = 1.0
    ef[..., 9:11] = torch.rand(A, B, E, 2, device=device)
    ef[..., 11] = 0.0
    ef[..., 18] = 0.8
    ef[..., 19] = 0.9

    states = [actor.initial_state(B, device=device) for actor in actors]
    stacked_state = _stack_states(states)

    executors = actors
    if args.compile_sequential:
        executors = [
            torch.compile(actor, mode="default", fullgraph=False)
            for actor in actors
        ]

    def sequential_forward():
        logits = []
        next_states = []
        for i, executor in enumerate(executors):
            state_i = _state_from_tensors(tuple(v[i] for v in stacked_state))
            out, ns, _ = executor(sf[i], ef[i], em[i], meta[i], state_i)
            logits.append(out)
            next_states.append(ns)
        return torch.stack(logits, 0), _stack_states(next_states)

    def one_actor_forward(p_i, b_i, sf_i, ef_i, em_i, meta_i, *state_i):
        state = _state_from_tensors(state_i)
        logits, ns, _ = functional_call(
            base,
            (p_i, b_i),
            (sf_i, ef_i, em_i, meta_i, state),
        )
        return (logits,) + tuple(getattr(ns, name) for name in STATE_FIELDS)

    vmapped = vmap(
        one_actor_forward,
        in_dims=(0, 0, 0, 0, 0, 0) + (0,) * len(STATE_FIELDS),
    )

    def vmap_forward_raw():
        out = vmapped(params, buffers, sf, ef, em, meta, *stacked_state)
        return out[0], out[1:]

    vmap_forward = vmap_forward_raw
    if args.compile_vmap:
        vmap_forward = torch.compile(vmap_forward_raw, mode="default", fullgraph=False)

    def make_step(forward_fn, parameter_source):
        def run():
            if args.backward:
                for p0 in parameter_source:
                    if p0.grad is not None:
                        p0.grad = None
            logits, state_out = forward_fn()
            loss = logits.square().mean() + 1e-4 * state_out[0].square().mean()
            if args.backward:
                loss.backward()
            return loss
        return run

    seq_params = [p0 for actor in actors for p0 in actor.parameters()]
    stacked_params = list(params.values())
    seq_step = make_step(sequential_forward, seq_params)
    vmap_step = make_step(vmap_forward, stacked_params)

    # Check execution equivalence before performance timing.
    with torch.no_grad():
        seq_logits, seq_state = sequential_forward()
        vm_logits, vm_state = vmap_forward()
    max_logits = float((seq_logits - vm_logits).abs().max().cpu())
    max_state = max(float((a - b).abs().max().cpu()) for a, b in zip(seq_state, vm_state))

    # Warmups absorb torch.compile and CUDA cold start. Timings below are steady-state.
    for _ in range(args.warmup):
        seq_step()
    for _ in range(args.warmup):
        vmap_step()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    seq_s, seq_loss = _time_iters(seq_step, args.iters, device)
    seq_peak = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0.0

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    vmap_s, vmap_loss = _time_iters(vmap_step, args.iters, device)
    vmap_peak = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0.0

    print(
        f"device={device} agents={A} batch={B} backward={args.backward} "
        f"compile_sequential={args.compile_sequential} compile_vmap={args.compile_vmap}"
    )
    print(f"max_logits_abs_diff={max_logits:.6e}")
    print(f"max_state_abs_diff={max_state:.6e}")
    print(f"sequential={seq_s * 1000.0:.3f} ms/iter")
    print(f"vmap={vmap_s * 1000.0:.3f} ms/iter")
    print(f"speedup={seq_s / vmap_s:.3f}x")
    print(f"seq_loss={float(seq_loss.detach().cpu()):.8f}")
    print(f"vmap_loss={float(vmap_loss.detach().cpu()):.8f}")
    if device.type == "cuda":
        print(f"seq_peak_allocated={seq_peak:.1f} MiB")
        print(f"vmap_peak_allocated={vmap_peak:.1f} MiB")
        print(f"cuda_reserved={torch.cuda.max_memory_reserved(device) / 2**20:.1f} MiB")


if __name__ == "__main__":
    main()
