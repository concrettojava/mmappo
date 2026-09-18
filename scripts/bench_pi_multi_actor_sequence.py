from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch
from torch.func import functional_call, stack_module_state, vmap

from fofe_mmapppo.models import PIActor, PIActorConfig, PIBeliefState
from fofe_mmapppo.algorithms.pi_sequence import blend_belief_state


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


def stack_states(states):
    return tuple(torch.stack([getattr(s, n) for s in states], 0) for n in STATE_FIELDS)


def make_state(values):
    return PIBeliefState(**dict(zip(STATE_FIELDS, values)))


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed(fn, iters, device):
    sync(device)
    t0 = time.perf_counter()
    out = None
    for _ in range(iters):
        out = fn()
    sync(device)
    return (time.perf_counter() - t0) / iters, out


def main():
    p = argparse.ArgumentParser(description="Benchmark 8 independent PI actors over a recurrent TBPTT chunk.")
    p.add_argument("--device", default="cuda")
    p.add_argument("--agents", type=int, default=8)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--compile-sequential", action="store_true")
    p.add_argument("--compile-vmap", action="store_true")
    args = p.parse_args()

    torch.manual_seed(17)
    device = torch.device(args.device)
    cfg = PIActorConfig()
    A, B, T, E = args.agents, args.batch, args.steps, cfg.n_entities

    actors = [PIActor(cfg).to(device) for _ in range(A)]
    params, buffers = stack_module_state(actors)
    base = copy.deepcopy(actors[0]).to("meta")

    sf = torch.randn(T, A, B, 14, device=device)
    ef = torch.randn(T, A, B, E, 20, device=device)
    em = torch.ones(T, A, B, E, device=device)
    meta = torch.zeros(T, A, B, E, 4, device=device)
    active = torch.ones(T, A, B, dtype=torch.bool, device=device)

    sf[..., 2:5] = 0.0
    sf[..., 2] = 1.0
    sf[..., 6:8] = torch.rand(T, A, B, 2, device=device)
    ef[..., 7] = 1.0
    ef[..., 9:11] = torch.rand(T, A, B, E, 2, device=device)
    ef[..., 18] = 0.8
    ef[..., 19] = 0.9

    states = [actor.initial_state(B, device=device) for actor in actors]
    stacked_init = stack_states(states)

    executors = actors
    if args.compile_sequential:
        executors = [torch.compile(a, mode="default", fullgraph=False) for a in actors]

    def sequential_forward():
        states_local = [make_state(tuple(v[i] for v in stacked_init)) for i in range(A)]
        logits_steps = []
        for t in range(T):
            row = []
            for i, executor in enumerate(executors):
                logits, cand, _ = executor(sf[t, i], ef[t, i], em[t, i], meta[t, i], states_local[i])
                states_local[i] = blend_belief_state(states_local[i], cand, active[t, i])
                row.append(logits)
            logits_steps.append(torch.stack(row, 0))
        return torch.stack(logits_steps, 0), stack_states(states_local)

    def one_actor_sequence(p_i, b_i, sf_i, ef_i, em_i, meta_i, active_i, *state_i):
        state = make_state(state_i)
        logits_steps = []
        for t in range(T):
            logits, cand, _ = functional_call(
                base,
                (p_i, b_i),
                (sf_i[t], ef_i[t], em_i[t], meta_i[t], state),
            )
            state = blend_belief_state(state, cand, active_i[t])
            logits_steps.append(logits)
        return (torch.stack(logits_steps, 0),) + tuple(getattr(state, n) for n in STATE_FIELDS)

    vmapped = vmap(
        one_actor_sequence,
        in_dims=(0, 0, 1, 1, 1, 1, 1) + (0,) * len(STATE_FIELDS),
        out_dims=(1,) + (0,) * len(STATE_FIELDS),
    )

    def vmap_forward_raw():
        out = vmapped(params, buffers, sf, ef, em, meta, active, *stacked_init)
        return out[0], out[1:]

    vmap_forward = vmap_forward_raw
    if args.compile_vmap:
        vmap_forward = torch.compile(vmap_forward_raw, mode="default", fullgraph=False)

    with torch.no_grad():
        seq_logits, seq_state = sequential_forward()
        vm_logits, vm_state = vmap_forward()
    max_logits = float((seq_logits - vm_logits).abs().max().cpu())
    max_state = max(float((a - b).abs().max().cpu()) for a, b in zip(seq_state, vm_state))

    seq_params = [p for a in actors for p in a.parameters()]
    vm_params = list(params.values())

    def seq_step():
        for p in seq_params:
            p.grad = None
        logits, state = sequential_forward()
        loss = logits.square().mean() + 1e-4 * state[0].square().mean()
        loss.backward()
        return loss

    def vm_step():
        for p in vm_params:
            p.grad = None
        logits, state = vmap_forward()
        loss = logits.square().mean() + 1e-4 * state[0].square().mean()
        loss.backward()
        return loss

    for _ in range(args.warmup):
        seq_step()
    for _ in range(args.warmup):
        vm_step()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    seq_s, seq_loss = timed(seq_step, args.iters, device)
    seq_peak = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0.0

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    vm_s, vm_loss = timed(vm_step, args.iters, device)
    vm_peak = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0.0

    print(
        f"device={device} agents={A} batch={B} steps={T} "
        f"compile_sequential={args.compile_sequential} compile_vmap={args.compile_vmap}"
    )
    print(f"max_logits_abs_diff={max_logits:.6e}")
    print(f"max_state_abs_diff={max_state:.6e}")
    print(f"sequential={seq_s:.4f} s/iter")
    print(f"vmap={vm_s:.4f} s/iter")
    print(f"speedup={seq_s / vm_s:.3f}x")
    print(f"seq_loss={float(seq_loss.detach().cpu()):.8f}")
    print(f"vmap_loss={float(vm_loss.detach().cpu()):.8f}")
    if device.type == "cuda":
        print(f"seq_peak_allocated={seq_peak:.1f} MiB")
        print(f"vmap_peak_allocated={vm_peak:.1f} MiB")


if __name__ == "__main__":
    main()
