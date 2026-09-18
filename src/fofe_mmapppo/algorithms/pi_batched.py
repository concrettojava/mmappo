"""Batched execution of independent PI actors; original modules own all weights."""
from __future__ import annotations

import copy
from dataclasses import fields
import torch
from torch.func import functional_call, vmap
from torch.distributions import Categorical
from torch.utils.checkpoint import checkpoint
from ..models import PIBeliefState
from .pi_sequence import blend_belief_state

STATE_FIELDS = tuple(f.name for f in fields(PIBeliefState))


def stack_beliefs(states):
    return tuple(torch.stack([getattr(s, key) for s in states]) for key in STATE_FIELDS)


class BatchedPIActors:
    def __init__(self, actors, compile_enabled=False, compile_mode="default", activation_checkpoint=True, saved_steps=0):
        self.actors = actors
        self.activation_checkpoint = activation_checkpoint
        if saved_steps < 0:
            raise ValueError("saved_steps must be non-negative")
        self.saved_steps = int(saved_steps)
        base = copy.deepcopy(actors[0]).to("meta")

        def one(params, buffers, sf, ef, em, meta, active, state):
            previous = PIBeliefState(**dict(zip(STATE_FIELDS, state)))
            logits, candidate, _ = functional_call(
                base, (params, buffers), (sf, ef, em, meta, previous))
            next_state = blend_belief_state(previous, candidate, active)
            return logits, tuple(getattr(next_state, k) for k in STATE_FIELDS)

        # Compile one step, not a Python-unrolled 50-step mega-graph.
        self.step = vmap(one)
        if compile_enabled:
            # Rollout E and replay minibatch B differ. Explicit static variants
            # avoid automatic symbolic-shape recompilation through vmap.
            self.step = torch.compile(
                self.step, mode=compile_mode, fullgraph=True, dynamic=False)

    def weights(self):
        # Differentiable stacking routes gradients to original parameters.
        params = [dict(a.named_parameters()) for a in self.actors]
        buffers = [dict(a.named_buffers()) for a in self.actors]
        return ({k: torch.stack([p[k] for p in params]) for k in params[0]},
                {k: torch.stack([b[k] for b in buffers]) for k in buffers[0]})

    def unroll(self, weights, inputs, active, state):
        logits = []
        for t in range(active.shape[0]):
            args = (*weights, *(x[t] for x in inputs), active[t], state)
            recompute = self.activation_checkpoint and t < active.shape[0] - self.saved_steps
            if recompute and torch.is_grad_enabled():
                # Recompute step activations on backward, preserving the full
                # recurrent gradient horizon without storing expanded vmap weights.
                row, state = checkpoint(self.step, *args, use_reentrant=False,
                                        preserve_rng_state=False)
            else:
                # Retain the final K steps when memory permits. Backward frees
                # these graphs first; the recurrent gradient span is unchanged.
                row, state = self.step(*args)
            logits.append(torch.where(active[t, ..., None], row, torch.zeros_like(row)))
        return torch.stack(logits), state


def update_batched(learner, data, metrics, chunk_length, backend, agent_start, agent_end):
    """Preserve per-agent objectives, shuffles, Adam states and gradient clipping."""
    cfg, device = learner.config, learner.device
    data = {k: v[:, :, agent_start:agent_end] for k, v in data.items()}
    actors = learner.actors[agent_start:agent_end]
    critics = learner.critics[agent_start:agent_end]
    actor_optimizers = learner.actor_optimizers[agent_start:agent_end]
    critic_optimizers = learner.critic_optimizers[agent_start:agent_end]
    T, E, A = data["active"].shape
    active = data["active"].permute(0, 2, 1) > 0.5
    inputs = [data[k].transpose(1, 2) for k in (
        "self_features", "entity_features", "evidence_mask", "evidence_meta")]
    advantages = data["advantages"].transpose(1, 2).clone()
    pools, orders = [], []
    # Legacy agent/epoch RNG order makes controlled comparisons reproducible.
    for a in range(A):
        pool = torch.nonzero(active[:, a].any(0), as_tuple=False).flatten()
        pools.append(pool)
        valid = advantages[:, a][active[:, a]]
        if valid.numel() > 1:
            advantages[:, a] = torch.where(active[:, a],
                (advantages[:, a] - valid.mean()) / valid.std(unbiased=False).clamp_min(1e-8), 0)
        orders.append([pool[torch.randperm(pool.numel(), device=device)]
                       for _ in range(cfg.ppo_epochs)] if pool.numel() else [])
    if not any(p.numel() for p in pools):
        return
    with torch.no_grad():
        state = stack_beliefs([a.initial_state(E, device=device) for a in backend.actors])
        boundaries = {0: state}
        weights = backend.weights()
        for t in range(0, T - chunk_length, chunk_length):
            _, state = backend.unroll(weights, [x[t:t+chunk_length] for x in inputs],
                                      active[t:t+chunk_length], state)
            boundaries[t + chunk_length] = tuple(x.detach() for x in state)

    B = min(cfg.sequence_env_minibatch_size, max(p.numel() for p in pools))
    agents = torch.arange(A, device=device)[:, None]
    for epoch in range(cfg.ppo_epochs):
        for start in range(0, max(p.numel() for p in pools), B):
            idx = torch.zeros(A, B, dtype=torch.long, device=device)
            valid_env = torch.zeros(A, B, dtype=torch.bool, device=device)
            for a in range(A):
                if orders[a]:
                    chosen = orders[a][epoch][start:start+B]
                    idx[a, :chosen.numel()] = chosen
                    valid_env[a, :chosen.numel()] = True
            mask_all = active[:, agents, idx] & valid_env[None]
            counts = mask_all.sum((0, 2))
            denominator = counts.clamp_min(1)
            for opt in actor_optimizers:
                opt.zero_grad(set_to_none=True)
            sums = torch.zeros(4, A, device=device, dtype=advantages.dtype)
            for t0 in range(0, T, chunk_length):
                t1 = min(T, t0 + chunk_length)
                mask = mask_all[t0:t1]
                state = tuple(x[agents, idx] for x in boundaries[t0])
                logits, _ = backend.unroll(backend.weights(),
                    [x[t0:t1, agents, idx] for x in inputs], mask, state)
                dist = Categorical(logits=logits)
                actions = data["actions"][t0:t1].transpose(1, 2)[:, agents, idx].long()
                lp = dist.log_prob(actions)
                ent = dist.entropy()
                old = data["log_probs"][t0:t1].transpose(1, 2)[:, agents, idx]
                adv = torch.where(mask, advantages[t0:t1, agents, idx], 0)
                ratio = torch.where(mask, lp - old, 0).exp()
                objective = -torch.minimum(ratio * adv,
                    ratio.clamp(1-cfg.clip_epsilon, 1+cfg.clip_epsilon) * adv)
                loss = ((objective - cfg.entropy_coef * ent) * mask).sum((0, 2)) / denominator
                loss.sum().backward()
                sums[0] += loss.detach()
                sums[1] += (ent.detach() * mask).sum((0, 2)) / denominator
                sums[2] += (ratio.detach() * mask).sum((0, 2)) / denominator
                sums[3] += (((ratio.detach()-1).abs() > cfg.clip_epsilon) * mask).sum((0, 2)) / denominator
            for a in range(A):
                if not bool(counts[a]):
                    continue  # Never advance Adam moments for absent agents.
                torch.nn.utils.clip_grad_norm_(actors[a].parameters(), cfg.max_grad_norm)
                actor_optimizers[a].step()
                mask = mask_all[:, a]
                states = data["states"][:, idx[a], a][mask]
                returns = data["returns"][:, idx[a], a][mask]
                critic_loss = cfg.value_coef * (critics[a](states) - returns).square().mean()
                opt = critic_optimizers[a]
                opt.zero_grad(set_to_none=True)
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(critics[a].parameters(), cfg.max_grad_norm)
                opt.step()
                for key, value in zip(("actor_loss", "entropy", "ratio_mean", "clip_fraction"), sums[:, a]):
                    metrics[key].append(float(value.cpu()))
                metrics["critic_loss"].append(float(critic_loss.detach().cpu()))
