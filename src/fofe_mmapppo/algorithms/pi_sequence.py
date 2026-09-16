"""Sequence-safe unrolling utilities for the stateful PI-Net actor.

The feed-forward MAPPO baseline can shuffle individual timesteps.  PI-Net
cannot: its persistent entity belief is part of the policy state.  This module
provides the minimal temporal primitive needed before recurrent PPO is added.

The key invariant is that replaying an unchanged actor over the exact same
observation sequence and initial belief state must reproduce rollout logits and
log-probabilities.  PPO policy ratios are meaningless if this invariant fails.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch.distributions import Categorical

from ..models import PIActor, PIBeliefState


@dataclass
class PISequenceOutput:
    """Outputs from a temporally ordered PI actor replay."""

    logits: torch.Tensor                 # [T,B,7]
    final_state: PIBeliefState
    log_probs: Optional[torch.Tensor] = None  # [T,B]
    entropy: Optional[torch.Tensor] = None    # [T,B]


def blend_belief_state(
    previous: PIBeliefState,
    candidate: PIBeliefState,
    update_mask: torch.Tensor,
) -> PIBeliefState:
    """Select candidate rows where ``update_mask`` is true.

    In synchronized parallel rollouts, some environments or agents terminate
    before others.  Padding timesteps must not continue evolving their recurrent
    belief state.  This helper makes that rule explicit for every state tensor.
    """
    mask = update_mask.to(device=previous.latent.device, dtype=torch.bool).reshape(-1)
    if mask.shape[0] != previous.latent.shape[0]:
        raise ValueError(
            f"update_mask batch {mask.shape[0]} does not match state batch "
            f"{previous.latent.shape[0]}"
        )

    values = {}
    for name, old_value in previous.__dict__.items():
        new_value = getattr(candidate, name)
        if old_value.shape != new_value.shape:
            raise ValueError(
                f"belief field {name!r} shape changed from {old_value.shape} "
                f"to {new_value.shape}"
            )
        view = mask.reshape(mask.shape[0], *([1] * (old_value.dim() - 1)))
        values[name] = torch.where(view, new_value, old_value)
    return PIBeliefState(**values)


def unroll_pi_actor(
    actor: PIActor,
    self_features: torch.Tensor,
    entity_features: torch.Tensor,
    evidence_mask: torch.Tensor,
    evidence_meta: torch.Tensor,
    *,
    initial_state: PIBeliefState | None = None,
    actions: torch.Tensor | None = None,
    active: torch.Tensor | None = None,
    reset_before: torch.Tensor | None = None,
) -> PISequenceOutput:
    """Replay PI-Net over a contiguous time-major sequence.

    Parameters use the following shapes:

    - ``self_features``: ``[T,B,14]``
    - ``entity_features``: ``[T,B,14,20]``
    - ``evidence_mask``: ``[T,B,14]``
    - ``evidence_meta``: ``[T,B,14,4]``
    - ``actions`` (optional): ``[T,B]``
    - ``active`` (optional): ``[T,B]``; inactive rows are padding and do not
      mutate belief state
    - ``reset_before`` (optional): ``[T,B]``; reset recurrent state immediately
      before processing that observation, e.g. at episode boundaries

    The function intentionally does not return the large per-step PI lattices.
    They remain part of the autograd graph required to produce logits but are
    not accumulated as Python references, reducing peak memory during sequence
    PPO updates.
    """
    if self_features.dim() != 3:
        raise ValueError(f"self_features must be [T,B,F], got {self_features.shape}")
    T, B, _ = self_features.shape
    if entity_features.shape[:2] != (T, B):
        raise ValueError("entity_features time/batch dimensions do not match")
    if evidence_mask.shape[:2] != (T, B):
        raise ValueError("evidence_mask time/batch dimensions do not match")
    if evidence_meta.shape[:2] != (T, B):
        raise ValueError("evidence_meta time/batch dimensions do not match")
    if actions is not None and tuple(actions.shape) != (T, B):
        raise ValueError(f"actions must be [T,B], got {actions.shape}")

    device = self_features.device
    if active is None:
        active = torch.ones(T, B, device=device, dtype=torch.bool)
    else:
        if tuple(active.shape) != (T, B):
            raise ValueError(f"active must be [T,B], got {active.shape}")
        active = active.to(device=device, dtype=torch.bool)

    if reset_before is None:
        reset_before = torch.zeros(T, B, device=device, dtype=torch.bool)
    else:
        if tuple(reset_before.shape) != (T, B):
            raise ValueError(f"reset_before must be [T,B], got {reset_before.shape}")
        reset_before = reset_before.to(device=device, dtype=torch.bool)

    state = initial_state
    if state is None:
        state = actor.initial_state(B, device=device, dtype=self_features.dtype)

    logits_steps = []
    log_prob_steps = []
    entropy_steps = []

    for t in range(T):
        if bool(reset_before[t].any()):
            state = state.reset_where(reset_before[t])

        logits_t, candidate_state, _ = actor(
            self_features[t],
            entity_features[t],
            evidence_mask[t],
            evidence_meta[t],
            state,
        )
        state = blend_belief_state(state, candidate_state, active[t])

        # Padding rows are not part of the PPO objective.  Zeroing their logits
        # also makes accidental downstream use conspicuous/deterministic.
        logits_t = torch.where(active[t, :, None], logits_t, torch.zeros_like(logits_t))
        logits_steps.append(logits_t)

        if actions is not None:
            dist = Categorical(logits=logits_t)
            lp = dist.log_prob(actions[t].long())
            ent = dist.entropy()
            lp = torch.where(active[t], lp, torch.zeros_like(lp))
            ent = torch.where(active[t], ent, torch.zeros_like(ent))
            log_prob_steps.append(lp)
            entropy_steps.append(ent)

    return PISequenceOutput(
        logits=torch.stack(logits_steps, dim=0),
        final_state=state,
        log_probs=torch.stack(log_prob_steps, dim=0) if log_prob_steps else None,
        entropy=torch.stack(entropy_steps, dim=0) if entropy_steps else None,
    )
