"""Execution-equivalent PIActor with sparse type dispatch.

The original :class:`pi_actor.PIActor` applied each of the three type-specific
MLPs (teammate, target, threat) to every entity slot, stacked the three results,
and then selected one result with a one-hot type mask.  Because PI-Net uses
stable contiguous entity slots, two thirds of that work is provably unused.

This subclass preserves the exact parameters, checkpoint layout and forward
semantics while dispatching each contiguous slot range only to its matching MLP.
It is intentionally isolated from the reference implementation so equivalence
can be tested directly and the optimization can be reverted without changing
model semantics.
"""
from __future__ import annotations

import torch
from torch import nn

from .pi_actor import PIActor as ReferencePIActor


class PIActor(ReferencePIActor):
    """PIActor with mathematically equivalent sparse type-specific MLP routing."""

    def _type_apply(self, modules: nn.ModuleList, x: torch.Tensor) -> torch.Tensor:
        if len(modules) != 3:
            raise ValueError(f"PI-Net typed dispatch expects exactly 3 modules, got {len(modules)}")

        # Stable PI entity layout is contiguous by type:
        #   [0:n_teammates)                         teammates
        #   [n_teammates:n_teammates+n_targets)    targets
        #   [..n_entities)                         threats
        # The entity axis differs between current-belief tensors and the
        # action x entity x hypothesis x horizon interaction lattice.
        if x.dim() == 4:
            entity_dim = 1  # [B,E,K,D]
        elif x.dim() == 6:
            entity_dim = 2  # [B,A,E,K,H,D]
        else:
            raise ValueError(f"unsupported typed tensor rank {x.dim()}")

        c = self.config
        split_sizes = (c.n_teammates, c.n_targets, c.n_threats)
        if x.shape[entity_dim] != sum(split_sizes):
            raise ValueError(
                f"entity axis has size {x.shape[entity_dim]}, expected {sum(split_sizes)}"
            )

        typed_inputs = torch.split(x, split_sizes, dim=entity_dim)
        typed_outputs = [
            module(typed_x)
            for module, typed_x in zip(modules, typed_inputs)
        ]
        return torch.cat(typed_outputs, dim=entity_dim)
