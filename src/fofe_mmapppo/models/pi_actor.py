"""PI-Net V1: persistent belief + physical/information interaction lattice.

This module is deliberately independent from the PPO trainer.  The actor is a
stateful function with explicit state input/output, which makes the temporal
semantics testable before recurrent-PPO integration.

Execution-time inputs contain only decentralized local evidence produced by
``EntityTensorizer``.  No jammer coordinates/intensity, global communication
graph, hidden target truth or future simulator state enters this module.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict

import torch
from torch import nn
from torch.distributions import Categorical
import torch.nn.functional as F

from .entity_tensorizer import ENTITY_DIM, EVIDENCE_META_DIM, SELF_DIM


def _linear(in_dim: int, out_dim: int, gain: float = 1.0) -> nn.Linear:
    layer = nn.Linear(in_dim, out_dim)
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0.0)
    return layer


def _mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        _linear(in_dim, hidden_dim, gain=math.sqrt(2.0)),
        nn.SiLU(),
        _linear(hidden_dim, out_dim, gain=1.0),
    )


def _wrap_angle(x: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(x), torch.cos(x))


@dataclass(frozen=True)
class PIActorConfig:
    n_teammates: int = 7
    n_targets: int = 4
    n_threats: int = 3
    n_hypotheses: int = 3
    horizon: int = 8
    belief_dim: int = 64
    context_dim: int = 32
    relation_dim: int = 64
    reactive_dim: int = 256
    world_size: float = 4000.0
    max_age: float = 200.0
    action_decay: float = 0.70
    uncertainty_temperature: float = 2.0

    @property
    def n_entities(self) -> int:
        return self.n_teammates + self.n_targets + self.n_threats


@dataclass
class PIBeliefState:
    """Explicit recurrent state for one PI actor across a batch of environments."""

    latent: torch.Tensor
    mode_logits: torch.Tensor
    position: torch.Tensor
    yaw: torch.Tensor
    speed: torch.Tensor
    turn_limit: torch.Tensor
    age: torch.Tensor
    known: torch.Tensor
    alive: torch.Tensor
    info_context: torch.Tensor

    def detach(self) -> "PIBeliefState":
        return PIBeliefState(**{k: v.detach() for k, v in self.__dict__.items()})

    def reset_where(self, reset_mask: torch.Tensor) -> "PIBeliefState":
        """Zero state rows where ``reset_mask`` is true (episode boundaries)."""
        mask = reset_mask.to(dtype=torch.bool, device=self.latent.device).reshape(-1)
        if not bool(mask.any()):
            return self
        values = {}
        for name, value in self.__dict__.items():
            cloned = value.clone()
            cloned[mask] = 0.0
            values[name] = cloned
        return PIBeliefState(**values)


class PIActor(nn.Module):
    """Physical-information lattice actor with persistent entity beliefs."""

    ACTION_VALUES = (-1.0, -0.4, -0.1, 0.0, 0.1, 0.4, 1.0)

    def __init__(self, config: PIActorConfig | None = None):
        super().__init__()
        self.config = config or PIActorConfig()
        c = self.config
        if c.n_entities != 14:
            raise ValueError("PI-Net V1 currently expects 7 teammates, 4 targets and 3 threats")
        if c.n_hypotheses < 2:
            raise ValueError("PI-Net requires at least two hypotheses")
        if c.horizon <= 0:
            raise ValueError("horizon must be positive")

        D, C, K = c.belief_dim, c.context_dim, c.n_hypotheses
        self.evidence_encoder = _mlp(ENTITY_DIM, D, D)
        self.mode_embedding = nn.Parameter(torch.empty(K, D))
        nn.init.normal_(self.mode_embedding, mean=0.0, std=0.05)

        dyn_in = D + D + D + 1
        self.context_to_belief = _linear(C, D, gain=1.0)
        self.dynamics = nn.ModuleList([_mlp(dyn_in, D, D) for _ in range(3)])
        self.turn_heads = nn.ModuleList([_mlp(dyn_in, D, 1) for _ in range(3)])

        correction_in = D + D + 1 + EVIDENCE_META_DIM
        self.evidence_scores = nn.ModuleList(
            [_mlp(correction_in, D, 1) for _ in range(3)]
        )
        self.evidence_gates = nn.ModuleList(
            [_mlp(correction_in, D, 1) for _ in range(3)]
        )

        context_input_dim = SELF_DIM + 3
        self.context_delta = _mlp(context_input_dim + C, C, C)
        self.context_gate = _mlp(context_input_dim + C, C, C)

        info_in = 3 + 2 + 3 + 1 + C
        self.observe_predictor = _mlp(info_in, C, 1)
        self.communicate_predictor = _mlp(info_in, C, 1)

        # dx,dy,distance + heading(sin/cos) + mode probability + age +
        # uncertainty + refresh + category(3) + known + alive + horizon = 15.
        relation_in = 15
        self.relation_encoders = nn.ModuleList(
            [_mlp(relation_in, c.relation_dim, c.relation_dim) for _ in range(3)]
        )
        self.task_heads = nn.ModuleList(
            [_mlp(c.relation_dim, c.relation_dim, 1) for _ in range(3)]
        )
        self.info_heads = nn.ModuleList(
            [_mlp(c.relation_dim, c.relation_dim, 1) for _ in range(3)]
        )

        gate_in = C + 4
        self.cognitive_gate = _mlp(gate_in, C, 1)
        self.self_action_head = nn.Sequential(
            _linear(SELF_DIM, D, gain=math.sqrt(2.0)),
            nn.Tanh(),
            _linear(D, len(self.ACTION_VALUES), gain=0.01),
        )

        # Capacity-first V2: current decentralized evidence has a direct route
        # to action logits. PI reasoning is a residual augmentation instead of
        # the only entity-to-action path.
        reactive_in = SELF_DIM + c.n_entities * (ENTITY_DIM + 1 + EVIDENCE_META_DIM)
        # Match the proven MAPPO baseline's two-hidden-layer depth so the
        # direct path is not a weaker bottleneck than the baseline it is meant
        # to recover.  The structured input is richer, but remains entirely
        # decentralized.
        tanh_gain = nn.init.calculate_gain("tanh")
        self.current_evidence_head = nn.Sequential(
            _linear(reactive_in, c.reactive_dim, gain=tanh_gain),
            nn.Tanh(),
            _linear(c.reactive_dim, c.reactive_dim, gain=tanh_gain),
            nn.Tanh(),
            _linear(c.reactive_dim, len(self.ACTION_VALUES), gain=0.01),
        )
        # Unknown targets have no invented position; their absence is encoded
        # explicitly so a search policy can learn before first discovery.
        self.search_head = nn.Sequential(
            _linear(SELF_DIM + 8, D, gain=math.sqrt(2.0)),
            nn.SiLU(),
            _linear(D, len(self.ACTION_VALUES), gain=0.01),
        )
        # Start the difficult counterfactual PI path as a small learnable
        # residual while basic reactive control is being acquired.
        # ReZero-style residual gate: start the expensive PI branch at exactly
        # zero contribution, with unit gradient through the scalar gate.  This
        # guarantees that untrained counterfactual reasoning cannot corrupt the
        # baseline-capacity direct policy at initialization; PI reasoning is
        # admitted only as PPO finds it useful.
        self._pi_residual_gate = nn.Parameter(torch.tensor(0.0))
        self._search_residual_logit = nn.Parameter(torch.tensor(-0.8472979))

        self._dispersion_gain = nn.Parameter(torch.tensor(-2.0))
        self._age_gain = nn.Parameter(torch.tensor(-3.0))
        self._reset_uncertainty_logit = nn.Parameter(torch.tensor(-3.0))

        category = torch.zeros(c.n_entities, 3, dtype=torch.float32)
        category[: c.n_teammates, 0] = 1.0
        category[c.n_teammates : c.n_teammates + c.n_targets, 1] = 1.0
        category[c.n_teammates + c.n_targets :, 2] = 1.0
        self.register_buffer("entity_category", category, persistent=False)
        self.register_buffer(
            "action_values", torch.tensor(self.ACTION_VALUES, dtype=torch.float32),
            persistent=False,
        )

    def initial_state(
        self,
        batch_size: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> PIBeliefState:
        c = self.config
        if device is None:
            device = next(self.parameters()).device
        shape = (batch_size, c.n_entities, c.n_hypotheses)
        return PIBeliefState(
            latent=torch.zeros(*shape, c.belief_dim, device=device, dtype=dtype),
            mode_logits=torch.zeros(*shape, device=device, dtype=dtype),
            position=torch.zeros(*shape, 2, device=device, dtype=dtype),
            yaw=torch.zeros(*shape, 1, device=device, dtype=dtype),
            speed=torch.zeros(*shape, 1, device=device, dtype=dtype),
            turn_limit=torch.zeros(*shape, 1, device=device, dtype=dtype),
            age=torch.zeros(batch_size, c.n_entities, 1, device=device, dtype=dtype),
            known=torch.zeros(batch_size, c.n_entities, 1, device=device, dtype=dtype),
            alive=torch.zeros(batch_size, c.n_entities, 1, device=device, dtype=dtype),
            info_context=torch.zeros(batch_size, c.context_dim, device=device, dtype=dtype),
        )

    def _type_apply(self, modules: nn.ModuleList, x: torch.Tensor) -> torch.Tensor:
        outputs = torch.stack([module(x) for module in modules], dim=-2)
        if x.dim() == 4:
            cat = self.entity_category[None, :, None, :, None]
        elif x.dim() == 6:
            cat = self.entity_category[None, None, :, None, None, :, None]
        else:
            raise ValueError(f"unsupported typed tensor rank {x.dim()}")
        return (outputs * cat).sum(dim=-2)

    def _update_context(
        self,
        self_features: torch.Tensor,
        evidence_mask: torch.Tensor,
        state: PIBeliefState,
    ) -> torch.Tensor:
        known = state.known.squeeze(-1)
        visible_fraction = evidence_mask.mean(dim=1, keepdim=True)
        known_fraction = known.mean(dim=1, keepdim=True)
        age_norm = (state.age.squeeze(-1) / self.config.max_age).clamp(0.0, 1.0)
        mean_age = (age_norm * known).sum(dim=1, keepdim=True) / known.sum(
            dim=1, keepdim=True
        ).clamp_min(1.0)
        summary = torch.cat([self_features, visible_fraction, known_fraction, mean_age], dim=-1)
        x = torch.cat([summary, state.info_context], dim=-1)
        delta = torch.tanh(self.context_delta(x))
        gate = torch.sigmoid(self.context_gate(x))
        return state.info_context + gate * delta

    def _predict_current(
        self,
        state: PIBeliefState,
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        c = self.config
        B, E, K, _ = state.latent.shape
        mode = self.mode_embedding[None, None, :, :].expand(B, E, K, -1)
        ctx = self.context_to_belief(context)[:, None, None, :].expand(B, E, K, -1)
        age_norm = (state.age / c.max_age).clamp(0.0, 1.0)
        age = age_norm[:, :, None, :].expand(B, E, K, 1)
        dyn_input = torch.cat([state.latent, mode, ctx, age], dim=-1)
        delta = self._type_apply(self.dynamics, dyn_input)
        prior_latent = state.latent + 0.20 * torch.tanh(delta)
        turn_unit = torch.tanh(self._type_apply(self.turn_heads, dyn_input))
        yaw = _wrap_angle(state.yaw + turn_unit * state.turn_limit)
        dx = state.speed * torch.sin(yaw)
        dy = state.speed * torch.cos(yaw)
        position = state.position + torch.cat([dx, dy], dim=-1)
        return prior_latent, position, yaw, state.mode_logits

    def _correct_with_evidence(
        self,
        prior_latent: torch.Tensor,
        prior_position: torch.Tensor,
        prior_yaw: torch.Tensor,
        prior_logits: torch.Tensor,
        state: PIBeliefState,
        entity_features: torch.Tensor,
        evidence_mask: torch.Tensor,
        evidence_meta: torch.Tensor,
    ) -> PIBeliefState:
        c = self.config
        B, E, K, D = prior_latent.shape
        encoded = self.evidence_encoder(entity_features)
        evidence = encoded[:, :, None, :].expand(B, E, K, D)
        age = (state.age / c.max_age).clamp(0.0, 1.0)
        age_k = age[:, :, None, :].expand(B, E, K, 1)
        meta_k = evidence_meta[:, :, None, :].expand(B, E, K, EVIDENCE_META_DIM)
        corr_input = torch.cat([prior_latent, evidence, age_k, meta_k], dim=-1)
        scores = self._type_apply(self.evidence_scores, corr_input).squeeze(-1)
        learned_gate = torch.sigmoid(
            self._type_apply(self.evidence_gates, corr_input).squeeze(-1)
        )

        evidence_k = evidence_mask[:, :, None]
        first_seen = (state.known.squeeze(-1) < 0.5)[:, :, None] & (evidence_k > 0.5)
        gate = torch.where(first_seen, torch.ones_like(learned_gate), learned_gate)
        gate = gate * evidence_k
        latent = prior_latent * (1.0 - gate[..., None]) + evidence * gate[..., None]

        flattened = prior_logits * 0.97
        logits = torch.where(evidence_k > 0.5, flattened + scores, flattened)

        # Evidence may be stale.  ``entity_features`` stores the location at the
        # evidence timestamp, not necessarily the current environment step.  Before
        # correction, dead-reckon that evidence to the present under the simplest
        # execution-time model available to the decentralized actor: constant
        # speed and heading.  This is exact for static threats, neutral for fresh
        # evidence (age=0), and deliberately does not use hidden simulator truth.
        supplied_age = evidence_meta[:, :, 0:1] * c.max_age
        obs_pos_now = entity_features[:, :, 9:11]
        obs_yaw_now = entity_features[:, :, 11:12] * math.pi
        obs_speed_now = entity_features[:, :, 18:19] * 50.0 / c.world_size
        stale_delta = torch.cat(
            [
                obs_speed_now * supplied_age * torch.sin(obs_yaw_now),
                obs_speed_now * supplied_age * torch.cos(obs_yaw_now),
            ],
            dim=-1,
        )
        aligned_pos = (obs_pos_now + stale_delta).clamp(0.0, 1.0)
        obs_pos = aligned_pos[:, :, None, :].expand(B, E, K, 2)
        obs_yaw = obs_yaw_now[:, :, None, :].expand(B, E, K, 1)
        physical_gate = gate[..., None]
        position = prior_position * (1.0 - physical_gate) + obs_pos * physical_gate
        sin_yaw = (1.0 - physical_gate) * torch.sin(prior_yaw) + physical_gate * torch.sin(obs_yaw)
        cos_yaw = (1.0 - physical_gate) * torch.cos(prior_yaw) + physical_gate * torch.cos(obs_yaw)
        yaw = torch.atan2(sin_yaw, cos_yaw)

        obs_speed = obs_speed_now[:, :, None, :].expand(B, E, K, 1)
        obs_turn = entity_features[:, :, 19:20] * 40.0 * math.pi / 180.0
        obs_turn = obs_turn[:, :, None, :].expand(B, E, K, 1)
        physical_evidence = evidence_mask[:, :, None, None]
        speed = torch.where(physical_evidence > 0.5, obs_speed, state.speed)
        turn_limit = torch.where(physical_evidence > 0.5, obs_turn, state.turn_limit)

        age_new = torch.where(
            evidence_mask[:, :, None] > 0.5,
            supplied_age,
            (state.age + 1.0).clamp_max(c.max_age),
        )
        known = torch.maximum(state.known, evidence_mask[:, :, None])
        obs_alive = entity_features[:, :, 7:8]
        alive = torch.where(evidence_mask[:, :, None] > 0.5, obs_alive, state.alive)

        return PIBeliefState(
            latent=latent,
            mode_logits=logits,
            position=position,
            yaw=yaw,
            speed=speed,
            turn_limit=turn_limit,
            age=age_new,
            known=known,
            alive=alive,
            info_context=state.info_context,
        )

    def update_belief(
        self,
        self_features: torch.Tensor,
        entity_features: torch.Tensor,
        evidence_mask: torch.Tensor,
        evidence_meta: torch.Tensor,
        state: PIBeliefState,
    ) -> PIBeliefState:
        context = self._update_context(self_features, evidence_mask, state)
        prior_latent, prior_position, prior_yaw, prior_logits = self._predict_current(
            state, context
        )
        corrected = self._correct_with_evidence(
            prior_latent,
            prior_position,
            prior_yaw,
            prior_logits,
            state,
            entity_features,
            evidence_mask,
            evidence_meta,
        )
        corrected.info_context = context
        return corrected

    def _physical_future(
        self, state: PIBeliefState
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        c = self.config
        B, E, K, _ = state.latent.shape
        latent = state.latent
        position = state.position
        yaw = state.yaw
        mode = self.mode_embedding[None, None, :, :].expand(B, E, K, -1)
        context = self.context_to_belief(state.info_context)[:, None, None, :].expand(
            B, E, K, -1
        )
        positions, yaws, latents = [], [], []
        for h in range(c.horizon):
            age = ((state.age + float(h + 1)) / c.max_age).clamp(0.0, 1.0)
            age = age[:, :, None, :].expand(B, E, K, 1)
            dyn_input = torch.cat([latent, mode, context, age], dim=-1)
            latent = latent + 0.20 * torch.tanh(self._type_apply(self.dynamics, dyn_input))
            turn = torch.tanh(self._type_apply(self.turn_heads, dyn_input)) * state.turn_limit
            yaw = _wrap_angle(yaw + turn)
            position = position + torch.cat(
                [state.speed * torch.sin(yaw), state.speed * torch.cos(yaw)], dim=-1
            )
            positions.append(position)
            yaws.append(yaw)
            latents.append(latent)
        return (
            torch.stack(positions, dim=3),
            torch.stack(yaws, dim=3),
            torch.stack(latents, dim=3),
        )

    def _action_future(self, self_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        c = self.config
        B = self_features.shape[0]
        subtype = self_features[:, 2:5]
        speed_mps = subtype @ self_features.new_tensor([40.0, 36.0, 38.0])
        max_turn = subtype @ self_features.new_tensor([36.0, 18.0, 24.0])
        speed = speed_mps / c.world_size
        max_turn = max_turn * math.pi / 180.0

        x = self_features[:, 6:7].expand(B, len(self.ACTION_VALUES)).clone()
        y = self_features[:, 7:8].expand(B, len(self.ACTION_VALUES)).clone()
        yaw = (self_features[:, 8:9] * math.pi).expand(B, len(self.ACTION_VALUES)).clone()
        action = self.action_values[None, :].expand(B, -1)
        pos_seq, yaw_seq = [], []
        for h in range(c.horizon):
            turn = action * max_turn[:, None] * (c.action_decay ** h)
            yaw = _wrap_angle(yaw + turn)
            x = x + speed[:, None] * torch.sin(yaw)
            y = y + speed[:, None] * torch.cos(yaw)
            pos_seq.append(torch.stack([x, y], dim=-1))
            yaw_seq.append(yaw[..., None])
        return torch.stack(pos_seq, dim=2), torch.stack(yaw_seq, dim=2)

    def _discovery_stats(
        self,
        evidence_mask: torch.Tensor,
        state: PIBeliefState,
    ) -> torch.Tensor:
        """Observable/belief summary for search and first-discovery behaviour.

        Feature order:
        teammate_visible, target_visible, threat_visible,
        overall_visible, overall_known, target_known,
        unknown_target, mean_known_target_age.
        """
        c = self.config
        t0 = c.n_teammates
        t1 = t0 + c.n_targets
        h0 = t1
        h1 = h0 + c.n_threats
        known = state.known.squeeze(-1)

        teammate_visible = evidence_mask[:, :t0].mean(dim=1, keepdim=True)
        target_visible = evidence_mask[:, t0:t1].mean(dim=1, keepdim=True)
        threat_visible = evidence_mask[:, h0:h1].mean(dim=1, keepdim=True)
        overall_visible = evidence_mask.mean(dim=1, keepdim=True)
        overall_known = known.mean(dim=1, keepdim=True)
        target_known_vec = known[:, t0:t1]
        target_known = target_known_vec.mean(dim=1, keepdim=True)
        unknown_target = 1.0 - target_known
        target_age = (state.age[:, t0:t1, 0] / c.max_age).clamp(0.0, 1.0)
        mean_target_age = (target_age * target_known_vec).sum(
            dim=1, keepdim=True
        ) / target_known_vec.sum(dim=1, keepdim=True).clamp_min(1.0)
        return torch.cat(
            [
                teammate_visible,
                target_visible,
                threat_visible,
                overall_visible,
                overall_known,
                target_known,
                unknown_target,
                mean_target_age,
            ],
            dim=-1,
        )

    def _reactive_logits(
        self,
        self_features: torch.Tensor,
        entity_features: torch.Tensor,
        evidence_mask: torch.Tensor,
        evidence_meta: torch.Tensor,
        state: PIBeliefState,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Direct local-evidence controller plus explicit discovery controller."""
        mask = evidence_mask[..., None]
        masked_entities = entity_features * mask
        masked_meta = evidence_meta * mask
        reactive_input = torch.cat(
            [
                self_features,
                masked_entities.reshape(self_features.shape[0], -1),
                evidence_mask,
                masked_meta.reshape(self_features.shape[0], -1),
            ],
            dim=-1,
        )
        reactive = self.self_action_head(self_features) + self.current_evidence_head(
            reactive_input
        )
        discovery = self._discovery_stats(evidence_mask, state)
        search = self.search_head(torch.cat([self_features, discovery], dim=-1))
        return reactive, search, discovery

    def _base_dispersion(
        self, future_pos: torch.Tensor, mode_probs: torch.Tensor
    ) -> torch.Tensor:
        w = mode_probs[:, :, :, None, None]
        mean = (future_pos * w).sum(dim=2, keepdim=True)
        var = ((future_pos - mean).square().sum(dim=-1) * mode_probs[:, :, :, None]).sum(dim=2)
        return var

    def _information_future(
        self,
        action_pos: torch.Tensor,
        action_yaw: torch.Tensor,
        future_pos: torch.Tensor,
        state: PIBeliefState,
        mode_probs: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        c = self.config
        B, A, H, _ = action_pos.shape
        E, K = c.n_entities, c.n_hypotheses
        category = self.entity_category[None, None, :, None, :].expand(B, A, E, K, 3)
        context = state.info_context[:, None, None, None, :].expand(B, A, E, K, -1)
        age = state.age.squeeze(-1)[:, None, :].expand(B, A, E)
        known = state.known.squeeze(-1)
        dispersion = self._base_dispersion(future_pos, mode_probs)
        mode_weight = mode_probs[:, None, :, :].expand(B, A, E, K)

        age_steps, uncertainty_steps = [], []
        q_obs_steps, q_com_steps, q_refresh_steps = [], [], []
        q_obs_hyp_steps, q_com_hyp_steps, q_refresh_hyp_steps = [], [], []
        current_u = dispersion[:, :, 0][:, None, :].expand(B, A, E).clone()
        dispersion_gain = F.softplus(self._dispersion_gain)
        age_gain = F.softplus(self._age_gain)
        reset_u = torch.sigmoid(self._reset_uncertainty_logit) * 0.10

        for h in range(H):
            # Keep the K physical futures separate until after observation and
            # communication refresh have been estimated.  Averaging positions
            # first can create a fictitious target location that corresponds to
            # none of the hypotheses.
            entity = future_pos[:, None, :, :, h, :].expand(B, A, E, K, 2)
            own = action_pos[:, :, h, :][:, :, None, None, :].expand(B, A, E, K, 2)
            delta = entity - own
            dist = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
            own_yaw = action_yaw[:, :, h, :][:, :, None, None, :].expand(B, A, E, K, 1)
            heading = torch.cat([torch.sin(own_yaw), torch.cos(own_yaw)], dim=-1)
            age_norm = (age / c.max_age).clamp(0.0, 1.0)[:, :, :, None, None]
            age_norm = age_norm.expand(B, A, E, K, 1)
            info_input = torch.cat([delta, dist, heading, category, age_norm, context], dim=-1)
            q_obs_hyp = torch.sigmoid(self.observe_predictor(info_input)).squeeze(-1)
            q_com_hyp = torch.sigmoid(self.communicate_predictor(info_input)).squeeze(-1)
            known_hyp = known[:, None, :, None].expand(B, A, E, K)
            q_obs_hyp = q_obs_hyp * known_hyp
            q_com_hyp = q_com_hyp * known_hyp
            q_refresh_hyp = 1.0 - (1.0 - q_obs_hyp) * (1.0 - q_com_hyp)

            q_obs = (mode_weight * q_obs_hyp).sum(dim=3)
            q_com = (mode_weight * q_com_hyp).sum(dim=3)
            q_refresh = (mode_weight * q_refresh_hyp).sum(dim=3)

            age = (1.0 - q_refresh) * (age + 1.0)
            disp_h = dispersion[:, :, h][:, None, :].expand(B, A, E)
            grow = current_u + dispersion_gain * disp_h + age_gain * (age / c.max_age)
            current_u = (1.0 - q_refresh) * grow + q_refresh * reset_u

            age_steps.append(age)
            uncertainty_steps.append(current_u)
            q_obs_steps.append(q_obs)
            q_com_steps.append(q_com)
            q_refresh_steps.append(q_refresh)
            q_obs_hyp_steps.append(q_obs_hyp)
            q_com_hyp_steps.append(q_com_hyp)
            q_refresh_hyp_steps.append(q_refresh_hyp)

        return {
            "expected_age": torch.stack(age_steps, dim=-1),
            "uncertainty": torch.stack(uncertainty_steps, dim=-1),
            "q_obs": torch.stack(q_obs_steps, dim=-1),
            "q_com": torch.stack(q_com_steps, dim=-1),
            "q_refresh": torch.stack(q_refresh_steps, dim=-1),
            "q_obs_hypothesis": torch.stack(q_obs_hyp_steps, dim=-1),
            "q_com_hypothesis": torch.stack(q_com_hyp_steps, dim=-1),
            "q_refresh_hypothesis": torch.stack(q_refresh_hyp_steps, dim=-1),
        }

    def _interaction_lattice(
        self,
        action_pos: torch.Tensor,
        action_yaw: torch.Tensor,
        future_pos: torch.Tensor,
        state: PIBeliefState,
        info: Dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        c = self.config
        B, A, H, _ = action_pos.shape
        E, K = c.n_entities, c.n_hypotheses
        u = info["uncertainty"]
        temperature = 1.0 + c.uncertainty_temperature * u.clamp_min(0.0)
        logits = state.mode_logits[:, None, :, :, None].expand(B, A, E, K, H)
        action_mode_probs = torch.softmax(logits / temperature[:, :, :, None, :], dim=3)

        entity = future_pos[:, None, :, :, :, :].expand(B, A, E, K, H, 2)
        own = action_pos[:, :, None, None, :, :].expand(B, A, E, K, H, 2)
        delta = entity - own
        dist = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
        own_yaw = action_yaw[:, :, None, None, :, :].expand(B, A, E, K, H, 1)
        heading = torch.cat([torch.sin(own_yaw), torch.cos(own_yaw)], dim=-1)
        mode_p = action_mode_probs[..., None]
        age = (info["expected_age"] / c.max_age).clamp(0.0, 1.0)[:, :, :, None, :, None]
        age = age.expand(B, A, E, K, H, 1)
        uncertainty = info["uncertainty"][:, :, :, None, :, None].expand(B, A, E, K, H, 1)
        refresh = info["q_refresh_hypothesis"][..., None]
        category = self.entity_category[None, None, :, None, None, :].expand(B, A, E, K, H, 3)
        known = state.known[:, None, :, None, None, :].expand(B, A, E, K, H, 1)
        alive = state.alive[:, None, :, None, None, :].expand(B, A, E, K, H, 1)
        horizon = torch.linspace(
            1.0 / H, 1.0, H, device=action_pos.device, dtype=action_pos.dtype
        )[None, None, None, None, :, None].expand(B, A, E, K, H, 1)
        relation = torch.cat(
            [delta, dist, heading, mode_p, age, uncertainty, refresh, category, known, alive, horizon],
            dim=-1,
        )
        lattice = self._type_apply(self.relation_encoders, relation)
        task_cell = self._type_apply(self.task_heads, lattice).squeeze(-1)
        info_cell = self._type_apply(self.info_heads, lattice).squeeze(-1)

        discount = torch.pow(
            action_pos.new_tensor(0.90),
            torch.arange(H, device=action_pos.device, dtype=action_pos.dtype),
        )[None, None, None, None, :]
        validity = (state.known * state.alive).squeeze(-1)
        weight = action_mode_probs * discount * validity[:, None, :, None, None]
        denominator = weight.sum(dim=(2, 3, 4)).clamp_min(1e-6)
        task_value = (task_cell * weight).sum(dim=(2, 3, 4)) / denominator
        info_value = (info_cell * weight).sum(dim=(2, 3, 4)) / denominator
        return lattice, task_value, info_value

    def forward(
        self,
        self_features: torch.Tensor,
        entity_features: torch.Tensor,
        evidence_mask: torch.Tensor,
        evidence_meta: torch.Tensor,
        state: PIBeliefState | None = None,
    ) -> tuple[torch.Tensor, PIBeliefState, Dict[str, torch.Tensor]]:
        if self_features.dim() != 2 or self_features.shape[-1] != SELF_DIM:
            raise ValueError(f"self_features must be [B,{SELF_DIM}]")
        if entity_features.dim() != 3 or entity_features.shape[1:] != (
            self.config.n_entities,
            ENTITY_DIM,
        ):
            raise ValueError(
                f"entity_features must be [B,{self.config.n_entities},{ENTITY_DIM}]"
            )
        B = self_features.shape[0]
        if state is None:
            state = self.initial_state(B, self_features.device, self_features.dtype)

        next_state = self.update_belief(
            self_features, entity_features, evidence_mask, evidence_meta, state
        )
        future_pos, future_yaw, _ = self._physical_future(next_state)
        action_pos, action_yaw = self._action_future(self_features)
        base_probs = torch.softmax(next_state.mode_logits, dim=-1)
        info = self._information_future(
            action_pos, action_yaw, future_pos, next_state, base_probs
        )
        lattice, task_value, info_value = self._interaction_lattice(
            action_pos, action_yaw, future_pos, next_state, info
        )

        known = next_state.known.squeeze(-1)
        current_dispersion = self._base_dispersion(
            next_state.position[:, :, :, None, :], base_probs
        ).squeeze(-1)
        mean_age = (
            (next_state.age.squeeze(-1) / self.config.max_age) * known
        ).sum(dim=1, keepdim=True) / known.sum(dim=1, keepdim=True).clamp_min(1.0)
        mean_u = (current_dispersion * known).sum(dim=1, keepdim=True) / known.sum(
            dim=1, keepdim=True
        ).clamp_min(1.0)
        known_fraction = known.mean(dim=1, keepdim=True)
        visible_fraction = evidence_mask.mean(dim=1, keepdim=True)
        gate_input = torch.cat(
            [next_state.info_context, mean_age, mean_u, known_fraction, visible_fraction], dim=-1
        )
        cognitive_gate = torch.sigmoid(self.cognitive_gate(gate_input))

        reactive_logits, search_logits, discovery_stats = self._reactive_logits(
            self_features,
            entity_features,
            evidence_mask,
            evidence_meta,
            next_state,
        )
        pi_residual = task_value + cognitive_gate * info_value
        pi_residual_scale = torch.tanh(self._pi_residual_gate)
        search_residual_scale = torch.sigmoid(self._search_residual_logit)
        logits = (
            reactive_logits
            + search_residual_scale * search_logits
            + pi_residual_scale * pi_residual
        )
        aux = {
            **info,
            "future_position": future_pos,
            "future_yaw": future_yaw,
            "action_position": action_pos,
            "action_yaw": action_yaw,
            "mode_probs": base_probs,
            "lattice": lattice,
            "task_value": task_value,
            "info_value": info_value,
            "cognitive_gate": cognitive_gate,
            "reactive_logits": reactive_logits,
            "search_logits": search_logits,
            "discovery_stats": discovery_stats,
            "pi_residual": pi_residual,
            "pi_residual_scale": pi_residual_scale,
            "search_residual_scale": search_residual_scale,
        }
        return logits, next_state, aux

    def distribution(self, *args, **kwargs) -> tuple[Categorical, PIBeliefState, Dict[str, torch.Tensor]]:
        logits, state, aux = self(*args, **kwargs)
        return Categorical(logits=logits), state, aux

    def act(
        self,
        self_features: torch.Tensor,
        entity_features: torch.Tensor,
        evidence_mask: torch.Tensor,
        evidence_meta: torch.Tensor,
        state: PIBeliefState | None = None,
        deterministic: bool = False,
    ):
        dist, next_state, aux = self.distribution(
            self_features, entity_features, evidence_mask, evidence_meta, state
        )
        action = torch.argmax(dist.logits, dim=-1) if deterministic else dist.sample()
        return action, dist.log_prob(action), dist.entropy(), next_state, aux

    def evaluate_actions(
        self,
        self_features: torch.Tensor,
        entity_features: torch.Tensor,
        evidence_mask: torch.Tensor,
        evidence_meta: torch.Tensor,
        state: PIBeliefState,
        actions: torch.Tensor,
    ):
        dist, next_state, aux = self.distribution(
            self_features, entity_features, evidence_mask, evidence_meta, state
        )
        return dist.log_prob(actions), dist.entropy(), next_state, aux
