"""Correctness-first recurrent MAPPO integration for PI-Net.

This learner deliberately preserves complete time sequences during actor replay.
It does *not* flatten recurrent actor observations across time.  For Phase 8,
parallel rollout batches are required to start at episode boundaries, so each
replayed environment starts from a zero ``PIBeliefState``.  Environment
minibatching is allowed; timestep shuffling is not.

The implementation is intentionally conservative before performance work:
full episode sequences are replayed for each selected environment minibatch.
Truncated BPTT/chunk caching can be added only after this exact path has passed
policy-ratio and episode-isolation tests.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Sequence

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from ..models import MLPCritic, PIActor, PIActorConfig, PIBeliefState
from .pi_parallel_buffer import PIParallelRolloutBuffer
from .pi_sequence import blend_belief_state, unroll_pi_actor


@dataclass
class PIMAPPOConfig:
    gamma: float = 0.95
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.1
    ppo_epochs: int = 15
    learning_rate: float = 4e-5
    hidden_dim: int = 256
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    sequence_env_minibatch_size: int = 4
    fused_adam: bool = True


class PIMAPPO:
    """MAPPO with stateful PI-Net actors and the original centralized MLP critics."""

    def __init__(
        self,
        state_dim: int,
        n_agents: int = 8,
        action_dim: int = 7,
        config: PIMAPPOConfig | None = None,
        actor_config: PIActorConfig | None = None,
        device: str | torch.device = "cpu",
    ):
        self.config = config or PIMAPPOConfig()
        self.actor_config = actor_config or PIActorConfig()
        self.n_agents = int(n_agents)
        self.action_dim = int(action_dim)
        self.device = torch.device(device)
        if self.action_dim != len(PIActor.ACTION_VALUES):
            raise ValueError("PI-Net V1 requires the paper's seven steering actions")

        self.actors = nn.ModuleList(
            [PIActor(self.actor_config) for _ in range(self.n_agents)]
        ).to(self.device)
        self.critics = nn.ModuleList(
            [MLPCritic(state_dim, self.config.hidden_dim) for _ in range(self.n_agents)]
        ).to(self.device)

        adam_kwargs = {
            "lr": self.config.learning_rate,
            "fused": bool(self.config.fused_adam and self.device.type == "cuda"),
        }
        self.actor_optimizers = [
            torch.optim.Adam(actor.parameters(), **adam_kwargs) for actor in self.actors
        ]
        self.critic_optimizers = [
            torch.optim.Adam(critic.parameters(), **adam_kwargs) for critic in self.critics
        ]

    def initial_belief_states(self, n_envs: int) -> list[PIBeliefState]:
        return [
            actor.initial_state(n_envs, device=self.device)
            for actor in self.actors
        ]

    @torch.inference_mode()
    def act_batch(
        self,
        self_features: np.ndarray,
        entity_features: np.ndarray,
        evidence_mask: np.ndarray,
        evidence_meta: np.ndarray,
        state_vectors: np.ndarray,
        active: np.ndarray,
        belief_states: Sequence[PIBeliefState] | None = None,
        deterministic: bool = False,
    ):
        """Act across parallel environments while preserving per-agent belief state.

        Structured actor inputs have leading shape ``[E,N,...]``.  Critic state
        remains the same centralized fixed vector used by the MAPPO baseline.
        """
        self_features = np.asarray(self_features, dtype=np.float32)
        entity_features = np.asarray(entity_features, dtype=np.float32)
        evidence_mask = np.asarray(evidence_mask, dtype=np.float32)
        evidence_meta = np.asarray(evidence_meta, dtype=np.float32)
        state_vectors = np.asarray(state_vectors, dtype=np.float32)
        active = np.asarray(active, dtype=np.float32)
        n_envs = self_features.shape[0]
        if self_features.shape[:2] != (n_envs, self.n_agents):
            raise ValueError("self_features must have leading shape [E,N]")
        if active.shape != (n_envs, self.n_agents):
            raise ValueError("active must be [E,N]")
        if belief_states is None:
            belief_states = self.initial_belief_states(n_envs)
        if len(belief_states) != self.n_agents:
            raise ValueError("one PIBeliefState is required per agent actor")

        actions = np.full((n_envs, self.n_agents), 3, dtype=np.int64)
        log_probs = np.zeros((n_envs, self.n_agents), dtype=np.float32)
        values = np.zeros((n_envs, self.n_agents), dtype=np.float32)
        next_beliefs: list[PIBeliefState] = []

        for i in range(self.n_agents):
            sf = torch.as_tensor(self_features[:, i], device=self.device)
            ef = torch.as_tensor(entity_features[:, i], device=self.device)
            em = torch.as_tensor(evidence_mask[:, i], device=self.device)
            meta = torch.as_tensor(evidence_meta[:, i], device=self.device)
            active_i = torch.as_tensor(active[:, i] > 0.5, device=self.device)
            logits, candidate_state, _ = self.actors[i](
                sf, ef, em, meta, belief_states[i]
            )
            next_state = blend_belief_state(belief_states[i], candidate_state, active_i)
            next_beliefs.append(next_state.detach())

            dist = Categorical(logits=logits)
            sampled = torch.argmax(logits, dim=-1) if deterministic else dist.sample()
            lp = dist.log_prob(sampled)
            sampled = torch.where(active_i, sampled, torch.full_like(sampled, 3))
            lp = torch.where(active_i, lp, torch.zeros_like(lp))

            centralized = torch.as_tensor(state_vectors[:, i], device=self.device)
            value = self.critics[i](centralized)
            value = torch.where(active_i, value, torch.zeros_like(value))

            actions[:, i] = sampled.cpu().numpy()
            log_probs[:, i] = lp.cpu().numpy()
            values[:, i] = value.cpu().numpy()

        return actions, log_probs, values, next_beliefs

    def _agent_replay(
        self,
        data: dict[str, torch.Tensor],
        agent: int,
        env_idx: torch.Tensor,
    ):
        return unroll_pi_actor(
            self.actors[agent],
            data["self_features"][:, env_idx, agent],
            data["entity_features"][:, env_idx, agent],
            data["evidence_mask"][:, env_idx, agent],
            data["evidence_meta"][:, env_idx, agent],
            initial_state=self.actors[agent].initial_state(
                int(env_idx.numel()), device=self.device
            ),
            actions=data["actions"][:, env_idx, agent],
            active=data["active"][:, env_idx, agent] > 0.5,
        )

    @torch.inference_mode()
    def replay_diagnostics(self, buffer: PIParallelRolloutBuffer) -> Dict[str, float]:
        """Verify that an unchanged policy reconstructs rollout probabilities."""
        if len(buffer) == 0:
            return {
                "max_abs_logprob_error": 0.0,
                "max_abs_ratio_error": 0.0,
                "ratio_mean": 1.0,
            }
        data = PIParallelRolloutBuffer.to_torch(buffer.as_arrays(), self.device)
        log_errors = []
        ratio_errors = []
        ratios = []
        for agent in range(self.n_agents):
            active = data["active"][:, :, agent] > 0.5
            env_idx = torch.nonzero(active.any(dim=0), as_tuple=False).squeeze(-1)
            if env_idx.numel() == 0:
                continue
            replay = self._agent_replay(data, agent, env_idx)
            mask = active[:, env_idx]
            old_lp = data["log_probs"][:, env_idx, agent]
            diff = replay.log_probs[mask] - old_lp[mask]
            ratio = torch.exp(diff)
            log_errors.append(diff.abs().max())
            ratio_errors.append((ratio - 1.0).abs().max())
            ratios.append(ratio)
        if not ratios:
            return {
                "max_abs_logprob_error": 0.0,
                "max_abs_ratio_error": 0.0,
                "ratio_mean": 1.0,
            }
        return {
            "max_abs_logprob_error": float(torch.stack(log_errors).max().cpu()),
            "max_abs_ratio_error": float(torch.stack(ratio_errors).max().cpu()),
            "ratio_mean": float(torch.cat(ratios).mean().cpu()),
        }

    def update_parallel(self, buffer: PIParallelRolloutBuffer) -> Dict[str, float]:
        """Sequence-preserving PPO update over complete environment trajectories."""
        if len(buffer) == 0:
            return {
                "actor_loss": 0.0,
                "critic_loss": 0.0,
                "entropy": 0.0,
                "ratio_mean": 1.0,
                "clip_fraction": 0.0,
            }

        data_np = buffer.compute_gae(self.config.gamma, self.config.gae_lambda)
        data = PIParallelRolloutBuffer.to_torch(data_np, self.device)
        metrics = {
            "actor_loss": [],
            "critic_loss": [],
            "entropy": [],
            "ratio_mean": [],
            "clip_fraction": [],
        }

        for agent in range(self.n_agents):
            active_all = data["active"][:, :, agent] > 0.5
            env_pool = torch.nonzero(active_all.any(dim=0), as_tuple=False).squeeze(-1)
            if env_pool.numel() == 0:
                continue

            advantages = data["advantages"][:, :, agent]
            valid_adv = advantages[active_all]
            if valid_adv.numel() > 1:
                mean = valid_adv.mean()
                std = valid_adv.std(unbiased=False).clamp_min(1e-8)
                advantages = torch.where(
                    active_all, (advantages - mean) / std, torch.zeros_like(advantages)
                )

            env_mb = min(self.config.sequence_env_minibatch_size, int(env_pool.numel()))
            for _ in range(self.config.ppo_epochs):
                order = env_pool[torch.randperm(env_pool.numel(), device=self.device)]
                for start in range(0, int(order.numel()), env_mb):
                    env_idx = order[start : start + env_mb]
                    replay = self._agent_replay(data, agent, env_idx)
                    mask = active_all[:, env_idx]
                    if not bool(mask.any()):
                        continue

                    new_lp = replay.log_probs[mask]
                    old_lp = data["log_probs"][:, env_idx, agent][mask]
                    adv = advantages[:, env_idx][mask]
                    ratio = torch.exp(new_lp - old_lp)
                    surr1 = ratio * adv
                    surr2 = torch.clamp(
                        ratio,
                        1.0 - self.config.clip_epsilon,
                        1.0 + self.config.clip_epsilon,
                    ) * adv
                    entropy = replay.entropy[mask]
                    actor_loss = -torch.min(surr1, surr2).mean()
                    actor_loss = actor_loss - self.config.entropy_coef * entropy.mean()

                    self.actor_optimizers[agent].zero_grad(set_to_none=True)
                    actor_loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.actors[agent].parameters(), self.config.max_grad_norm
                    )
                    self.actor_optimizers[agent].step()

                    critic_states = data["states"][:, env_idx, agent][mask]
                    target_returns = data["returns"][:, env_idx, agent][mask]
                    predicted = self.critics[agent](critic_states)
                    critic_loss = self.config.value_coef * torch.mean(
                        (predicted - target_returns) ** 2
                    )
                    self.critic_optimizers[agent].zero_grad(set_to_none=True)
                    critic_loss.backward()
                    nn.utils.clip_grad_norm_(
                        self.critics[agent].parameters(), self.config.max_grad_norm
                    )
                    self.critic_optimizers[agent].step()

                    clip_fraction = (
                        (ratio - 1.0).abs() > self.config.clip_epsilon
                    ).float().mean()
                    metrics["actor_loss"].append(float(actor_loss.detach().cpu()))
                    metrics["critic_loss"].append(float(critic_loss.detach().cpu()))
                    metrics["entropy"].append(float(entropy.mean().detach().cpu()))
                    metrics["ratio_mean"].append(float(ratio.mean().detach().cpu()))
                    metrics["clip_fraction"].append(float(clip_fraction.detach().cpu()))

        return {
            key: float(np.mean(values)) if values else (1.0 if key == "ratio_mean" else 0.0)
            for key, values in metrics.items()
        }

    def checkpoint(self, include_optimizers: bool = False) -> dict:
        checkpoint = {
            "algorithm": "pi_mappo_v1",
            "config": asdict(self.config),
            "actor_config": asdict(self.actor_config),
            "n_agents": self.n_agents,
            "action_dim": self.action_dim,
            "actors": self.actors.state_dict(),
            "critics": self.critics.state_dict(),
        }
        if include_optimizers:
            checkpoint["actor_optimizers"] = [
                optimizer.state_dict() for optimizer in self.actor_optimizers
            ]
            checkpoint["critic_optimizers"] = [
                optimizer.state_dict() for optimizer in self.critic_optimizers
            ]
        return checkpoint

    def load_checkpoint(self, checkpoint: dict, load_optimizers: bool = False) -> bool:
        self.actors.load_state_dict(checkpoint["actors"])
        self.critics.load_state_dict(checkpoint["critics"])
        if load_optimizers:
            actor_states = checkpoint.get("actor_optimizers")
            critic_states = checkpoint.get("critic_optimizers")
            if actor_states is not None and critic_states is not None:
                for optimizer, state in zip(self.actor_optimizers, actor_states):
                    optimizer.load_state_dict(state)
                for optimizer, state in zip(self.critic_optimizers, critic_states):
                    optimizer.load_state_dict(state)
                return True
        return False
