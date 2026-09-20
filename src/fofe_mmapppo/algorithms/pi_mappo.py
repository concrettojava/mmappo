"""Correctness-first recurrent MAPPO integration for PI-Net.

PI-Net carries a persistent per-entity belief, so PPO replay must preserve time
order.  The learner supports both the original full-episode BPTT path and a
chunked truncated-BPTT path.  Chunking never resets the forward belief: chunk
boundaries are reconstructed once with the unchanged behaviour policy, detached,
and then reused as recurrent initial states during PPO epochs.  Thus the policy
still carries information across the complete episode while gradients are
bounded to a configurable temporal window.

The eager PIActor modules remain the source of truth for parameters, optimizers
and checkpoints. Batched updates stack their parameters differentiably and
checkpoint per-step activations without shortening the recurrent gradient span.
On CUDA, optional ``torch.compile`` execution views point to
the same parameters and are used only for forward/replay execution.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import os
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
    ppo_epochs: int = 4
    learning_rate: float = 8e-5
    actor_learning_rate: float | None = None
    critic_learning_rate: float | None = None
    hidden_dim: int = 256
    entropy_coef: float = 0.005
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    sequence_env_minibatch_size: int = 4
    # Maximum actor BPTT span.  50 keeps the full 200-step forward belief while
    # cutting autograd history at 50-step boundaries.  Set >= rollout length to
    # recover the previous full-episode BPTT implementation exactly.
    tbptt_chunk_length: int = 50
    fused_adam: bool = True
    compile_actors: bool = True
    actor_compile_mode: str = "default"
    batch_actor_updates: bool = False
    actor_batch_size: int = 8
    actor_saved_steps: int = 0


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
        if int(self.config.tbptt_chunk_length) <= 0:
            raise ValueError("tbptt_chunk_length must be positive")

        self.actors = nn.ModuleList(
            [PIActor(self.actor_config) for _ in range(self.n_agents)]
        ).to(self.device)
        self.critics = nn.ModuleList(
            [MLPCritic(state_dim, self.config.hidden_dim) for _ in range(self.n_agents)]
        ).to(self.device)

        actor_lr = (
            self.config.learning_rate
            if self.config.actor_learning_rate is None
            else self.config.actor_learning_rate
        )
        critic_lr = (
            self.config.learning_rate
            if self.config.critic_learning_rate is None
            else self.config.critic_learning_rate
        )
        common_adam_kwargs = {
            "fused": bool(self.config.fused_adam and self.device.type == "cuda"),
        }
        self.actor_optimizers = [
            torch.optim.Adam(actor.parameters(), lr=actor_lr, **common_adam_kwargs)
            for actor in self.actors
        ]
        self.critic_optimizers = [
            torch.optim.Adam(critic.parameters(), lr=critic_lr, **common_adam_kwargs)
            for critic in self.critics
        ]

        disable_compile = os.environ.get("PI_DISABLE_TORCH_COMPILE", "").strip().lower()
        disable_compile = disable_compile in {"1", "true", "yes", "on"}
        self.compiled_actors_enabled = bool(
            self.config.compile_actors
            and self.device.type == "cuda"
            and not disable_compile
            and hasattr(torch, "compile")
        )
        if self.compiled_actors_enabled:
            self._actor_executors = [
                torch.compile(
                    actor,
                    mode=self.config.actor_compile_mode,
                    fullgraph=False,
                )
                for actor in self.actors
            ]
        else:
            self._actor_executors = list(self.actors)

        self._batched_groups = []
        if self.config.actor_batch_size <= 0:
            raise ValueError("actor_batch_size must be positive")
        if self.config.actor_saved_steps < 0:
            raise ValueError("actor_saved_steps must be non-negative")
        if self.config.batch_actor_updates:
            from .pi_batched import BatchedPIActors
            for start in range(0, self.n_agents, self.config.actor_batch_size):
                end = min(self.n_agents, start + self.config.actor_batch_size)
                backend = BatchedPIActors(
                    self.actors[start:end], self.compiled_actors_enabled,
                    self.config.actor_compile_mode,
                    saved_steps=self.config.actor_saved_steps)
                self._batched_groups.append((start, end, backend))

    @property
    def actor_execution_backend(self) -> str:
        if self.compiled_actors_enabled:
            return f"torch.compile:{self.config.actor_compile_mode}"
        return "eager"

    def initial_belief_states(self, n_envs: int) -> list[PIBeliefState]:
        return [actor.initial_state(n_envs, device=self.device) for actor in self.actors]

    @staticmethod
    def _index_belief_state(state: PIBeliefState, env_idx: torch.Tensor) -> PIBeliefState:
        """Take environment rows from every tensor in a belief state."""
        return PIBeliefState(
            **{
                name: value.index_select(0, env_idx)
                for name, value in state.__dict__.items()
            }
        )

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
            logits, candidate_state, _ = self._actor_executors[i](
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
            self._actor_executors[agent],
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

    @torch.no_grad()
    def _agent_chunk_boundaries(
        self,
        data: dict[str, torch.Tensor],
        agent: int,
        chunk_length: int,
    ) -> dict[int, PIBeliefState]:
        """Reconstruct behaviour-policy recurrent states at TBPTT boundaries.

        ``no_grad`` is deliberate rather than ``inference_mode``: the detached
        boundary tensors are later inputs to grad-enabled chunk replays, and
        ordinary tensors avoid PyTorch's inference-tensor autograd restriction.
        """
        T, E = data["active"].shape[:2]
        state = self.actors[agent].initial_state(E, device=self.device)
        boundaries: dict[int, PIBeliefState] = {0: state.detach()}
        executor = self._actor_executors[agent]
        for start in range(0, T, chunk_length):
            end = min(T, start + chunk_length)
            replay = unroll_pi_actor(
                executor,
                data["self_features"][start:end, :, agent],
                data["entity_features"][start:end, :, agent],
                data["evidence_mask"][start:end, :, agent],
                data["evidence_meta"][start:end, :, agent],
                initial_state=state,
                active=data["active"][start:end, :, agent] > 0.5,
            )
            state = replay.final_state.detach()
            if end < T:
                boundaries[end] = state
        return boundaries

    @torch.inference_mode()
    def replay_diagnostics(self, buffer: PIParallelRolloutBuffer) -> Dict[str, float]:
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

    def _update_agent_full_bptt(
        self,
        data: dict[str, torch.Tensor],
        agent: int,
        env_pool: torch.Tensor,
        active_all: torch.Tensor,
        advantages: torch.Tensor,
        metrics: dict[str, list[float]],
    ) -> None:
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

    def _update_agent_chunked(
        self,
        data: dict[str, torch.Tensor],
        agent: int,
        env_pool: torch.Tensor,
        active_all: torch.Tensor,
        advantages: torch.Tensor,
        metrics: dict[str, list[float]],
        chunk_length: int,
    ) -> None:
        """TBPTT PPO: full forward memory, bounded backward graph.

        One optimizer step is still taken per environment minibatch per PPO epoch,
        exactly as in the legacy path.  Gradients from temporal chunks are
        accumulated before clipping/stepping, so chunking changes only the
        recurrent gradient horizon rather than multiplying optimizer steps.
        """
        T = int(active_all.shape[0])
        boundaries = self._agent_chunk_boundaries(data, agent, chunk_length)
        env_mb = min(self.config.sequence_env_minibatch_size, int(env_pool.numel()))

        for _ in range(self.config.ppo_epochs):
            order = env_pool[torch.randperm(env_pool.numel(), device=self.device)]
            for mb_start in range(0, int(order.numel()), env_mb):
                env_idx = order[mb_start : mb_start + env_mb]
                full_mask = active_all[:, env_idx]
                total_count = int(full_mask.sum().item())
                if total_count == 0:
                    continue

                self.actor_optimizers[agent].zero_grad(set_to_none=True)
                loss_parts = []
                entropy_sums = []
                ratio_sums = []
                clip_sums = []

                for t0 in range(0, T, chunk_length):
                    t1 = min(T, t0 + chunk_length)
                    mask = active_all[t0:t1, env_idx]
                    count = int(mask.sum().item())
                    if count == 0:
                        continue

                    initial_state = self._index_belief_state(boundaries[t0], env_idx)
                    replay = unroll_pi_actor(
                        self._actor_executors[agent],
                        data["self_features"][t0:t1, env_idx, agent],
                        data["entity_features"][t0:t1, env_idx, agent],
                        data["evidence_mask"][t0:t1, env_idx, agent],
                        data["evidence_meta"][t0:t1, env_idx, agent],
                        initial_state=initial_state,
                        actions=data["actions"][t0:t1, env_idx, agent],
                        active=mask,
                    )

                    new_lp = replay.log_probs[mask]
                    old_lp = data["log_probs"][t0:t1, env_idx, agent][mask]
                    adv = advantages[t0:t1, env_idx][mask]
                    ratio = torch.exp(new_lp - old_lp)
                    surr1 = ratio * adv
                    surr2 = torch.clamp(
                        ratio,
                        1.0 - self.config.clip_epsilon,
                        1.0 + self.config.clip_epsilon,
                    ) * adv
                    entropy = replay.entropy[mask]
                    chunk_loss = (
                        -torch.min(surr1, surr2).sum()
                        - self.config.entropy_coef * entropy.sum()
                    ) / float(total_count)
                    chunk_loss.backward()
                    loss_parts.append(chunk_loss.detach())
                    entropy_sums.append(entropy.detach().sum())
                    ratio_sums.append(ratio.detach().sum())
                    clip_sums.append(
                        ((ratio.detach() - 1.0).abs() > self.config.clip_epsilon)
                        .float()
                        .sum()
                    )

                nn.utils.clip_grad_norm_(
                    self.actors[agent].parameters(), self.config.max_grad_norm
                )
                self.actor_optimizers[agent].step()

                critic_mask = full_mask
                critic_states = data["states"][:, env_idx, agent][critic_mask]
                target_returns = data["returns"][:, env_idx, agent][critic_mask]
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

                actor_loss = torch.stack(loss_parts).sum()
                entropy_mean = torch.stack(entropy_sums).sum() / float(total_count)
                ratio_mean = torch.stack(ratio_sums).sum() / float(total_count)
                clip_fraction = torch.stack(clip_sums).sum() / float(total_count)
                metrics["actor_loss"].append(float(actor_loss.cpu()))
                metrics["critic_loss"].append(float(critic_loss.detach().cpu()))
                metrics["entropy"].append(float(entropy_mean.cpu()))
                metrics["ratio_mean"].append(float(ratio_mean.cpu()))
                metrics["clip_fraction"].append(float(clip_fraction.cpu()))

    def update_parallel(self, buffer: PIParallelRolloutBuffer) -> Dict[str, float]:
        """Sequence-preserving PPO with optional chunked truncated BPTT."""
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
        T = int(data["active"].shape[0])
        chunk_length = min(int(self.config.tbptt_chunk_length), T)
        use_chunked = chunk_length < T

        if self._batched_groups:
            from .pi_batched import update_batched
            for start, end, backend in self._batched_groups:
                update_batched(self, data, metrics, chunk_length, backend, start, end)

        for agent in range(self.n_agents) if not self._batched_groups else ():
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

            if use_chunked:
                self._update_agent_chunked(
                    data,
                    agent,
                    env_pool,
                    active_all,
                    advantages,
                    metrics,
                    chunk_length,
                )
            else:
                self._update_agent_full_bptt(
                    data,
                    agent,
                    env_pool,
                    active_all,
                    advantages,
                    metrics,
                )

        result = {
            key: float(np.mean(values)) if values else (1.0 if key == "ratio_mean" else 0.0)
            for key, values in metrics.items()
        }
        result["tbptt_chunk_length"] = float(chunk_length)
        result["tbptt_chunks"] = float((T + chunk_length - 1) // chunk_length)
        result["pi_residual_scale"] = float(
            np.mean(
                [
                    torch.tanh(actor._pi_residual_gate).detach().cpu().item()
                    for actor in self.actors
                ]
            )
        )
        result["search_residual_scale"] = float(
            np.mean(
                [
                    torch.sigmoid(actor._search_residual_logit).detach().cpu().item()
                    for actor in self.actors
                ]
            )
        )
        return result

    def checkpoint(self, include_optimizers: bool = False) -> dict:
        checkpoint = {
            "algorithm": "pi_mappo_v2_1_capacity_first",
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
