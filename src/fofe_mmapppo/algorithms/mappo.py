"""Minimal MAPPO implementation for the fixed-vector baseline.

This stage intentionally excludes FOFE and Mamba.  Each RSUAV owns its own
actor and critic.  The critic receives the centralized Eq.(9) state vector; the
actor receives only the local Eq.(8) observation vector.

Besides the original single-environment path, this module provides batched
multi-environment inference/update methods.  Those are engineering
accelerations, not claimed paper settings.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict

import numpy as np
import torch
from torch import nn

from ..models import MLPActor, MLPCritic
from .buffer import RolloutBuffer
from .parallel_buffer import ParallelRolloutBuffer


@dataclass
class MAPPOConfig:
    gamma: float = 0.95
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.1
    ppo_epochs: int = 15
    learning_rate: float = 4e-5
    hidden_dim: int = 256
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    minibatch_size: int = 64


class MAPPO:
    def __init__(self, obs_dim: int, state_dim: int, n_agents: int = 8,
                 action_dim: int = 7, config: MAPPOConfig | None = None,
                 device: str | torch.device = "cpu"):
        self.config = config or MAPPOConfig()
        self.n_agents = int(n_agents)
        self.action_dim = int(action_dim)
        self.device = torch.device(device)

        self.actors = nn.ModuleList([
            MLPActor(obs_dim, action_dim, self.config.hidden_dim)
            for _ in range(self.n_agents)
        ]).to(self.device)
        self.critics = nn.ModuleList([
            MLPCritic(state_dim, self.config.hidden_dim)
            for _ in range(self.n_agents)
        ]).to(self.device)

        self.actor_optimizers = [
            torch.optim.Adam(actor.parameters(), lr=self.config.learning_rate)
            for actor in self.actors
        ]
        self.critic_optimizers = [
            torch.optim.Adam(critic.parameters(), lr=self.config.learning_rate)
            for critic in self.critics
        ]

    @torch.no_grad()
    def act(self, obs_vectors: np.ndarray, state_vectors: np.ndarray,
            active: np.ndarray, deterministic: bool = False):
        actions = np.full(self.n_agents, 3, dtype=np.int64)
        log_probs = np.zeros(self.n_agents, dtype=np.float32)
        values = np.zeros(self.n_agents, dtype=np.float32)

        for i in range(self.n_agents):
            if not bool(active[i]):
                continue
            obs = torch.as_tensor(obs_vectors[i], dtype=torch.float32, device=self.device).unsqueeze(0)
            state = torch.as_tensor(state_vectors[i], dtype=torch.float32, device=self.device).unsqueeze(0)
            action, log_prob, _ = self.actors[i].act(obs, deterministic=deterministic)
            value = self.critics[i](state)
            actions[i] = int(action.item())
            log_probs[i] = float(log_prob.item())
            values[i] = float(value.item())
        return actions, log_probs, values

    @torch.no_grad()
    def act_batch(self, obs_vectors: np.ndarray, state_vectors: np.ndarray,
                  active: np.ndarray, deterministic: bool = False):
        """Batched inference across environments.

        Inputs have shapes ``[E,N,obs_dim]``, ``[E,N,state_dim]`` and ``[E,N]``.
        Each agent still owns an independent actor/critic, but all active
        environments for that agent are evaluated in one GPU call.
        """
        obs_vectors = np.asarray(obs_vectors, dtype=np.float32)
        state_vectors = np.asarray(state_vectors, dtype=np.float32)
        active = np.asarray(active, dtype=np.float32)
        n_envs = obs_vectors.shape[0]

        actions = np.full((n_envs, self.n_agents), 3, dtype=np.int64)
        log_probs = np.zeros((n_envs, self.n_agents), dtype=np.float32)
        values = np.zeros((n_envs, self.n_agents), dtype=np.float32)

        for i in range(self.n_agents):
            env_idx = np.flatnonzero(active[:, i] > 0.5)
            if env_idx.size == 0:
                continue
            obs = torch.as_tensor(obs_vectors[env_idx, i], dtype=torch.float32, device=self.device)
            state = torch.as_tensor(state_vectors[env_idx, i], dtype=torch.float32, device=self.device)
            action, log_prob, _ = self.actors[i].act(obs, deterministic=deterministic)
            value = self.critics[i](state)
            actions[env_idx, i] = action.cpu().numpy()
            log_probs[env_idx, i] = log_prob.cpu().numpy()
            values[env_idx, i] = value.cpu().numpy()
        return actions, log_probs, values

    @torch.no_grad()
    def values(self, state_vectors: np.ndarray, active: np.ndarray):
        values = np.zeros(self.n_agents, dtype=np.float32)
        for i in range(self.n_agents):
            if not bool(active[i]):
                continue
            state = torch.as_tensor(state_vectors[i], dtype=torch.float32, device=self.device).unsqueeze(0)
            values[i] = float(self.critics[i](state).item())
        return values

    def _update_flat_agent(self, agent: int, obs, states, actions,
                           old_log_probs, returns, advantages, metrics):
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        count = obs.shape[0]
        mb = min(self.config.minibatch_size, count)
        for _ in range(self.config.ppo_epochs):
            permutation = torch.randperm(count, device=self.device)
            for start in range(0, count, mb):
                idx = permutation[start:start + mb]
                new_log_probs, entropy = self.actors[agent].evaluate_actions(obs[idx], actions[idx])
                ratio = torch.exp(new_log_probs - old_log_probs[idx])
                surr1 = ratio * advantages[idx]
                surr2 = torch.clamp(
                    ratio,
                    1.0 - self.config.clip_epsilon,
                    1.0 + self.config.clip_epsilon,
                ) * advantages[idx]
                actor_loss = -torch.min(surr1, surr2).mean() - self.config.entropy_coef * entropy.mean()

                self.actor_optimizers[agent].zero_grad(set_to_none=True)
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actors[agent].parameters(), self.config.max_grad_norm)
                self.actor_optimizers[agent].step()

                predicted = self.critics[agent](states[idx])
                critic_loss = self.config.value_coef * torch.mean((predicted - returns[idx]) ** 2)
                self.critic_optimizers[agent].zero_grad(set_to_none=True)
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critics[agent].parameters(), self.config.max_grad_norm)
                self.critic_optimizers[agent].step()

                metrics["actor_loss"].append(float(actor_loss.detach().cpu()))
                metrics["critic_loss"].append(float(critic_loss.detach().cpu()))
                metrics["entropy"].append(float(entropy.mean().detach().cpu()))

    def update(self, buffer: RolloutBuffer) -> Dict[str, float]:
        if len(buffer) == 0:
            return {"actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0}

        data_np = buffer.compute_gae(self.config.gamma, self.config.gae_lambda)
        data = RolloutBuffer.to_torch(data_np, self.device)
        metrics = {"actor_loss": [], "critic_loss": [], "entropy": []}

        for agent in range(self.n_agents):
            active_idx = torch.nonzero(data["active"][:, agent] > 0.5, as_tuple=False).squeeze(-1)
            if active_idx.numel() == 0:
                continue
            self._update_flat_agent(
                agent,
                data["obs"][active_idx, agent],
                data["states"][active_idx, agent],
                data["actions"][active_idx, agent],
                data["log_probs"][active_idx, agent],
                data["returns"][active_idx, agent],
                data["advantages"][active_idx, agent],
                metrics,
            )
        return {key: float(np.mean(v)) if v else 0.0 for key, v in metrics.items()}

    def update_parallel(self, buffer: ParallelRolloutBuffer) -> Dict[str, float]:
        """PPO update from ``[T,E,N,...]`` parallel trajectories."""
        if len(buffer) == 0:
            return {"actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0}

        data_np = buffer.compute_gae(self.config.gamma, self.config.gae_lambda)
        data = ParallelRolloutBuffer.to_torch(data_np, self.device)
        metrics = {"actor_loss": [], "critic_loss": [], "entropy": []}

        for agent in range(self.n_agents):
            mask = data["active"][:, :, agent] > 0.5
            if not bool(mask.any()):
                continue
            self._update_flat_agent(
                agent,
                data["obs"][:, :, agent][mask],
                data["states"][:, :, agent][mask],
                data["actions"][:, :, agent][mask],
                data["log_probs"][:, :, agent][mask],
                data["returns"][:, :, agent][mask],
                data["advantages"][:, :, agent][mask],
                metrics,
            )
        return {key: float(np.mean(v)) if v else 0.0 for key, v in metrics.items()}

    def checkpoint(self, include_optimizers: bool = False):
        checkpoint = {
            "config": asdict(self.config),
            "n_agents": self.n_agents,
            "action_dim": self.action_dim,
            "actors": self.actors.state_dict(),
            "critics": self.critics.state_dict(),
        }
        if include_optimizers:
            checkpoint["actor_optimizers"] = [opt.state_dict() for opt in self.actor_optimizers]
            checkpoint["critic_optimizers"] = [opt.state_dict() for opt in self.critic_optimizers]
        return checkpoint

    def load_checkpoint(self, checkpoint: dict, load_optimizers: bool = False):
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
