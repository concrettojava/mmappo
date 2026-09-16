from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

from .algorithms import MAPPO, MAPPOConfig
from .envs import CooperativeUAVEnv
from .models import DirectFixedVectorizer, FixedVectorizer


@torch.inference_mode()
def policy_actions_batch(
    learner: MAPPO,
    obs_vectors: np.ndarray,
    active: np.ndarray,
    deterministic: bool = True,
) -> np.ndarray:
    """Run actor-only batched inference.

    This intentionally bypasses the critics. Under CTDE the critic is only
    required during training; decentralized evaluation/execution uses actors.
    """
    obs_vectors = np.asarray(obs_vectors, dtype=np.float32)
    active = np.asarray(active, dtype=np.float32)
    n_envs = obs_vectors.shape[0]
    actions = np.full((n_envs, learner.n_agents), 3, dtype=np.int64)

    for i in range(learner.n_agents):
        env_idx = np.flatnonzero(active[:, i] > 0.5)
        if env_idx.size == 0:
            continue
        obs = torch.as_tensor(
            obs_vectors[env_idx, i], dtype=torch.float32, device=learner.device
        )
        action, _, _ = learner.actors[i].act(obs, deterministic=deterministic)
        actions[env_idx, i] = action.cpu().numpy()
    return actions


def load_fixed_mappo_checkpoint(
    checkpoint_path: str | Path,
    device: str,
) -> Tuple[MAPPO, FixedVectorizer, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    vec_cfg = checkpoint.get("vectorizer", {})
    vectorizer = FixedVectorizer(**vec_cfg) if vec_cfg else FixedVectorizer()
    config = MAPPOConfig(**checkpoint.get("config", {}))
    learner = MAPPO(
        vectorizer.observation_dim,
        vectorizer.state_dim,
        n_agents=int(checkpoint.get("n_agents", 8)),
        action_dim=int(checkpoint.get("action_dim", 7)),
        config=config,
        device=device,
    )
    learner.load_checkpoint(checkpoint)
    return learner, vectorizer, checkpoint


def _stats(values: np.ndarray) -> Dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {"mean": float(values.mean()), "std": float(values.std())}


def evaluate_fixed_mappo(
    learner: MAPPO,
    vectorizer: FixedVectorizer,
    episodes: int,
    seed: int = 10000,
    deterministic: bool = True,
    batch_size: int | None = None,
) -> Dict[str, Dict[str, float]]:
    """Evaluate fixed-vector MAPPO with actor-only inference.

    Environments are evaluated in batches solely for speed. Each episode uses
    seed ``seed + episode_index`` and therefore remains independent.
    """
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    if batch_size is None:
        batch_size = episodes
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    direct = DirectFixedVectorizer(vectorizer)
    all_completion = []
    all_survival = []
    all_steps = []
    all_returns = []

    for start in range(0, episodes, batch_size):
        count = min(batch_size, episodes - start)
        envs = [CooperativeUAVEnv(seed=seed + start + e) for e in range(count)]
        obs = np.zeros((count, learner.n_agents, vectorizer.observation_dim), dtype=np.float32)
        state = np.zeros((count, learner.n_agents, vectorizer.state_dim), dtype=np.float32)
        active = np.zeros((count, learner.n_agents), dtype=np.float32)
        for e, env in enumerate(envs):
            obs[e], state[e], active[e] = env.reset_vectors(direct, seed=seed + start + e)

        finished = np.zeros(count, dtype=bool)
        returns = np.zeros((count, learner.n_agents), dtype=np.float32)
        infos = [None] * count

        while not bool(finished.all()):
            actions = policy_actions_batch(learner, obs, active, deterministic=deterministic)
            next_obs = np.zeros_like(obs)
            next_state = np.zeros_like(state)
            next_active = np.zeros_like(active)

            for e, env in enumerate(envs):
                if finished[e]:
                    continue
                o, s, a, rewards, done, info = env.step_vectors(actions[e], direct)
                next_obs[e], next_state[e], next_active[e] = o, s, a
                returns[e] += np.fromiter(
                    (rewards[i] for i in range(learner.n_agents)),
                    dtype=np.float32,
                    count=learner.n_agents,
                )
                if done:
                    finished[e] = True
                    infos[e] = info

            obs, state, active = next_obs, next_state, next_active

        for e, info in enumerate(infos):
            all_completion.append(float(info["completion_ratio"]))
            all_survival.append(float(info["survival_ratio"]))
            all_steps.append(float(info["step"]))
            all_returns.append(float(returns[e].mean()))

    return {
        "completion_ratio": _stats(np.asarray(all_completion)),
        "survival_ratio": _stats(np.asarray(all_survival)),
        "completion_time_steps": _stats(np.asarray(all_steps)),
        "mean_agent_return": _stats(np.asarray(all_returns)),
    }
