from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from fofe_mmapppo.algorithms import (
    PIParallelRolloutBuffer,
    PIMAPPO,
    PIMAPPOConfig,
)
from fofe_mmapppo.models import MLPCritic, PIActorConfig


class PIRecurrentMAPPOTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(23)
        np.random.seed(23)
        self.T = 4
        self.E = 2
        self.N = 2
        self.state_dim = 9
        self.actor_cfg = PIActorConfig(
            n_hypotheses=3,
            horizon=2,
            belief_dim=8,
            context_dim=8,
            relation_dim=8,
        )
        self.ppo_cfg = PIMAPPOConfig(
            ppo_epochs=1,
            learning_rate=5e-4,
            hidden_dim=16,
            entropy_coef=0.0,
            sequence_env_minibatch_size=2,
            fused_adam=False,
        )
        self.learner = PIMAPPO(
            self.state_dim,
            n_agents=self.N,
            config=self.ppo_cfg,
            actor_config=self.actor_cfg,
            device="cpu",
        )

    def _actor_inputs(self, t: int):
        sf = np.zeros((self.E, self.N, 16), dtype=np.float32)
        sf[..., 0] = 1.0
        sf[..., 2] = 1.0  # Stk
        sf[..., 5] = 1.0
        sf[..., 6] = 0.20 + 0.01 * t
        sf[..., 7] = 0.15
        sf[..., 8] = 0.0

        entities = np.zeros((self.E, self.N, 14, 20), dtype=np.float32)
        mask = np.zeros((self.E, self.N, 14), dtype=np.float32)
        meta = np.zeros((self.E, self.N, 14, 4), dtype=np.float32)
        target = 7
        # Target is seen at t=0 and t=3, with missing evidence in between.
        if t in (0, 3):
            entities[:, :, target, 0] = 1.0
            entities[:, :, target, 2] = 1.0
            entities[:, :, target, 7] = 1.0
            entities[:, :, target, 9] = 0.45 + 0.01 * t
            entities[:, :, target, 10] = 0.55
            entities[:, :, target, 11] = 0.0
            entities[:, :, target, 18] = 8.0 / 50.0
            entities[:, :, target, 19] = 10.0 / 40.0
            mask[:, :, target] = 1.0
        return sf, entities, mask, meta

    def _build_rollout(self):
        buffer = PIParallelRolloutBuffer(n_envs=self.E, n_agents=self.N)
        beliefs = self.learner.initial_belief_states(self.E)
        for t in range(self.T):
            sf, entities, mask, meta = self._actor_inputs(t)
            states = np.zeros((self.E, self.N, self.state_dim), dtype=np.float32)
            states[..., 0] = t / max(1, self.T - 1)
            states[..., 1] = np.arange(self.E, dtype=np.float32)[:, None]
            states[..., 2] = np.arange(self.N, dtype=np.float32)[None, :]

            active = np.ones((self.E, self.N), dtype=np.float32)
            if t >= 3:
                active[0, 0] = 0.0  # synchronized-rollout padding for one trajectory

            actions, log_probs, values, beliefs = self.learner.act_batch(
                sf,
                entities,
                mask,
                meta,
                states,
                active,
                belief_states=beliefs,
                deterministic=False,
            )
            rewards = np.zeros((self.E, self.N), dtype=np.float32)
            rewards[:, :,] = 0.1 * (t + 1)
            rewards[1, 0] += 0.2
            rewards[0, 1] -= 0.1
            rewards *= active

            dones = np.zeros((self.E, self.N), dtype=np.float32)
            if t == 2:
                dones[0, 0] = 1.0
            if t == self.T - 1:
                dones[active > 0.5] = 1.0

            buffer.add(
                sf,
                entities,
                mask,
                meta,
                states,
                actions,
                log_probs,
                rewards,
                dones,
                values,
                active,
            )
        return buffer

    def test_critic_remains_original_centralized_mlp(self):
        self.assertTrue(all(isinstance(critic, MLPCritic) for critic in self.learner.critics))

    def test_buffer_preserves_time_environment_agent_axes(self):
        buffer = self._build_rollout()
        data = buffer.as_arrays()
        self.assertEqual(data["self_features"].shape, (self.T, self.E, self.N, 16))
        self.assertEqual(data["entity_features"].shape, (self.T, self.E, self.N, 14, 20))
        self.assertEqual(data["actions"].shape, (self.T, self.E, self.N))
        self.assertEqual(data["active"].shape, (self.T, self.E, self.N))

    def test_preupdate_replay_ratio_is_one(self):
        buffer = self._build_rollout()
        diagnostics = self.learner.replay_diagnostics(buffer)
        self.assertLess(diagnostics["max_abs_logprob_error"], 1e-5)
        self.assertLess(diagnostics["max_abs_ratio_error"], 1e-5)
        self.assertAlmostEqual(diagnostics["ratio_mean"], 1.0, places=5)

    def test_one_sequence_ppo_update_is_finite_and_changes_policy(self):
        buffer = self._build_rollout()
        before_diag = self.learner.replay_diagnostics(buffer)
        before = [p.detach().clone() for p in self.learner.actors[0].parameters()]

        metrics = self.learner.update_parallel(buffer)
        for value in metrics.values():
            self.assertTrue(math.isfinite(value))
        self.assertGreaterEqual(metrics["clip_fraction"], 0.0)
        self.assertLessEqual(metrics["clip_fraction"], 1.0)

        after = list(self.learner.actors[0].parameters())
        changed = any(not torch.allclose(a, b.detach()) for a, b in zip(before, after))
        self.assertTrue(changed)

        after_diag = self.learner.replay_diagnostics(buffer)
        self.assertLess(before_diag["max_abs_ratio_error"], 1e-5)
        self.assertGreater(after_diag["max_abs_ratio_error"], 1e-8)


if __name__ == "__main__":
    unittest.main()
