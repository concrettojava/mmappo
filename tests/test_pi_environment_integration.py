from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from fofe_mmapppo.algorithms import PIParallelRolloutBuffer, PIMAPPO, PIMAPPOConfig
from fofe_mmapppo.envs import CooperativeUAVEnv
from fofe_mmapppo.models import EntityTensorizer, FixedVectorizer, PIActorConfig


class PIEnvironmentIntegrationTests(unittest.TestCase):
    """Exercise the recurrent PI policy against the real UAV environment API.

    These tests intentionally use very short episodes and small networks.  They
    are not performance tests; their purpose is to catch interface/temporal
    errors that synthetic tensor tests cannot reveal.
    """

    def setUp(self):
        torch.manual_seed(31)
        np.random.seed(31)
        self.n_envs = 2
        self.n_agents = 8
        self.max_steps = 6
        self.fixed = FixedVectorizer()
        self.tensorizer = EntityTensorizer()
        self.actor_cfg = PIActorConfig(
            n_hypotheses=3,
            horizon=2,
            belief_dim=8,
            context_dim=8,
            relation_dim=8,
        )
        self.ppo_cfg = PIMAPPOConfig(
            ppo_epochs=1,
            learning_rate=2e-4,
            hidden_dim=16,
            entropy_coef=0.0,
            sequence_env_minibatch_size=2,
            fused_adam=False,
        )

    def _learner(self) -> PIMAPPO:
        return PIMAPPO(
            self.fixed.state_dim,
            n_agents=self.n_agents,
            config=self.ppo_cfg,
            actor_config=self.actor_cfg,
            device="cpu",
        )

    def _collect(self, learner: PIMAPPO, scenario: str) -> tuple[PIParallelRolloutBuffer, list[dict]]:
        envs = [
            CooperativeUAVEnv(
                seed=700 + e,
                scenario=scenario,
                max_steps=self.max_steps,
            )
            for e in range(self.n_envs)
        ]
        observations = []
        global_states = []
        for env in envs:
            obs, states = env.reset()
            observations.append(obs)
            global_states.append(states)

        finished = np.zeros(self.n_envs, dtype=bool)
        beliefs = learner.initial_belief_states(self.n_envs)
        buffer = PIParallelRolloutBuffer(
            n_envs=self.n_envs,
            n_agents=self.n_agents,
        )
        final_infos: list[dict | None] = [None] * self.n_envs

        while not bool(finished.all()):
            structured = self.tensorizer.parallel(observations, finished=finished)
            state_vectors = np.zeros(
                (self.n_envs, self.n_agents, self.fixed.state_dim), dtype=np.float32
            )
            for e in range(self.n_envs):
                if not finished[e]:
                    state_vectors[e] = self.fixed.batch_states(global_states[e])

            active = structured.active.astype(np.float32, copy=False)
            actions, log_probs, values, beliefs = learner.act_batch(
                structured.self_features,
                structured.entity_features,
                structured.evidence_mask,
                structured.evidence_meta,
                state_vectors,
                active,
                belief_states=beliefs,
                deterministic=False,
            )

            reward_batch = np.zeros((self.n_envs, self.n_agents), dtype=np.float32)
            done_batch = np.ones((self.n_envs, self.n_agents), dtype=np.float32)
            next_observations = list(observations)
            next_states = list(global_states)

            for e, env in enumerate(envs):
                if finished[e]:
                    continue
                next_obs, next_state, rewards, done, info = env.step(actions[e])
                reward_batch[e] = np.fromiter(
                    (rewards[i] for i in range(self.n_agents)),
                    dtype=np.float32,
                    count=self.n_agents,
                )
                done_batch[e] = np.asarray(
                    [
                        1.0 if done or next_obs[i] is None else 0.0
                        for i in range(self.n_agents)
                    ],
                    dtype=np.float32,
                )
                next_observations[e] = next_obs
                next_states[e] = next_state
                if done:
                    finished[e] = True
                    final_infos[e] = info

            buffer.add(
                structured.self_features,
                structured.entity_features,
                structured.evidence_mask,
                structured.evidence_meta,
                state_vectors,
                actions,
                log_probs,
                reward_batch,
                done_batch,
                values,
                active,
            )
            observations = next_observations
            global_states = next_states

        return buffer, [info for info in final_infos if info is not None]

    def _assert_buffer_sane(self, buffer: PIParallelRolloutBuffer) -> None:
        data = buffer.as_arrays()
        self.assertEqual(data["self_features"].shape[0], self.max_steps)
        self.assertEqual(data["self_features"].shape[1:3], (self.n_envs, self.n_agents))
        self.assertEqual(data["states"].shape[-1], self.fixed.state_dim)
        for key, value in data.items():
            self.assertTrue(np.isfinite(value).all(), msg=f"non-finite values in {key}")
        self.assertTrue((data["actions"] >= 0).all())
        self.assertTrue((data["actions"] < 7).all())
        self.assertGreater(float(data["active"].sum()), 0.0)

    def test_reference_environment_rollout_replays_exactly(self):
        learner = self._learner()
        buffer, infos = self._collect(learner, "reference")
        self._assert_buffer_sane(buffer)
        self.assertEqual(len(infos), self.n_envs)
        self.assertTrue(all(info["scenario"] == "reference" for info in infos))
        diagnostics = learner.replay_diagnostics(buffer)
        self.assertLess(diagnostics["max_abs_logprob_error"], 1e-5)
        self.assertLess(diagnostics["max_abs_ratio_error"], 1e-5)
        self.assertAlmostEqual(diagnostics["ratio_mean"], 1.0, places=5)

    def test_contested_environment_rollout_replays_exactly(self):
        learner = self._learner()
        buffer, infos = self._collect(learner, "contested")
        self._assert_buffer_sane(buffer)
        self.assertEqual(len(infos), self.n_envs)
        self.assertTrue(all(info["scenario"] == "contested" for info in infos))
        diagnostics = learner.replay_diagnostics(buffer)
        self.assertLess(diagnostics["max_abs_logprob_error"], 1e-5)
        self.assertLess(diagnostics["max_abs_ratio_error"], 1e-5)

    def test_real_contested_rollout_supports_one_sequence_ppo_update(self):
        learner = self._learner()
        buffer, _ = self._collect(learner, "contested")
        before = learner.replay_diagnostics(buffer)
        self.assertLess(before["max_abs_ratio_error"], 1e-5)

        params_before = [p.detach().clone() for p in learner.actors[0].parameters()]
        metrics = learner.update_parallel(buffer)
        for key, value in metrics.items():
            self.assertTrue(math.isfinite(value), msg=f"non-finite metric {key}={value}")
        self.assertGreaterEqual(metrics["clip_fraction"], 0.0)
        self.assertLessEqual(metrics["clip_fraction"], 1.0)

        params_after = list(learner.actors[0].parameters())
        self.assertTrue(
            any(
                not torch.allclose(old, new.detach())
                for old, new in zip(params_before, params_after)
            )
        )
        after = learner.replay_diagnostics(buffer)
        self.assertGreater(after["max_abs_ratio_error"], 1e-8)


if __name__ == "__main__":
    unittest.main()
