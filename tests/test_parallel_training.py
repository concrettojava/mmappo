from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from fofe_mmapppo.algorithms import MAPPO, MAPPOConfig, ParallelRolloutBuffer
from fofe_mmapppo.envs import CooperativeUAVEnv
from fofe_mmapppo.models import FixedVectorizer


class ParallelGAETests(unittest.TestCase):
    def test_env_trajectories_do_not_leak_into_each_other(self):
        b = ParallelRolloutBuffer(n_envs=2, n_agents=1)
        dummy_obs = np.zeros((2, 1, 1), dtype=np.float32)
        dummy_state = np.zeros((2, 1, 1), dtype=np.float32)
        actions = np.zeros((2, 1), dtype=np.int64)
        logp = np.zeros((2, 1), dtype=np.float32)
        values = np.zeros((2, 1), dtype=np.float32)
        active = np.ones((2, 1), dtype=np.float32)

        # env0 terminates immediately with reward 1; env1 continues and gets 10.
        b.add(dummy_obs, dummy_state, actions, logp,
              np.array([[1.0], [0.0]], dtype=np.float32),
              np.array([[1.0], [0.0]], dtype=np.float32), values, active)
        b.add(dummy_obs, dummy_state, actions, logp,
              np.array([[0.0], [10.0]], dtype=np.float32),
              np.array([[1.0], [1.0]], dtype=np.float32), values,
              np.array([[0.0], [1.0]], dtype=np.float32))
        data = b.compute_gae(gamma=1.0, gae_lambda=1.0)
        self.assertAlmostEqual(float(data["returns"][0, 0, 0]), 1.0)
        self.assertAlmostEqual(float(data["returns"][0, 1, 0]), 10.0)


class BatchedMAPPOTests(unittest.TestCase):
    def setUp(self):
        self.envs = [CooperativeUAVEnv(seed=10 + i) for i in range(3)]
        reset = [env.reset() for env in self.envs]
        self.obs = [x[0] for x in reset]
        self.states = [x[1] for x in reset]
        self.vec = FixedVectorizer(n_uavs=8, n_targets=4, n_threats=3)
        self.obs_vec = np.stack([self.vec.batch_observations(x) for x in self.obs])
        self.state_vec = np.stack([self.vec.batch_states(x) for x in self.states])
        self.active = np.stack([
            np.asarray([x[i] is not None for i in range(8)], dtype=np.float32)
            for x in self.obs
        ])
        self.learner = MAPPO(
            self.vec.observation_dim, self.vec.state_dim,
            n_agents=8, action_dim=7,
            config=MAPPOConfig(ppo_epochs=1, minibatch_size=32),
            device="cpu",
        )

    def test_batched_inference_shapes(self):
        actions, logp, values = self.learner.act_batch(
            self.obs_vec, self.state_vec, self.active
        )
        self.assertEqual(actions.shape, (3, 8))
        self.assertEqual(logp.shape, (3, 8))
        self.assertEqual(values.shape, (3, 8))
        self.assertTrue(np.all((actions >= 0) & (actions < 7)))

    def test_one_parallel_update_is_finite(self):
        actions, logp, values = self.learner.act_batch(
            self.obs_vec, self.state_vec, self.active
        )
        reward = np.zeros((3, 8), dtype=np.float32)
        dones = np.ones((3, 8), dtype=np.float32)
        buffer = ParallelRolloutBuffer(n_envs=3, n_agents=8)
        buffer.add(
            self.obs_vec, self.state_vec, actions, logp,
            reward, dones, values, self.active,
        )
        losses = self.learner.update_parallel(buffer)
        for value in losses.values():
            self.assertTrue(math.isfinite(value))


if __name__ == "__main__":
    unittest.main()
