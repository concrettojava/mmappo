from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from fofe_mmapppo.algorithms import MAPPO, MAPPOConfig, RolloutBuffer
from fofe_mmapppo.envs import CooperativeUAVEnv
from fofe_mmapppo.models import FixedVectorizer


class VectorizerTests(unittest.TestCase):
    def setUp(self):
        self.env = CooperativeUAVEnv(seed=7)
        self.obs, self.states = self.env.reset()
        self.vec = FixedVectorizer()

    def test_fixed_dimensions(self):
        self.assertEqual(self.vec.observation_dim, 182)
        self.assertEqual(self.vec.state_dim, 280)
        obs = self.vec.batch_observations(self.obs)
        states = self.vec.batch_states(self.states)
        self.assertEqual(obs.shape, (8, 182))
        self.assertEqual(states.shape, (8, 280))
        self.assertTrue(np.isfinite(obs).all())
        self.assertTrue(np.isfinite(states).all())

    def test_missing_local_objects_are_zero_padded(self):
        # Isolate U0 so it has no neighbors and move all targets/threats far away.
        for u in self.env.uavs[1:]:
            u.x, u.y = 3900.0, 3900.0
        u0 = self.env.uavs[0]
        u0.x, u0.y = 100.0, 100.0
        for target in self.env.targets:
            target.x, target.y = 3500.0, 3500.0
        for threat in self.env.threats:
            threat.x, threat.y = 3500.0, 3500.0
        obs = self.env.get_observations()[0]
        vector = self.vec.observation(obs)
        # Self record is present; at least all padded record-presence bits are zero.
        self.assertEqual(vector[0], 1.0)
        neighbor_start = self.vec.uav_dim
        for slot in range(self.vec.n_uavs - 1):
            self.assertEqual(vector[neighbor_start + slot * self.vec.uav_dim], 0.0)

    def test_dead_agent_vector_is_all_zero(self):
        self.env.uavs[0].alive = False
        obs = self.env.get_observations()
        vector = self.vec.observation(obs[0])
        self.assertTrue(np.array_equal(vector, np.zeros(self.vec.observation_dim, dtype=np.float32)))


class GAETests(unittest.TestCase):
    def test_terminal_transition_does_not_bootstrap(self):
        buffer = RolloutBuffer(n_agents=1)
        zobs = np.zeros((1, 2), dtype=np.float32)
        zstate = np.zeros((1, 2), dtype=np.float32)
        for reward, done, value in [(1.0, 0.0, 0.5), (2.0, 1.0, 0.25)]:
            buffer.add(
                zobs, zstate,
                np.array([0]), np.array([0.0]), np.array([reward]),
                np.array([done]), np.array([value]), np.array([1.0]),
            )
        data = buffer.compute_gae(gamma=1.0, gae_lambda=1.0, last_values=np.array([99.0]))
        # At t=1 done=1, so the fake bootstrap value 99 must be ignored.
        self.assertAlmostEqual(float(data["returns"][1, 0]), 2.0, places=6)
        self.assertAlmostEqual(float(data["returns"][0, 0]), 3.0, places=6)


class MAPPOSmokeTests(unittest.TestCase):
    def test_actor_critic_action_shapes_and_one_update(self):
        torch.manual_seed(0)
        env = CooperativeUAVEnv(seed=0)
        obs, states = env.reset()
        vec = FixedVectorizer()
        learner = MAPPO(
            vec.observation_dim,
            vec.state_dim,
            config=MAPPOConfig(ppo_epochs=1, minibatch_size=8, hidden_dim=32),
            device="cpu",
        )
        buffer = RolloutBuffer(n_agents=8)

        for _ in range(3):
            obs_vec = vec.batch_observations(obs)
            state_vec = vec.batch_states(states)
            active = np.asarray([obs[i] is not None for i in range(8)], dtype=np.float32)
            actions, log_probs, values = learner.act(obs_vec, state_vec, active)
            self.assertEqual(actions.shape, (8,))
            self.assertTrue(((actions >= 0) & (actions < 7)).all())

            next_obs, next_states, rewards, done, _ = env.step(actions)
            reward_vec = np.asarray([rewards[i] for i in range(8)], dtype=np.float32)
            agent_dones = np.asarray([done or next_obs[i] is None for i in range(8)], dtype=np.float32)
            buffer.add(obs_vec, state_vec, actions, log_probs, reward_vec, agent_dones, values, active)
            obs, states = next_obs, next_states
            if done:
                break

        metrics = learner.update(buffer)
        for value in metrics.values():
            self.assertTrue(math.isfinite(value))


if __name__ == "__main__":
    unittest.main()
