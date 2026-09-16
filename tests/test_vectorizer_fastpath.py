from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from fofe_mmapppo.envs import CooperativeUAVEnv
from fofe_mmapppo.models import FixedVectorizer


class FastVectorizerTests(unittest.TestCase):
    def test_into_matches_public_single_env_api(self):
        env = CooperativeUAVEnv(seed=123)
        observations, states = env.reset(seed=123)
        vec = FixedVectorizer(n_uavs=8, n_targets=4, n_threats=3)

        for i in range(8):
            obs_expected = vec.observation(observations[i])
            obs_out = np.empty(vec.observation_dim, dtype=np.float32)
            vec.observation_into(observations[i], obs_out)
            np.testing.assert_array_equal(obs_out, obs_expected)

            state_expected = vec.state(states[i])
            state_out = np.empty(vec.state_dim, dtype=np.float32)
            vec.state_into(states[i], state_out)
            np.testing.assert_array_equal(state_out, state_expected)

    def test_parallel_batch_matches_per_environment_batches(self):
        envs = [CooperativeUAVEnv(seed=20 + i) for i in range(4)]
        reset = [env.reset(seed=20 + i) for i, env in enumerate(envs)]
        observations = [item[0] for item in reset]
        states = [item[1] for item in reset]
        vec = FixedVectorizer(n_uavs=8, n_targets=4, n_threats=3)

        expected_obs = np.stack([vec.batch_observations(x) for x in observations])
        expected_states = np.stack([vec.batch_states(x) for x in states])
        expected_active = np.ones((4, 8), dtype=np.float32)

        obs, state, active = vec.parallel_batch(observations, states)
        np.testing.assert_array_equal(obs, expected_obs)
        np.testing.assert_array_equal(state, expected_states)
        np.testing.assert_array_equal(active, expected_active)

        finished = np.array([False, True, False, True])
        obs, state, active = vec.parallel_batch(observations, states, finished)
        self.assertTrue(np.all(obs[1] == 0.0))
        self.assertTrue(np.all(state[1] == 0.0))
        self.assertTrue(np.all(active[1] == 0.0))
        self.assertTrue(np.all(obs[3] == 0.0))
        self.assertTrue(np.all(state[3] == 0.0))
        self.assertTrue(np.all(active[3] == 0.0))


if __name__ == "__main__":
    unittest.main()
