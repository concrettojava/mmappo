from __future__ import annotations

import unittest

import numpy as np

from fofe_mmapppo.envs import CooperativeUAVEnv
from fofe_mmapppo.models import DirectFixedVectorizer, FixedVectorizer


class DirectVectorTrainingPathTests(unittest.TestCase):
    def setUp(self):
        self.base = FixedVectorizer()
        self.direct = DirectFixedVectorizer(self.base)

    def _reference_vectors(self, env):
        observations = env.get_observations()
        states = env.get_global_state()
        obs = self.base.batch_observations(observations)
        state = self.base.batch_states(states)
        active = np.asarray(
            [observations[i] is not None for i in range(self.base.n_uavs)],
            dtype=np.float32,
        )
        return obs, state, active

    def test_reset_vectors_match_structured_reference(self):
        ref_env = CooperativeUAVEnv(seed=123)
        fast_env = CooperativeUAVEnv(seed=123)
        ref_env.reset(seed=123)
        fast_obs, fast_state, fast_active = fast_env.reset_vectors(self.direct, seed=123)
        ref_obs, ref_state, ref_active = self._reference_vectors(ref_env)

        np.testing.assert_allclose(fast_obs, ref_obs, rtol=0.0, atol=1e-6)
        np.testing.assert_allclose(fast_state, ref_state, rtol=0.0, atol=1e-6)
        np.testing.assert_array_equal(fast_active, ref_active)

    def test_step_vectors_match_structured_reference(self):
        ref_env = CooperativeUAVEnv(seed=321)
        fast_env = CooperativeUAVEnv(seed=321)
        ref_env.reset(seed=321)
        fast_env.reset_vectors(self.direct, seed=321)
        rng = np.random.default_rng(99)

        for _ in range(12):
            actions = rng.integers(0, 7, size=8)
            ref_obs_raw, ref_state_raw, ref_rewards, ref_done, ref_info = ref_env.step(actions)
            fast_obs, fast_state, fast_active, fast_rewards, fast_done, fast_info = fast_env.step_vectors(
                actions, self.direct
            )

            ref_obs = self.base.batch_observations(ref_obs_raw)
            ref_state = self.base.batch_states(ref_state_raw)
            ref_active = np.asarray(
                [ref_obs_raw[i] is not None for i in range(8)], dtype=np.float32
            )

            np.testing.assert_allclose(fast_obs, ref_obs, rtol=0.0, atol=1e-6)
            np.testing.assert_allclose(fast_state, ref_state, rtol=0.0, atol=1e-6)
            np.testing.assert_array_equal(fast_active, ref_active)
            np.testing.assert_allclose(
                [fast_rewards[i] for i in range(8)],
                [ref_rewards[i] for i in range(8)],
                rtol=0.0,
                atol=1e-8,
            )
            self.assertEqual(fast_done, ref_done)
            self.assertEqual(fast_info["step"], ref_info["step"])
            self.assertEqual(fast_info["completion_ratio"], ref_info["completion_ratio"])
            self.assertEqual(fast_info["survival_ratio"], ref_info["survival_ratio"])
            if ref_done:
                break


if __name__ == "__main__":
    unittest.main()
