import unittest

import numpy as np

from fofe_mmapppo.envs import CooperativeUAVEnv
from fofe_mmapppo.models import DirectFixedVectorizer, FixedVectorizer


class ContestedScenarioTests(unittest.TestCase):
    def test_reference_scenario_is_unchanged(self):
        env = CooperativeUAVEnv(seed=7, scenario="reference")
        env.reset(seed=7)
        self.assertEqual(env.jammers, [])
        for u in env.uavs:
            self.assertEqual(env.communication_factor(u), 1.0)
            self.assertEqual(env.reconnaissance_factor(u), 1.0)

    def test_contested_scenario_spawns_deterministic_jammers(self):
        a = CooperativeUAVEnv(seed=11, scenario="contested")
        b = CooperativeUAVEnv(seed=11, scenario="contested")
        a.reset(seed=11)
        b.reset(seed=11)
        self.assertEqual(len(a.jammers), 2)
        self.assertEqual(a.jammers, b.jammers)
        for j in a.jammers:
            self.assertGreaterEqual(j["radius"], a.jammer_min_radius)
            self.assertLessEqual(j["radius"], a.jammer_max_radius)
            self.assertGreaterEqual(j["strength"], a.jammer_min_strength)
            self.assertLessEqual(j["strength"], a.jammer_max_strength)

    def test_jammer_field_reduces_link_and_sensor_factors_near_source(self):
        env = CooperativeUAVEnv(seed=5, scenario="contested")
        env.reset(seed=5)
        jammer = env.jammers[0]
        u = env.uavs[0]
        u.x, u.y = jammer["x"], jammer["y"]
        self.assertLess(env.communication_factor(u), 1.0)
        self.assertLess(env.reconnaissance_factor(u), 1.0)
        self.assertGreaterEqual(env.communication_factor(u), env.contested_min_comm_factor)
        self.assertGreaterEqual(env.reconnaissance_factor(u), env.contested_min_recon_factor)

    def test_only_self_observation_exposes_local_equipment_quality(self):
        env = CooperativeUAVEnv(seed=5, scenario="contested")
        env.reset(seed=5)
        jammer = env.jammers[0]
        u = env.uavs[0]
        u.x, u.y = jammer["x"], jammer["y"]

        self_record = env.get_observations()[u.idx]["self"]
        self.assertEqual(self_record["comm_quality"], env.communication_factor(u))
        self.assertEqual(self_record["recon_quality"], env.reconnaissance_factor(u))
        for neighbor in env.get_observations()[u.idx]["neighbors"]:
            self.assertNotIn("comm_quality", neighbor)
            self.assertNotIn("recon_quality", neighbor)

    def test_direct_vectorizer_matches_structured_path_in_contested_scenario(self):
        env = CooperativeUAVEnv(seed=19, scenario="contested")
        base = FixedVectorizer()
        direct = DirectFixedVectorizer(base)

        obs, states = env.reset(seed=19)
        obs_ref = base.batch_observations(obs)
        state_ref = base.batch_states(states)

        env2 = CooperativeUAVEnv(seed=19, scenario="contested")
        obs_fast, state_fast, active_fast = env2.reset_vectors(direct, seed=19)

        np.testing.assert_allclose(obs_fast, obs_ref, rtol=0.0, atol=1e-6)
        np.testing.assert_allclose(state_fast, state_ref, rtol=0.0, atol=1e-6)
        np.testing.assert_array_equal(active_fast, np.ones(8, dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
