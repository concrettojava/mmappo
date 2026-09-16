import unittest

import numpy as np

from fofe_mmapppo.envs import CooperativeUAVEnv
from fofe_mmapppo.envs.batch_geometry import install_geometry_cache


class TestBatchGeometryCache(unittest.TestCase):
    def test_cached_distances_match_reference(self):
        envs = [CooperativeUAVEnv(seed=10 + i) for i in range(4)]
        for i, env in enumerate(envs):
            env.reset(seed=10 + i)
            env.prepare_step(np.full(8, 3, dtype=np.int64))
        install_geometry_cache(envs)

        for env in envs:
            for a in env.uavs:
                for b in env.uavs:
                    expected = float(np.hypot(a.x - b.x, a.y - b.y))
                    self.assertAlmostEqual(env._dist(a, b), expected, places=10)
            for a in env.uavs:
                for t in env.targets:
                    expected = float(np.hypot(a.x - t.x, a.y - t.y))
                    self.assertAlmostEqual(env._dist(a, t), expected, places=10)
                for th in env.threats:
                    expected = float(np.hypot(a.x - th.x, a.y - th.y))
                    self.assertAlmostEqual(env._dist(a, th), expected, places=10)

    def test_split_batch_step_matches_reference(self):
        seeds = [31, 32, 33, 34]
        refs = [CooperativeUAVEnv(seed=s) for s in seeds]
        bats = [CooperativeUAVEnv(seed=s) for s in seeds]
        for env, s in zip(refs, seeds):
            env.reset(seed=s)
        for env, s in zip(bats, seeds):
            env.reset(seed=s)

        rng = np.random.default_rng(123)
        for _ in range(12):
            actions = rng.integers(0, 7, size=(len(seeds), 8), dtype=np.int64)
            reference_results = [env.step(actions[i]) for i, env in enumerate(refs)]

            contexts = [env.prepare_step(actions[i]) for i, env in enumerate(bats)]
            install_geometry_cache(bats)
            batched_results = [env.finish_step(contexts[i]) for i, env in enumerate(bats)]

            for ref_env, bat_env, ref_result, bat_result in zip(
                refs, bats, reference_results, batched_results
            ):
                ref_obs, ref_state, ref_rewards, ref_done, ref_info = ref_result
                bat_obs, bat_state, bat_rewards, bat_done, bat_info = bat_result
                self.assertEqual(ref_done, bat_done)
                self.assertEqual(ref_info["alive_uavs"], bat_info["alive_uavs"])
                self.assertEqual(ref_info["alive_targets"], bat_info["alive_targets"])
                for i in range(8):
                    self.assertAlmostEqual(ref_rewards[i], bat_rewards[i], places=8)
                for ru, bu in zip(ref_env.uavs, bat_env.uavs):
                    self.assertEqual(ru.alive, bu.alive)
                    self.assertAlmostEqual(ru.x, bu.x, places=12)
                    self.assertAlmostEqual(ru.y, bu.y, places=12)
                    self.assertAlmostEqual(ru.yaw, bu.yaw, places=12)
                    self.assertEqual(ru.threat_memory, bu.threat_memory)
                for rt, bt in zip(ref_env.targets, bat_env.targets):
                    self.assertEqual(rt.alive, bt.alive)
                    self.assertAlmostEqual(rt.x, bt.x, places=12)
                    self.assertAlmostEqual(rt.y, bt.y, places=12)
                    self.assertAlmostEqual(rt.yaw, bt.yaw, places=12)

                if ref_done:
                    # Reinitialize both sides identically if a mission happened
                    # to terminate before the fixed test horizon.
                    seed = seeds[refs.index(ref_env)] + 1000
                    ref_env.reset(seed=seed)
                    bat_env.reset(seed=seed)


if __name__ == "__main__":
    unittest.main()
