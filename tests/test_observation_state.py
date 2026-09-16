"""Rule-level tests for Eq. (8)/(9) observation and state construction."""
from __future__ import annotations

import math
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fofe_mmapppo.envs import CooperativeUAVEnv
from fofe_mmapppo.envs.coordinates import body_pose


class CoordinateTests(unittest.TestCase):
    def setUp(self):
        self.env = CooperativeUAVEnv(seed=7)
        self.env.reset()

    def test_north_zero_body_axes(self):
        u0 = self.env.uavs[0]  # (1700, 200), yaw=0 (North)
        u1 = self.env.uavs[1]  # 200 m East of U0
        pose = body_pose(u0, u1.x, u1.y, u1.yaw)
        self.assertAlmostEqual(pose["x"], 0.0, places=7)
        self.assertAlmostEqual(pose["y"], 200.0, places=7)
        self.assertAlmostEqual(pose["range"], 200.0, places=7)
        self.assertAlmostEqual(pose["bearing"], math.pi / 2.0, places=7)

    def test_self_body_pose_is_zero(self):
        u0 = self.env.uavs[0]
        pose = body_pose(u0, u0.x, u0.y, u0.yaw)
        self.assertAlmostEqual(pose["x"], 0.0)
        self.assertAlmostEqual(pose["y"], 0.0)
        self.assertAlmostEqual(pose["yaw"], 0.0)
        self.assertAlmostEqual(pose["range"], 0.0)
        self.assertAlmostEqual(pose["bearing"], 0.0)


class ObservationTests(unittest.TestCase):
    def setUp(self):
        self.env = CooperativeUAVEnv(seed=7)
        self.env.reset()

    def test_observation_has_four_flexible_channels(self):
        obs = self.env.get_observations()[0]
        self.assertEqual(set(obs), {"self", "neighbors", "targets", "threats"})
        self.assertEqual(obs["self"]["idx"], 0)
        self.assertIn("geo", obs["self"]["pose"])
        self.assertIn("body", obs["self"]["pose"])

    def test_multihop_subgroup_shares_target_detection(self):
        # Put M0 inside U4's large Rec range but outside U0's direct Stk range.
        u0 = self.env.uavs[0]
        u4 = self.env.uavs[4]
        target = self.env.targets[0]
        target.x = u4.x
        target.y = u4.y + 500.0
        target.yaw = 0.0

        self.assertGreater(self.env._dist(u0, target), u0.p["recon_range"])
        self.assertLess(self.env._dist(u4, target), u4.p["recon_range"])

        obs0 = self.env.get_observations()[0]
        self.assertIn(0, {item["idx"] for item in obs0["targets"]})

    def test_disconnected_agent_does_not_receive_target(self):
        # Move U0 far away and isolate it.  Put M0 near U4 only.
        u0 = self.env.uavs[0]
        u0.x, u0.y = 0.0, 0.0
        u4 = self.env.uavs[4]
        target = self.env.targets[0]
        target.x = u4.x
        target.y = u4.y + 400.0
        target.yaw = 0.0

        self.assertEqual(self.env.communication_components()[0], {0})
        obs0 = self.env.get_observations()[0]
        self.assertNotIn(0, {item["idx"] for item in obs0["targets"]})

    def test_threat_memory_survives_loss_of_direct_detection(self):
        u0 = self.env.uavs[0]
        threat = self.env.threats[0]
        threat.x, threat.y = u0.x, u0.y + 100.0
        self.env.get_observations()  # discover and remember
        self.assertIn(0, u0.threat_memory)

        threat.x, threat.y = 3900.0, 3900.0
        obs0 = self.env.get_observations()[0]
        self.assertIn(0, {item["idx"] for item in obs0["threats"]})


class GlobalStateTests(unittest.TestCase):
    def setUp(self):
        self.env = CooperativeUAVEnv(seed=7)
        self.env.reset()

    def test_global_state_removes_observability_constraints(self):
        states = self.env.get_global_state()
        self.assertEqual(set(states), set(range(8)))
        state0 = states[0]
        self.assertEqual(len(state0["uavs"]), 8)
        self.assertEqual(len(state0["neighbors"]), 7)
        self.assertEqual(len(state0["targets"]), 4)
        self.assertEqual(len(state0["threats"]), 3)

    def test_global_state_is_agent_relative(self):
        states = self.env.get_global_state()
        # The same world target has different body coordinates from two UAVs.
        t0_from_u0 = states[0]["targets"][0]["pose"]["body"]
        t0_from_u1 = states[1]["targets"][0]["pose"]["body"]
        self.assertNotEqual((t0_from_u0["x"], t0_from_u0["y"]),
                            (t0_from_u1["x"], t0_from_u1["y"]))


if __name__ == "__main__":
    unittest.main()
