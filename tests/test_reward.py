from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from fofe_mmapppo.envs import CooperativeUAVEnv
from fofe_mmapppo.envs.reward import (
    ABILITY_RATIOS,
    RewardContext,
    action_reward,
    boundary_reward,
    build_reward_context,
    general_reward,
    mission_reward,
    strike_reward,
)


class GeneralRewardTests(unittest.TestCase):
    def test_eq11_reference_points(self):
        self.assertAlmostEqual(general_reward(0.0, 100.0), -1.0)
        self.assertEqual(general_reward(100.0, 100.0), 0.0)
        self.assertEqual(general_reward(101.0, 100.0), 0.0)
        self.assertEqual(general_reward(1.0, 0.0), 0.0)

    def test_eq11_is_clipped_at_minus_ten(self):
        self.assertEqual(general_reward(-10000.0, 100.0), -10.0)


class CoefficientTests(unittest.TestCase):
    def test_heterogeneous_ability_ratios_match_paper(self):
        self.assertEqual(ABILITY_RATIOS["Stk"], {"Stk": 0.7, "Rec": 0.2, "Com": 0.1})
        self.assertEqual(ABILITY_RATIOS["Rec"], {"Stk": 0.1, "Rec": 0.7, "Com": 0.2})
        self.assertEqual(ABILITY_RATIOS["Com"], {"Stk": 0.2, "Rec": 0.1, "Com": 0.7})
        for ratios in ABILITY_RATIOS.values():
            self.assertAlmostEqual(sum(ratios.values()), 1.0)


class MissionBoundaryActionTests(unittest.TestCase):
    def setUp(self):
        self.env = CooperativeUAVEnv(seed=7)
        self.env.reset()

    def test_initial_mission_reward_is_zero_at_t0(self):
        self.assertAlmostEqual(mission_reward(self.env), 0.0)

    def test_destroying_target_increases_mission_component(self):
        before = mission_reward(self.env)
        self.env.targets[0].alive = False
        after = mission_reward(self.env)
        self.assertGreater(after, before)

    def test_losing_uav_decreases_mission_component(self):
        before = mission_reward(self.env)
        self.env.uavs[0].alive = False
        after = mission_reward(self.env)
        self.assertLess(after, before)

    def test_boundary_penalty_only_near_boundary(self):
        u = self.env.uavs[0]
        u.x = 2000.0
        u.y = 2000.0
        self.assertAlmostEqual(boundary_reward(self.env, u), 0.0)
        u.x = 0.0
        self.assertAlmostEqual(boundary_reward(self.env, u), -1.0)

    def test_action_reward_eq15(self):
        u = self.env.uavs[0]
        u.last_action_u = 1.0
        self.assertAlmostEqual(action_reward(u, previous_u=0.0), -1.5)
        u.last_action_u = 0.0
        self.assertAlmostEqual(action_reward(u, previous_u=0.0), 0.0)


class AbilityRewardTests(unittest.TestCase):
    def setUp(self):
        self.env = CooperativeUAVEnv(seed=0)
        self.env.reset()

    def test_context_marks_target_in_strike_sector(self):
        # Isolate U0 and put M0 200 m directly North.  yaw=0 means North in
        # the validated scene convention.
        for u in self.env.uavs[1:]:
            u.alive = False
        for t in self.env.targets[1:]:
            t.alive = False

        u = self.env.uavs[0]
        t = self.env.targets[0]
        u.x, u.y, u.yaw = 1000.0, 1000.0, 0.0
        t.x, t.y, t.yaw = 1000.0, 1200.0, 0.0

        ctx = build_reward_context(self.env, {u.idx: 0.0})
        self.assertEqual(ctx.strikeable_targets[u.idx], {t.idx})

    def test_strike_effectiveness_term_is_present_pre_destruction(self):
        for u in self.env.uavs[1:]:
            u.alive = False
        for t in self.env.targets[1:]:
            t.alive = False

        u = self.env.uavs[0]
        t = self.env.targets[0]
        u.x, u.y, u.yaw = 1000.0, 1000.0, 0.0
        t.x, t.y = 1000.0, 1200.0

        ctx = build_reward_context(self.env, {u.idx: 0.0})
        # -lambda_dist*200 + 10*1, with no same-type peer alive.
        self.assertAlmostEqual(strike_reward(self.env, u, ctx), 9.9, places=6)


class EnvironmentIntegrationTests(unittest.TestCase):
    def test_step_returns_per_agent_reward_and_breakdown(self):
        env = CooperativeUAVEnv(seed=42)
        env.reset()
        result = env.step(np.full(len(env.uavs), 3, dtype=int))
        self.assertEqual(len(result), 5)
        obs, state, rewards, done, info = result
        self.assertEqual(set(rewards), set(range(8)))
        self.assertEqual(set(info["reward_breakdown"]), set(range(8)))
        for uid, value in rewards.items():
            self.assertTrue(math.isfinite(value), uid)
            self.assertAlmostEqual(value, info["reward_breakdown"][uid]["total"])
        self.assertIsInstance(done, bool)
        self.assertEqual(set(obs), set(range(8)))
        self.assertEqual(set(state), set(range(8)))


if __name__ == "__main__":
    unittest.main()
