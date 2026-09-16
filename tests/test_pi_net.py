from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from fofe_mmapppo.envs import CooperativeUAVEnv
from fofe_mmapppo.envs.coordinates import dual_pose
from fofe_mmapppo.models import EntityTensorizer, PIActor, PIActorConfig


class EntityTensorizerTests(unittest.TestCase):
    def setUp(self):
        self.env = CooperativeUAVEnv(seed=11, scenario="contested")
        self.obs, _ = self.env.reset()
        self.tensorizer = EntityTensorizer()

    def test_environment_shapes_and_finite_values(self):
        batch = self.tensorizer.environment(self.obs)
        self.assertEqual(batch.self_features.shape, (8, 14))
        self.assertEqual(batch.entity_features.shape, (8, 14, 20))
        self.assertEqual(batch.evidence_mask.shape, (8, 14))
        self.assertEqual(batch.evidence_meta.shape, (8, 14, 4))
        self.assertEqual(batch.active.shape, (8,))
        self.assertTrue(np.isfinite(batch.self_features).all())
        self.assertTrue(np.isfinite(batch.entity_features).all())
        self.assertTrue(np.isfinite(batch.evidence_meta).all())

    def test_target_identity_uses_stable_slot(self):
        observer_idx = 0
        observer = self.env.uavs[observer_idx]
        target = self.env.targets[2]
        self_record = self.obs[observer_idx]["self"]
        target_record = {
            "idx": 2,
            "alive": True,
            "pose": dual_pose(observer, target),
        }
        local = {
            "self": self_record,
            "neighbors": [],
            "targets": [target_record],
            "threats": [],
        }
        encoded = self.tensorizer.agent(local, observer_idx)
        expected_slot = self.tensorizer.target_offset + 2
        present = np.flatnonzero(encoded.evidence_mask > 0.5)
        self.assertEqual(present.tolist(), [expected_slot])
        self.assertEqual(encoded.entity_features[expected_slot, 2], 1.0)

    def test_optional_age_and_source_metadata_are_preserved(self):
        observer_idx = 0
        observer = self.env.uavs[observer_idx]
        target = self.env.targets[1]
        record = {
            "idx": 1,
            "alive": True,
            "age": 40.0,
            "source": "relayed",
            "pose": dual_pose(observer, target),
        }
        local = {
            "self": self.obs[observer_idx]["self"],
            "neighbors": [],
            "targets": [record],
            "threats": [],
        }
        encoded = self.tensorizer.agent(local, observer_idx)
        slot = self.tensorizer.target_offset + 1
        self.assertAlmostEqual(float(encoded.evidence_meta[slot, 0]), 0.2, places=6)
        self.assertTrue(
            np.array_equal(
                encoded.evidence_meta[slot, 1:4],
                np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
            )
        )


class PIActorTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)
        self.cfg = PIActorConfig(
            n_hypotheses=3,
            horizon=4,
            belief_dim=16,
            context_dim=8,
            relation_dim=16,
        )
        self.actor = PIActor(self.cfg)
        self.B = 2
        self.self_features = torch.zeros(self.B, 14)
        self.self_features[:, 0] = 1.0
        self.self_features[:, 2] = 1.0  # Stk-enhanced
        self.self_features[:, 5] = 1.0
        self.self_features[:, 6] = 0.50
        self.self_features[:, 7] = 0.10
        self.self_features[:, 8] = 0.0  # north

        self.entities = torch.zeros(self.B, 14, 20)
        self.mask = torch.zeros(self.B, 14)
        self.meta = torch.zeros(self.B, 14, 4)
        self.target_slot = 7
        self.entities[:, self.target_slot, 0] = 1.0
        self.entities[:, self.target_slot, 2] = 1.0  # target category
        self.entities[:, self.target_slot, 7] = 1.0
        self.entities[:, self.target_slot, 9] = 0.50
        self.entities[:, self.target_slot, 10] = 0.50
        self.entities[:, self.target_slot, 11] = 0.0
        self.entities[:, self.target_slot, 18] = 8.0 / 50.0
        self.entities[:, self.target_slot, 19] = 10.0 / 40.0
        self.mask[:, self.target_slot] = 1.0

    def test_forward_contract_and_probabilities(self):
        logits, state, aux = self.actor(
            self.self_features, self.entities, self.mask, self.meta
        )
        self.assertEqual(tuple(logits.shape), (self.B, 7))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertEqual(tuple(state.latent.shape), (self.B, 14, 3, 16))
        self.assertEqual(tuple(aux["future_position"].shape), (self.B, 14, 3, 4, 2))
        self.assertEqual(tuple(aux["action_position"].shape), (self.B, 7, 4, 2))
        self.assertEqual(tuple(aux["q_refresh"].shape), (self.B, 7, 14, 4))
        self.assertEqual(tuple(aux["q_refresh_hypothesis"].shape), (self.B, 7, 14, 3, 4))
        self.assertEqual(tuple(aux["uncertainty"].shape), (self.B, 7, 14, 4))
        self.assertEqual(tuple(aux["lattice"].shape), (self.B, 7, 14, 3, 4, 16))
        sums = aux["mode_probs"].sum(dim=-1)
        self.assertTrue(torch.allclose(sums, torch.ones_like(sums), atol=1e-6))
        self.assertTrue(((aux["q_refresh"] >= 0.0) & (aux["q_refresh"] <= 1.0)).all())

    def test_information_refresh_is_aggregated_after_hypothesis_prediction(self):
        _, _, aux = self.actor(
            self.self_features, self.entities, self.mask, self.meta
        )
        weights = aux["mode_probs"][:, None, :, :, None]
        expected_obs = (aux["q_obs_hypothesis"] * weights).sum(dim=3)
        expected_com = (aux["q_com_hypothesis"] * weights).sum(dim=3)
        expected_refresh = (aux["q_refresh_hypothesis"] * weights).sum(dim=3)
        self.assertTrue(torch.allclose(aux["q_obs"], expected_obs, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(aux["q_com"], expected_com, atol=1e-6, rtol=1e-6))
        self.assertTrue(
            torch.allclose(aux["q_refresh"], expected_refresh, atol=1e-6, rtol=1e-6)
        )

    def test_missing_evidence_ages_belief_and_fresh_evidence_resets_it(self):
        _, state1, _ = self.actor(
            self.self_features, self.entities, self.mask, self.meta
        )
        self.assertTrue(torch.allclose(state1.age[:, self.target_slot, 0], torch.zeros(self.B)))
        self.assertTrue(torch.all(state1.known[:, self.target_slot, 0] == 1.0))

        missing_entities = torch.zeros_like(self.entities)
        missing_mask = torch.zeros_like(self.mask)
        missing_meta = torch.zeros_like(self.meta)
        _, state2, _ = self.actor(
            self.self_features, missing_entities, missing_mask, missing_meta, state1
        )
        self.assertTrue(torch.allclose(state2.age[:, self.target_slot, 0], torch.ones(self.B)))
        self.assertTrue(torch.all(state2.known[:, self.target_slot, 0] == 1.0))

        _, state3, _ = self.actor(
            self.self_features, self.entities, self.mask, self.meta, state2
        )
        self.assertTrue(torch.allclose(state3.age[:, self.target_slot, 0], torch.zeros(self.B)))

    def test_stale_evidence_is_dead_reckoned_to_current_time_before_correction(self):
        stale_meta = self.meta.clone()
        stale_age = 5.0
        stale_meta[:, self.target_slot, 0] = stale_age / self.cfg.max_age
        _, state, _ = self.actor(
            self.self_features, self.entities, self.mask, stale_meta
        )
        # Target yaw=0 means north in this environment convention.  A 5-second
        # old observation at y=.50 with target speed 8m/s should be aligned to
        # y=.50 + 5*8/4000 = .51 before the first-seen correction.
        expected_y = 0.50 + stale_age * 8.0 / self.cfg.world_size
        target_position = state.position[:, self.target_slot]
        self.assertTrue(
            torch.allclose(
                target_position[..., 0],
                torch.full_like(target_position[..., 0], 0.50),
                atol=1e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                target_position[..., 1],
                torch.full_like(target_position[..., 1], expected_y),
                atol=1e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                state.age[:, self.target_slot, 0],
                torch.full((self.B,), stale_age),
                atol=1e-6,
            )
        )

    def test_straight_action_primitive_respects_north_heading(self):
        pos, _ = self.actor._action_future(self.self_features[:1])
        straight = pos[0, 3]
        self.assertTrue(torch.allclose(straight[:, 0], torch.full((4,), 0.50), atol=1e-6))
        expected_first_y = 0.10 + 40.0 / 4000.0
        self.assertAlmostEqual(float(straight[0, 1]), expected_first_y, places=6)
        self.assertGreater(float(straight[-1, 1]), float(straight[0, 1]))

    def test_forward_is_differentiable(self):
        logits, _, aux = self.actor(
            self.self_features, self.entities, self.mask, self.meta
        )
        loss = logits.square().mean() + 0.001 * aux["lattice"].square().mean()
        loss.backward()
        grads = [p.grad for p in self.actor.parameters() if p.requires_grad and p.grad is not None]
        self.assertGreater(len(grads), 0)
        self.assertTrue(all(torch.isfinite(g).all() for g in grads))
        self.assertIsNotNone(self.actor.task_heads[0][0].weight.grad)

    def test_reset_where_clears_only_selected_environments(self):
        _, state, _ = self.actor(
            self.self_features, self.entities, self.mask, self.meta
        )
        reset = state.reset_where(torch.tensor([True, False]))
        self.assertEqual(reset.known[0].sum().detach().item(), 0.0)
        self.assertGreater(reset.known[1].sum().detach().item(), 0.0)
        self.assertEqual(reset.info_context[0].abs().sum().detach().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
