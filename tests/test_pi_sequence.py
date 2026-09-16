from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from torch.distributions import Categorical

from fofe_mmapppo.algorithms import unroll_pi_actor
from fofe_mmapppo.models import PIActor, PIActorConfig


class PISequenceTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.cfg = PIActorConfig(
            n_hypotheses=3,
            horizon=3,
            belief_dim=12,
            context_dim=8,
            relation_dim=12,
        )
        self.actor = PIActor(self.cfg)
        self.T = 5
        self.B = 2
        self.target_slot = 7

        self.self_features = torch.zeros(self.T, self.B, 14)
        self.self_features[..., 0] = 1.0
        self.self_features[..., 2] = 1.0  # Stk
        self.self_features[..., 5] = 1.0
        self.self_features[..., 6] = 0.25
        self.self_features[..., 7] = 0.15
        self.self_features[..., 8] = 0.0  # north

        self.entities = torch.zeros(self.T, self.B, 14, 20)
        self.mask = torch.zeros(self.T, self.B, 14)
        self.meta = torch.zeros(self.T, self.B, 14, 4)

        # One target is seen at t=0, disappears for two steps, then reappears.
        for t in (0, 3, 4):
            self.entities[t, :, self.target_slot, 0] = 1.0
            self.entities[t, :, self.target_slot, 2] = 1.0
            self.entities[t, :, self.target_slot, 7] = 1.0
            self.entities[t, :, self.target_slot, 9] = 0.45 + 0.01 * t
            self.entities[t, :, self.target_slot, 10] = 0.55 + 0.005 * t
            self.entities[t, :, self.target_slot, 11] = 0.0
            self.entities[t, :, self.target_slot, 18] = 8.0 / 50.0
            self.entities[t, :, self.target_slot, 19] = 10.0 / 40.0
            self.mask[t, :, self.target_slot] = 1.0

    def test_unchanged_policy_replay_reproduces_rollout_log_probs(self):
        state = self.actor.initial_state(self.B)
        rollout_logits = []
        rollout_actions = []
        rollout_log_probs = []

        with torch.inference_mode():
            for t in range(self.T):
                logits, state, _ = self.actor(
                    self.self_features[t],
                    self.entities[t],
                    self.mask[t],
                    self.meta[t],
                    state,
                )
                dist = Categorical(logits=logits)
                action = torch.argmax(logits, dim=-1)
                rollout_logits.append(logits)
                rollout_actions.append(action)
                rollout_log_probs.append(dist.log_prob(action))

        rollout_logits = torch.stack(rollout_logits)
        rollout_actions = torch.stack(rollout_actions)
        rollout_log_probs = torch.stack(rollout_log_probs)

        replay = unroll_pi_actor(
            self.actor,
            self.self_features,
            self.entities,
            self.mask,
            self.meta,
            initial_state=self.actor.initial_state(self.B),
            actions=rollout_actions,
        )

        self.assertTrue(torch.allclose(replay.logits, rollout_logits, atol=1e-6, rtol=1e-6))
        self.assertTrue(
            torch.allclose(replay.log_probs, rollout_log_probs, atol=1e-6, rtol=1e-6)
        )
        ratio = torch.exp(replay.log_probs - rollout_log_probs)
        self.assertTrue(torch.allclose(ratio, torch.ones_like(ratio), atol=1e-6, rtol=1e-6))

    def test_inactive_padding_does_not_advance_belief_state(self):
        # Use only the first observation as evidence, then remove the target.
        mask = self.mask.clone()
        entities = self.entities.clone()
        mask[1:] = 0.0
        entities[1:] = 0.0

        active = torch.ones(self.T, self.B, dtype=torch.bool)
        active[1:, 0] = False
        result = unroll_pi_actor(
            self.actor,
            self.self_features,
            entities,
            mask,
            self.meta,
            active=active,
        )

        # Environment 0 freezes immediately after t=0, so target age remains 0.
        self.assertAlmostEqual(
            result.final_state.age[0, self.target_slot, 0].detach().item(), 0.0, places=6
        )
        # Environment 1 stays active for four missing-evidence steps.
        self.assertAlmostEqual(
            result.final_state.age[1, self.target_slot, 0].detach().item(), 4.0, places=6
        )
        self.assertTrue(torch.equal(result.logits[1:, 0], torch.zeros_like(result.logits[1:, 0])))

    def test_reset_before_clears_prior_episode_memory(self):
        # Only t=0 observes the target. Reset environment 0 before t=2, and do
        # not provide evidence afterwards: the old target must no longer be known.
        mask = self.mask.clone()
        entities = self.entities.clone()
        mask[1:] = 0.0
        entities[1:] = 0.0
        reset = torch.zeros(self.T, self.B, dtype=torch.bool)
        reset[2, 0] = True

        result = unroll_pi_actor(
            self.actor,
            self.self_features,
            entities,
            mask,
            self.meta,
            reset_before=reset,
        )
        self.assertEqual(
            result.final_state.known[0, self.target_slot, 0].detach().item(), 0.0
        )
        self.assertEqual(
            result.final_state.known[1, self.target_slot, 0].detach().item(), 1.0
        )

    def test_sequence_log_prob_path_is_differentiable(self):
        actions = torch.full((self.T, self.B), 3, dtype=torch.long)
        replay = unroll_pi_actor(
            self.actor,
            self.self_features,
            self.entities,
            self.mask,
            self.meta,
            actions=actions,
        )
        loss = -replay.log_probs.mean()
        loss.backward()
        grads = [
            parameter.grad
            for parameter in self.actor.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        self.assertGreater(len(grads), 0)
        self.assertTrue(all(torch.isfinite(grad).all() for grad in grads))


if __name__ == "__main__":
    unittest.main()
