from __future__ import annotations
import copy
import unittest
from unittest.mock import patch
import torch
import test_pi_recurrent_ppo as fixture_module
from fofe_mmapppo.algorithms.pi_parallel_buffer import PIParallelRolloutBuffer
from fofe_mmapppo.algorithms import PIMAPPO
from fofe_mmapppo.algorithms.pi_batched import BatchedPIActors, stack_beliefs
from fofe_mmapppo.algorithms.pi_sequence import unroll_pi_actor


class BatchedPIUpdateTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.fixture = fixture_module.PIRecurrentMAPPOTests()
        self.fixture.setUp()

    def compare_update(self, chunk, uneven=False, absent=False, tail=False):
        f = self.fixture
        serial = f.learner
        serial.config.tbptt_chunk_length = chunk
        serial.config.ppo_epochs = 2
        serial.config.sequence_env_minibatch_size = 2 if tail else 1
        if tail:
            f.E = 3
        buffer = f._build_rollout()
        if uneven:
            # Remove one actor's entire environment without changing tensor sizes.
            for row in buffer.active:
                row[0, 1] = 0
        if absent:
            for row in buffer.active:
                row[:, 1] = 0
        for old_lp, active in zip(buffer.log_probs, buffer.active):
            old_lp[active < .5] = -1e4  # Padding must never enter the PPO ratio.
        cfg = copy.deepcopy(serial.config)
        cfg.batch_actor_updates = True
        batched = PIMAPPO(f.state_dim, n_agents=f.N, config=cfg, actor_config=f.actor_cfg)
        batched.load_checkpoint(copy.deepcopy(serial.checkpoint()))
        # Float64 separates algorithmic equivalence from Adam amplification of
        # roundoff on mathematically zero softmax-shift gradients in float32.
        for model in (serial, batched):
            model.actors.double()
            model.critics.double()
        original_to_torch = PIParallelRolloutBuffer.to_torch
        def as_double(*args, **kwargs):
            return {k: v.double() if v.is_floating_point() else v
                    for k, v in original_to_torch(*args, **kwargs).items()}
        conversion = patch.object(PIParallelRolloutBuffer, 'to_torch', side_effect=as_double)
        conversion.start()
        self.addCleanup(conversion.stop)
        torch.manual_seed(123)
        m1 = serial.update_parallel(buffer)
        torch.manual_seed(123)
        m2 = batched.update_parallel(buffer)
        for key in m1:
            self.assertAlmostEqual(m1[key], m2[key], delta=2e-5, msg=key)
        for group in ('actors', 'critics'):
            for name, p in getattr(serial, group).named_parameters():
                q = dict(getattr(batched, group).named_parameters())[name]
                torch.testing.assert_close(p, q, atol=1e-9, rtol=1e-7, msg=lambda message: name + ": " + message)
        for opts1, opts2 in ((serial.actor_optimizers, batched.actor_optimizers),
                             (serial.critic_optimizers, batched.critic_optimizers)):
            for o1, o2 in zip(opts1, opts2):
                s1, s2 = o1.state_dict()['state'], o2.state_dict()['state']
                self.assertEqual(s1.keys(), s2.keys())
                for key in s1:
                    for field in s1[key]:
                        torch.testing.assert_close(s1[key][field], s2[key][field], atol=2e-6, rtol=2e-4)
        restored = PIMAPPO(f.state_dim, n_agents=f.N, config=cfg, actor_config=f.actor_cfg)
        restored.actors.double()
        restored.critics.double()
        self.assertTrue(restored.load_checkpoint(copy.deepcopy(batched.checkpoint(True)), True))
        torch.manual_seed(987)
        batched.update_parallel(buffer)
        torch.manual_seed(987)
        restored.update_parallel(buffer)
        for p, q in zip(batched.actors.parameters(), restored.actors.parameters()):
            torch.testing.assert_close(p, q)

    def test_full_sequence_matches_serial_and_resumes(self):
        self.compare_update(4)

    def test_chunked_sequence_matches_serial_and_resumes(self):
        self.compare_update(2)

    def test_partial_chunk_matches_serial(self):
        self.compare_update(3)

    def test_uneven_environment_pools(self):
        self.compare_update(2, uneven=True)

    def test_absent_agent_does_not_advance_adam(self):
        self.compare_update(2, absent=True)

    def test_padded_last_minibatch(self):
        self.compare_update(2, uneven=True, tail=True)

    def test_separate_actor_groups_match_serial(self):
        self.fixture.learner.config.actor_batch_size = 1
        self.compare_update(2, uneven=True, tail=True)

    def test_float32_updated_policy_matches(self):
        f = self.fixture
        buffer = f._build_rollout()
        serial = f.learner
        serial.config.ppo_epochs = 2
        serial.config.tbptt_chunk_length = 2
        cfg = copy.deepcopy(serial.config)
        cfg.batch_actor_updates = True
        batched = PIMAPPO(f.state_dim, n_agents=f.N, config=cfg, actor_config=f.actor_cfg)
        batched.load_checkpoint(copy.deepcopy(serial.checkpoint()))
        torch.manual_seed(77)
        first = serial.update_parallel(buffer)
        torch.manual_seed(77)
        second = batched.update_parallel(buffer)
        for k in first:
            self.assertAlmostEqual(first[k], second[k], delta=2e-5)
        data = PIParallelRolloutBuffer.to_torch(buffer.as_arrays(), serial.device)
        for a in range(f.N):
            idx = torch.arange(f.E)
            out1 = serial._agent_replay(data, a, idx)
            out2 = batched._agent_replay(data, a, idx)
            torch.testing.assert_close(out1.log_probs, out2.log_probs, atol=2e-5, rtol=1e-5)

    def test_logits_beliefs_and_gradients_match(self):
        f = self.fixture
        data = f._build_rollout().as_arrays()
        inputs = [torch.tensor(data[k]).transpose(1, 2) for k in (
            'self_features', 'entity_features', 'evidence_mask', 'evidence_meta')]
        mask = torch.tensor(data['active']).transpose(1, 2) > .5
        backend = BatchedPIActors(f.learner.actors)
        initial = f.learner.initial_belief_states(f.E)
        logits, state = backend.unroll(backend.weights(), inputs, mask, stack_beliefs(initial))
        loss = logits.square().sum() + state[0].square().sum() * .001
        loss.backward()
        gradients = [p.grad.clone() for p in f.learner.actors.parameters()]
        for p in f.learner.actors.parameters():
            p.grad = None
        serial_loss = 0
        for a, actor in enumerate(f.learner.actors):
            out = unroll_pi_actor(actor, *(x[:, a] for x in inputs),
                                  initial_state=initial[a], active=mask[:, a])
            torch.testing.assert_close(logits[:, a], out.logits, atol=1e-6, rtol=1e-5)
            for value, name in zip(state, initial[a].__dict__):
                torch.testing.assert_close(value[a], getattr(out.final_state, name), atol=1e-6, rtol=1e-5)
            serial_loss = serial_loss + out.logits.square().sum() + out.final_state.latent.square().sum() * .001
        serial_loss.backward()
        for p, grad in zip(f.learner.actors.parameters(), gradients):
            torch.testing.assert_close(p.grad, grad, atol=1e-6, rtol=1e-4)


if __name__ == '__main__':
    unittest.main()
