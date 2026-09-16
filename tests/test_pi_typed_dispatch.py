from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from fofe_mmapppo.models.pi_actor import PIActor as ReferencePIActor
from fofe_mmapppo.models.pi_actor import PIActorConfig
from fofe_mmapppo.models.pi_actor_optimized import PIActor as SparsePIActor


class SparseTypedDispatchTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1234)
        self.cfg = PIActorConfig(
            belief_dim=16,
            context_dim=8,
            relation_dim=16,
            horizon=2,
        )
        self.reference = ReferencePIActor(self.cfg)
        self.sparse = SparsePIActor(self.cfg)
        self.sparse.load_state_dict(self.reference.state_dict())

    def _compare_forward_backward(self, module_name: str, shape: tuple[int, ...]):
        x_ref = torch.randn(*shape, requires_grad=True)
        x_sparse = x_ref.detach().clone().requires_grad_(True)
        ref_modules = getattr(self.reference, module_name)
        sparse_modules = getattr(self.sparse, module_name)

        y_ref = self.reference._type_apply(ref_modules, x_ref)
        y_sparse = self.sparse._type_apply(sparse_modules, x_sparse)
        self.assertEqual(tuple(y_ref.shape), tuple(y_sparse.shape))
        self.assertTrue(torch.allclose(y_ref, y_sparse, atol=1e-7, rtol=1e-6))

        # Use the same non-uniform upstream gradient so equality covers the
        # complete Jacobian rather than only a symmetric sum reduction.
        upstream = torch.randn_like(y_ref)
        (y_ref * upstream).sum().backward()
        (y_sparse * upstream).sum().backward()
        self.assertTrue(torch.allclose(x_ref.grad, x_sparse.grad, atol=1e-7, rtol=1e-6))

        ref_params = dict(self.reference.named_parameters())
        sparse_params = dict(self.sparse.named_parameters())
        for name in ref_params:
            grad_ref = ref_params[name].grad
            grad_sparse = sparse_params[name].grad
            if grad_ref is None and grad_sparse is None:
                continue
            self.assertIsNotNone(grad_ref, name)
            self.assertIsNotNone(grad_sparse, name)
            self.assertTrue(
                torch.allclose(grad_ref, grad_sparse, atol=1e-7, rtol=1e-6),
                name,
            )

    def test_rank4_belief_dispatch_matches_reference(self):
        d = self.cfg.belief_dim
        dyn_in = d + d + d + 1
        self._compare_forward_backward(
            "dynamics",
            (2, self.cfg.n_entities, self.cfg.n_hypotheses, dyn_in),
        )

    def test_rank6_lattice_dispatch_matches_reference(self):
        self._compare_forward_backward(
            "relation_encoders",
            (1, 3, self.cfg.n_entities, self.cfg.n_hypotheses, 2, 15),
        )


if __name__ == "__main__":
    unittest.main()
