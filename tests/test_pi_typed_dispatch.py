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

    def _models(self, *, dtype: torch.dtype):
        reference = ReferencePIActor(self.cfg).to(dtype=dtype)
        sparse = SparsePIActor(self.cfg).to(dtype=dtype)
        sparse.load_state_dict(reference.state_dict())
        return reference, sparse

    def _compare_forward_backward(
        self,
        module_name: str,
        shape: tuple[int, ...],
        *,
        dtype: torch.dtype,
        atol: float,
        rtol: float,
    ):
        # Recreate the pair for each check so gradients from one test cannot leak
        # into the next one.
        reference, sparse = self._models(dtype=dtype)
        x_ref = torch.randn(*shape, dtype=dtype, requires_grad=True)
        x_sparse = x_ref.detach().clone().requires_grad_(True)
        ref_modules = getattr(reference, module_name)
        sparse_modules = getattr(sparse, module_name)

        y_ref = reference._type_apply(ref_modules, x_ref)
        y_sparse = sparse._type_apply(sparse_modules, x_sparse)
        self.assertEqual(tuple(y_ref.shape), tuple(y_sparse.shape))
        self.assertTrue(
            torch.allclose(y_ref, y_sparse, atol=atol, rtol=rtol),
            f"forward max_abs={(y_ref - y_sparse).abs().max().item():.3e}",
        )

        # Same non-uniform upstream gradient: this exercises the complete
        # Jacobian, not merely a symmetric sum reduction.
        upstream = torch.randn_like(y_ref)
        (y_ref * upstream).sum().backward()
        (y_sparse * upstream).sum().backward()
        self.assertTrue(
            torch.allclose(x_ref.grad, x_sparse.grad, atol=atol, rtol=rtol),
            f"input_grad max_abs={(x_ref.grad - x_sparse.grad).abs().max().item():.3e}",
        )

        ref_params = dict(reference.named_parameters())
        sparse_params = dict(sparse.named_parameters())
        for name in ref_params:
            grad_ref = ref_params[name].grad
            grad_sparse = sparse_params[name].grad
            if grad_ref is None and grad_sparse is None:
                continue
            self.assertIsNotNone(grad_ref, name)
            self.assertIsNotNone(grad_sparse, name)
            max_abs = (grad_ref - grad_sparse).abs().max().item()
            denom = grad_ref.abs().max().item()
            max_rel = max_abs / max(denom, 1e-30)
            self.assertTrue(
                torch.allclose(grad_ref, grad_sparse, atol=atol, rtol=rtol),
                f"{name}: max_abs={max_abs:.3e} max_rel={max_rel:.3e}",
            )

    def _run_rank4(self, *, dtype, atol, rtol):
        d = self.cfg.belief_dim
        dyn_in = d + d + d + 1
        self._compare_forward_backward(
            "dynamics",
            (2, self.cfg.n_entities, self.cfg.n_hypotheses, dyn_in),
            dtype=dtype,
            atol=atol,
            rtol=rtol,
        )

    def _run_rank6(self, *, dtype, atol, rtol):
        self._compare_forward_backward(
            "relation_encoders",
            (1, 3, self.cfg.n_entities, self.cfg.n_hypotheses, 2, 15),
            dtype=dtype,
            atol=atol,
            rtol=rtol,
        )

    def test_rank4_float64_proves_mathematical_equivalence(self):
        # Float64 makes the proof insensitive to different GEMM reduction orders.
        self._run_rank4(dtype=torch.float64, atol=1e-10, rtol=1e-9)

    def test_rank6_float64_proves_mathematical_equivalence(self):
        self._run_rank6(dtype=torch.float64, atol=1e-10, rtol=1e-9)

    def test_rank4_float32_training_numerics_are_close(self):
        # Sparse dispatch changes the GEMM reduction shape: the reference path
        # reduces over rows whose upstream gradient is exactly zero, whereas the
        # sparse path omits those rows.  The mathematical gradient is identical,
        # but float32 parameter-gradient accumulation can differ by a few ulps.
        self._run_rank4(dtype=torch.float32, atol=2e-5, rtol=2e-5)

    def test_rank6_float32_training_numerics_are_close(self):
        self._run_rank6(dtype=torch.float32, atol=2e-5, rtol=2e-5)


if __name__ == "__main__":
    unittest.main()
