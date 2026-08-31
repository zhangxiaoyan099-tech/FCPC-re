from __future__ import annotations

import unittest

try:
    import torch
except ImportError:  # pragma: no cover - exercised only in minimal environments
    torch = None

from src.algorithms.fedprox import FedProxAdapter
from src.fcpc.regularizer import fcpc_regularization, weighted_state_center


@unittest.skipIf(torch is None, "PyTorch is not installed")
class RegularizerGradientTests(unittest.TestCase):
    def test_weighted_pair_center_uses_sample_counts(self) -> None:
        state_a = {
            "weight": torch.tensor([1.0, 3.0]),
            "num_batches_tracked": torch.tensor(2, dtype=torch.long),
        }
        state_b = {
            "weight": torch.tensor([5.0, 7.0]),
            "num_batches_tracked": torch.tensor(9, dtype=torch.long),
        }
        fallback = {
            "weight": torch.zeros(2),
            "num_batches_tracked": torch.tensor(0, dtype=torch.long),
        }
        center = weighted_state_center(
            state_a,
            state_b,
            count_a=1,
            count_b=3,
            fallback_state=fallback,
        )

        self.assertTrue(torch.allclose(center["weight"], torch.tensor([4.0, 6.0])))
        self.assertEqual(int(center["num_batches_tracked"]), 2)
        self.assertFalse(center["weight"].requires_grad)

    def test_missing_pair_states_fall_back_to_global_model(self) -> None:
        fallback = {"weight": torch.tensor([2.0, 4.0])}
        center = weighted_state_center(
            None,
            None,
            count_a=10,
            count_b=30,
            fallback_state=fallback,
        )
        self.assertTrue(torch.equal(center["weight"], fallback["weight"]))
        self.assertIsNot(center["weight"], fallback["weight"])

    def test_fcpc_regularizer_contributes_gradient(self) -> None:
        model = torch.nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            model.weight.zero_()
        partner_state = {"weight": torch.ones_like(model.weight)}

        penalty = fcpc_regularization(
            dict(model.named_parameters()),
            partner_state,
            beta=0.2,
        )

        self.assertTrue(penalty.requires_grad)
        penalty.backward()
        expected = torch.full_like(model.weight, -0.4)
        self.assertTrue(torch.allclose(model.weight.grad, expected))

    def test_fedprox_regularizer_contributes_gradient(self) -> None:
        model = torch.nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(1.0)
        global_state = {"weight": torch.zeros_like(model.weight)}
        task_loss = model.weight.sum() * 0.0

        penalty = FedProxAdapter(mu=0.1).extra_loss(
            model,
            batch=None,
            task_loss=task_loss,
            context={"global_state": global_state},
        )

        self.assertTrue(penalty.requires_grad)
        penalty.backward()
        expected = torch.full_like(model.weight, 0.1)
        self.assertTrue(torch.allclose(model.weight.grad, expected))


if __name__ == "__main__":
    unittest.main()
