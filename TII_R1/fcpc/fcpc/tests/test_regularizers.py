from __future__ import annotations

import unittest

try:
    import torch
except ImportError:  # pragma: no cover - exercised only in minimal environments
    torch = None

from src.algorithms.fedprox import FedProxAdapter
from src.fcpc.regularizer import (
    clip_state_center_to_global,
    fcpc_regularization,
    proximal_center_step,
    state_l2_distance,
    weighted_state_center,
)


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

    def test_proximal_center_step_contracts_pair_disagreement(self) -> None:
        left = torch.nn.Linear(2, 1, bias=False)
        right = torch.nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            left.weight.copy_(torch.tensor([[2.0, -1.0]]))
            right.weight.copy_(torch.tensor([[-2.0, 3.0]]))
        center = {"weight": torch.tensor([[10.0, 10.0]])}
        before = torch.linalg.vector_norm(left.weight - right.weight)

        left_factor = proximal_center_step(
            dict(left.named_parameters()),
            center,
            beta=0.5,
            learning_rate=0.2,
        )
        right_factor = proximal_center_step(
            dict(right.named_parameters()),
            center,
            beta=0.5,
            learning_rate=0.2,
        )

        after = torch.linalg.vector_norm(left.weight - right.weight)
        expected_factor = 1.0 / 1.2
        self.assertAlmostEqual(left_factor, expected_factor)
        self.assertAlmostEqual(right_factor, expected_factor)
        self.assertTrue(torch.allclose(after, before * expected_factor))

    def test_center_clipping_enforces_global_radius(self) -> None:
        global_state = {
            "weight": torch.tensor([0.0, 0.0]),
            "buffer": torch.tensor(7, dtype=torch.long),
        }
        center = {
            "weight": torch.tensor([3.0, 4.0]),
            "buffer": torch.tensor(9, dtype=torch.long),
        }

        clipped, original_distance, scale = clip_state_center_to_global(
            center,
            global_state,
            max_distance=2.0,
            parameter_names={"weight"},
        )

        self.assertAlmostEqual(original_distance, 5.0)
        self.assertAlmostEqual(scale, 0.4)
        self.assertAlmostEqual(
            state_l2_distance(clipped, global_state, {"weight"}),
            2.0,
        )
        self.assertEqual(int(clipped["buffer"]), 9)

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
