from __future__ import annotations

import unittest

try:
    import torch
except ImportError:  # pragma: no cover - exercised only in minimal environments
    torch = None

from src.algorithms.fedprox import FedProxAdapter
from src.fcpc.regularizer import fcpc_regularization


@unittest.skipIf(torch is None, "PyTorch is not installed")
class RegularizerGradientTests(unittest.TestCase):
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
