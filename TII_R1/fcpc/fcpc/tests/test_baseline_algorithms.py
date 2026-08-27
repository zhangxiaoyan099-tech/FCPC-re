from __future__ import annotations

import unittest

import numpy as np

from src.algorithms import build_algorithm
from src.algorithms.fblg import FBLGAdapter


class _Client:
    def __init__(self, count):
        self.sample_count = count


class BaselineRegistrationTests(unittest.TestCase):
    def test_all_baselines_are_real_adapters(self):
        expected = {
            "fedavg": "src.algorithms.fedavg",
            "fedprox": "src.algorithms.fedprox",
            "moon": "src.algorithms.moon",
            "feddyn": "src.algorithms.feddyn",
            "fblg": "src.algorithms.fblg",
            "fedcfa": "src.algorithms.fedcfa",
        }
        for name, module in expected.items():
            with self.subTest(name=name):
                self.assertEqual(build_algorithm(name).__class__.__module__, module)

    def test_fblg_subset_uses_graph_and_sample_scores(self):
        adapter = FBLGAdapter()
        adapter._distance_matrix = np.asarray(
            [
                [0.0, 0.1, 1.0],
                [0.1, 0.0, 0.5],
                [1.0, 0.5, 0.0],
            ]
        )
        clients = [_Client(10), _Client(10), _Client(10)]
        self.assertEqual(adapter._optimize_subset([0, 1, 2], 2, clients), [0, 2])


try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None


if nn is not None:
    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Linear(4, 3)
            self.classifier = nn.Linear(3, 2)

        def forward_with_representation(self, inputs):
            representation = torch.tanh(self.features(inputs))
            return self.classifier(representation), representation

        def forward(self, inputs):
            return self.forward_with_representation(inputs)[0]
else:
    TinyModel = object


@unittest.skipIf(torch is None, "PyTorch is not installed")
class BaselineGradientTests(unittest.TestCase):

    def test_moon_contrastive_term_has_gradient(self):
        model = TinyModel()
        global_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        previous_state = {key: value.detach().clone() + 0.1 for key, value in model.state_dict().items()}
        adapter = build_algorithm("moon", mu=1.0, temperature=0.5)
        adapter.begin_local_train(
            model=model,
            global_state=global_state,
            previous_local_state=previous_state,
        )
        inputs = torch.randn(5, 4)
        targets = torch.randint(0, 2, (5,))
        logits, context = adapter.forward(model, inputs)
        task = nn.CrossEntropyLoss()(logits, targets)
        loss = adapter.extra_loss(model, (inputs, targets), task, context)
        loss.backward()
        self.assertGreater(sum(parameter.grad.abs().sum().item() for parameter in model.parameters()), 0.0)

    def test_feddyn_history_changes_custom_aggregation(self):
        model = TinyModel()
        global_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        local_state = {key: value.detach().clone() + 0.1 for key, value in model.state_dict().items()}
        adapter = build_algorithm("feddyn", alpha=0.01)
        adapter.begin_round(clients=[object(), object()])
        adapter.after_local_train(client_id=0, local_state=local_state, global_state=global_state)
        result = adapter.aggregate(
            selected_client_ids=[0],
            client_states=[local_state],
            default_state=local_state,
        )
        first_float = next(key for key, value in local_state.items() if value.is_floating_point())
        self.assertFalse(torch.equal(result[first_float], local_state[first_float]))


if __name__ == "__main__":
    unittest.main()
