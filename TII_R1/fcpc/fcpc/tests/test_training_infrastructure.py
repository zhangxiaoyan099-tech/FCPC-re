from __future__ import annotations

import unittest

import numpy as np

try:
    import torch
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:  # pragma: no cover
    torch = None

from src.algorithms.fedavg import FedAvgAdapter
from src.data.split import stratified_holdout_indices
from src.federated.client import Client
from src.federated.trainer import Trainer


class StratifiedSplitTests(unittest.TestCase):
    def test_split_is_reproducible_disjoint_and_complete(self) -> None:
        labels = [label for label in range(3) for _ in range(10)]
        first = stratified_holdout_indices(labels, validation_fraction=0.2, seed=7)
        second = stratified_holdout_indices(labels, validation_fraction=0.2, seed=7)

        self.assertEqual(first, second)
        train_indices, validation_indices = first
        self.assertFalse(set(train_indices) & set(validation_indices))
        self.assertEqual(set(train_indices) | set(validation_indices), set(range(30)))
        validation_labels = [labels[index] for index in validation_indices]
        self.assertEqual({label: validation_labels.count(label) for label in range(3)}, {0: 2, 1: 2, 2: 2})

    def test_zero_fraction_preserves_training_set(self) -> None:
        train_indices, validation_indices = stratified_holdout_indices([0, 1, 1], 0.0, seed=1)
        self.assertEqual(train_indices, [0, 1, 2])
        self.assertEqual(validation_indices, [])


@unittest.skipIf(torch is None, "PyTorch is not installed")
class ClientMetricTests(unittest.TestCase):
    def test_local_training_reports_separate_fcpc_losses(self) -> None:
        torch.manual_seed(3)
        inputs = torch.randn(8, 2)
        targets = torch.randint(0, 2, (8,))
        loader = DataLoader(TensorDataset(inputs, targets), batch_size=4, shuffle=False)
        model = torch.nn.Linear(2, 2)
        global_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
        partner_state = {
            name: value.detach().clone() + 0.1
            for name, value in model.state_dict().items()
        }
        client = Client(client_id=0, train_loader=loader, sample_count=8, label_histogram=np.ones(2))

        state, metrics = client.local_train(
            model,
            FedAvgAdapter(),
            global_state,
            paired_previous_state=partner_state,
            use_fcpc=True,
            beta=0.01,
            lr=0.01,
            optimizer_name="sgd",
            momentum=0.9,
            weight_decay=5e-4,
            local_epochs=1,
            device="cpu",
            return_metrics=True,
        )

        self.assertIn("weight", state)
        self.assertEqual(metrics["processed_examples"], 8)
        self.assertEqual(metrics["processed_batches"], 2)
        self.assertGreater(metrics["task_loss"], 0.0)
        self.assertGreater(metrics["fcpc_raw_loss"], 0.0)
        self.assertAlmostEqual(
            metrics["fcpc_weighted_loss"],
            0.01 * metrics["fcpc_raw_loss"],
            places=6,
        )


class LearningRateScheduleTests(unittest.TestCase):
    def test_cosine_schedule_starts_at_base_and_ends_at_minimum(self) -> None:
        config = {"name": "cosine", "min_lr": 0.001}
        self.assertAlmostEqual(Trainer._learning_rate_for_round(0.1, 0, 5, config), 0.1)
        self.assertAlmostEqual(Trainer._learning_rate_for_round(0.1, 4, 5, config), 0.001)


class FCPCBetaScheduleTests(unittest.TestCase):
    def test_constant_beta_does_not_change(self) -> None:
        values = [
            Trainer._beta_for_round(0.01, round_idx, 5, "constant", 0.0)
            for round_idx in range(5)
        ]
        self.assertEqual(values, [0.01] * 5)

    def test_cosine_beta_starts_at_base_and_ends_at_minimum(self) -> None:
        values = [
            Trainer._beta_for_round(0.01, round_idx, 5, "cosine_decay", 0.0)
            for round_idx in range(5)
        ]
        self.assertAlmostEqual(values[0], 0.01)
        self.assertAlmostEqual(values[2], 0.005)
        self.assertAlmostEqual(values[-1], 0.0)


class FCPCPartnerWeightTests(unittest.TestCase):
    def test_uniform_weight_is_one(self) -> None:
        self.assertEqual(Trainer._partner_weight_for_pair(100, 900, "uniform"), 1.0)

    def test_sample_ratio_has_no_factor_two(self) -> None:
        small_from_large = Trainer._partner_weight_for_pair(100, 900, "sample_ratio")
        large_from_small = Trainer._partner_weight_for_pair(900, 100, "sample_ratio")
        self.assertAlmostEqual(small_from_large, 0.9)
        self.assertAlmostEqual(large_from_small, 0.1)
        self.assertAlmostEqual(small_from_large + large_from_small, 1.0)


if __name__ == "__main__":
    unittest.main()
