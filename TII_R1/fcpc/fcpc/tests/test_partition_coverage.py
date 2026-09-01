from __future__ import annotations

import unittest

from src.data.partition import build_client_indices, validate_complete_partition


class FullCoveragePartitionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.labels = [label for label in range(10) for _ in range(50)]

    def test_dual_skew_assigns_every_sample_exactly_once(self) -> None:
        first = build_client_indices(
            self.labels,
            num_clients=10,
            partition="dual_skew",
            alpha=0.1,
            seed=42,
        )
        second = build_client_indices(
            self.labels,
            num_clients=10,
            partition="dual_skew",
            alpha=0.1,
            seed=42,
        )
        summary = validate_complete_partition(first, len(self.labels))

        self.assertEqual(first, second)
        self.assertEqual(summary["assigned_examples"], len(self.labels))
        self.assertEqual(summary["unique_examples"], len(self.labels))
        self.assertGreater(summary["client_examples_max"], summary["client_examples_min"])
        self.assertGreater(summary["client_examples_min"], 0)

    def test_quantity_skew_assigns_every_sample_exactly_once(self) -> None:
        partition = build_client_indices(
            self.labels,
            num_clients=10,
            partition="quantity_skew",
            seed=7,
        )
        summary = validate_complete_partition(partition, len(self.labels))
        self.assertEqual(summary["assigned_examples"], len(self.labels))
        self.assertGreater(summary["client_examples_max"], summary["client_examples_min"])

    def test_validator_rejects_duplicate_assignments(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_complete_partition({0: [0, 1], 1: [1, 2]}, 3)


if __name__ == "__main__":
    unittest.main()
