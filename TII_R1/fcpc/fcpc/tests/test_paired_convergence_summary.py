from __future__ import annotations

import unittest

import numpy as np

from scripts.summarize_paired_convergence import (
    paired_differences,
    summarize_differences,
)


class PairedConvergenceSummaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = [
            {"method": "fcpc_grad", "seed": seed, "val_auc_50": value, "round_to_0.6": round_value}
            for seed, value, round_value in ((1, 0.5, 20), (2, 0.6, 25), (3, 0.7, 30))
        ] + [
            {"method": "fedavg", "seed": seed, "val_auc_50": value, "round_to_0.6": round_value}
            for seed, value, round_value in ((1, 0.4, 30), (2, 0.5, 35), (3, 0.6, 40))
        ]

    def test_positive_always_means_candidate_is_faster(self) -> None:
        auc = paired_differences(
            self.rows, candidate="fcpc_grad", baseline="fedavg", metric="val_auc_50"
        )
        rounds = paired_differences(
            self.rows, candidate="fcpc_grad", baseline="fedavg", metric="round_to_0.6"
        )
        self.assertEqual([round(value, 8) for _, value in auc], [0.1, 0.1, 0.1])
        self.assertEqual([value for _, value in rounds], [10.0, 10.0, 10.0])

    def test_paired_lcb_uses_seed_differences(self) -> None:
        result = summarize_differences(
            [(1, 0.1), (2, 0.1), (3, 0.1)],
            bootstrap_resamples=100,
            rng=np.random.default_rng(1),
        )
        self.assertEqual(result["n"], 3)
        self.assertAlmostEqual(result["mean_delta"], 0.1)
        self.assertAlmostEqual(result["paired_t_lcb95"], 0.1)


if __name__ == "__main__":
    unittest.main()
