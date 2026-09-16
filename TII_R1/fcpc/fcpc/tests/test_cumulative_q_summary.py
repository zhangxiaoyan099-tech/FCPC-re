from __future__ import annotations

import unittest

from scripts.summarize_cumulative_q import (
    build_cumulative_curves,
    summarize_cumulative_curves,
)


def row(round_number: int, batch_seed: int, q: float, grad_sq: float = 2.0) -> dict:
    return {
        "model_seed": "42",
        "checkpoint_round": str(round_number),
        "panel": "raw",
        "pairing_strategy": "optimal",
        "pairing_seed": "",
        "batch_seed": str(batch_seed),
        "method": "ungated",
        "step_scale": "0.5",
        "smoothness_L": "0.0",
        "Q_counterfactual": str(q),
        "Q_proxy_vs_baseline": str(q + 0.1),
        "Q_observed_loss_gain": str(q - 0.1),
        "global_gradient_norm_sq": str(grad_sq),
    }


class CumulativeQSummaryTests(unittest.TestCase):
    def test_accumulates_inside_trace_before_averaging(self) -> None:
        rows = [
            row(5, 100, 1.0),
            row(10, 100, -0.5),
            row(5, 101, 2.0),
            row(10, 101, 1.0),
        ]
        curves = build_cumulative_curves(rows, first_round=0, expected_step=5)
        final = [item for item in curves if item["checkpoint_round"] == 10]
        self.assertEqual(len(final), 2)
        self.assertAlmostEqual(final[0]["round_weighted_cumulative_Q"], 2.5)
        self.assertAlmostEqual(final[1]["round_weighted_cumulative_Q"], 15.0)

        summary = summarize_cumulative_curves(
            curves, bootstrap_resamples=100, bootstrap_seed=1
        )
        endpoint = [item for item in summary if item["checkpoint_round"] == 10][0]
        self.assertEqual(endpoint["trace_n"], 2)
        self.assertEqual(endpoint["model_seed_n"], 1)
        self.assertAlmostEqual(endpoint["cumulative_Q_mean"], 8.75)
        self.assertEqual(endpoint["cumulative_Q_positive_fraction"], 1.0)

    def test_rejects_missing_five_round_checkpoint(self) -> None:
        with self.assertRaisesRegex(ValueError, "checkpoint interval"):
            build_cumulative_curves(
                [row(5, 100, 1.0), row(15, 100, 1.0)],
                first_round=0,
                expected_step=5,
            )


if __name__ == "__main__":
    unittest.main()
