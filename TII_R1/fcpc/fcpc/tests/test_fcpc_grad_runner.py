from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_fcpc_grad_convergence import _fcpc_grad_override
from scripts.run_fcpc_grad_tuning import (
    BASE_CONFIG,
    _build_config,
    _parse_float_grid,
    _validation_stats,
)
from scripts.summarize_fcpc_grad_convergence import _first_round_at, _normalized_auc


class FCPCGradTuningTests(unittest.TestCase):
    def test_float_grid_validation(self) -> None:
        self.assertEqual(
            _parse_float_grid("0,0.5,1", name="mix", lower=0.0, upper=1.0),
            [0.0, 0.5, 1.0],
        )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            _parse_float_grid("0.5,0.5", name="mix", lower=0.0, upper=1.0)
        with self.assertRaisesRegex(ValueError, "must be in"):
            _parse_float_grid("1.1", name="mix", lower=0.0, upper=1.0)

    def test_tuning_config_changes_center_only_within_fcpc_grad(self) -> None:
        base = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
        config = _build_config(
            base,
            beta=0.05,
            gradient_mix=0.5,
            step_scale=0.25,
            seed=42,
            rounds=50,
        )
        self.assertEqual(config["fcpc"]["reference_strategy"], "pair_grad_center")
        self.assertEqual(config["fcpc"]["update_rule"], "proximal")
        self.assertEqual(config["fcpc"]["grad_center_mix"], 0.5)
        self.assertEqual(config["fcpc"]["grad_center_step_scale"], 0.25)
        self.assertFalse(config["evaluation"]["evaluate_test"])
        self.assertEqual(config["partition"], base["partition"])
        self.assertEqual(config["optimizer"], base["optimizer"])

    def test_validation_selection_statistics_do_not_read_test_accuracy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["round", "val_acc", "test_acc"])
                writer.writeheader()
                writer.writerows(
                    [
                        {"round": 1, "val_acc": 0.2, "test_acc": 0.99},
                        {"round": 2, "val_acc": 0.4, "test_acc": 0.01},
                    ]
                )
            stats = _validation_stats(path)
        self.assertAlmostEqual(stats["val_auc"], 0.3)
        self.assertEqual(stats["best_round"], 2)
        self.assertAlmostEqual(stats["best_val_acc"], 0.4)

    def test_convergence_override_uses_selected_values(self) -> None:
        override = _fcpc_grad_override(
            {"beta": 0.05, "grad_center_mix": 0.5, "grad_center_step_scale": 0.25}
        )
        self.assertEqual(override["fcpc"]["reference_strategy"], "pair_grad_center")
        self.assertEqual(override["fcpc"]["beta"], 0.05)
        self.assertEqual(override["fcpc"]["grad_center_mix"], 0.5)


class FCPCGradSummaryTests(unittest.TestCase):
    def test_auc_and_threshold_round(self) -> None:
        self.assertAlmostEqual(_normalized_auc([0.2, 0.4, 0.6], 3), 0.4)
        self.assertEqual(_first_round_at([0.2, 0.4, 0.6], 0.5), 3)
        self.assertEqual(_first_round_at([0.2, 0.4], 0.5), "")


if __name__ == "__main__":
    unittest.main()
