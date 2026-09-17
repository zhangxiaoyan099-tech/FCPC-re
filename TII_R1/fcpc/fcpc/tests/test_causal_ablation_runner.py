from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_cifar10_full_comparison import BASE_CONFIG
from scripts.analyze_causal_ablation import (
    _index_rows,
    component_differences,
    interaction_differences,
)
from scripts.run_fcpc_grad_causal_ablation import (
    INTERACTION_METHODS,
    METHOD_OVERRIDES,
    SCREEN_METHODS,
    _completed_rounds,
    _has_final_test_metrics,
    _resolve_methods,
    build_config,
)


class CausalAblationRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.base = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))

    def test_all_variants_share_the_training_protocol(self) -> None:
        configs = {
            method: build_config(self.base, method, seed=45, rounds=50)
            for method in METHOD_OVERRIDES
        }
        reference = configs["fedavg"]
        for method, config in configs.items():
            self.assertEqual(config["seed"], 45, method)
            self.assertEqual(config["federated"], reference["federated"], method)
            self.assertEqual(config["optimizer"], reference["optimizer"], method)
            self.assertEqual(config["scheduler"], reference["scheduler"], method)
            self.assertEqual(config["partition"], reference["partition"], method)
            self.assertEqual(config["model"], reference["model"], method)

    def test_direction_controls_change_only_the_direction_field(self) -> None:
        real = METHOD_OVERRIDES["grad_real"]["fcpc"]
        shuffled = METHOD_OVERRIDES["grad_shuffled"]["fcpc"]
        reversed_direction = METHOD_OVERRIDES["grad_reversed"]["fcpc"]
        differences = lambda other: {
            key for key in real if real.get(key) != other.get(key)
        } | {key for key in other if real.get(key) != other.get(key)}
        self.assertEqual(differences(shuffled), {"grad_center_direction"})
        self.assertEqual(differences(reversed_direction), {"grad_center_direction"})

    def test_component_controls_are_pre_registered(self) -> None:
        self.assertEqual(
            METHOD_OVERRIDES["global_prox"]["fcpc"]["reference_strategy"],
            "global_center",
        )
        self.assertEqual(
            METHOD_OVERRIDES["pair_center"]["fcpc"]["reference_strategy"],
            "pair_center",
        )
        self.assertEqual(
            METHOD_OVERRIDES["grad_random_pairing"]["fcpc"]["pairing_strategy"],
            "random",
        )
        self.assertEqual(
            METHOD_OVERRIDES["grad_constant_beta"]["fcpc"]["beta_schedule"],
            "constant",
        )
        self.assertEqual(
            METHOD_OVERRIDES["grad_local_end"]["fcpc"]["proximal_frequency"],
            "local_end_matched",
        )
        self.assertIsNone(
            METHOD_OVERRIDES["grad_no_clip"]["fcpc"][
                "center_max_relative_distance"
            ]
        )

    def test_screen_and_interaction_presets_are_distinct(self) -> None:
        self.assertEqual(_resolve_methods("screen"), list(SCREEN_METHODS))
        self.assertEqual(
            _resolve_methods("interactions"), list(INTERACTION_METHODS)
        )
        self.assertNotIn("pair_center_random", SCREEN_METHODS)
        self.assertIn("pair_center_random", INTERACTION_METHODS)

    def test_component_and_difference_in_differences(self) -> None:
        rows = [
            {"method": "grad_real", "seed": 1, "val_auc_50": 0.60},
            {"method": "grad_random_pairing", "seed": 1, "val_auc_50": 0.50},
            {"method": "pair_center", "seed": 1, "val_auc_50": 0.48},
            {"method": "pair_center_random", "seed": 1, "val_auc_50": 0.45},
            {"method": "grad_real", "seed": 2, "val_auc_50": 0.58},
            {"method": "grad_random_pairing", "seed": 2, "val_auc_50": 0.50},
            {"method": "pair_center", "seed": 2, "val_auc_50": 0.49},
            {"method": "pair_center_random", "seed": 2, "val_auc_50": 0.47},
        ]
        indexed = _index_rows(rows)
        components = component_differences(
            indexed,
            treatment="grad_real",
            control="grad_random_pairing",
            metric="val_auc_50",
        )
        self.assertEqual([seed for seed, _ in components], [1, 2])
        self.assertAlmostEqual(components[0][1], 0.10)
        self.assertAlmostEqual(components[1][1], 0.08)
        interactions = interaction_differences(
            indexed,
            full_treatment="grad_real",
            full_control="grad_random_pairing",
            base_treatment="pair_center",
            base_control="pair_center_random",
            metric="val_auc_50",
        )
        self.assertAlmostEqual(interactions[0][1], 0.07)
        self.assertAlmostEqual(interactions[1][1], 0.06)

    def test_completion_check_rejects_corrupt_or_unfinished_runs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv_path = root / "run.csv"
            console_path = root / "run.log"
            csv_path.write_text("round,val_acc\n1,0.1\n2,0.2\n", encoding="utf-8")
            console_path.write_text(
                "test_acc: 0.2\nlast_test_acc: 0.19\n", encoding="utf-8"
            )
            self.assertEqual(_completed_rounds(csv_path), 2)
            self.assertTrue(_has_final_test_metrics(console_path))

            csv_path.write_text("round,val_acc\n1,0.1\n3,0.2\n", encoding="utf-8")
            self.assertEqual(_completed_rounds(csv_path), 0)
            console_path.write_text("test_acc: 0.2\n", encoding="utf-8")
            self.assertFalse(_has_final_test_metrics(console_path))


if __name__ == "__main__":
    unittest.main()
