from __future__ import annotations

import json
import unittest
from pathlib import Path

from scripts.run_cifar10_full_comparison import BASE_CONFIG
from scripts.run_original_fcpc_ablation import METHODS, build_config, method_override


class OriginalFCPCAblationRunnerTests(unittest.TestCase):
    def test_completion_check_rejects_corrupt_csv(self) -> None:
        from tempfile import TemporaryDirectory

        from scripts.run_original_fcpc_ablation import completed_rounds

        with TemporaryDirectory() as directory:
            path = Path(directory) / "interrupted.csv"
            path.write_bytes(b"round,val_acc\n1,0.1\n\x00broken\n")
            self.assertEqual(completed_rounds(path), 0)

    def setUp(self) -> None:
        self.base = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))

    def test_exactly_seven_registered_cells(self) -> None:
        self.assertEqual(len(METHODS), 7)
        self.assertEqual(len(set(METHODS)), 7)

    def test_all_cells_share_the_training_protocol(self) -> None:
        configs = {
            method: build_config(
                self.base,
                method,
                seed=42,
                rounds=50,
                beta=0.01,
            )
            for method in METHODS
        }
        reference = configs["fedavg"]
        for method, config in configs.items():
            self.assertTrue(config["reproducibility"]["deterministic"], method)
            self.assertEqual(config["federated"], reference["federated"], method)
            self.assertEqual(config["optimizer"], reference["optimizer"], method)
            self.assertEqual(config["scheduler"], reference["scheduler"], method)
            self.assertEqual(config["partition"], reference["partition"], method)
            self.assertEqual(config["model"], reference["model"], method)

    def test_active_cells_use_only_original_penalty(self) -> None:
        for method in METHODS:
            config = method_override(method, 0.01)
            fcpc = config["fcpc"]
            if method == "fedavg":
                self.assertFalse(fcpc["enabled"])
                continue
            self.assertEqual(fcpc["metric"], "jsdn", method)
            self.assertEqual(fcpc["update_rule"], "penalty", method)
            self.assertEqual(fcpc["beta_schedule"], "constant", method)
            self.assertEqual(fcpc["partner_weighting"], "uniform", method)
            self.assertNotIn("grad_center_mix", fcpc, method)
            self.assertNotIn("proximal_frequency", fcpc, method)

    def test_controls_isolate_reference_and_pairing(self) -> None:
        beta = 0.01
        self.assertEqual(
            method_override("global_anchor", beta)["fcpc"]["reference_strategy"],
            "global_center",
        )
        self.assertEqual(
            method_override("self_history", beta)["fcpc"]["reference_strategy"],
            "self_history",
        )
        self.assertEqual(
            method_override("random_partner", beta)["fcpc"]["pairing_strategy"],
            "random",
        )
        self.assertEqual(
            method_override("similar_partner", beta)["fcpc"]["pairing_strategy"],
            "similar",
        )
        self.assertEqual(
            method_override("jsdn_partner", beta)["fcpc"]["pairing_strategy"],
            "fair_greedy_dissimilar",
        )
        self.assertEqual(method_override("jsdn_beta0", beta)["fcpc"]["beta"], 0.0)


if __name__ == "__main__":
    unittest.main()
