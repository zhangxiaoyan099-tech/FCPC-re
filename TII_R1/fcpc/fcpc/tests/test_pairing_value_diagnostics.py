from __future__ import annotations

import unittest

import numpy as np

from scripts.run_fcpc_grad_causal_ablation import build_config as build_causal_config
from scripts.run_pairing_value_experiment import (
    METHOD_OVERRIDES,
    build_config as build_pairing_config,
)
from src.fcpc.diagnostics import (
    classification_fairness_metrics,
    pairing_distribution_metrics,
)
from src.fcpc.pairing import PairingResult


class PairingDistributionDiagnosticTests(unittest.TestCase):
    def test_weighted_kl_decomposition_is_exact(self) -> None:
        pairing = PairingResult(
            pairs=[(0, 1)],
            pair_map={0: 1, 1: 0},
            unpaired=[],
        )
        metrics = pairing_distribution_metrics(
            pairing,
            label_histograms=[[10.0, 0.0], [0.0, 10.0]],
            sample_counts=[10, 10],
            pairing_matrix=np.array([[0.0, 1.0], [1.0, 0.0]]),
        )
        self.assertAlmostEqual(metrics["pair_individual_kl_residual"], np.log(2.0))
        self.assertAlmostEqual(metrics["pair_mixture_kl_residual"], 0.0)
        self.assertAlmostEqual(metrics["pair_complementarity_gain"], np.log(2.0))
        self.assertAlmostEqual(metrics["pair_kl_identity_error"], 0.0)
        self.assertAlmostEqual(metrics["mean_pair_label_entropy"], 1.0)
        self.assertAlmostEqual(metrics["mean_pair_class_coverage"], 1.0)

    def test_classification_fairness_is_label_shift_proxy(self) -> None:
        metrics = classification_fairness_metrics(
            [[8, 2], [1, 9]],
            client_label_histograms=[[9, 1], [1, 9]],
        )
        self.assertAlmostEqual(metrics["test_macro_recall"], 0.85)
        self.assertAlmostEqual(metrics["client_accuracy_proxy_min"], 0.81)
        self.assertAlmostEqual(metrics["client_accuracy_proxy_mean"], 0.85)
        self.assertAlmostEqual(metrics["client_accuracy_proxy_p10"], 0.818)
        self.assertLess(metrics["client_accuracy_proxy_jain"], 1.0)


class PairingValueConfigTests(unittest.TestCase):
    @staticmethod
    def _base() -> dict:
        return {
            "seed": 1,
            "algorithm": {"name": "fedavg"},
            "federated": {"rounds": 10, "clients_per_round": 6},
            "evaluation": {"validation_seed": 1},
            "fcpc": {},
            "logging": {},
        }

    def test_pairing_cells_only_change_pairing_strategy(self) -> None:
        configs = {
            method: build_pairing_config(self._base(), method, seed=45, rounds=50)
            for method in METHOD_OVERRIDES
        }
        strategies = {
            method: config["fcpc"]["pairing_strategy"]
            for method, config in configs.items()
        }
        self.assertEqual(
            strategies,
            {
                "pairing_jsd": "optimal",
                "pairing_random": "random",
                "pairing_similar": "similar",
            },
        )
        for config in configs.values():
            self.assertEqual(config["fcpc"]["beta"], 0.2)
            self.assertEqual(config["fcpc"]["beta_schedule"], "constant")
            self.assertEqual(config["fcpc"]["reference_strategy"], "pair_grad_center")
            self.assertEqual(config["fcpc"]["proximal_frequency"], "batch")

    def test_constant_beta_causal_cells_remain_constant(self) -> None:
        for method in (
            "global_prox_constant_beta",
            "grad_constant_beta",
            "grad_constant_beta_reversed",
            "grad_constant_beta_random_pairing",
            "grad_constant_beta_local_end",
        ):
            config = build_causal_config(self._base(), method, seed=45, rounds=50)
            self.assertEqual(config["fcpc"]["beta"], 0.2)
            self.assertEqual(config["fcpc"]["beta_schedule"], "constant")


if __name__ == "__main__":
    unittest.main()
