from __future__ import annotations

import json
import unittest

from scripts.run_new_fcpc_beta_ablation import (
    BASE_CONFIG,
    _beta_tag,
    _build_config,
    _parse_betas,
)
from scripts.run_cifar10_full_comparison import METHOD_OVERRIDES


class BetaAblationRunnerTests(unittest.TestCase):
    def test_formal_comparison_uses_validation_selected_beta(self) -> None:
        new_fcpc = METHOD_OVERRIDES["new_fcpc"]["fcpc"]
        self.assertEqual(new_fcpc["beta"], 0.001)
        self.assertEqual(new_fcpc["beta_schedule"], "cosine_decay")
        self.assertEqual(new_fcpc["min_beta"], 0.0)

    def test_beta_parser_and_tags_are_stable(self) -> None:
        self.assertEqual(_parse_betas("0.005,0.01,0.2"), [0.005, 0.01, 0.2])
        self.assertEqual(_beta_tag(0.005), "0p005")
        self.assertEqual(_beta_tag(0.2), "0p2")

    def test_rejects_duplicate_or_negative_betas(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate"):
            _parse_betas("0.01,0.01")
        with self.assertRaisesRegex(ValueError, "non-negative"):
            _parse_betas("-0.01")

    def test_build_config_changes_only_ablation_fields(self) -> None:
        base = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
        config = _build_config(base, beta=0.02, seed=42, rounds=200)

        self.assertEqual(config["fcpc"]["beta"], 0.02)
        self.assertEqual(config["fcpc"]["metric"], "pair_complementarity")
        self.assertEqual(config["fcpc"]["reference_strategy"], "pair_center")
        self.assertEqual(config["fcpc"]["pairing_strategy"], "optimal")
        self.assertEqual(config["seed"], 42)
        self.assertEqual(config["evaluation"]["validation_seed"], 10042)
        self.assertEqual(config["federated"]["rounds"], 200)
        self.assertEqual(config["partition"], base["partition"])
        self.assertEqual(config["optimizer"], base["optimizer"])
        self.assertEqual(config["scheduler"], base["scheduler"])


if __name__ == "__main__":
    unittest.main()
