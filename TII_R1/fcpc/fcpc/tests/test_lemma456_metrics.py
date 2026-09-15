from __future__ import annotations

import unittest

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from src.experiments.lemma456 import (
    compute_gain_metrics,
    compute_pair_oracle_gates,
    compute_proxy_chain_metrics,
    proximal_transfer_coefficient,
)
from src.fcpc.pairing import PairingResult


@unittest.skipIf(torch is None, "PyTorch is not installed")
class Lemma456MetricTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pairing = PairingResult(
            pairs=[(0, 1)], pair_map={0: 1, 1: 0}, unpaired=[]
        )
        self.global_state = {"weight": torch.tensor([0.0, 0.0])}
        self.starts = {
            0: {"weight": torch.tensor([0.0, 0.0])},
            1: {"weight": torch.tensor([0.0, 0.0])},
        }
        self.ends = {
            0: {"weight": torch.tensor([-0.1, 0.0])},
            1: {"weight": torch.tensor([-0.1, 0.0])},
        }
        self.gradients = {
            0: torch.tensor([1.0, 0.0]),
            1: torch.tensor([1.0, 0.0]),
        }

    def test_exact_history_proxy_satisfies_lemma4_and_lemma6_identity(self) -> None:
        metrics, rows, proxy = compute_proxy_chain_metrics(
            global_state=self.global_state,
            previous_states=self.ends,
            previous_global_states=self.starts,
            client_gradients=self.gradients,
            sample_counts={0: 1, 1: 1},
            pairing=self.pairing,
            parameter_names=["weight"],
            history_gamma=0.1,
            learning_rate=0.1,
            local_steps=2,
            effective_betas={0: 0.2, 1: 0.2},
            gradient_mix=1.0,
            step_scale=0.5,
        )
        self.assertTrue(metrics["all_pairs_lemma4_sufficient"])
        self.assertTrue(metrics["all_pairs_proxy_favorable"])
        self.assertAlmostEqual(rows[0]["epsilon_pair"], 0.0, places=7)
        self.assertAlmostEqual(metrics["lemma6_identity_gap"], 0.0, places=7)
        self.assertGreater(metrics["lemma6_projection"], 0.0)
        self.assertGreater(float(proxy.norm().item()), 0.0)

    def test_proximal_coefficient_matches_closed_form(self) -> None:
        value = proximal_transfer_coefficient(
            beta=0.2, learning_rate=0.1, local_steps=2
        )
        contraction = 1.0 / 1.04
        self.assertAlmostEqual(value, 1.0 - contraction**2, places=12)

    def test_oracle_gate_rejects_harmful_pair_proxy(self) -> None:
        pairing = PairingResult(
            pairs=[(0, 1), (2, 3)],
            pair_map={0: 1, 1: 0, 2: 3, 3: 2},
            unpaired=[],
        )
        starts = {index: {"weight": torch.zeros(2)} for index in range(4)}
        ends = {
            0: {"weight": torch.tensor([-0.1, 0.0])},
            1: {"weight": torch.tensor([-0.1, 0.0])},
            2: {"weight": torch.tensor([0.1, 0.0])},
            3: {"weight": torch.tensor([0.1, 0.0])},
        }
        gradients = {index: torch.tensor([1.0, 0.0]) for index in range(4)}
        gates, rows = compute_pair_oracle_gates(
            previous_states=ends,
            previous_global_states=starts,
            client_gradients=gradients,
            sample_counts={index: 1 for index in range(4)},
            pairing=pairing,
            parameter_names=["weight"],
        )
        self.assertEqual(gates[(0, 1)], 1.0)
        self.assertEqual(gates[(2, 3)], 0.0)
        self.assertTrue(rows[0]["oracle_gate"])
        self.assertFalse(rows[1]["oracle_gate"])

    def test_proxy_component_respects_pair_gate(self) -> None:
        metrics, rows, proxy = compute_proxy_chain_metrics(
            global_state=self.global_state,
            previous_states=self.ends,
            previous_global_states=self.starts,
            client_gradients=self.gradients,
            sample_counts={0: 1, 1: 1},
            pairing=self.pairing,
            parameter_names=["weight"],
            history_gamma=0.1,
            learning_rate=0.1,
            local_steps=2,
            effective_betas={0: 0.2, 1: 0.2},
            gradient_mix=1.0,
            step_scale=0.5,
            pair_gate_values={(0, 1): 0.0},
        )
        self.assertEqual(rows[0]["proxy_gate"], 0.0)
        self.assertEqual(metrics["proxy_gate_fraction"], 0.0)
        self.assertAlmostEqual(float(proxy.norm().item()), 0.0, places=7)

    def test_gain_inequality_uses_squared_gradient_norm(self) -> None:
        proxy = torch.tensor([-0.02, 0.0])
        baseline = {
            0: {"weight": torch.tensor([-0.1, 0.0])},
            1: {"weight": torch.tensor([-0.1, 0.0])},
        }
        grad = {
            0: {"weight": torch.tensor([-0.12, 0.0])},
            1: {"weight": torch.tensor([-0.12, 0.0])},
        }
        metrics = compute_gain_metrics(
            global_state=self.global_state,
            grad_states=grad,
            baseline_states=baseline,
            client_gradients=self.gradients,
            sample_counts={0: 1, 1: 1},
            parameter_names=["weight"],
            proxy_component=proxy,
            smoothness_l=1.0,
            q_candidate=0.01,
        )
        self.assertAlmostEqual(metrics["trajectory_error_norm"], 0.0, places=7)
        self.assertGreater(metrics["Q_counterfactual"], 0.0)
        self.assertTrue(metrics["inequality_holds"])


if __name__ == "__main__":
    unittest.main()
