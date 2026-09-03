from __future__ import annotations

import unittest

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from src.experiments.at_m import compute_matching_metrics, pair_mixture_kl_residual
from src.fcpc.jsdn import build_pair_complementarity_matrix
from src.fcpc.pairing import PairingResult, pairing_weight


class AtMDistributionTests(unittest.TestCase):
    def test_weighted_js_identity_matches_pair_residual(self) -> None:
        histograms = {
            0: np.array([8.0, 2.0]),
            1: np.array([1.0, 9.0]),
            2: np.array([6.0, 4.0]),
            3: np.array([2.0, 8.0]),
        }
        counts = {client_id: float(hist.sum()) for client_id, hist in histograms.items()}
        pairing = PairingResult(
            pairs=[(0, 1), (2, 3)],
            pair_map={0: 1, 1: 0, 2: 3, 3: 2},
            unpaired=[],
        )
        matrix = build_pair_complementarity_matrix(
            [histograms[index] for index in range(4)],
            [counts[index] for index in range(4)],
        )
        total = sum(counts.values())
        p_bar = sum(
            (counts[index] / total) * (histograms[index] / histograms[index].sum())
            for index in range(4)
        )

        def kl_bits(p, q):
            positive = p > 0
            return float(np.sum(p[positive] * np.log2(p[positive] / q[positive])))

        individual = sum(
            (counts[index] / total)
            * kl_bits(histograms[index] / histograms[index].sum(), p_bar)
            for index in range(4)
        )
        residual = pair_mixture_kl_residual(histograms, counts, pairing)
        self.assertAlmostEqual(
            individual,
            pairing_weight(pairing, matrix) + residual,
            places=10,
        )


@unittest.skipIf(torch is None, "PyTorch is not installed")
class AtMMetricsTests(unittest.TestCase):

    def test_global_update_error_is_bounded_by_A(self) -> None:
        global_state = {"weight": torch.tensor([0.0, 0.0])}
        client_states = {
            0: {"weight": torch.tensor([-0.10, 0.00])},
            1: {"weight": torch.tensor([-0.08, -0.02])},
            2: {"weight": torch.tensor([0.01, -0.09])},
            3: {"weight": torch.tensor([-0.01, -0.11])},
        }
        gradients = {
            0: torch.tensor([1.0, 0.0]),
            1: torch.tensor([1.0, 0.0]),
            2: torch.tensor([0.0, 1.0]),
            3: torch.tensor([0.0, 1.0]),
        }
        pairing = PairingResult(
            pairs=[(0, 1), (2, 3)],
            pair_map={0: 1, 1: 0, 2: 3, 3: 2},
            unpaired=[],
        )
        metrics, pair_rows = compute_matching_metrics(
            global_state=global_state,
            client_states=client_states,
            client_gradients=gradients,
            sample_counts={0: 1, 1: 1, 2: 1, 3: 1},
            pairing=pairing,
            gamma=0.1,
            parameter_names=["weight"],
        )
        self.assertEqual(len(pair_rows), 2)
        self.assertGreaterEqual(metrics["U_le_A_gap"], -1e-10)
        self.assertLessEqual(metrics["U_t_M"], metrics["A_t_M"] + 1e-10)

    def test_singleton_groups_preserve_complete_coverage(self) -> None:
        pairing = PairingResult(
            pairs=[(0, 1)],
            pair_map={0: 1, 1: 0},
            unpaired=[2],
        )
        metrics, rows = compute_matching_metrics(
            global_state={"weight": torch.tensor([0.0])},
            client_states={
                0: {"weight": torch.tensor([-0.1])},
                1: {"weight": torch.tensor([-0.1])},
                2: {"weight": torch.tensor([-0.1])},
            },
            client_gradients={
                0: torch.tensor([1.0]),
                1: torch.tensor([1.0]),
                2: torch.tensor([1.0]),
            },
            sample_counts={0: 2, 1: 3, 2: 5},
            pairing=pairing,
            gamma=0.1,
            parameter_names=["weight"],
        )
        self.assertEqual(len(rows), 2)
        self.assertAlmostEqual(metrics["A_t_M"], 0.0, places=10)
        self.assertAlmostEqual(metrics["U_t_M"], 0.0, places=10)


if __name__ == "__main__":
    unittest.main()
