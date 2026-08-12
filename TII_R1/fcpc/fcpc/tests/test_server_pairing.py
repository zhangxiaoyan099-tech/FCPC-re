from __future__ import annotations

import unittest

import numpy as np

from src.federated.server import Server


class FairOddPairingTests(unittest.TestCase):
    def test_unpaired_role_rotates_without_dropping_clients(self) -> None:
        server = Server(clients=[object(), object(), object()])
        server.pairing_matrix = np.array(
            [
                [0.0, 0.9, 0.2],
                [0.9, 0.0, 0.8],
                [0.2, 0.8, 0.0],
            ]
        )

        unpaired = []
        for round_idx in range(3):
            result = server.pair_selected(
                [0, 1, 2],
                strategy="fair_greedy_dissimilar",
                seed=42 + round_idx,
            )
            unpaired.extend(result.unpaired)
            participating = {client for pair in result.pairs for client in pair}
            participating.update(result.unpaired)
            self.assertEqual(participating, {0, 1, 2})

        self.assertEqual(unpaired, [0, 1, 2])
        self.assertEqual(server.unpaired_counts, {0: 1, 1: 1, 2: 1})


if __name__ == "__main__":
    unittest.main()
