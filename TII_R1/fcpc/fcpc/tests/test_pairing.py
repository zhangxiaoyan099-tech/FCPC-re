from __future__ import annotations

import unittest

import numpy as np

from src.fcpc.pairing import (
    greedy_high_dissimilarity_pairing,
    greedy_similarity_pairing,
    optimal_high_dissimilarity_pairing,
    pairing_weight,
    random_pairing,
)


class PairingTests(unittest.TestCase):
    def test_even_number_of_clients_are_all_paired(self) -> None:
        matrix = np.array(
            [
                [0.0, 0.9, 0.2, 0.1],
                [0.9, 0.0, 0.3, 0.2],
                [0.2, 0.3, 0.0, 0.8],
                [0.1, 0.2, 0.8, 0.0],
            ]
        )
        result = greedy_high_dissimilarity_pairing(matrix)
        self.assertEqual(result.pairs, [(0, 1), (2, 3)])
        self.assertEqual(result.unpaired, [])
        self.assertEqual(result.pair_map[0], 1)
        self.assertEqual(result.pair_map[3], 2)

    def test_odd_client_is_unpaired_not_discarded(self) -> None:
        matrix = np.array(
            [
                [0.0, 0.9, 0.1],
                [0.9, 0.0, 0.2],
                [0.1, 0.2, 0.0],
            ]
        )
        result = greedy_high_dissimilarity_pairing(matrix)
        self.assertEqual(result.pairs, [(0, 1)])
        self.assertEqual(result.unpaired, [2])
        self.assertNotIn(2, result.pair_map)

    def test_similarity_pairing_is_a_distinct_ablation(self) -> None:
        matrix = np.array(
            [
                [0.0, 0.9, 0.1, 0.2],
                [0.9, 0.0, 0.3, 0.4],
                [0.1, 0.3, 0.0, 0.8],
                [0.2, 0.4, 0.8, 0.0],
            ]
        )
        result = greedy_similarity_pairing(matrix)
        self.assertEqual(result.pairs, [(0, 2), (1, 3)])

    def test_optimal_pairing_can_improve_on_greedy(self) -> None:
        matrix = np.array(
            [
                [0.0, 10.0, 9.0, 0.0],
                [10.0, 0.0, 0.0, 9.0],
                [9.0, 0.0, 0.0, 1.0],
                [0.0, 9.0, 1.0, 0.0],
            ]
        )
        greedy = greedy_high_dissimilarity_pairing(matrix)
        optimal = optimal_high_dissimilarity_pairing(matrix)
        self.assertEqual(pairing_weight(greedy, matrix), 11.0)
        self.assertEqual(pairing_weight(optimal, matrix), 18.0)

    def test_random_pairing_is_reproducible(self) -> None:
        self.assertEqual(random_pairing(5, seed=7), random_pairing(5, seed=7))


if __name__ == "__main__":
    unittest.main()
