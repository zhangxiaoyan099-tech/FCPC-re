from __future__ import annotations

import unittest

import numpy as np

from src.fcpc.jsdn import build_jsdn_matrix, jsdn_score


class VectorizedJSDNTests(unittest.TestCase):
    def test_matrix_matches_scalar_definition(self) -> None:
        distributions = [
            [8, 2, 0],
            [1, 3, 6],
            [0, 5, 5],
        ]
        counts = [10, 20, 8]
        matrix = build_jsdn_matrix(distributions, counts, lambda_jsdn=0.3, block_size=2)
        for i in range(3):
            self.assertEqual(matrix[i, i], 0.0)
            for j in range(i + 1, 3):
                expected = jsdn_score(
                    distributions[i],
                    distributions[j],
                    counts[i],
                    counts[j],
                    lambda_jsdn=0.3,
                )
                self.assertAlmostEqual(matrix[i, j], expected, places=12)
                self.assertAlmostEqual(matrix[j, i], expected, places=12)

    def test_matrix_is_bounded_and_finite(self) -> None:
        rng = np.random.default_rng(42)
        distributions = rng.dirichlet(np.ones(10), size=25)
        counts = rng.integers(1, 1000, size=25)
        matrix = build_jsdn_matrix(distributions, counts)
        self.assertTrue(np.all(np.isfinite(matrix)))
        self.assertTrue(np.all(matrix >= 0.0))
        self.assertTrue(np.all(matrix <= 1.0))


if __name__ == "__main__":
    unittest.main()
