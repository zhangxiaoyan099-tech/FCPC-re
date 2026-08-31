from __future__ import annotations

import unittest

import numpy as np

from src.fcpc.jsdn import (
    build_jsdn_matrix,
    build_pair_complementarity_matrix,
    jsdn_score,
    pair_complementarity_score,
)


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


class PairComplementarityTests(unittest.TestCase):
    def test_matrix_matches_scalar_definition(self) -> None:
        distributions = [[9, 1, 0], [1, 2, 7], [0, 4, 6]]
        counts = [10, 30, 20]
        matrix = build_pair_complementarity_matrix(
            distributions,
            counts,
            block_size=2,
        )
        for i in range(3):
            self.assertEqual(matrix[i, i], 0.0)
            for j in range(i + 1, 3):
                expected = pair_complementarity_score(
                    distributions[i],
                    distributions[j],
                    counts[i],
                    counts[j],
                    total_count=sum(counts),
                )
                self.assertAlmostEqual(matrix[i, j], expected, places=12)
                self.assertAlmostEqual(matrix[j, i], expected, places=12)

    def test_score_is_exact_pair_mixture_kl_reduction(self) -> None:
        distributions = np.asarray(
            [
                [0.90, 0.10, 0.00],
                [0.05, 0.15, 0.80],
                [0.60, 0.30, 0.10],
                [0.10, 0.70, 0.20],
            ],
            dtype=float,
        )
        counts = np.asarray([10.0, 30.0, 20.0, 40.0])
        total = float(counts.sum())
        global_distribution = (counts[:, None] * distributions).sum(axis=0) / total

        def kl(p, q):
            mask = p > 0.0
            return float(np.sum(p[mask] * np.log2(p[mask] / q[mask])))

        fixed_client_kl = sum(
            (counts[i] / total) * kl(distributions[i], global_distribution)
            for i in range(4)
        )
        for pairs in [[(0, 1), (2, 3)], [(0, 2), (1, 3)], [(0, 3), (1, 2)]]:
            score_sum = 0.0
            pair_mixture_kl = 0.0
            for i, j in pairs:
                pair_count = counts[i] + counts[j]
                mixture = (
                    counts[i] * distributions[i]
                    + counts[j] * distributions[j]
                ) / pair_count
                score_sum += pair_complementarity_score(
                    distributions[i],
                    distributions[j],
                    counts[i],
                    counts[j],
                    total_count=total,
                )
                pair_mixture_kl += (pair_count / total) * kl(
                    mixture,
                    global_distribution,
                )
            self.assertAlmostEqual(
                fixed_client_kl,
                score_sum + pair_mixture_kl,
                places=10,
            )


if __name__ == "__main__":
    unittest.main()
