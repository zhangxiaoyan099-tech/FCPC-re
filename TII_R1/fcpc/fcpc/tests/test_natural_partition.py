from __future__ import annotations

import unittest

from src.data.partition import natural_client_partition


class NaturalClientPartitionTests(unittest.TestCase):
    def test_subject_ids_become_clients_without_sample_loss(self) -> None:
        partition = natural_client_partition([7, 7, 3, 9, 3])
        self.assertEqual(partition, {0: [2, 4], 1: [0, 1], 2: [3]})
        assigned = sorted(index for indices in partition.values() for index in indices)
        self.assertEqual(assigned, [0, 1, 2, 3, 4])


if __name__ == "__main__":
    unittest.main()
