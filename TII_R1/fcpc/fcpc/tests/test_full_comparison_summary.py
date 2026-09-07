from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from scripts.summarize_full_comparison import summarize_run


class FullComparisonSummaryTests(unittest.TestCase):
    def test_summary_parses_unified_run_and_validation_curve(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv_path = root / "cifar10_full_a0p1_cpr6_r3_fcpc_grad_seed45.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["round", "val_acc", "round_time_s", "cumulative_total_bytes"],
                )
                writer.writeheader()
                writer.writerows(
                    [
                        {"round": 1, "val_acc": 0.2, "round_time_s": 1, "cumulative_total_bytes": 10},
                        {"round": 2, "val_acc": 0.4, "round_time_s": 2, "cumulative_total_bytes": 20},
                        {"round": 3, "val_acc": 0.6, "round_time_s": 3, "cumulative_total_bytes": 30},
                    ]
                )
            row = summarize_run(csv_path, root, [0.5])

        self.assertEqual(row["method"], "fcpc_grad")
        self.assertEqual(row["seed"], 45)
        self.assertEqual(row["clients_per_round"], 6)
        self.assertAlmostEqual(row["val_auc_50"], 0.4)
        self.assertEqual(row["round_to_0.5"], 3)
        self.assertEqual(row["total_round_time_s"], 6.0)
        self.assertEqual(row["total_bytes"], 30.0)


if __name__ == "__main__":
    unittest.main()
