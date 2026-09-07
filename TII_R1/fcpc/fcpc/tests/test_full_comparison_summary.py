from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from scripts.summarize_full_comparison import _selected_paths, summarize_run


class FullComparisonSummaryTests(unittest.TestCase):
    def test_summary_parses_unified_run_and_validation_curve(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv_path = root / "cifar10_full_a0p1_cpr6_r3_fcpc_grad_seed45.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "round",
                        "val_acc",
                        "round_time_s",
                        "cumulative_total_bytes",
                        "process_cpu_mean_pct",
                        "process_cpu_peak_pct",
                        "rss_peak_mib",
                        "gpu_util_mean_pct",
                        "gpu_util_peak_pct",
                        "gpu_memory_peak_mib",
                    ],
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
        self.assertEqual(row["time_to_0.5_s"], 6.0)
        self.assertEqual(row["bytes_to_0.5"], 30.0)
        self.assertEqual(row["mean_round_time_s"], 2.0)
        self.assertEqual(row["total_round_time_s"], 6.0)
        self.assertEqual(row["total_bytes"], 30.0)

    def test_path_filters_exclude_old_rounds_and_seeds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            names = (
                "cifar10_full_a0p1_cpr6_r200_fedavg_seed45.csv",
                "cifar10_full_a0p1_cpr6_r50_fedavg_seed45.csv",
                "cifar10_full_a0p1_cpr2_r200_fedavg_seed45.csv",
                "cifar10_full_a0p1_cpr6_r200_fedavg_seed46.csv",
            )
            for name in names:
                (root / name).touch()
            selected = _selected_paths(
                root,
                seeds={45},
                rounds=200,
                clients_per_round=6,
            )

        self.assertEqual(
            [path.name for path in selected],
            ["cifar10_full_a0p1_cpr6_r200_fedavg_seed45.csv"],
        )


if __name__ == "__main__":
    unittest.main()
