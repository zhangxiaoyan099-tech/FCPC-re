"""Print the key normalized partner-intervention diagnostics."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "summary",
        nargs="?",
        default="outputs/d_m_quick/at_m_summary.csv",
        help="Path to an at_m_summary.csv file",
    )
    return parser.parse_args()


def percent(value: str) -> str:
    return f"{100.0 * float(value):8.3f}%"


def main() -> None:
    path = Path(parse_args().summary)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "D_normalized_mean",
        "D_client_normalized_mean",
        "D_retention_ratio_mean",
        "correction_alignment_cosine_mean",
        "correction_gain_relative_mean",
        "correction_gain_positive_fraction",
    }
    missing = sorted(required - set(rows[0] if rows else []))
    if missing:
        raise ValueError(f"summary does not contain D(M) fields: {missing}")

    print(
        f"{'strategy':<12} {'D/server':>10} {'D/client':>10} "
        f"{'retained':>10} {'align':>9} {'U gain':>10} {'gain>0':>10}"
    )
    for row in rows:
        print(
            f"{row['pairing_strategy']:<12} "
            f"{percent(row['D_normalized_mean'])} "
            f"{percent(row['D_client_normalized_mean'])} "
            f"{percent(row['D_retention_ratio_mean'])} "
            f"{float(row['correction_alignment_cosine_mean']):9.4f} "
            f"{percent(row['correction_gain_relative_mean'])} "
            f"{percent(row['correction_gain_positive_fraction'])}"
        )


if __name__ == "__main__":
    main()
