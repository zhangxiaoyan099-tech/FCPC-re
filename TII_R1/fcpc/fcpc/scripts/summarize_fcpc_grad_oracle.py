"""Print the clean FCPC-grad oracle counterfactual summary."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", default="outputs/fcpc_grad_oracle_seed42/oracle_metrics.csv")
    parser.add_argument("--panel", default="raw")
    parser.add_argument("--strategy", default="optimal")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.path)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row["panel"] == args.panel
            and row["pairing_strategy"] == args.strategy
        ]
    grouped = defaultdict(list)
    for row in rows:
        grouped[(int(row["checkpoint_round"]), row["method"], float(row["smoothness_L"]))].append(row)

    print(
        f"{'round':>5} {'method':>9} {'L':>6} {'gate':>8} {'-<g,P>':>11} "
        f"{'cos(P,-g)':>11} {'Q(P)':>11} {'Q(Z)':>11} {'obs gain':>11} "
        f"{'P>0':>7} {'Z>0':>7} {'obs>0':>7}"
    )
    for (round_index, method, smoothness_l), values in sorted(grouped.items()):
        mean = lambda name: float(np.mean([float(row[name]) for row in values]))
        fraction = lambda name: float(
            np.mean([float(row[name]) > 0.0 for row in values])
        )
        print(
            f"{round_index:5d} {method:>9} {smoothness_l:6.2g} "
            f"{mean('proxy_gate_fraction'):8.1%} "
            f"{mean('lemma6_projection'):11.3e} "
            f"{mean('P_t_descent_cosine'):11.4f} "
            f"{mean('Q_proxy_vs_baseline'):11.3e} "
            f"{mean('Q_counterfactual'):11.3e} "
            f"{mean('Q_observed_loss_gain'):11.3e} "
            f"{fraction('Q_proxy_vs_baseline'):7.1%} "
            f"{fraction('Q_counterfactual'):7.1%} "
            f"{fraction('Q_observed_loss_gain'):7.1%}"
        )
    print(f"source: {path}")


if __name__ == "__main__":
    main()
