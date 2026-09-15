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
        grouped[
            (
                int(row["checkpoint_round"]),
                row["method"],
                float(row["step_scale"]),
                float(row["smoothness_L"]),
            )
        ].append(row)

    print(
        f"{'round':>5} {'method':>9} {'s':>5} {'L':>6} {'gate':>8} "
        f"{'L4norm':>7} {'L4par':>7} {'server':>7} {'srv slack':>10} {'-<g,P>':>11} "
        f"{'cos(P,-g)':>11} {'Q(P)':>11} {'Q(Z)':>11} {'obs gain':>11} "
        f"{'||e||':>10} {'epsP':>10} {'qP':>9} {'qZ':>9} {'qZ-LCB':>9} "
        f"{'qP-eps':>9} {'Q(P)>0':>8} {'Q(Z)>0':>8} {'obs>0':>7}"
    )
    for (round_index, method, step_scale, smoothness_l), values in sorted(grouped.items()):
        mean = lambda name: float(np.mean([float(row[name]) for row in values]))
        fraction = lambda name: float(
            np.mean([float(row[name]) > 0.0 for row in values])
        )
        print(
            f"{round_index:5d} {method:>9} {step_scale:5.2g} {smoothness_l:6.2g} "
            f"{mean('proxy_gate_fraction'):8.1%} "
            f"{mean('lemma4_sufficient_fraction'):7.1%} "
            f"{mean('lemma4_parallel_sufficient_fraction'):7.1%} "
            f"{mean('server_parallel_condition_holds'):7.1%} "
            f"{mean('server_parallel_condition_slack'):10.3e} "
            f"{mean('lemma6_projection'):11.3e} "
            f"{mean('P_t_descent_cosine'):11.4f} "
            f"{mean('Q_proxy_vs_baseline'):11.3e} "
            f"{mean('Q_counterfactual'):11.3e} "
            f"{mean('Q_observed_loss_gain'):11.3e} "
            f"{mean('trajectory_error_norm'):10.3e} "
            f"{mean('epsilon_trajectory_bound'):10.3e} "
            f"{mean('proxy_Q_over_grad_sq'):9.3f} "
            f"{mean('Q_over_grad_sq'):9.3f} "
            f"{mean('Q_over_grad_sq') - 1.96 * float(np.std([float(row['Q_over_grad_sq']) for row in values], ddof=1 if len(values) > 1 else 0)) / max(len(values), 1) ** 0.5:9.3f} "
            f"{mean('proxy_minus_epsilon_over_grad_sq'):9.3f} "
            f"{fraction('Q_proxy_vs_baseline'):7.1%} "
            f"{fraction('Q_counterfactual'):7.1%} "
            f"{fraction('Q_observed_loss_gain'):7.1%}"
        )
    print(f"source: {path}")


if __name__ == "__main__":
    main()
