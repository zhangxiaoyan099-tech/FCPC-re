"""Print the main pass/fail diagnostics from a Lemma 4--6 audit."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        nargs="?",
        default="outputs/lemma456_quick_seed42/lemma456_metrics.csv",
    )
    parser.add_argument("--panel", default="raw")
    parser.add_argument("--strategy", default="optimal")
    args = parser.parse_args()
    path = Path(args.path)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row["panel"] == args.panel
            and row["pairing_strategy"] == args.strategy
        ]
    if not rows:
        raise SystemExit("no rows matched the requested panel and strategy")
    grouped = defaultdict(list)
    for row in rows:
        grouped[(int(row["checkpoint_round"]), float(row["smoothness_L"]))].append(row)
    print(
        "round      L      L4%    favorable%    -<g,P>    cos(P,-g)      E[Q]  "
        "obs gain    eps_P     q(proxy)    q(actual)  q-gate  bound"
    )
    for (round_number, smoothness_l), values in sorted(grouped.items()):
        mean = lambda name: float(np.mean([float(row[name]) for row in values]))
        print(
            f"{round_number:5d} {smoothness_l:6.2g} "
            f"{100*mean('lemma4_sufficient_fraction'):7.2f} "
            f"{100*mean('proxy_favorable_fraction'):11.2f} "
            f"{mean('lemma6_projection'):10.3e} "
            f"{mean('P_t_descent_cosine'):11.4f} "
            f"{mean('Q_counterfactual'):9.3e} "
            f"{mean('Q_observed_loss_gain'):9.3e} "
            f"{mean('epsilon_trajectory_bound'):9.3e} "
            f"{mean('proxy_Q_over_grad_sq'):11.3e} "
            f"{mean('Q_over_grad_sq'):11.3e} "
            f"{100*mean('proxy_q_condition_holds'):6.1f}% "
            f"{100*mean('inequality_holds'):6.1f}%"
        )
    print(f"source: {path}")
    print(
        "Note: L=0 isolates the first-order projection. Positive L rows are "
        "sensitivity checks unless a valid smoothness bound has been established."
    )


if __name__ == "__main__":
    main()
