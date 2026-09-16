"""Paired-seed convergence-speed comparison for FCPC-grad and baselines.

Positive deltas always favor the candidate: accuracy/AUC uses
``candidate - baseline`` while rounds/time-to-threshold uses
``baseline - candidate``.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np


T975 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
    26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        nargs="?",
        default="outputs/cifar10_full_comparison/comparison_summary.csv",
    )
    parser.add_argument("--candidate", default="fcpc_grad")
    parser.add_argument(
        "--baselines",
        default="fedavg,fedprox,moon,feddyn_dynamicreg,fblg,fedcfa",
    )
    parser.add_argument(
        "--metrics",
        default="val_auc_50,val_auc_100,round_to_0.6,round_to_0.65,round_to_0.7",
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260916)
    parser.add_argument(
        "--output",
        default="outputs/cifar10_full_comparison/paired_convergence_lcb.csv",
    )
    return parser.parse_args()


def _numeric(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def paired_differences(
    rows: list[Mapping[str, Any]],
    *,
    candidate: str,
    baseline: str,
    metric: str,
) -> list[tuple[int, float]]:
    indexed: dict[tuple[str, int], Mapping[str, Any]] = {}
    for row in rows:
        key = (str(row["method"]), int(row["seed"]))
        if key in indexed:
            raise ValueError(f"duplicate method/seed row: {key}")
        indexed[key] = row
    seeds = sorted(
        {seed for method, seed in indexed if method == candidate}
        & {seed for method, seed in indexed if method == baseline}
    )
    output = []
    reverse = metric.startswith("round_to_") or metric.startswith("time_to_")
    for seed in seeds:
        candidate_value = _numeric(indexed[(candidate, seed)].get(metric))
        baseline_value = _numeric(indexed[(baseline, seed)].get(metric))
        if candidate_value is None or baseline_value is None:
            continue
        delta = (
            baseline_value - candidate_value
            if reverse
            else candidate_value - baseline_value
        )
        output.append((seed, delta))
    return output


def summarize_differences(
    differences: list[tuple[int, float]],
    *,
    bootstrap_resamples: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    values = np.asarray([value for _, value in differences], dtype=np.float64)
    if values.size == 0:
        return {
            "n": 0, "mean_delta": float("nan"), "std_delta": float("nan"),
            "paired_t_lcb95": float("nan"), "bootstrap_lcb95": float("nan"),
            "positive_fraction": float("nan"), "seeds": "",
        }
    mean = float(values.mean())
    if values.size < 2:
        std = 0.0
        t_lcb = float("nan")
        bootstrap_lcb = float("nan")
    else:
        std = float(values.std(ddof=1))
        df = values.size - 1
        critical = T975.get(df, 1.96)
        t_lcb = mean - critical * std / math.sqrt(values.size)
        if bootstrap_resamples <= 0:
            bootstrap_lcb = float("nan")
        else:
            draws = rng.choice(
                values, size=(bootstrap_resamples, values.size), replace=True
            )
            bootstrap_lcb = float(np.quantile(draws.mean(axis=1), 0.025))
    return {
        "n": int(values.size),
        "mean_delta": mean,
        "std_delta": std,
        "paired_t_lcb95": t_lcb,
        "bootstrap_lcb95": bootstrap_lcb,
        "positive_fraction": float(np.mean(values > 0.0)),
        "seeds": ",".join(str(seed) for seed, _ in differences),
    }


def main() -> None:
    args = _parse_args()
    with Path(args.path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    baselines = [value.strip() for value in args.baselines.split(",") if value.strip()]
    metrics = [value.strip() for value in args.metrics.split(",") if value.strip()]
    rng = np.random.default_rng(args.bootstrap_seed)
    output = []
    for baseline in baselines:
        for metric in metrics:
            differences = paired_differences(
                rows,
                candidate=args.candidate,
                baseline=baseline,
                metric=metric,
            )
            output.append(
                {
                    "candidate": args.candidate,
                    "baseline": baseline,
                    "metric": metric,
                    **summarize_differences(
                        differences,
                        bootstrap_resamples=args.bootstrap_resamples,
                        rng=rng,
                    ),
                }
            )
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)

    print(
        f"{'baseline':<20} {'metric':<18} {'n':>3} {'mean delta':>12} "
        f"{'t-LCB95':>12} {'boot-LCB95':>12} {'delta>0':>9}"
    )
    for row in output:
        print(
            f"{row['baseline']:<20} {row['metric']:<18} {row['n']:3d} "
            f"{row['mean_delta']:12.4f} {row['paired_t_lcb95']:12.4f} "
            f"{row['bootstrap_lcb95']:12.4f} {row['positive_fraction']:9.1%}"
        )
    print(f"summary_path: {path}")


if __name__ == "__main__":
    main()
