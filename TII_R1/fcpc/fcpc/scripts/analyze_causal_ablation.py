"""Estimate paired component effects and selected FCPC-grad interactions.

Positive effects always favor the first/named mechanism.  Accuracy and AUC
use treatment minus control; rounds/time to a target use control minus
treatment so that faster convergence remains positive.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from scripts.summarize_paired_convergence import summarize_differences


COMPONENT_CONTRASTS = (
    ("global_proximal", "global_prox", "fedavg"),
    ("pair_history_center", "pair_center", "global_prox"),
    ("gradient_proxy", "grad_real", "pair_center"),
    ("correct_pair_assignment", "grad_real", "grad_shuffled"),
    ("correct_proxy_sign", "grad_real", "grad_reversed"),
    ("complementary_pairing", "grad_real", "grad_random_pairing"),
    ("cosine_beta", "grad_real", "grad_constant_beta"),
    ("batchwise_proximal", "grad_real", "grad_local_end"),
    ("center_clipping", "grad_real", "grad_no_clip"),
)

# Difference in differences:
# (gradient-center effect with mechanism) - (pair-center effect with mechanism).
INTERACTION_CONTRASTS = (
    (
        "pairing_x_gradient_proxy",
        "grad_real",
        "grad_random_pairing",
        "pair_center",
        "pair_center_random",
    ),
    (
        "cosine_beta_x_gradient_proxy",
        "grad_real",
        "grad_constant_beta",
        "pair_center",
        "pair_center_constant_beta",
    ),
    (
        "batchwise_proximal_x_gradient_proxy",
        "grad_real",
        "grad_local_end",
        "pair_center",
        "pair_center_local_end",
    ),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        nargs="?",
        default="outputs/fcpc_grad_causal_ablation/causal_ablation_detail.csv",
    )
    parser.add_argument(
        "--metrics",
        default="val_auc_50,best_val_acc,round_to_0.5,round_to_0.6",
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260917)
    parser.add_argument(
        "--output",
        default="outputs/fcpc_grad_causal_ablation/causal_effects.csv",
    )
    return parser.parse_args()


def _numeric(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _higher_is_better(metric: str) -> bool:
    return not (metric.startswith("round_to_") or metric.startswith("time_to_"))


def _oriented_delta(treatment: float, control: float, metric: str) -> float:
    return treatment - control if _higher_is_better(metric) else control - treatment


def _index_rows(rows: list[Mapping[str, Any]]) -> dict[tuple[str, int], Mapping[str, Any]]:
    indexed = {}
    for row in rows:
        key = (str(row["method"]), int(row["seed"]))
        if key in indexed:
            raise ValueError(f"duplicate method/seed row: {key}")
        indexed[key] = row
    return indexed


def component_differences(
    indexed: Mapping[tuple[str, int], Mapping[str, Any]],
    *,
    treatment: str,
    control: str,
    metric: str,
) -> list[tuple[int, float]]:
    seeds = sorted(
        {seed for method, seed in indexed if method == treatment}
        & {seed for method, seed in indexed if method == control}
    )
    output = []
    for seed in seeds:
        treatment_value = _numeric(indexed[(treatment, seed)].get(metric))
        control_value = _numeric(indexed[(control, seed)].get(metric))
        if treatment_value is None or control_value is None:
            continue
        output.append(
            (seed, _oriented_delta(treatment_value, control_value, metric))
        )
    return output


def interaction_differences(
    indexed: Mapping[tuple[str, int], Mapping[str, Any]],
    *,
    full_treatment: str,
    full_control: str,
    base_treatment: str,
    base_control: str,
    metric: str,
) -> list[tuple[int, float]]:
    method_names = {
        full_treatment,
        full_control,
        base_treatment,
        base_control,
    }
    seed_sets = [
        {seed for method, seed in indexed if method == name}
        for name in method_names
    ]
    seeds = sorted(set.intersection(*seed_sets)) if seed_sets else []
    output = []
    for seed in seeds:
        values = [
            _numeric(indexed[(name, seed)].get(metric))
            for name in (
                full_treatment,
                full_control,
                base_treatment,
                base_control,
            )
        ]
        if any(value is None for value in values):
            continue
        ft, fc, bt, bc = (float(value) for value in values)
        full_effect = _oriented_delta(ft, fc, metric)
        base_effect = _oriented_delta(bt, bc, metric)
        output.append((seed, full_effect - base_effect))
    return output


def main() -> None:
    args = _parse_args()
    with Path(args.path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    indexed = _index_rows(rows)
    metrics = [item.strip() for item in args.metrics.split(",") if item.strip()]
    rng = np.random.default_rng(args.bootstrap_seed)
    output = []

    for name, treatment, control in COMPONENT_CONTRASTS:
        for metric in metrics:
            differences = component_differences(
                indexed,
                treatment=treatment,
                control=control,
                metric=metric,
            )
            output.append(
                {
                    "effect_type": "component",
                    "effect": name,
                    "treatment": treatment,
                    "control": control,
                    "metric": metric,
                    **summarize_differences(
                        differences,
                        bootstrap_resamples=args.bootstrap_resamples,
                        rng=rng,
                    ),
                }
            )

    for name, full_treatment, full_control, base_treatment, base_control in INTERACTION_CONTRASTS:
        for metric in metrics:
            differences = interaction_differences(
                indexed,
                full_treatment=full_treatment,
                full_control=full_control,
                base_treatment=base_treatment,
                base_control=base_control,
                metric=metric,
            )
            output.append(
                {
                    "effect_type": "interaction",
                    "effect": name,
                    "treatment": f"({full_treatment}-{full_control})",
                    "control": f"({base_treatment}-{base_control})",
                    "metric": metric,
                    **summarize_differences(
                        differences,
                        bootstrap_resamples=args.bootstrap_resamples,
                        rng=rng,
                    ),
                }
            )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)

    print(
        f"{'type':<11} {'effect':<38} {'metric':<16} {'n':>3} "
        f"{'mean':>11} {'t-LCB95':>11} {'boot-LCB':>11} {'>0':>7}"
    )
    for row in output:
        print(
            f"{row['effect_type']:<11} {row['effect']:<38} "
            f"{row['metric']:<16} {row['n']:3d} "
            f"{row['mean_delta']:11.4f} {row['paired_t_lcb95']:11.4f} "
            f"{row['bootstrap_lcb95']:11.4f} {row['positive_fraction']:7.1%}"
        )
    print("Positive values favor the named component or positive interaction.")
    print(f"summary_path: {output_path}")


if __name__ == "__main__":
    main()
