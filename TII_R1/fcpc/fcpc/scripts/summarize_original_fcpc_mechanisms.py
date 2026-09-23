"""Summarize accuracy and update-geometry signals for the seven FCPC cells."""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path


RUN_PATTERN = re.compile(
    r"cifar10_full_a0p1_cpr(?P<cpr>\d+)_r(?P<rounds>\d+)_"
    r"(?P<method>.+)_seed(?P<seed>\d+)$"
)
METHOD_ORDER = (
    "fedavg",
    "jsdn_beta0",
    "global_anchor",
    "self_history",
    "random_partner",
    "similar_partner",
    "jsdn_partner",
)


def _float(row: dict[str, str], field: str) -> float:
    value = row.get(field, "")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return sum(finite) / len(finite) if finite else float("nan")


def _auc(values: list[float], limit: int) -> float:
    selected = values[:limit]
    if not selected:
        return float("nan")
    if len(selected) == 1:
        return selected[0]
    return sum(
        0.5 * (left + right) for left, right in zip(selected, selected[1:])
    ) / (len(selected) - 1)


def _mean_field(rows: list[dict[str, str]], field: str, stop: int | None = None) -> float:
    selected = rows if stop is None else rows[:stop]
    return _mean([_float(row, field) for row in selected])


def summarize(path: Path) -> dict[str, float | int | str]:
    match = RUN_PATTERN.match(path.stem)
    if not match:
        raise ValueError(f"unexpected run name: {path.name}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    expected_rounds = int(match.group("rounds"))
    if len(rows) != expected_rounds:
        raise ValueError(f"incomplete run {path.name}: {len(rows)}/{expected_rounds}")
    val = [_float(row, "val_acc") for row in rows]
    best_index = max(range(len(val)), key=val.__getitem__)
    output: dict[str, float | int | str] = {
        "method": match.group("method"),
        "seed": int(match.group("seed")),
        "rounds": len(rows),
        "val_auc_10": _auc(val, 10),
        "val_auc_20": _auc(val, 20),
        "val_auc_50": _auc(val, 50),
        "best_val_acc": val[best_index],
        "best_val_round": best_index + 1,
        "last_val_acc": val[-1],
        "mean_round_time_s": _mean_field(rows, "round_time_s"),
        "early_update_variance": _mean_field(rows, "client_update_variance", 20),
        "early_update_cancellation": _mean_field(
            rows, "update_cancellation_fraction", 20
        ),
        "early_pair_update_disagreement": _mean_field(
            rows, "mean_pair_update_disagreement", 20
        ),
        "mean_fcpc_weighted_loss": _mean_field(rows, "train_fcpc_weighted_loss"),
        "mean_pairing_score": _mean_field(rows, "mean_selected_pairing_score"),
        "mean_pair_complementarity": _mean_field(
            rows, "pair_complementarity_gain_normalized"
        ),
    }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", default="outputs/original_fcpc_ablation/logs")
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        default="outputs/original_fcpc_ablation/mechanism_summary_seed42_r50.csv",
    )
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    summaries: list[dict[str, float | int | str]] = []
    for method in METHOD_ORDER:
        path = log_dir / (
            f"cifar10_full_a0p1_cpr6_r{args.rounds}_{method}_seed{args.seed}.csv"
        )
        summaries.append(summarize(path))

    baseline = summaries[0]
    for row in summaries:
        for field in ("val_auc_10", "val_auc_20", "val_auc_50", "best_val_acc"):
            row[f"delta_{field}_vs_fedavg_pp"] = 100.0 * (
                float(row[field]) - float(baseline[field])
            )
        baseline_variance = float(baseline["early_update_variance"])
        row["early_update_variance_ratio_vs_fedavg"] = (
            float(row["early_update_variance"]) / baseline_variance
            if baseline_variance > 0.0
            else float("nan")
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    print(
        f"{'method':<17} {'AUC10':>7} {'AUC20':>7} {'AUC50':>7} "
        f"{'best':>7} {'dAUC50':>8} {'var/FedAvg':>11} {'cancel':>8}"
    )
    for row in summaries:
        print(
            f"{str(row['method']):<17} "
            f"{100 * float(row['val_auc_10']):7.2f} "
            f"{100 * float(row['val_auc_20']):7.2f} "
            f"{100 * float(row['val_auc_50']):7.2f} "
            f"{100 * float(row['best_val_acc']):7.2f} "
            f"{float(row['delta_val_auc_50_vs_fedavg_pp']):8.2f} "
            f"{float(row['early_update_variance_ratio_vs_fedavg']):11.3f} "
            f"{float(row['early_update_cancellation']):8.3f}"
        )
    print(f"summary_path: {output}")


if __name__ == "__main__":
    main()
