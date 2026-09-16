"""Summarize the every-k-round FCPC-grad counterfactual gain audit.

Each ``batch_seed`` is first treated as one complete replay trace.  The script
accumulates Q inside that trace and only then estimates uncertainty across
traces.  It therefore does not incorrectly treat correlated checkpoints from
one trajectory as independent observations.

The round-weighted quantity is a right-endpoint estimate

    Q_hat(1:T) = sum_k (t_k - t_{k-1}) Q_{t_k}.

With checkpoints every five rounds this is ``5 * sum_k Q_{5k}``.  It estimates
the all-round sum; the unweighted sampled sum is also retained explicitly.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import NormalDist
from typing import Any, Iterable, Mapping

import numpy as np


TRACE_KEYS = (
    "model_seed",
    "panel",
    "pairing_strategy",
    "pairing_seed",
    "batch_seed",
    "method",
    "step_scale",
    "smoothness_L",
)

GROUP_KEYS = (
    "panel",
    "pairing_strategy",
    "method",
    "step_scale",
    "smoothness_L",
    "checkpoint_round",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="one or more oracle_metrics.csv files")
    parser.add_argument("--panel", default="raw")
    parser.add_argument("--strategy", default="optimal")
    parser.add_argument("--method", default="ungated")
    parser.add_argument("--step-scale", type=float, default=0.5)
    parser.add_argument("--smoothness-L", type=float, default=0.0)
    parser.add_argument("--expected-step", type=int, default=5)
    parser.add_argument("--first-round", type=int, default=0)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260916)
    parser.add_argument(
        "--output-dir", default="outputs/fcpc_grad_cumulative_summary"
    )
    return parser.parse_args()


def _float(row: Mapping[str, Any], field: str) -> float:
    value = float(row[field])
    if not math.isfinite(value):
        raise ValueError(f"non-finite {field}: {row[field]!r}")
    return value


def _trace_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(row.get(field, "") for field in TRACE_KEYS)


def build_cumulative_curves(
    rows: Iterable[Mapping[str, Any]],
    *,
    first_round: int = 0,
    expected_step: int | None = 5,
) -> list[dict[str, Any]]:
    """Accumulate sampled and round-weighted gains within each replay trace."""
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_trace_key(row)].append(row)

    curves: list[dict[str, Any]] = []
    for key, values in grouped.items():
        ordered = sorted(values, key=lambda row: int(row["checkpoint_round"]))
        previous_round = int(first_round)
        sampled_q = 0.0
        weighted_q = 0.0
        weighted_proxy_q = 0.0
        weighted_observed_q = 0.0
        weighted_grad_sq = 0.0
        seen_rounds: set[int] = set()
        for row in ordered:
            round_number = int(row["checkpoint_round"])
            if round_number in seen_rounds:
                raise ValueError(f"duplicate checkpoint round {round_number} for trace {key}")
            seen_rounds.add(round_number)
            interval = round_number - previous_round
            if interval <= 0:
                raise ValueError(f"non-increasing checkpoints for trace {key}")
            if expected_step is not None and interval != int(expected_step):
                raise ValueError(
                    f"checkpoint interval {interval} != expected {expected_step} "
                    f"at round {round_number} for trace {key}"
                )

            q_actual = _float(row, "Q_counterfactual")
            q_proxy = _float(row, "Q_proxy_vs_baseline")
            q_observed = _float(row, "Q_observed_loss_gain")
            grad_sq = _float(row, "global_gradient_norm_sq")
            sampled_q += q_actual
            weighted_q += interval * q_actual
            weighted_proxy_q += interval * q_proxy
            weighted_observed_q += interval * q_observed
            weighted_grad_sq += interval * grad_sq
            normalized_q = (
                weighted_q / weighted_grad_sq
                if weighted_grad_sq > np.finfo(np.float64).eps
                else float("nan")
            )
            curves.append(
                {
                    **dict(zip(TRACE_KEYS, key)),
                    "checkpoint_round": round_number,
                    "interval_rounds": interval,
                    "sampled_cumulative_Q": sampled_q,
                    "round_weighted_cumulative_Q": weighted_q,
                    "round_weighted_cumulative_Q_proxy": weighted_proxy_q,
                    "round_weighted_cumulative_observed_gain": weighted_observed_q,
                    "round_weighted_cumulative_grad_sq": weighted_grad_sq,
                    "cumulative_Q_over_grad_sq": normalized_q,
                }
            )
            previous_round = round_number
    return curves


def _mean_lcb(
    values: list[float],
    *,
    confidence: float,
    bootstrap_resamples: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "lcb": float("nan"), "bootstrap_lcb": float("nan")}
    mean = float(array.mean())
    if array.size < 2:
        return {"mean": mean, "std": 0.0, "lcb": float("nan"), "bootstrap_lcb": float("nan")}
    std = float(array.std(ddof=1))
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    lcb = mean - z * std / math.sqrt(array.size)
    if bootstrap_resamples <= 0:
        bootstrap_lcb = float("nan")
    else:
        draws = rng.choice(array, size=(bootstrap_resamples, array.size), replace=True)
        alpha = (1.0 - confidence) / 2.0
        bootstrap_lcb = float(np.quantile(draws.mean(axis=1), alpha))
    return {"mean": mean, "std": std, "lcb": lcb, "bootstrap_lcb": bootstrap_lcb}


def summarize_cumulative_curves(
    curves: Iterable[Mapping[str, Any]],
    *,
    confidence: float = 0.95,
    bootstrap_resamples: int = 10_000,
    bootstrap_seed: int = 20260916,
) -> list[dict[str, Any]]:
    """Report replay-level and model-seed-level cumulative statistics."""
    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in curves:
        grouped[tuple(row[field] for field in GROUP_KEYS)].append(row)

    rng = np.random.default_rng(bootstrap_seed)
    summary: list[dict[str, Any]] = []
    for key, values in sorted(grouped.items()):
        q_values = [float(row["round_weighted_cumulative_Q"]) for row in values]
        q_norm_values = [float(row["cumulative_Q_over_grad_sq"]) for row in values]
        proxy_values = [
            float(row["round_weighted_cumulative_Q_proxy"]) for row in values
        ]
        observed_values = [
            float(row["round_weighted_cumulative_observed_gain"]) for row in values
        ]
        replay_q = _mean_lcb(
            q_values,
            confidence=confidence,
            bootstrap_resamples=bootstrap_resamples,
            rng=rng,
        )
        replay_q_norm = _mean_lcb(
            q_norm_values,
            confidence=confidence,
            bootstrap_resamples=bootstrap_resamples,
            rng=rng,
        )
        replay_proxy = _mean_lcb(
            proxy_values,
            confidence=confidence,
            bootstrap_resamples=bootstrap_resamples,
            rng=rng,
        )
        replay_observed = _mean_lcb(
            observed_values,
            confidence=confidence,
            bootstrap_resamples=bootstrap_resamples,
            rng=rng,
        )

        by_model: dict[str, list[float]] = defaultdict(list)
        for row in values:
            by_model[str(row["model_seed"])].append(
                float(row["round_weighted_cumulative_Q"])
            )
        model_values = [float(np.mean(items)) for items in by_model.values()]
        model_q = _mean_lcb(
            model_values,
            confidence=confidence,
            bootstrap_resamples=bootstrap_resamples,
            rng=rng,
        )
        summary.append(
            {
                **dict(zip(GROUP_KEYS, key)),
                "trace_n": len(q_values),
                "model_seed_n": len(by_model),
                "cumulative_Q_mean": replay_q["mean"],
                "cumulative_Q_std": replay_q["std"],
                "cumulative_Q_lcb95_replay": replay_q["lcb"],
                "cumulative_Q_bootstrap_lcb95_replay": replay_q["bootstrap_lcb"],
                "cumulative_Q_positive_fraction": float(np.mean(np.asarray(q_values) > 0.0)),
                "cumulative_q_mean": replay_q_norm["mean"],
                "cumulative_q_lcb95_replay": replay_q_norm["lcb"],
                "cumulative_proxy_Q_mean": replay_proxy["mean"],
                "cumulative_proxy_Q_lcb95_replay": replay_proxy["lcb"],
                "cumulative_observed_gain_mean": replay_observed["mean"],
                "cumulative_observed_gain_lcb95_replay": replay_observed["lcb"],
                "model_level_cumulative_Q_mean": model_q["mean"],
                "model_level_cumulative_Q_lcb95": model_q["lcb"],
                "model_level_cumulative_Q_bootstrap_lcb95": model_q["bootstrap_lcb"],
                "confidence": confidence,
            }
        )
    return summary


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refusing to write empty summary: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = _parse_args()
    if not 0.0 < args.confidence < 1.0:
        raise ValueError("--confidence must lie strictly between zero and one")
    rows: list[dict[str, str]] = []
    for value in args.paths:
        with Path(value).open(newline="", encoding="utf-8") as handle:
            rows.extend(
                row
                for row in csv.DictReader(handle)
                if row["panel"] == args.panel
                and row["pairing_strategy"] == args.strategy
                and row["method"] == args.method
                and math.isclose(float(row["step_scale"]), args.step_scale)
                and math.isclose(float(row["smoothness_L"]), args.smoothness_L)
            )
    if not rows:
        raise SystemExit("no audit rows matched the requested filters")

    curves = build_cumulative_curves(
        rows, first_round=args.first_round, expected_step=args.expected_step
    )
    summary = summarize_cumulative_curves(
        curves,
        confidence=args.confidence,
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
    )
    output_dir = Path(args.output_dir)
    curve_path = output_dir / "cumulative_q_curves.csv"
    summary_path = output_dir / "cumulative_q_summary.csv"
    _write_csv(curve_path, curves)
    _write_csv(summary_path, summary)

    final_round = max(int(row["checkpoint_round"]) for row in summary)
    print(
        f"{'round':>5} {'method':>9} {'L':>6} {'traces':>7} {'models':>7} "
        f"{'cum Q mean':>12} {'LCB95 replay':>13} {'LCB95 model':>12} "
        f"{'q mean':>10} {'q LCB95':>10} {'obs LCB95':>11} {'Q>0':>8}"
    )
    for row in summary:
        if int(row["checkpoint_round"]) != final_round:
            continue
        print(
            f"{int(row['checkpoint_round']):5d} {str(row['method']):>9} "
            f"{float(row['smoothness_L']):6.2g} {int(row['trace_n']):7d} "
            f"{int(row['model_seed_n']):7d} {float(row['cumulative_Q_mean']):12.3e} "
            f"{float(row['cumulative_Q_lcb95_replay']):13.3e} "
            f"{float(row['model_level_cumulative_Q_lcb95']):12.3e} "
            f"{float(row['cumulative_q_mean']):10.3e} "
            f"{float(row['cumulative_q_lcb95_replay']):10.3e} "
            f"{float(row['cumulative_observed_gain_lcb95_replay']):11.3e} "
            f"{float(row['cumulative_Q_positive_fraction']):8.1%}"
        )
    print(f"curve_path: {curve_path}")
    print(f"summary_path: {summary_path}")
    if len({str(row['model_seed']) for row in curves}) < 2:
        print(
            "warning: model-level LCB is unavailable with one model seed; "
            "the replay-level LCB is conditional on the frozen trajectory."
        )


if __name__ == "__main__":
    main()
