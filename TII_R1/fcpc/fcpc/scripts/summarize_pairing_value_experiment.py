"""Summarize pairing value, fairness proxies, and convergence under fixed traces."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

from scripts.summarize_fcpc_grad_convergence import _console_value
from scripts.summarize_full_comparison import (
    RUN_PATTERN,
    _mean,
    _sample_std,
    _selected_paths,
    summarize_run,
)


PAIR_FIELDS = (
    "pair_mixture_kl_residual_normalized",
    "pair_complementarity_gain_normalized",
    "pair_kl_identity_error",
    "mean_pair_label_entropy",
    "min_pair_label_entropy",
    "mean_pair_class_coverage",
    "min_pair_class_coverage",
    "mean_selected_pairing_score",
)

FINAL_FIELDS = (
    "test_macro_f1",
    "test_macro_recall",
    "test_worst_class_recall",
    "client_accuracy_proxy_mean",
    "client_accuracy_proxy_min",
    "client_accuracy_proxy_p10",
    "client_accuracy_proxy_std",
    "client_accuracy_proxy_jain",
    "client_minority_recall_proxy_mean",
    "client_minority_recall_proxy_min",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="outputs/fcpc_grad_pairing_value")
    parser.add_argument("--seeds", default="45,46,47")
    parser.add_argument("--rounds", type=int, default=200)
    parser.add_argument("--clients-per-round", type=int, default=6)
    parser.add_argument("--thresholds", default="0.50,0.60,0.65,0.70")
    parser.add_argument("--output", default="")
    return parser.parse_args()


def _mean_field(rows: list[dict[str, str]], field: str) -> float:
    values = []
    for row in rows:
        try:
            value = float(row.get(field, ""))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
    return _mean(values)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _verify_selection_traces(
    paths: list[Path],
    *,
    expected_rounds: int,
) -> None:
    by_seed: dict[int, dict[str, tuple[str, ...]]] = defaultdict(dict)
    for path in paths:
        match = RUN_PATTERN.match(path.stem)
        if match is None:
            continue
        rows = _read_rows(path)
        trace = tuple(row.get("selected_clients", "") for row in rows)
        if len(trace) != expected_rounds or any(not item for item in trace):
            raise ValueError(
                f"{path.name} has no complete selected_clients trace; rerun with "
                "the updated trainer"
            )
        by_seed[int(match.group("seed"))][match.group("method")] = trace

    for seed, traces in sorted(by_seed.items()):
        if len(traces) < 2:
            continue
        reference_method, reference = next(iter(traces.items()))
        for method, trace in traces.items():
            if trace != reference:
                mismatch = next(
                    index + 1
                    for index, (left, right) in enumerate(zip(reference, trace))
                    if left != right
                )
                raise ValueError(
                    f"selection trace mismatch for seed {seed}, round {mismatch}: "
                    f"{reference_method}={reference[mismatch - 1]!r}, "
                    f"{method}={trace[mismatch - 1]!r}"
                )


def main() -> None:
    args = _parse_args()
    root = Path(args.root)
    log_dir = root / "logs"
    console_dir = root / "console"
    seeds = {int(item.strip()) for item in args.seeds.split(",") if item.strip()}
    thresholds = [
        float(item.strip()) for item in args.thresholds.split(",") if item.strip()
    ]
    paths = _selected_paths(
        log_dir,
        seeds=seeds,
        rounds=args.rounds,
        clients_per_round=args.clients_per_round,
    )
    paths = [
        path
        for path in paths
        if RUN_PATTERN.match(path.stem)
        and RUN_PATTERN.match(path.stem).group("method")
        in {"pairing_jsd", "pairing_random", "pairing_similar"}
    ]
    if not paths:
        raise SystemExit(f"no pairing-value logs found under {log_dir}")
    _verify_selection_traces(paths, expected_rounds=args.rounds)

    output_rows = []
    rejected = []
    for path in paths:
        rows = _read_rows(path)
        base = summarize_run(path, console_dir, thresholds)
        if int(base["rounds"]) != args.rounds or not math.isfinite(
            float(base["selected_test_acc"])
        ):
            rejected.append(path.name)
            continue
        console_path = console_dir / f"{path.stem}.log"
        diagnostics = {f"mean_{field}": _mean_field(rows, field) for field in PAIR_FIELDS}
        final_metrics = {
            field: _console_value(console_path, field) for field in FINAL_FIELDS
        }
        output_rows.append({**base, **diagnostics, **final_metrics})
    if not output_rows:
        raise SystemExit("all pairing-value runs were incomplete")

    output_path = (
        Path(args.output) if args.output else root / "pairing_value_summary.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in output_rows:
        grouped[str(row["method"])].append(row)
    print(
        f"{'method':<18} {'n':>3} {'AUC50':>13} {'KL residual':>13} "
        f"{'JS gain':>13} {'macro-F1':>13} {'client min':>13} {'minor recall':>13}"
    )
    for method in sorted(grouped):
        values = grouped[method]

        def cell(field: str, scale: float = 1.0) -> str:
            raw = [float(row[field]) * scale for row in values]
            return f"{_mean(raw):.3e}+/-{_sample_std(raw):.2e}"

        print(
            f"{method:<18} {len(values):3d} "
            f"{cell('val_auc_50', 100):>13} "
            f"{cell('mean_pair_mixture_kl_residual_normalized'):>13} "
            f"{cell('mean_pair_complementarity_gain_normalized'):>13} "
            f"{cell('test_macro_f1', 100):>13} "
            f"{cell('client_accuracy_proxy_min', 100):>13} "
            f"{cell('client_minority_recall_proxy_mean', 100):>13}"
        )
    print("selection_traces_verified: True")
    print("client metrics are label-shift proxies, not observed local-test accuracy")
    print(f"summary_path: {output_path}")
    if rejected:
        print("warning: excluded incomplete runs: " + ", ".join(rejected))


if __name__ == "__main__":
    main()
