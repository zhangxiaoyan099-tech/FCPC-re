"""Summarize the unified CIFAR-10 comparison across methods and seeds."""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path

from scripts.summarize_fcpc_grad_convergence import (
    _console_value,
    _first_round_at,
    _normalized_auc,
)


RUN_PATTERN = re.compile(
    r"cifar10_full_a0p1_cpr(?P<cpr>\d+)_r(?P<rounds>\d+)_"
    r"(?P<method>.+)_seed(?P<seed>\d+)$"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", default="outputs/cifar10_full_comparison/logs")
    parser.add_argument("--console-dir", default="outputs/cifar10_full_comparison/console")
    parser.add_argument("--thresholds", default="0.50,0.60,0.65,0.70")
    parser.add_argument(
        "--seeds",
        default="",
        help="optional comma-separated seed filter, e.g. 45 or 45,46,47",
    )
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--clients-per-round", type=int, default=None)
    parser.add_argument(
        "--output",
        default="outputs/cifar10_full_comparison/comparison_summary.csv",
    )
    return parser.parse_args()


def _selected_paths(
    log_dir: Path,
    *,
    seeds: set[int],
    rounds: int | None,
    clients_per_round: int | None,
) -> list[Path]:
    selected = []
    for path in sorted(log_dir.glob("cifar10_full_a0p1_cpr*_r*_seed*.csv")):
        match = RUN_PATTERN.match(path.stem)
        if not match:
            continue
        if seeds and int(match.group("seed")) not in seeds:
            continue
        if rounds is not None and int(match.group("rounds")) != rounds:
            continue
        if clients_per_round is not None and int(match.group("cpr")) != clients_per_round:
            continue
        selected.append(path)
    return selected


def _mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return sum(finite) / len(finite) if finite else float("nan")


def _sample_std(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    if len(finite) < 2:
        return 0.0 if finite else float("nan")
    center = sum(finite) / len(finite)
    return math.sqrt(sum((value - center) ** 2 for value in finite) / (len(finite) - 1))


def _float_or_zero(row: dict, field: str) -> float:
    value = row.get(field)
    return float(value) if value not in (None, "") else 0.0


def summarize_run(csv_path: Path, console_dir: Path, thresholds: list[float]) -> dict:
    match = RUN_PATTERN.match(csv_path.stem)
    if not match:
        raise ValueError(f"unexpected comparison filename: {csv_path.name}")
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty comparison CSV: {csv_path}")
    val_rows = [row for row in rows if row.get("val_acc") not in (None, "")]
    val_acc = [float(row["val_acc"]) for row in val_rows]
    if not val_acc:
        raise ValueError(f"no validation curve in {csv_path}")
    best_index = max(range(len(val_acc)), key=val_acc.__getitem__)
    last_row = rows[-1]
    round_times = [_float_or_zero(row, "round_time_s") for row in rows]
    output = {
        "method": match.group("method"),
        "seed": int(match.group("seed")),
        "clients_per_round": int(match.group("cpr")),
        "rounds": len(rows),
        "val_auc_50": _normalized_auc(val_acc, 50),
        "val_auc_100": _normalized_auc(val_acc, 100),
        "val_auc_all": _normalized_auc(val_acc, len(val_acc)),
        "best_val_acc": val_acc[best_index],
        "best_val_round": best_index + 1,
        "last_val_acc": val_acc[-1],
        "selected_test_acc": _console_value(console_dir / f"{csv_path.stem}.log", "test_acc"),
        "last_test_acc": _console_value(console_dir / f"{csv_path.stem}.log", "last_test_acc"),
        "mean_round_time_s": sum(round_times) / len(round_times),
        "total_round_time_s": sum(round_times),
        "total_bytes": float(last_row.get("cumulative_total_bytes") or 0.0),
        "bytes_per_round": float(last_row.get("cumulative_total_bytes") or 0.0) / len(rows),
        "process_cpu_mean_pct": _mean(
            [_float_or_zero(row, "process_cpu_mean_pct") for row in rows]
        ),
        "process_cpu_peak_pct": max(
            _float_or_zero(row, "process_cpu_peak_pct") for row in rows
        ),
        "rss_peak_mib": max(_float_or_zero(row, "rss_peak_mib") for row in rows),
        "gpu_util_mean_pct": _mean(
            [_float_or_zero(row, "gpu_util_mean_pct") for row in rows]
        ),
        "gpu_util_peak_pct": max(
            _float_or_zero(row, "gpu_util_peak_pct") for row in rows
        ),
        "gpu_memory_peak_mib": max(
            _float_or_zero(row, "gpu_memory_peak_mib") for row in rows
        ),
    }
    for threshold in thresholds:
        threshold_round = _first_round_at(val_acc, threshold)
        output[f"round_to_{threshold:g}"] = threshold_round
        if isinstance(threshold_round, int):
            threshold_index = threshold_round - 1
            output[f"time_to_{threshold:g}_s"] = sum(round_times[:threshold_round])
            output[f"bytes_to_{threshold:g}"] = float(
                val_rows[threshold_index].get("cumulative_total_bytes") or 0.0
            )
        else:
            output[f"time_to_{threshold:g}_s"] = ""
            output[f"bytes_to_{threshold:g}"] = ""
    return output


def main() -> None:
    args = _parse_args()
    thresholds = [float(item.strip()) for item in args.thresholds.split(",") if item.strip()]
    seeds = {int(item.strip()) for item in args.seeds.split(",") if item.strip()}
    log_dir = Path(args.log_dir)
    console_dir = Path(args.console_dir)
    rows = [
        summarize_run(path, console_dir, thresholds)
        for path in _selected_paths(
            log_dir,
            seeds=seeds,
            rounds=args.rounds,
            clients_per_round=args.clients_per_round,
        )
    ]
    if not rows:
        raise SystemExit(f"no comparison CSV files found under {log_dir}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["method"])].append(row)
    print(
        f"{'method':<20} {'n':>3} {'AUC50':>15} {'AUC100':>15} "
        f"{'best val':>15} {'selected test':>15} {'sec/round':>12} {'total GB':>12}"
    )
    for method in sorted(grouped):
        values = grouped[method]
        columns = []
        for field in ("val_auc_50", "val_auc_100", "best_val_acc", "selected_test_acc"):
            raw = [float(row[field]) for row in values]
            columns.append(f"{100 * _mean(raw):6.2f}+/-{100 * _sample_std(raw):5.2f}")
        seconds_per_round = [float(row["mean_round_time_s"]) for row in values]
        total_gb = [float(row["total_bytes"]) / 1e9 for row in values]
        time_text = f"{_mean(seconds_per_round):5.2f}+/-{_sample_std(seconds_per_round):4.2f}"
        bytes_text = f"{_mean(total_gb):5.2f}+/-{_sample_std(total_gb):4.2f}"
        print(
            f"{method:<20} {len(values):>3} "
            + " ".join(f"{value:>15}" for value in columns)
            + f" {time_text:>12} {bytes_text:>12}"
        )
    print(f"summary_path: {output_path}")


if __name__ == "__main__":
    main()
