"""Summarize finite-round convergence without using test accuracy for selection."""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path


RUN_PATTERN = re.compile(r"cifar10_r(?P<rounds>\d+)_(?P<method>.+)_seed(?P<seed>\d+)$")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log-dir",
        default="outputs/fcpc_grad_convergence/logs",
    )
    parser.add_argument(
        "--console-dir",
        default="outputs/fcpc_grad_convergence/console",
    )
    parser.add_argument("--thresholds", default="0.50,0.60,0.65")
    parser.add_argument(
        "--output",
        default="outputs/fcpc_grad_convergence/convergence_summary.csv",
    )
    return parser.parse_args()


def _normalized_auc(values: list[float], limit: int) -> float:
    values = values[:limit]
    if not values:
        return float("nan")
    if len(values) == 1:
        return values[0]
    return sum(
        0.5 * (left + right) for left, right in zip(values, values[1:])
    ) / (len(values) - 1)


def _first_round_at(values: list[float], threshold: float) -> int | str:
    for index, value in enumerate(values, start=1):
        if value >= threshold:
            return index
    return ""


def _console_value(path: Path, name: str) -> float:
    if not path.exists():
        return float("nan")
    pattern = re.compile(rf"^{re.escape(name)}:\s*(\S+)\s*$")
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.match(line.strip())
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return float("nan")
    return float("nan")


def _mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return sum(finite) / len(finite) if finite else float("nan")


def _sample_std(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    if len(finite) < 2:
        return 0.0 if finite else float("nan")
    center = sum(finite) / len(finite)
    return math.sqrt(sum((value - center) ** 2 for value in finite) / (len(finite) - 1))


def summarize_run(csv_path: Path, console_dir: Path, thresholds: list[float]) -> dict:
    match = RUN_PATTERN.match(csv_path.stem)
    if not match:
        raise ValueError(f"unexpected convergence filename: {csv_path.name}")
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    val_acc = [float(row["val_acc"]) for row in rows if row.get("val_acc") not in (None, "")]
    if not val_acc:
        raise ValueError(f"no validation curve in {csv_path}")
    best_index = max(range(len(val_acc)), key=lambda index: val_acc[index])
    last_row = rows[-1]
    output = {
        "method": match.group("method"),
        "seed": int(match.group("seed")),
        "rounds": len(rows),
        "val_auc_50": _normalized_auc(val_acc, 50),
        "val_auc_100": _normalized_auc(val_acc, 100),
        "val_auc_all": _normalized_auc(val_acc, len(val_acc)),
        "best_val_acc": val_acc[best_index],
        "best_val_round": best_index + 1,
        "last_val_acc": val_acc[-1],
        "selected_test_acc": _console_value(console_dir / f"{csv_path.stem}.log", "test_acc"),
        "last_test_acc": _console_value(console_dir / f"{csv_path.stem}.log", "last_test_acc"),
        "total_round_time_s": sum(float(row["round_time_s"]) for row in rows),
        "total_bytes": float(last_row.get("cumulative_total_bytes") or 0.0),
    }
    for threshold in thresholds:
        output[f"round_to_{threshold:g}"] = _first_round_at(val_acc, threshold)
    return output


def main() -> None:
    args = _parse_args()
    thresholds = [float(item.strip()) for item in args.thresholds.split(",") if item.strip()]
    log_dir = Path(args.log_dir)
    console_dir = Path(args.console_dir)
    rows = [
        summarize_run(path, console_dir, thresholds)
        for path in sorted(log_dir.glob("cifar10_r*_seed*.csv"))
    ]
    if not rows:
        raise SystemExit(f"no convergence CSV files found under {log_dir}")
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
        f"{'best val':>15} {'selected test':>15}"
    )
    for method in sorted(grouped):
        values = grouped[method]
        columns = []
        for field in ("val_auc_50", "val_auc_100", "best_val_acc", "selected_test_acc"):
            raw = [float(row[field]) for row in values]
            columns.append(f"{100*_mean(raw):6.2f}±{100*_sample_std(raw):5.2f}")
        print(f"{method:<20} {len(values):>3} " + " ".join(f"{value:>15}" for value in columns))
    print(f"summary_path: {output_path}")


if __name__ == "__main__":
    main()
