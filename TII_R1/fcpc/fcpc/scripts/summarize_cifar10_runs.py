"""Summarize validation performance and FCPC loss scale from training CSVs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean


def optional_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def summarize(path: Path, tail: int) -> dict[str, object]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty CSV: {path}")

    val_values = [value for row in rows if (value := optional_float(row.get("val_acc"))) is not None]
    test_values = [value for row in rows if (value := optional_float(row.get("test_acc"))) is not None]
    ratios = []
    for row in rows:
        task = optional_float(row.get("train_task_loss"))
        weighted = optional_float(row.get("train_fcpc_weighted_loss"))
        if task is not None and weighted is not None and task > 0:
            ratios.append(weighted / task)

    return {
        "run": path.stem,
        "rounds": len(rows),
        "last_val": val_values[-1] if val_values else None,
        "tail_val_mean": mean(val_values[-tail:]) if val_values else None,
        "best_val": max(val_values) if val_values else None,
        "final_test": test_values[-1] if test_values else None,
        "tail_fcpc_task_ratio": mean(ratios[-tail:]) if ratios else None,
    }


def format_percent(value: object) -> str:
    return "-" if value is None else f"{100.0 * float(value):.2f}%"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument("--directory", type=Path, default=Path("outputs/cifar10_plan"))
    parser.add_argument("--pattern", default="*.csv")
    parser.add_argument("--tail", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = args.paths or sorted(args.directory.glob(args.pattern))
    if not paths:
        raise SystemExit("no matching CSV files")
    summaries = [summarize(path, max(args.tail, 1)) for path in paths]
    summaries.sort(
        key=lambda row: float(row["tail_val_mean"])
        if row["tail_val_mean"] is not None
        else float("-inf"),
        reverse=True,
    )
    print(
        f"{'run':58} {'rnd':>4} {'last val':>10} {'tail val':>10} "
        f"{'best val':>10} {'test':>10} {'FCPC/task':>10}"
    )
    for row in summaries:
        ratio = row["tail_fcpc_task_ratio"]
        ratio_text = "-" if ratio is None else f"{float(ratio):.3e}"
        print(
            f"{str(row['run'])[:58]:58} {int(row['rounds']):4d} "
            f"{format_percent(row['last_val']):>10} "
            f"{format_percent(row['tail_val_mean']):>10} "
            f"{format_percent(row['best_val']):>10} "
            f"{format_percent(row['final_test']):>10} "
            f"{ratio_text:>10}"
        )


if __name__ == "__main__":
    main()
