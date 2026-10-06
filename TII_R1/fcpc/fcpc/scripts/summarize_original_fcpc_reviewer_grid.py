"""Summarize the original-FCPC reviewer scale/sensitivity grid."""

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path


METRICS = (
    "val_auc",
    "best_val_acc",
    "last_val_acc",
    "last_test_acc",
    "mean_round_time_s",
    "mean_pairing_time_s",
    "cumulative_total_mib",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/reviewer_scale_sensitivity"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="one manifest CSV; default combines manifest_*.csv",
    )
    parser.add_argument("--auc-rounds", type=int, default=50)
    return parser.parse_args()


def value(row: dict[str, str], field: str) -> float | None:
    raw = row.get(field, "")
    try:
        number = float(raw)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else float("nan")


def sd(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) >= 2 else float("nan")


def trapezoid_auc(values: list[float], limit: int) -> float:
    selected = values[:limit]
    if not selected:
        return float("nan")
    if len(selected) == 1:
        return selected[0]
    return sum(
        0.5 * (left + right) for left, right in zip(selected, selected[1:])
    ) / (len(selected) - 1)


def console_metric(path: Path, name: str) -> float:
    if not path.exists():
        return float("nan")
    text = path.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(
        rf"(?m)^{re.escape(name)}:\s*([-+0-9.eE]+)\s*$",
        text,
    )
    return float(matches[-1]) if matches else float("nan")


def read_manifests(root: Path, manifest: Path | None) -> list[dict[str, str]]:
    paths = [manifest] if manifest else sorted(root.glob("manifest_*.csv"))
    if not paths:
        raise FileNotFoundError(f"no manifest CSV in {root}")
    by_run: dict[str, dict[str, str]] = {}
    for path in paths:
        if path is None or not path.exists():
            raise FileNotFoundError(path)
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                by_run[row["run_name"]] = row
    return [by_run[name] for name in sorted(by_run)]


def summarize_run(root: Path, manifest_row: dict[str, str], auc_rounds: int) -> dict[str, object]:
    run_name = manifest_row["run_name"]
    log_path = root / "logs" / f"{run_name}.csv"
    console_path = root / "console" / f"{run_name}.log"
    if not log_path.exists():
        rows: list[dict[str, str]] = []
    else:
        with log_path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    validation = [number for row in rows if (number := value(row, "val_acc")) is not None]
    round_time = [number for row in rows if (number := value(row, "round_time_s")) is not None]
    pairing_time = [number for row in rows if (number := value(row, "pairing_time_s")) is not None]
    cumulative = [
        number
        for row in rows
        if (number := value(row, "cumulative_total_bytes")) is not None
    ]
    expected = int(manifest_row["rounds"])
    return {
        "run_name": run_name,
        "sources": manifest_row["sources"],
        "method": manifest_row["method"],
        "num_clients": int(manifest_row["num_clients"]),
        "alpha": float(manifest_row["alpha"]),
        "beta": float(manifest_row["beta"]),
        "lambda_jsdn": float(manifest_row["lambda_jsdn"]),
        "seed": int(manifest_row["seed"]),
        "expected_rounds": expected,
        "observed_rounds": len(rows),
        "complete": int(len(rows) == expected),
        "val_auc": trapezoid_auc(validation, min(auc_rounds, expected)),
        "best_val_acc": max(validation) if validation else float("nan"),
        "last_val_acc": validation[-1] if validation else float("nan"),
        "last_test_acc": console_metric(console_path, "last_test_acc"),
        "mean_round_time_s": mean(round_time),
        "mean_pairing_time_s": mean(pairing_time),
        "cumulative_total_mib": cumulative[-1] / (1024 * 1024) if cumulative else float("nan"),
    }


def group_runs(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    keys = ("sources", "method", "num_clients", "alpha", "beta", "lambda_jsdn")
    groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    output: list[dict[str, object]] = []
    for key, group in sorted(groups.items(), key=lambda item: tuple(str(x) for x in item[0])):
        record = dict(zip(keys, key))
        record["runs"] = len(group)
        record["seeds"] = ";".join(str(row["seed"]) for row in sorted(group, key=lambda x: int(x["seed"])))
        record["all_complete"] = int(all(int(row["complete"]) for row in group))
        for metric in METRICS:
            numbers = [
                float(row[metric])
                for row in group
                if math.isfinite(float(row[metric]))
            ]
            record[f"{metric}_mean"] = mean(numbers)
            record[f"{metric}_sd"] = sd(numbers)
        output.append(record)
    return output


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"no rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    manifests = read_manifests(args.output_root, args.manifest)
    runs = [summarize_run(args.output_root, row, args.auc_rounds) for row in manifests]
    grouped = group_runs(runs)
    write_csv(args.output_root / "summary_by_run.csv", runs)
    write_csv(args.output_root / "summary_by_treatment.csv", grouped)
    incomplete = [row["run_name"] for row in runs if not int(row["complete"])]
    print(f"runs={len(runs)}; treatments={len(grouped)}; incomplete={len(incomplete)}")
    if incomplete:
        print("INCOMPLETE")
        for name in incomplete:
            print(name)
    print(f"run_summary={args.output_root / 'summary_by_run.csv'}")
    print(f"treatment_summary={args.output_root / 'summary_by_treatment.csv'}")


if __name__ == "__main__":
    main()
