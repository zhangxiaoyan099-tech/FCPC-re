"""Summarize UCI HAR staged dynamic-client-join runs."""

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path


PATTERN = re.compile(
    r"uci_har_dynamic_(?P<method>.+)_n(?P<initial>\d+)to(?P<final>\d+)"
    r"_join(?P<join>\d+)_r(?P<rounds>\d+)_seed(?P<seed>\d+)$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/uci_har_dynamic_join"),
    )
    return parser.parse_args()


def number(row: dict[str, str], field: str) -> float:
    try:
        value = float(row.get(field, ""))
    except (TypeError, ValueError):
        return float("nan")
    return value if math.isfinite(value) else float("nan")


def finite(values):
    return [float(value) for value in values if math.isfinite(float(value))]


def mean(values):
    values = finite(values)
    return statistics.fmean(values) if values else float("nan")


def sd(values):
    values = finite(values)
    return statistics.stdev(values) if len(values) >= 2 else float("nan")


def console_metric(path: Path, name: str) -> float:
    if not path.exists():
        return float("nan")
    text = path.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(rf"(?m)^{re.escape(name)}:\s*([-+0-9.eE]+)\s*$", text)
    return float(matches[-1]) if matches else float("nan")


def summarize_run(path: Path, console_dir: Path) -> dict[str, object]:
    match = PATTERN.match(path.stem)
    if not match:
        raise ValueError(f"unexpected filename: {path.name}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected = int(match.group("rounds"))
    join_after = int(match.group("join"))
    validation = [number(row, "val_acc") for row in rows]
    pre = validation[:join_after]
    post = validation[join_after:]
    train_clients = [number(row, "train_clients") for row in rows]
    traffic = [number(row, "cumulative_total_bytes") for row in rows]
    return {
        "method": match.group("method"),
        "seed": int(match.group("seed")),
        "initial_clients": int(match.group("initial")),
        "final_clients": int(match.group("final")),
        "join_after_rounds": join_after,
        "expected_rounds": expected,
        "observed_rounds": len(rows),
        "complete": int(len(rows) == expected),
        "last_pre_join_val_acc": pre[-1] if pre else float("nan"),
        "first_post_join_val_acc": post[0] if post else float("nan"),
        "mean_post_join_val_acc": mean(post),
        "last_val_acc": validation[-1] if validation else float("nan"),
        "last_test_acc": console_metric(console_dir / f"{path.stem}.log", "last_test_acc"),
        "mean_pre_join_active_clients": mean(train_clients[:join_after]),
        "mean_post_join_active_clients": mean(train_clients[join_after:]),
        "mean_round_time_s": mean([number(row, "round_time_s") for row in rows]),
        "mean_pairing_time_s": mean([number(row, "pairing_time_s") for row in rows]),
        "cumulative_total_mib": traffic[-1] / (1024 * 1024) if traffic and math.isfinite(traffic[-1]) else float("nan"),
    }


def grouped(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[str(row["method"])].append(row)
    metrics = [
        key
        for key in rows[0]
        if key not in {
            "method",
            "seed",
            "initial_clients",
            "final_clients",
            "join_after_rounds",
            "expected_rounds",
            "observed_rounds",
            "complete",
        }
    ]
    output: list[dict[str, object]] = []
    for method, group in sorted(groups.items()):
        record: dict[str, object] = {
            "method": method,
            "runs": len(group),
            "seeds": ";".join(str(row["seed"]) for row in sorted(group, key=lambda x: int(x["seed"]))),
            "all_complete": int(all(int(row["complete"]) for row in group)),
        }
        for metric in metrics:
            values = [row[metric] for row in group]
            record[f"{metric}_mean"] = mean(values)
            record[f"{metric}_sd"] = sd(values)
        output.append(record)
    return output


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    log_dir = args.output_root / "logs"
    paths = sorted(log_dir.glob("uci_har_dynamic_*.csv"))
    if not paths:
        raise FileNotFoundError(f"no dynamic-join logs in {log_dir}")
    rows = [summarize_run(path, args.output_root / "console") for path in paths]
    groups = grouped(rows)
    write_csv(args.output_root / "summary_by_run.csv", rows)
    write_csv(args.output_root / "summary_by_method.csv", groups)
    print(f"runs={len(rows)}; methods={len(groups)}")
    for row in groups:
        print(
            f"{row['method']}: pre={float(row['last_pre_join_val_acc_mean']):.6f}"
            f"±{float(row['last_pre_join_val_acc_sd']):.6f}; "
            f"first_post={float(row['first_post_join_val_acc_mean']):.6f}"
            f"±{float(row['first_post_join_val_acc_sd']):.6f}; "
            f"mean_post={float(row['mean_post_join_val_acc_mean']):.6f}"
            f"±{float(row['mean_post_join_val_acc_sd']):.6f}"
        )


if __name__ == "__main__":
    main()
