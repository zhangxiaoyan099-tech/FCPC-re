"""Summarize reviewer-requested resource fields from completed FCPC CSV logs.

This script does not run training.  It validates the profiler columns already
stored in per-round logs, creates one row per run, and then reports mean and
sample standard deviation across seeds.  Empty or zero-only monitor fields are
flagged instead of being silently reported as measurements.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path


UCI_PATTERN = re.compile(
    r"uci_har_(?P<scenario>feature|concept)_(?P<method>.+)_seed"
    r"(?P<seed>\d+)_r(?P<expected_rounds>\d+)$"
)

MEAN_FIELDS = (
    "pairing_time_s",
    "round_time_s",
    "process_cpu_mean_pct",
    "gpu_util_mean_pct",
)
PEAK_FIELDS = (
    "process_cpu_peak_pct",
    "rss_peak_mib",
    "gpu_util_peak_pct",
    "gpu_memory_peak_mib",
)
BYTE_FIELDS = (
    "server_download_bytes",
    "server_upload_bytes",
    "peer_upload_bytes",
    "round_total_bytes",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("outputs/uci_har_stress_r20/logs"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/uci_har_stress_r20/resource_summary"),
    )
    return parser.parse_args()


def finite_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def mean_or_nan(values: list[float]) -> float:
    return statistics.fmean(values) if values else float("nan")


def max_or_nan(values: list[float]) -> float:
    return max(values) if values else float("nan")


def final_or_sum(rows: list[dict[str, str]]) -> float:
    cumulative = [
        value
        for row in rows
        if (value := finite_float(row.get("cumulative_total_bytes"))) is not None
    ]
    if cumulative:
        return cumulative[-1]
    per_round = [
        value
        for row in rows
        if (value := finite_float(row.get("round_total_bytes"))) is not None
    ]
    return sum(per_round) if per_round else float("nan")


def weighted_gpu_mean(rows: list[dict[str, str]]) -> float:
    numerator = 0.0
    denominator = 0.0
    fallback: list[float] = []
    for row in rows:
        value = finite_float(row.get("gpu_util_mean_pct"))
        if value is None:
            continue
        fallback.append(value)
        count = finite_float(row.get("gpu_sample_count"))
        if count is not None and count > 0:
            numerator += value * count
            denominator += count
    if denominator > 0:
        return numerator / denominator
    return mean_or_nan(fallback)


def parse_identity(path: Path) -> dict[str, object]:
    match = UCI_PATTERN.match(path.stem)
    if match:
        return {
            "scenario": match.group("scenario"),
            "method": match.group("method"),
            "seed": int(match.group("seed")),
            "expected_rounds": int(match.group("expected_rounds")),
        }
    return {
        "scenario": "unparsed",
        "method": path.stem,
        "seed": -1,
        "expected_rounds": -1,
    }


def summarize_run(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = set(reader.fieldnames or [])
    if not rows:
        raise ValueError(f"empty CSV: {path}")

    identity = parse_identity(path)
    summary: dict[str, object] = {
        "file": path.name,
        **identity,
        "observed_rounds": len(rows),
        "complete": int(
            int(identity["expected_rounds"]) < 0
            or len(rows) == int(identity["expected_rounds"])
        ),
    }
    missing = sorted(
        ({"gpu_monitor_backend", "gpu_sample_count", "cumulative_total_bytes"}
         | set(MEAN_FIELDS) | set(PEAK_FIELDS) | set(BYTE_FIELDS))
        - fieldnames
    )
    summary["missing_columns"] = ";".join(missing)

    for field in MEAN_FIELDS:
        values = [
            value
            for row in rows
            if (value := finite_float(row.get(field))) is not None
        ]
        summary[field] = mean_or_nan(values)
    summary["gpu_util_mean_pct"] = weighted_gpu_mean(rows)

    for field in PEAK_FIELDS:
        values = [
            value
            for row in rows
            if (value := finite_float(row.get(field))) is not None
        ]
        summary[field] = max_or_nan(values)

    for field in BYTE_FIELDS:
        values = [
            value
            for row in rows
            if (value := finite_float(row.get(field))) is not None
        ]
        summary[f"total_{field}"] = sum(values) if values else float("nan")

    summary["cumulative_total_bytes"] = final_or_sum(rows)
    summary["cumulative_total_mib"] = (
        float(summary["cumulative_total_bytes"]) / (1024 * 1024)
        if math.isfinite(float(summary["cumulative_total_bytes"]))
        else float("nan")
    )
    sample_counts = [
        value
        for row in rows
        if (value := finite_float(row.get("gpu_sample_count"))) is not None
    ]
    summary["gpu_sample_count_total"] = sum(sample_counts) if sample_counts else 0.0
    backends = sorted(
        {
            str(row.get("gpu_monitor_backend", "")).strip()
            for row in rows
            if str(row.get("gpu_monitor_backend", "")).strip()
        }
    )
    summary["gpu_monitor_backend"] = ";".join(backends)

    cpu_values = [
        float(summary[field])
        for field in ("process_cpu_mean_pct", "process_cpu_peak_pct", "rss_peak_mib")
        if math.isfinite(float(summary[field]))
    ]
    gpu_values = [
        float(summary[field])
        for field in ("gpu_util_mean_pct", "gpu_util_peak_pct", "gpu_memory_peak_mib")
        if math.isfinite(float(summary[field]))
    ]
    summary["cpu_monitor_ok"] = int(bool(cpu_values) and max(cpu_values) > 0)
    summary["gpu_monitor_ok"] = int(
        float(summary["gpu_sample_count_total"]) > 0
        and bool(gpu_values)
        and max(gpu_values) > 0
    )
    flags: list[str] = []
    if missing:
        flags.append("missing_columns")
    if not int(summary["complete"]):
        flags.append("incomplete_run")
    if not int(summary["cpu_monitor_ok"]):
        flags.append("cpu_monitor_unusable")
    if not int(summary["gpu_monitor_ok"]):
        flags.append("gpu_monitor_unusable_or_cpu_run")
    summary["validation_flags"] = ";".join(flags)
    return summary


def sample_sd(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) >= 2 else float("nan")


def summarize_groups(run_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in run_rows:
        groups[(str(row["scenario"]), str(row["method"]))].append(row)

    metrics = (
        "pairing_time_s",
        "round_time_s",
        "process_cpu_mean_pct",
        "process_cpu_peak_pct",
        "rss_peak_mib",
        "gpu_util_mean_pct",
        "gpu_util_peak_pct",
        "gpu_memory_peak_mib",
        "cumulative_total_bytes",
        "cumulative_total_mib",
    )
    output: list[dict[str, object]] = []
    for (scenario, method), rows in sorted(groups.items()):
        record: dict[str, object] = {
            "scenario": scenario,
            "method": method,
            "runs": len(rows),
            "seeds": ";".join(str(row["seed"]) for row in sorted(rows, key=lambda x: int(x["seed"]))),
            "all_complete": int(all(int(row["complete"]) for row in rows)),
            "all_cpu_monitor_ok": int(all(int(row["cpu_monitor_ok"]) for row in rows)),
            "all_gpu_monitor_ok": int(all(int(row["gpu_monitor_ok"]) for row in rows)),
        }
        for metric in metrics:
            values = [
                float(row[metric])
                for row in rows
                if math.isfinite(float(row[metric]))
            ]
            record[f"{metric}_mean"] = mean_or_nan(values)
            record[f"{metric}_sd"] = sample_sd(values)
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
    paths = sorted(args.log_dir.glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"no CSV logs found in {args.log_dir}")
    runs = [summarize_run(path) for path in paths]
    groups = summarize_groups(runs)
    run_output = args.output_dir / "resource_by_run.csv"
    group_output = args.output_dir / "resource_by_method.csv"
    write_csv(run_output, runs)
    write_csv(group_output, groups)

    print(f"run_summary: {run_output}")
    print(f"group_summary: {group_output}")
    print("\nVALIDATION")
    for row in runs:
        print(
            f"{row['file']}: complete={row['complete']}; "
            f"cpu_ok={row['cpu_monitor_ok']}; gpu_ok={row['gpu_monitor_ok']}; "
            f"flags={row['validation_flags'] or 'none'}"
        )
    print("\nGROUPED RESOURCE SUMMARY")
    for row in groups:
        print(
            f"{row['scenario']}/{row['method']}: "
            f"round={float(row['round_time_s_mean']):.6f}"
            f"±{float(row['round_time_s_sd']):.6f}s; "
            f"traffic={float(row['cumulative_total_mib_mean']):.3f}"
            f"±{float(row['cumulative_total_mib_sd']):.3f}MiB; "
            f"CPU_OK={row['all_cpu_monitor_ok']}; GPU_OK={row['all_gpu_monitor_ok']}"
        )


if __name__ == "__main__":
    main()
