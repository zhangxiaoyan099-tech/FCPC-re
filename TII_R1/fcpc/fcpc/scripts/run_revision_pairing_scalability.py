"""Repeated FCPC pairing scalability and dynamic-join microbenchmarks.

The benchmark measures the pairing subsystem only; it does not claim training
accuracy under dynamic arrival.  It reports repeated wall time, Python peak
allocation, pairing weight, optimality ratio where feasible, and the fraction
of incumbent clients whose partner changes after new clients join.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import time
import tracemalloc
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.fcpc.jsdn import build_jsdn_matrix
from src.fcpc.pairing import (
    greedy_high_dissimilarity_pairing,
    optimal_high_dissimilarity_pairing,
    pairing_weight,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clients", nargs="+", type=int, default=[10, 50, 100, 500, 1000])
    parser.add_argument("--classes", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--optimal-max-clients", type=int, default=100)
    parser.add_argument("--initial-fraction", type=float, default=0.8)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/revision_pairing_scalability"),
    )
    return parser.parse_args()


def timed_with_peak(function, *args):
    tracemalloc.start()
    started = time.perf_counter()
    result = function(*args)
    elapsed = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, elapsed, peak / (1024 * 1024)


def partner_map(pairing_result) -> dict[int, int]:
    """Return a plain integer partner map from a PairingResult."""
    return {
        int(client_id): int(partner_id)
        for client_id, partner_id in pairing_result.pair_map.items()
    }


def benchmark_once(
    *,
    num_clients: int,
    num_classes: int,
    seed: int,
    optimal_max_clients: int,
    initial_fraction: float,
) -> tuple[dict[str, object], dict[str, object]]:
    rng = np.random.default_rng(seed)
    distributions = rng.dirichlet(np.ones(num_classes), size=num_clients)
    counts = rng.integers(20, 20_000, size=num_clients)

    matrix, matrix_seconds, matrix_peak_mib = timed_with_peak(
        build_jsdn_matrix, distributions, counts
    )
    greedy, greedy_seconds, greedy_peak_mib = timed_with_peak(
        greedy_high_dissimilarity_pairing, matrix
    )
    greedy_weight = float(pairing_weight(greedy, matrix))
    optimal_seconds = float("nan")
    optimal_weight = float("nan")
    greedy_to_optimal_ratio = float("nan")
    if num_clients <= optimal_max_clients:
        started = time.perf_counter()
        optimal = optimal_high_dissimilarity_pairing(matrix)
        optimal_seconds = time.perf_counter() - started
        optimal_weight = float(pairing_weight(optimal, matrix))
        greedy_to_optimal_ratio = (
            greedy_weight / optimal_weight if optimal_weight > 0 else 1.0
        )

    static_row = {
        "num_clients": num_clients,
        "num_classes": num_classes,
        "seed": seed,
        "matrix_seconds": matrix_seconds,
        "matrix_peak_mib": matrix_peak_mib,
        "greedy_seconds": greedy_seconds,
        "greedy_peak_mib": greedy_peak_mib,
        "total_pairing_seconds": matrix_seconds + greedy_seconds,
        "combined_python_peak_upper_mib": matrix_peak_mib + greedy_peak_mib,
        "greedy_pair_count": len(greedy.pairs),
        "greedy_coverage": 2.0 * len(greedy.pairs) / num_clients,
        "greedy_weight": greedy_weight,
        "optimal_seconds": optimal_seconds,
        "optimal_weight": optimal_weight,
        "greedy_to_optimal_ratio": greedy_to_optimal_ratio,
    }

    initial_clients = max(2, min(num_clients - 1, int(round(num_clients * initial_fraction))))
    initial_matrix = build_jsdn_matrix(
        distributions[:initial_clients], counts[:initial_clients]
    )
    initial_pairs = greedy_high_dissimilarity_pairing(initial_matrix)
    initial_map = partner_map(initial_pairs)

    started = time.perf_counter()
    joined_matrix = build_jsdn_matrix(distributions, counts)
    joined_pairs = greedy_high_dissimilarity_pairing(joined_matrix)
    join_recompute_seconds = time.perf_counter() - started
    joined_map = partner_map(joined_pairs)
    incumbent_paired = [client for client in range(initial_clients) if client in initial_map]
    changed = sum(
        joined_map.get(client) != initial_map.get(client) for client in incumbent_paired
    )
    dynamic_row = {
        "final_num_clients": num_clients,
        "initial_num_clients": initial_clients,
        "new_clients": num_clients - initial_clients,
        "num_classes": num_classes,
        "seed": seed,
        "join_recompute_seconds": join_recompute_seconds,
        "initial_pair_count": len(initial_pairs.pairs),
        "joined_pair_count": len(joined_pairs.pairs),
        "joined_coverage": 2.0 * len(joined_pairs.pairs) / num_clients,
        "incumbent_clients_with_partner_before_join": len(incumbent_paired),
        "incumbent_partner_changes": changed,
        "incumbent_partner_churn": (
            changed / len(incumbent_paired) if incumbent_paired else float("nan")
        ),
        "joined_pairing_weight": float(pairing_weight(joined_pairs, joined_matrix)),
    }
    return static_row, dynamic_row


def finite(values):
    return [float(value) for value in values if math.isfinite(float(value))]


def mean(values):
    values = finite(values)
    return statistics.fmean(values) if values else float("nan")


def sd(values):
    values = finite(values)
    return statistics.stdev(values) if len(values) >= 2 else float("nan")


def grouped_summary(rows: list[dict[str, object]], key: str) -> list[dict[str, object]]:
    groups: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[int(row[key])].append(row)
    output: list[dict[str, object]] = []
    excluded = {key, "seed", "num_classes"}
    for group_value, group_rows in sorted(groups.items()):
        record: dict[str, object] = {key: group_value, "repeats": len(group_rows)}
        for field in group_rows[0]:
            if field in excluded:
                continue
            try:
                values = [float(row[field]) for row in group_rows]
            except (TypeError, ValueError):
                continue
            record[f"{field}_mean"] = mean(values)
            record[f"{field}_sd"] = sd(values)
        output.append(record)
    return output


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.repeats < 2:
        raise ValueError("use at least two repeats so a sample standard deviation is defined")
    if not 0.0 < args.initial_fraction < 1.0:
        raise ValueError("--initial-fraction must be in (0, 1)")
    if any(value < 2 for value in args.clients):
        raise ValueError("all client counts must be at least two")

    static_rows: list[dict[str, object]] = []
    dynamic_rows: list[dict[str, object]] = []
    for num_clients in args.clients:
        for repeat in range(args.repeats):
            seed = args.base_seed + repeat
            static_row, dynamic_row = benchmark_once(
                num_clients=num_clients,
                num_classes=args.classes,
                seed=seed,
                optimal_max_clients=args.optimal_max_clients,
                initial_fraction=args.initial_fraction,
            )
            static_rows.append(static_row)
            dynamic_rows.append(dynamic_row)
            print(
                f"n={num_clients}; seed={seed}; "
                f"pairing={float(static_row['total_pairing_seconds']):.6f}s; "
                f"join_recompute={float(dynamic_row['join_recompute_seconds']):.6f}s",
                flush=True,
            )

    static_summary = grouped_summary(static_rows, "num_clients")
    dynamic_summary = grouped_summary(dynamic_rows, "final_num_clients")
    write_csv(args.output_dir / "pairing_raw.csv", static_rows)
    write_csv(args.output_dir / "pairing_summary.csv", static_summary)
    write_csv(args.output_dir / "dynamic_join_raw.csv", dynamic_rows)
    write_csv(args.output_dir / "dynamic_join_summary.csv", dynamic_summary)

    print("\nPAIRING SUMMARY")
    for row in static_summary:
        print(
            f"n={row['num_clients']}: "
            f"total={float(row['total_pairing_seconds_mean']):.6f}"
            f"±{float(row['total_pairing_seconds_sd']):.6f}s; "
            f"peak_upper={float(row['combined_python_peak_upper_mib_mean']):.3f}"
            f"±{float(row['combined_python_peak_upper_mib_sd']):.3f}MiB"
        )
    print("\nDYNAMIC-JOIN SUMMARY")
    for row in dynamic_summary:
        print(
            f"n={row['final_num_clients']}: "
            f"recompute={float(row['join_recompute_seconds_mean']):.6f}"
            f"±{float(row['join_recompute_seconds_sd']):.6f}s; "
            f"incumbent_churn={float(row['incumbent_partner_churn_mean']):.4f}"
            f"±{float(row['incumbent_partner_churn_sd']):.4f}"
        )


if __name__ == "__main__":
    main()
