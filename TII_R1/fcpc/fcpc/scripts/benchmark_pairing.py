from __future__ import annotations

import argparse
import csv
import time
import tracemalloc
from pathlib import Path

import numpy as np

from src.fcpc.jsdn import build_jsdn_matrix
from src.fcpc.pairing import (
    greedy_high_dissimilarity_pairing,
    optimal_high_dissimilarity_pairing,
    pairing_weight,
)


def benchmark(num_clients: int, num_classes: int, seed: int, run_optimal: bool):
    rng = np.random.default_rng(seed)
    distributions = rng.dirichlet(np.ones(num_classes), size=num_clients)
    counts = rng.integers(20, 20_000, size=num_clients)

    tracemalloc.start()
    started = time.perf_counter()
    matrix = build_jsdn_matrix(distributions, counts)
    matrix_seconds = time.perf_counter() - started
    _, matrix_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    tracemalloc.start()
    started = time.perf_counter()
    greedy = greedy_high_dissimilarity_pairing(matrix)
    greedy_seconds = time.perf_counter() - started
    _, greedy_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    row = {
        "num_clients": num_clients,
        "num_classes": num_classes,
        "matrix_seconds": matrix_seconds,
        "matrix_peak_mib": matrix_peak / (1024 * 1024),
        "greedy_seconds": greedy_seconds,
        "greedy_peak_mib": greedy_peak / (1024 * 1024),
        "greedy_weight": pairing_weight(greedy, matrix),
        "optimal_seconds": "",
        "optimal_weight": "",
        "greedy_to_optimal_ratio": "",
    }
    if run_optimal:
        started = time.perf_counter()
        optimal = optimal_high_dissimilarity_pairing(matrix)
        optimal_seconds = time.perf_counter() - started
        optimal_weight = pairing_weight(optimal, matrix)
        row.update(
            {
                "optimal_seconds": optimal_seconds,
                "optimal_weight": optimal_weight,
                "greedy_to_optimal_ratio": (
                    row["greedy_weight"] / optimal_weight if optimal_weight else 1.0
                ),
            }
        )
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="FCPC pairing scalability benchmark")
    parser.add_argument("--clients", nargs="+", type=int, default=[10, 50, 100, 500, 1000])
    parser.add_argument("--classes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--optimal-max-clients", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("outputs/benchmarks/pairing.csv"))
    args = parser.parse_args()

    rows = [
        benchmark(
            num_clients=n,
            num_classes=args.classes,
            seed=args.seed,
            run_optimal=n <= args.optimal_max_clients,
        )
        for n in args.clients
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
