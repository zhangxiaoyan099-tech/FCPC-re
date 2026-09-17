"""Summarize FCPC-grad causal ablations and update-geometry diagnostics."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

from scripts.summarize_full_comparison import (
    RUN_PATTERN,
    _mean,
    _sample_std,
    _selected_paths,
    summarize_run,
)


GEOMETRY_FIELDS = (
    "client_update_second_moment",
    "client_update_variance",
    "server_update_norm",
    "update_cancellation_fraction",
    "mean_pair_update_disagreement",
    "mean_effective_proximal_contraction",
    "std_effective_proximal_contraction",
    "mean_center_clip_scale",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", default="outputs/fcpc_grad_causal_ablation"
    )
    parser.add_argument("--seeds", default="45,46,47")
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--clients-per-round", type=int, default=6)
    parser.add_argument("--thresholds", default="0.50,0.60,0.65,0.70")
    parser.add_argument("--tail", type=int, default=0)
    parser.add_argument("--output", default="")
    return parser.parse_args()


def _geometry_summary(path: Path, tail: int) -> dict[str, float]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if tail > 0:
        rows = rows[-tail:]
    result = {}
    for field in GEOMETRY_FIELDS:
        values = []
        for row in rows:
            try:
                value = float(row.get(field, ""))
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                values.append(value)
        result[f"mean_{field}"] = _mean(values)
    second = result["mean_client_update_second_moment"]
    variance = result["mean_client_update_variance"]
    result["variance_over_second_moment"] = (
        variance / second if math.isfinite(second) and second > 0.0 else float("nan")
    )
    return result


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
    rows = []
    rejected = []
    for path in paths:
        match = RUN_PATTERN.match(path.stem)
        if match is None:
            continue
        summary = summarize_run(path, console_dir, thresholds)
        if int(summary["rounds"]) != args.rounds or not math.isfinite(
            float(summary["selected_test_acc"])
        ):
            rejected.append(path.name)
            continue
        rows.append({**summary, **_geometry_summary(path, args.tail)})
    if not rows:
        raise SystemExit(f"no matching causal-ablation logs under {log_dir}")

    output = Path(args.output) if args.output else root / "causal_ablation_detail.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row["method"])].append(row)
    print(
        f"{'method':<22} {'n':>3} {'AUC50':>13} {'best val':>13} "
        f"{'upd var':>13} {'cancel':>13} {'pair diff':>13} {'rho_eff':>13}"
    )
    for method in sorted(grouped):
        values = grouped[method]

        def cell(field: str, scale: float = 1.0) -> str:
            raw = [float(row[field]) * scale for row in values]
            return f"{_mean(raw):.3e}+/-{_sample_std(raw):.2e}"

        print(
            f"{method:<22} {len(values):3d} "
            f"{cell('val_auc_50', 100):>13} "
            f"{cell('best_val_acc', 100):>13} "
            f"{cell('mean_client_update_variance'):>13} "
            f"{cell('mean_update_cancellation_fraction'):>13} "
            f"{cell('mean_mean_pair_update_disagreement'):>13} "
            f"{cell('mean_mean_effective_proximal_contraction'):>13}"
        )
    print(f"summary_path: {output}")
    if rejected:
        print(
            "warning: excluded incomplete runs: " + ", ".join(rejected)
        )
    print(
        "Interpretation: AUC/best-val higher is better; update variance, "
        "cancellation, and pair disagreement lower indicate stabilization."
    )


if __name__ == "__main__":
    main()
