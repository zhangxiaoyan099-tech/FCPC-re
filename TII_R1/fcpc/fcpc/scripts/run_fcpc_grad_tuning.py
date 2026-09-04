"""Tune FCPC-grad on one development seed using validation curves only."""

from __future__ import annotations

import argparse
import copy
import csv
import itertools
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = REPO_ROOT / "configs" / "cifar10_full_comparison_base.yaml"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "fcpc_grad_tuning"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--betas", default="0.01,0.05,0.1,0.2")
    parser.add_argument(
        "--mixes",
        default="0,0.5,1.0",
        help="0 is the matched historical-center control",
    )
    parser.add_argument("--scales", default="0.5")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument(
        "--selection-metric",
        choices=("val_auc", "best_val_acc"),
        default="val_auc",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def _parse_float_grid(value: str, *, name: str, lower: float, upper: float | None) -> list[float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError(f"{name} requires at least one value")
    if len(set(values)) != len(values):
        raise ValueError(f"duplicate {name} values are not allowed: {values}")
    if any(item < lower or (upper is not None and item > upper) for item in values):
        interval = f"[{lower}, {upper}]" if upper is not None else f"[{lower}, infinity)"
        raise ValueError(f"{name} values must be in {interval}: {values}")
    return values


def _tag(value: float) -> str:
    return format(float(value), ".12g").replace("-", "m").replace(".", "p")


def _completed_rounds(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def _validation_stats(path: Path) -> dict[str, float | int]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("val_acc") not in (None, "")]
    if not rows:
        raise ValueError(f"no validation measurements in {path}")
    values = [float(row["val_acc"]) for row in rows]
    best_index = max(range(len(rows)), key=lambda index: values[index])
    val_auc = values[0] if len(values) == 1 else sum(
        0.5 * (left + right) for left, right in zip(values, values[1:])
    ) / (len(values) - 1)
    return {
        "val_auc": val_auc,
        "best_val_acc": values[best_index],
        "best_round": int(rows[best_index]["round"]),
        "last_val_acc": values[-1],
    }


def _build_config(
    base: dict,
    *,
    beta: float,
    gradient_mix: float,
    step_scale: float,
    seed: int,
    rounds: int,
) -> dict:
    config = copy.deepcopy(base)
    config["seed"] = int(seed)
    config["federated"]["rounds"] = int(rounds)
    config["evaluation"]["validation_seed"] = int(seed) + 10_000
    # Hyperparameter search must not inspect the held-out test set.
    config["evaluation"]["evaluate_test"] = False
    config["algorithm"] = {"name": "fedavg"}
    config["fcpc"] = {
        "enabled": True,
        "metric": "pair_complementarity",
        "reference_strategy": "pair_grad_center",
        "update_rule": "proximal",
        "beta": float(beta),
        "beta_schedule": "cosine_decay",
        "min_beta": 0.0,
        "epsilon": 1.0,
        "pairing_strategy": "optimal",
        "partner_weighting": "uniform",
        "grad_center_mix": float(gradient_mix),
        "grad_center_step_scale": float(step_scale),
        "center_max_relative_distance": 0.05,
    }
    config["logging"]["output_dir"] = "outputs/fcpc_grad_tuning/logs"
    config["logging"]["checkpoint_dir"] = "outputs/fcpc_grad_tuning/checkpoints"
    return config


def _require_cuda() -> tuple[str, str]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("FCPC-grad tuning requires CUDA")
    return str(torch.cuda.get_device_name(0)), str(torch.version.cuda or "unknown")


def main() -> None:
    args = _parse_args()
    if args.rounds <= 0:
        raise ValueError("rounds must be positive")
    betas = _parse_float_grid(args.betas, name="beta", lower=0.0, upper=None)
    mixes = _parse_float_grid(args.mixes, name="mix", lower=0.0, upper=1.0)
    scales = _parse_float_grid(args.scales, name="scale", lower=0.0, upper=None)
    if args.dry_run:
        print("Dry run: CUDA availability is not required.", flush=True)
    else:
        gpu_name, cuda_version = _require_cuda()
        print(f"GPU: {gpu_name}; torch CUDA: {cuda_version}", flush=True)

    base = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    resolved_dir = OUTPUT_ROOT / "resolved_configs"
    console_dir = OUTPUT_ROOT / "console"
    resolved_dir.mkdir(parents=True, exist_ok=True)
    console_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, float | int | str]] = []
    failures: list[str] = []
    for beta, mix, scale in itertools.product(betas, mixes, scales):
        config = _build_config(
            base,
            beta=beta,
            gradient_mix=mix,
            step_scale=scale,
            seed=args.seed,
            rounds=args.rounds,
        )
        run_name = (
            f"cifar10_fcpc_grad_r{args.rounds}_b{_tag(beta)}"
            f"_x{_tag(mix)}_s{_tag(scale)}_seed{args.seed}"
        )
        config["logging"]["run_name"] = run_name
        config_path = resolved_dir / f"{run_name}.json"
        config_path.write_text(
            json.dumps(config, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        csv_path = OUTPUT_ROOT / "logs" / f"{run_name}.csv"
        if _completed_rounds(csv_path) < args.rounds or args.force:
            command = [
                sys.executable,
                "-u",
                "-m",
                "src.main",
                "--config",
                str(config_path),
            ]
            if args.dry_run:
                command.append("--dry-run")
            console_path = console_dir / f"{run_name}.log"
            print(f"START {run_name}", flush=True)
            with console_path.open("w", encoding="utf-8") as console:
                completed = subprocess.run(
                    command,
                    cwd=REPO_ROOT,
                    stdout=console,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            if completed.returncode:
                failures.append(run_name)
                print(f"FAILED {run_name}; inspect {console_path}", flush=True)
                if not args.continue_on_error:
                    raise SystemExit(completed.returncode)
                continue
        else:
            print(f"SKIP completed {run_name}", flush=True)

        if not args.dry_run and _completed_rounds(csv_path) >= args.rounds:
            stats = _validation_stats(csv_path)
            results.append(
                {
                    "run_name": run_name,
                    "beta": beta,
                    "grad_center_mix": mix,
                    "grad_center_step_scale": scale,
                    **stats,
                }
            )

    if args.dry_run:
        print("Dry-run configurations completed; no hyperparameter was selected.")
        return
    if not results:
        raise SystemExit(f"no completed tuning results; failures={failures}")

    summary_path = OUTPUT_ROOT / "tuning_summary.csv"
    fields = list(results[0])
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    selected = max(
        results,
        key=lambda row: (
            float(row[args.selection_metric]),
            float(row["best_val_acc"]),
            -float(row["beta"]),
        ),
    )
    selection = {
        "selection_metric": args.selection_metric,
        "tuning_seed": args.seed,
        "tuning_rounds": args.rounds,
        **selected,
    }
    selected_path = OUTPUT_ROOT / "selected_hparams.json"
    selected_path.write_text(
        json.dumps(selection, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print("\nVALIDATION-ONLY SUMMARY (test set was not evaluated)")
    print("beta\tmix\tscale\tval_auc\tbest_val\tbest_round")
    for row in sorted(results, key=lambda item: float(item[args.selection_metric]), reverse=True):
        print(
            f"{float(row['beta']):g}\t{float(row['grad_center_mix']):g}\t"
            f"{float(row['grad_center_step_scale']):g}\t{float(row['val_auc']):.6f}\t"
            f"{float(row['best_val_acc']):.6f}\t{int(row['best_round'])}"
        )
    print(f"SELECTED {json.dumps(selection, ensure_ascii=False)}")
    print(f"summary_path: {summary_path}")
    print(f"selected_hparams_path: {selected_path}")
    if failures:
        raise SystemExit(f"failed runs: {failures}")


if __name__ == "__main__":
    main()
