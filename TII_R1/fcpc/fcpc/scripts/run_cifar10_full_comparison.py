"""Run reproducible CIFAR-10 baselines on one full-coverage client partition."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = REPO_ROOT / "configs" / "cifar10_full_comparison_base.yaml"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "cifar10_full_comparison"

CORE_METHODS = ("fedavg", "fedprox", "original_fcpc", "new_fcpc")
ALL_METHODS = (
    "fedavg",
    "fedprox",
    "moon",
    "feddyn_dynamicreg",
    "fblg",
    "fedcfa",
    "original_fcpc",
    "new_fcpc",
)

METHOD_OVERRIDES = {
    "fedavg": {
        "algorithm": {"name": "fedavg"},
        "fcpc": {"enabled": False, "beta": 0.0},
    },
    "fedprox": {
        "algorithm": {"name": "fedprox", "mu": 0.01},
        "fcpc": {"enabled": False, "beta": 0.0},
    },
    "moon": {
        "algorithm": {"name": "moon", "mu": 1.0, "temperature": 0.5},
        "fcpc": {"enabled": False, "beta": 0.0},
    },
    "feddyn_dynamicreg": {
        "algorithm": {
            "name": "feddyn",
            "alpha": 0.01,
            "adaptive_alpha": True,
            "max_grad_norm": 10.0,
        },
        "fcpc": {"enabled": False, "beta": 0.0},
    },
    "fblg": {
        "algorithm": {
            "name": "fblg",
            "candidate_ratio": 0.8,
            "epsilon": 0.01,
            "sigma": 0.1,
        },
        "fcpc": {"enabled": False, "beta": 0.0},
    },
    "fedcfa": {
        "algorithm": {
            "name": "fedcfa",
            "topk": 24,
            "rates": "1:5:5",
            "mean_batch_size": 128,
        },
        "fcpc": {"enabled": False, "beta": 0.0},
    },
    "original_fcpc": {
        "algorithm": {"name": "fedavg"},
        "fcpc": {
            "enabled": True,
            "metric": "jsdn",
            "reference_strategy": "partner",
            "lambda_jsdn": 0.3,
            "beta": 0.01,
            "beta_schedule": "cosine_decay",
            "min_beta": 0.0,
            "epsilon": 1.0,
            "pairing_strategy": "fair_greedy_dissimilar",
            "partner_weighting": "sample_ratio",
        },
    },
    "new_fcpc": {
        "algorithm": {"name": "fedavg"},
        "fcpc": {
            "enabled": True,
            "metric": "pair_complementarity",
            "reference_strategy": "pair_center",
            "beta": 0.2,
            "beta_schedule": "cosine_decay",
            "min_beta": 0.0,
            "epsilon": 1.0,
            "pairing_strategy": "optimal",
            "partner_weighting": "uniform",
        },
    },
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sequential, GPU-required CIFAR-10 full-sample comparison"
    )
    parser.add_argument(
        "--methods",
        default="core",
        help="core, all, or comma-separated method names",
    )
    parser.add_argument(
        "--seeds",
        default="42",
        help="comma-separated seeds, e.g. 42,43,44",
    )
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def _resolve_methods(value: str) -> list[str]:
    normalized = value.strip().lower()
    if normalized == "core":
        return list(CORE_METHODS)
    if normalized == "all":
        return list(ALL_METHODS)
    methods = [item.strip().lower() for item in value.split(",") if item.strip()]
    unsupported = [name for name in methods if name not in METHOD_OVERRIDES]
    if unsupported:
        raise ValueError(f"unsupported methods: {unsupported}; choices={list(METHOD_OVERRIDES)}")
    return methods


def _deep_update(target: dict, updates: dict) -> None:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = copy.deepcopy(value)


def _completed_rounds(csv_path: Path) -> int:
    if not csv_path.exists():
        return 0
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def _require_cuda() -> tuple[str, str]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "This comparison requires CUDA, but torch.cuda.is_available() is False"
        )
    return str(torch.cuda.get_device_name(0)), str(torch.version.cuda or "unknown")


def main() -> None:
    args = _parse_args()
    methods = _resolve_methods(args.methods)
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    if not seeds:
        raise ValueError("at least one seed is required")

    gpu_name, cuda_version = _require_cuda()
    print(f"GPU required and detected: {gpu_name}; torch CUDA: {cuda_version}", flush=True)
    base = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    resolved_dir = OUTPUT_ROOT / "resolved_configs"
    console_dir = OUTPUT_ROOT / "console"
    resolved_dir.mkdir(parents=True, exist_ok=True)
    console_dir.mkdir(parents=True, exist_ok=True)

    failures: list[str] = []
    for seed in seeds:
        for method in methods:
            config = copy.deepcopy(base)
            _deep_update(config, METHOD_OVERRIDES[method])
            config["seed"] = seed
            config["evaluation"]["validation_seed"] = seed + 10_000
            if args.rounds is not None:
                config["federated"]["rounds"] = int(args.rounds)
            rounds = int(config["federated"]["rounds"])
            run_name = (
                f"cifar10_full_a0p1_cpr{config['federated']['clients_per_round']}"
                f"_r{rounds}_{method}_seed{seed}"
            )
            config["logging"]["run_name"] = run_name
            config_path = resolved_dir / f"{run_name}.json"
            config_path.write_text(
                json.dumps(config, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            csv_path = REPO_ROOT / config["logging"]["output_dir"] / f"{run_name}.csv"
            completed = _completed_rounds(csv_path)
            if completed >= rounds and not args.force:
                print(f"SKIP completed: {run_name} ({completed}/{rounds} rounds)", flush=True)
                continue

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
            print(f"START {run_name}; console={console_path}", flush=True)
            with console_path.open("w", encoding="utf-8") as console:
                completed_process = subprocess.run(
                    command,
                    cwd=REPO_ROOT,
                    stdout=console,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            if completed_process.returncode:
                failures.append(run_name)
                print(f"FAILED {run_name}; inspect {console_path}", flush=True)
                if not args.continue_on_error:
                    raise SystemExit(completed_process.returncode)
            else:
                print(f"DONE {run_name}", flush=True)

    if failures:
        raise SystemExit(f"failed runs: {failures}")
    print("All requested comparison runs completed.", flush=True)


if __name__ == "__main__":
    main()
