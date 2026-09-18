"""Run the pre-registered FCPC-grad pairing-value experiment on CIFAR-10.

All cells use the same seed, initial model, client sampler, optimizer, constant
beta, gradient center, and batchwise proximal update. Only the pairing strategy
changes. The summarizer verifies the selected-client trace round by round.
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path

from scripts.run_cifar10_full_comparison import BASE_CONFIG, REPO_ROOT, _deep_update
from scripts.run_fcpc_grad_causal_ablation import (
    _completed_rounds,
    _has_final_test_metrics,
    _require_cuda,
)


OUTPUT_ROOT = REPO_ROOT / "outputs" / "fcpc_grad_pairing_value"

FCPC_COMMON = {
    "enabled": True,
    "metric": "pair_complementarity",
    "reference_strategy": "pair_grad_center",
    "update_rule": "proximal",
    "beta": 0.2,
    "beta_schedule": "constant",
    "min_beta": 0.0,
    "epsilon": 1.0,
    "partner_weighting": "uniform",
    "grad_center_mix": 1.0,
    "grad_center_step_scale": 0.5,
    "grad_center_direction": "real",
    "center_max_relative_distance": 0.05,
    "proximal_frequency": "batch",
}

METHOD_OVERRIDES = {
    # Exact maximum-weight matching on the weighted JS complementarity graph.
    "pairing_jsd": {"fcpc": {**FCPC_COMMON, "pairing_strategy": "optimal"}},
    # Negative controls: same selected clients and local optimizer.
    "pairing_random": {"fcpc": {**FCPC_COMMON, "pairing_strategy": "random"}},
    "pairing_similar": {"fcpc": {**FCPC_COMMON, "pairing_strategy": "similar"}},
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", default="all")
    parser.add_argument("--seeds", default="45,46,47")
    parser.add_argument("--rounds", type=int, default=200)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def _resolve_methods(value: str) -> list[str]:
    normalized = value.strip().lower()
    if normalized == "all":
        return list(METHOD_OVERRIDES)
    methods = [item.strip().lower() for item in value.split(",") if item.strip()]
    unsupported = [method for method in methods if method not in METHOD_OVERRIDES]
    if unsupported:
        raise ValueError(
            f"unsupported methods: {unsupported}; choices={list(METHOD_OVERRIDES)}"
        )
    return methods


def build_config(base: dict, method: str, seed: int, rounds: int) -> dict:
    config = copy.deepcopy(base)
    _deep_update(config, {"algorithm": {"name": "fedavg"}})
    _deep_update(config, METHOD_OVERRIDES[method])
    config["seed"] = int(seed)
    config["evaluation"]["validation_seed"] = int(seed) + 10_000
    config["federated"]["rounds"] = int(rounds)
    config["logging"]["output_dir"] = "outputs/fcpc_grad_pairing_value/logs"
    config["logging"]["checkpoint_dir"] = (
        "outputs/fcpc_grad_pairing_value/checkpoints"
    )
    config["logging"]["run_name"] = (
        f"cifar10_full_a0p1_cpr{config['federated']['clients_per_round']}"
        f"_r{rounds}_{method}_seed{seed}"
    )
    return config


def main() -> None:
    args = _parse_args()
    methods = _resolve_methods(args.methods)
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    if not seeds:
        raise ValueError("at least one seed is required")
    if args.rounds <= 0:
        raise ValueError("--rounds must be positive")

    if args.dry_run:
        print("Dry run: CUDA availability is not required.", flush=True)
    else:
        gpu_name, cuda_version = _require_cuda()
        print(
            f"GPU required and detected: {gpu_name}; torch CUDA: {cuda_version}",
            flush=True,
        )

    base = json.loads(BASE_CONFIG.read_text(encoding="utf-8"))
    resolved_dir = OUTPUT_ROOT / "resolved_configs"
    console_dir = OUTPUT_ROOT / "console"
    resolved_dir.mkdir(parents=True, exist_ok=True)
    console_dir.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []

    for seed in seeds:
        for method in methods:
            config = build_config(base, method, seed, args.rounds)
            run_name = config["logging"]["run_name"]
            config_path = resolved_dir / f"{run_name}.json"
            config_path.write_text(
                json.dumps(config, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            csv_path = REPO_ROOT / config["logging"]["output_dir"] / f"{run_name}.csv"
            console_path = console_dir / f"{run_name}.log"
            completed = _completed_rounds(csv_path)
            complete = completed == args.rounds and _has_final_test_metrics(console_path)
            if complete and not args.force:
                print(
                    f"SKIP completed: {run_name} ({completed}/{args.rounds} rounds)",
                    flush=True,
                )
                continue
            if completed and not args.force:
                print(
                    f"RERUN incomplete: {run_name} "
                    f"({completed}/{args.rounds} rounds or final metrics missing)",
                    flush=True,
                )

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
            print(f"START {run_name}; console={console_path}", flush=True)
            with console_path.open("w", encoding="utf-8") as console:
                result = subprocess.run(
                    command,
                    cwd=REPO_ROOT,
                    stdout=console,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            if result.returncode:
                failures.append(run_name)
                print(f"FAILED {run_name}; inspect {console_path}", flush=True)
                if not args.continue_on_error:
                    raise SystemExit(result.returncode)
            else:
                print(f"DONE {run_name}", flush=True)

    if failures:
        raise SystemExit(f"failed runs: {failures}")
    print("All requested pairing-value runs completed.", flush=True)


if __name__ == "__main__":
    main()
