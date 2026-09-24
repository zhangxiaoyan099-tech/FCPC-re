"""Run the seven-cell causal ablation for the original FCPC update.

Every treatment uses the original quadratic partner penalty.  The cells only
change whether the penalty is active and which frozen model is used as its
reference.  No FCPC-grad center, exact proximal step, or maximum-weight
matching is used here.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import subprocess
import sys
from pathlib import Path

from scripts.run_cifar10_full_comparison import BASE_CONFIG, REPO_ROOT, _deep_update


OUTPUT_ROOT = REPO_ROOT / "outputs" / "original_fcpc_ablation"


def _original_fcpc(*, beta: float, pairing: str, reference: str) -> dict:
    return {
        "algorithm": {"name": "fedavg"},
        "fcpc": {
            "enabled": True,
            "metric": "jsdn",
            "reference_strategy": reference,
            "update_rule": "penalty",
            "lambda_jsdn": 0.3,
            "beta": float(beta),
            "beta_schedule": "constant",
            "min_beta": 0.0,
            "epsilon": 1.0,
            "pairing_strategy": pairing,
            "partner_weighting": "uniform",
        },
    }


METHODS = (
    "fedavg",
    "jsdn_beta0",
    "global_anchor",
    "self_history",
    "random_partner",
    "similar_partner",
    "jsdn_partner",
)


def method_override(method: str, beta: float) -> dict:
    if method == "fedavg":
        return {
            "algorithm": {"name": "fedavg"},
            "fcpc": {"enabled": False, "beta": 0.0},
        }
    if method == "jsdn_beta0":
        return _original_fcpc(
            beta=0.0,
            pairing="fair_greedy_dissimilar",
            reference="partner",
        )
    if method == "global_anchor":
        return _original_fcpc(
            beta=beta,
            pairing="random",
            reference="global_center",
        )
    if method == "self_history":
        return _original_fcpc(
            beta=beta,
            pairing="random",
            reference="self_history",
        )
    if method == "random_partner":
        return _original_fcpc(beta=beta, pairing="random", reference="partner")
    if method == "similar_partner":
        return _original_fcpc(beta=beta, pairing="similar", reference="partner")
    if method == "jsdn_partner":
        return _original_fcpc(
            beta=beta,
            pairing="fair_greedy_dissimilar",
            reference="partner",
        )
    raise ValueError(f"unsupported method: {method}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--methods",
        default="all",
        help="all or a comma-separated subset of the seven registered cells",
    )
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument(
        "--beta",
        type=float,
        default=0.01,
        help="common penalty coefficient for all active-regularizer cells",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def resolve_methods(value: str) -> list[str]:
    if value.strip().lower() == "all":
        return list(METHODS)
    methods = [item.strip().lower() for item in value.split(",") if item.strip()]
    unsupported = [method for method in methods if method not in METHODS]
    if unsupported:
        raise ValueError(f"unsupported methods: {unsupported}; choices={list(METHODS)}")
    return methods


def completed_rounds(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except (csv.Error, OSError, UnicodeError):
        # Interrupted writes (for example, a host reboot during a round) can
        # leave a truncated CSV or embedded NUL bytes.  Such a file is not a
        # completed run and will be replaced by the normal execution path.
        return 0
    try:
        rounds = [int(row["round"]) for row in rows]
    except (KeyError, TypeError, ValueError):
        return 0
    return len(rounds) if rounds == list(range(1, len(rounds) + 1)) else 0


def has_final_test_metrics(path: Path) -> bool:
    if not path.exists():
        return False
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return any(line.startswith("test_acc:") for line in lines) and any(
        line.startswith("last_test_acc:") for line in lines
    )


def require_cuda() -> tuple[str, str]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the original-FCPC ablation")
    return str(torch.cuda.get_device_name(0)), str(torch.version.cuda or "unknown")


def build_config(base: dict, method: str, *, seed: int, rounds: int, beta: float) -> dict:
    config = copy.deepcopy(base)
    _deep_update(config, method_override(method, beta))
    config["seed"] = int(seed)
    # Multi-seed treatment comparisons require the same initialization,
    # augmentation streams, and CUDA kernels within every seed.
    config["reproducibility"] = {"deterministic": True}
    config["evaluation"]["validation_seed"] = int(seed) + 10_000
    config["federated"]["rounds"] = int(rounds)
    config["logging"]["output_dir"] = "outputs/original_fcpc_ablation/logs"
    config["logging"]["checkpoint_dir"] = (
        "outputs/original_fcpc_ablation/checkpoints"
    )
    run_name = (
        f"cifar10_full_a0p1_cpr{config['federated']['clients_per_round']}"
        f"_r{rounds}_{method}_seed{seed}"
    )
    config["logging"]["run_name"] = run_name
    return config


def main() -> None:
    args = parse_args()
    methods = resolve_methods(args.methods)
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    if not seeds:
        raise ValueError("at least one seed is required")
    if args.rounds <= 0:
        raise ValueError("--rounds must be positive")
    if args.beta < 0.0:
        raise ValueError("--beta must be non-negative")

    if args.dry_run:
        print("Dry run: CUDA availability is not required.", flush=True)
    else:
        gpu_name, cuda_version = require_cuda()
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
            config = build_config(
                base,
                method,
                seed=seed,
                rounds=args.rounds,
                beta=args.beta,
            )
            run_name = str(config["logging"]["run_name"])
            config_path = resolved_dir / f"{run_name}.json"
            config_path.write_text(
                json.dumps(config, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            csv_path = REPO_ROOT / config["logging"]["output_dir"] / f"{run_name}.csv"
            console_path = console_dir / f"{run_name}.log"
            done = completed_rounds(csv_path)
            if (
                done >= args.rounds
                and has_final_test_metrics(console_path)
                and not args.force
            ):
                print(f"SKIP completed: {run_name} ({done}/{args.rounds})", flush=True)
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
    print("All requested original-FCPC ablation runs completed.", flush=True)


if __name__ == "__main__":
    main()
