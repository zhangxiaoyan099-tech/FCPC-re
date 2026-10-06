"""Run the minimal UCI HAR staged dynamic-client-join experiment.

The first ``initial_clients`` subject clients participate for
``join_after_rounds`` rounds; the remaining subjects become available from the
next round onward.  This isolates training behavior under staged availability.
The separate pairing microbenchmark measures metadata-registration/rebuild
cost, because this runner preconstructs the pairwise matrix for reproducibility.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = REPO_ROOT / "configs" / "uci_har_natural_fcpc.yaml"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "uci_har_dynamic_join"
METHODS = ("fedavg", "high_jsdn")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, default=BASE_CONFIG)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--join-after-rounds", type=int, default=10)
    parser.add_argument("--initial-clients", type=int, default=14)
    parser.add_argument("--final-clients", type=int, default=21)
    parser.add_argument("--beta", type=float, default=0.2)
    parser.add_argument("--lambda-jsdn", type=float, default=0.3)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument(
        "--data-root",
        default="./data/uci_har_official/UCI HAR Dataset",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def parse_list(value: str, cast):
    output = [cast(item.strip()) for item in value.split(",") if item.strip()]
    if not output:
        raise ValueError("a non-empty comma-separated list is required")
    return output


def validate(args: argparse.Namespace) -> None:
    if not 0 < args.initial_clients < args.final_clients:
        raise ValueError("require 0 < initial_clients < final_clients")
    if not 0 < args.join_after_rounds < args.rounds:
        raise ValueError("join-after-rounds must lie strictly between 0 and rounds")
    if args.beta < 0:
        raise ValueError("beta must be non-negative")
    if not 0.0 <= args.lambda_jsdn <= 1.0:
        raise ValueError("lambda-jsdn must lie in [0, 1]")
    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("validation-fraction must lie in (0, 1)")


def completed_rounds(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        rounds = [int(row["round"]) for row in rows]
    except (OSError, UnicodeError, csv.Error, KeyError, TypeError, ValueError):
        return 0
    return len(rounds) if rounds == list(range(1, len(rounds) + 1)) else 0


def has_final_test(path: Path) -> bool:
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8", errors="replace")
    return "last_test_acc:" in text


def require_cuda() -> tuple[str, str]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the formal dynamic-join runs")
    return str(torch.cuda.get_device_name(0)), str(torch.version.cuda or "unknown")


def build_config(base: dict, method: str, seed: int, args: argparse.Namespace) -> tuple[dict, str]:
    config = copy.deepcopy(base)
    config["seed"] = seed
    config["reproducibility"] = {"deterministic": True}
    config.setdefault("dataset", {})["root"] = args.data_root
    config.setdefault("federated", {}).update(
        {
            "num_clients": args.final_clients,
            "clients_per_round": args.final_clients,
            "rounds": args.rounds,
            "aggregation": "weighted",
            "device": "cuda:0",
        }
    )
    config["algorithm"] = {
        "name": "dynamic_fedavg",
        "initial_clients": args.initial_clients,
        # zero-based index: join after exactly this many completed rounds
        "join_round": args.join_after_rounds,
        "final_clients": args.final_clients,
    }
    config["evaluation"] = {
        "validation_fraction": args.validation_fraction,
        "validation_seed": seed + 10_000,
        "evaluate_test": True,
        "test_every_round": False,
    }
    if method == "fedavg":
        config["fcpc"] = {
            "enabled": False,
            "metric": "jsdn",
            "reference_strategy": "partner",
            "update_rule": "penalty",
            "lambda_jsdn": args.lambda_jsdn,
            "beta": 0.0,
            "beta_schedule": "constant",
            "min_beta": 0.0,
            "epsilon": 1.0,
            "pairing_strategy": "fair_greedy_dissimilar",
            "partner_weighting": "uniform",
        }
    elif method == "high_jsdn":
        config["fcpc"] = {
            "enabled": True,
            "metric": "jsdn",
            "reference_strategy": "partner",
            "update_rule": "penalty",
            "lambda_jsdn": args.lambda_jsdn,
            "beta": args.beta,
            "beta_schedule": "constant",
            "min_beta": 0.0,
            "epsilon": 1.0,
            "pairing_strategy": "fair_greedy_dissimilar",
            "partner_weighting": "uniform",
        }
    else:
        raise ValueError(f"unsupported method: {method}")

    run_name = (
        f"uci_har_dynamic_{method}_n{args.initial_clients}to{args.final_clients}"
        f"_join{args.join_after_rounds}_r{args.rounds}_seed{seed}"
    )
    config.setdefault("logging", {})["output_dir"] = str(
        (args.output_root / "logs").relative_to(REPO_ROOT)
    ).replace("\\", "/")
    config["logging"]["checkpoint_dir"] = str(
        (args.output_root / "checkpoints").relative_to(REPO_ROOT)
    ).replace("\\", "/")
    config["logging"]["run_name"] = run_name
    return config, run_name


def main() -> None:
    args = parse_args()
    validate(args)
    if not args.base_config.is_absolute():
        args.base_config = REPO_ROOT / args.base_config
    if not args.output_root.is_absolute():
        args.output_root = REPO_ROOT / args.output_root
    try:
        args.output_root.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise ValueError("--output-root must remain inside the repository") from exc

    methods = parse_list(args.methods, str)
    unsupported = sorted(set(methods) - set(METHODS))
    if unsupported:
        raise ValueError(f"unsupported methods: {unsupported}; choices={METHODS}")
    seeds = parse_list(args.seeds, int)
    if args.dry_run:
        print("Dry run: CUDA availability is not required.", flush=True)
    else:
        gpu_name, cuda_version = require_cuda()
        print(f"GPU: {gpu_name}; torch CUDA: {cuda_version}", flush=True)

    base = json.loads(args.base_config.read_text(encoding="utf-8"))
    resolved_dir = args.output_root / "resolved_configs"
    console_dir = args.output_root / "console"
    logs_dir = args.output_root / "logs"
    for directory in (resolved_dir, console_dir, logs_dir):
        directory.mkdir(parents=True, exist_ok=True)

    failures: list[str] = []
    for seed in seeds:
        for method in methods:
            config, run_name = build_config(base, method, seed, args)
            config_path = resolved_dir / f"{run_name}.json"
            csv_path = logs_dir / f"{run_name}.csv"
            console_path = console_dir / f"{run_name}.log"
            config_path.write_text(
                json.dumps(config, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            done = completed_rounds(csv_path)
            if done >= args.rounds and has_final_test(console_path) and not args.force:
                print(f"SKIP {run_name} ({done}/{args.rounds})", flush=True)
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
    print("All requested UCI HAR dynamic-join runs completed.", flush=True)


if __name__ == "__main__":
    main()
