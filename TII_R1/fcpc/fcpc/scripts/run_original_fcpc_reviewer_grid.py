"""Run the minimal reviewer-requested original-FCPC scale/sensitivity grid.

This runner intentionally selects only the original quadratic partner penalty:
``update_rule=penalty`` and ``reference_strategy=partner``.  It never selects
FCPC-grad, a pair center, or an exact proximal update.  Jobs run sequentially
so their timing/resource measurements are not contaminated by GPU contention.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_CONFIG = REPO_ROOT / "configs" / "cifar10_full_comparison_base.yaml"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "reviewer_scale_sensitivity"


@dataclass(frozen=True)
class Treatment:
    method: str
    num_clients: int
    alpha: float
    beta: float
    lambda_jsdn: float
    sources: tuple[str, ...]


def parse_number_list(value: str, cast):
    output = [cast(item.strip()) for item in value.split(",") if item.strip()]
    if not output:
        raise ValueError("a non-empty comma-separated list is required")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study",
        choices=("scale", "beta", "lambda", "all"),
        default="all",
    )
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--scale-clients", default="10,50,100")
    parser.add_argument("--sensitivity-clients", default="10,100")
    parser.add_argument("--alphas", default="0.05,0.5")
    parser.add_argument("--betas", default="0.05,0.2,0.3")
    parser.add_argument("--lambdas", default="0.1,0.3,0.4")
    parser.add_argument("--default-beta", type=float, default=0.2)
    parser.add_argument("--default-lambda", type=float, default=0.3)
    parser.add_argument("--seeds", default="45,46,47")
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument(
        "--keep-base-optimizer",
        action="store_true",
        help="keep optimizer/scheduler from the base config instead of plain SGD",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.rounds <= 0 or args.local_epochs <= 0 or args.batch_size <= 0:
        raise ValueError("rounds, local epochs, and batch size must be positive")
    if args.learning_rate <= 0:
        raise ValueError("learning rate must be positive")
    if args.default_beta < 0:
        raise ValueError("default beta must be non-negative")
    if not 0.0 <= args.default_lambda <= 1.0:
        raise ValueError("default lambda must lie in [0, 1]")


def build_treatments(args: argparse.Namespace) -> list[Treatment]:
    scale_clients = parse_number_list(args.scale_clients, int)
    sensitivity_clients = parse_number_list(args.sensitivity_clients, int)
    alphas = parse_number_list(args.alphas, float)
    betas = parse_number_list(args.betas, float)
    lambdas = parse_number_list(args.lambdas, float)
    if any(value < 2 for value in scale_clients + sensitivity_clients):
        raise ValueError("client counts must be at least two")
    if any(value <= 0 for value in alphas):
        raise ValueError("Dirichlet alpha values must be positive")
    if any(value < 0 for value in betas):
        raise ValueError("beta values must be non-negative")
    if any(not 0.0 <= value <= 1.0 for value in lambdas):
        raise ValueError("lambda values must lie in [0, 1]")

    cells: dict[tuple[str, int, float, float, float], set[str]] = {}

    def add(method, num_clients, alpha, beta, lambda_jsdn, source):
        key = (
            str(method),
            int(num_clients),
            float(alpha),
            float(beta),
            float(lambda_jsdn),
        )
        cells.setdefault(key, set()).add(str(source))

    if args.study in {"scale", "all"}:
        for num_clients in scale_clients:
            for alpha in alphas:
                add("fedavg", num_clients, alpha, 0.0, args.default_lambda, "scale")
                add(
                    "original_fcpc",
                    num_clients,
                    alpha,
                    args.default_beta,
                    args.default_lambda,
                    "scale",
                )

    if args.study in {"beta", "all"}:
        for num_clients in sensitivity_clients:
            for alpha in alphas:
                for beta in betas:
                    add(
                        "original_fcpc",
                        num_clients,
                        alpha,
                        beta,
                        args.default_lambda,
                        "beta",
                    )

    if args.study in {"lambda", "all"}:
        for num_clients in sensitivity_clients:
            for alpha in alphas:
                for lambda_jsdn in lambdas:
                    add(
                        "original_fcpc",
                        num_clients,
                        alpha,
                        args.default_beta,
                        lambda_jsdn,
                        "lambda",
                    )

    return [
        Treatment(*key, sources=tuple(sorted(sources)))
        for key, sources in sorted(cells.items())
    ]


def tag(value: float) -> str:
    return format(value, ".12g").replace("-", "m").replace(".", "p").replace("+", "")


def deep_update(target: dict, updates: dict) -> None:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            deep_update(target[key], value)
        else:
            target[key] = copy.deepcopy(value)


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


def has_final_test(console_path: Path) -> bool:
    if not console_path.exists():
        return False
    text = console_path.read_text(encoding="utf-8", errors="replace")
    return bool(re.search(r"(?m)^last_test_acc:", text))


def require_cuda() -> tuple[str, str]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the formal reviewer grid")
    return str(torch.cuda.get_device_name(0)), str(torch.version.cuda or "unknown")


def build_config(
    base: dict,
    treatment: Treatment,
    *,
    seed: int,
    args: argparse.Namespace,
) -> tuple[dict, str]:
    config = copy.deepcopy(base)
    config["seed"] = seed
    config["reproducibility"] = {"deterministic": True}
    config.setdefault("evaluation", {})["validation_seed"] = seed + 10_000
    deep_update(
        config,
        {
            "federated": {
                "num_clients": treatment.num_clients,
                "clients_per_round": treatment.num_clients,
                "rounds": args.rounds,
                "local_epochs": args.local_epochs,
                "batch_size": args.batch_size,
                "aggregation": "weighted",
                "device": "cuda:0",
            },
            "partition": {
                "mode": "dual_skew",
                "alpha": treatment.alpha,
                "quantity_skew": True,
                "preserve_all_samples": True,
            },
            "algorithm": {"name": "fedavg"},
        },
    )
    if not args.keep_base_optimizer:
        config["optimizer"] = {"name": "sgd", "lr": args.learning_rate}
        config["scheduler"] = {"name": "constant"}

    if treatment.method == "fedavg":
        config["fcpc"] = {
            "enabled": False,
            "metric": "jsdn",
            "reference_strategy": "partner",
            "update_rule": "penalty",
            "lambda_jsdn": treatment.lambda_jsdn,
            "beta": 0.0,
            "beta_schedule": "constant",
            "min_beta": 0.0,
            "epsilon": 1.0,
            "pairing_strategy": "fair_greedy_dissimilar",
            "partner_weighting": "uniform",
        }
    elif treatment.method == "original_fcpc":
        config["fcpc"] = {
            "enabled": True,
            "metric": "jsdn",
            "reference_strategy": "partner",
            "update_rule": "penalty",
            "lambda_jsdn": treatment.lambda_jsdn,
            "beta": treatment.beta,
            "beta_schedule": "constant",
            "min_beta": 0.0,
            "epsilon": 1.0,
            "pairing_strategy": "fair_greedy_dissimilar",
            "partner_weighting": "uniform",
        }
    else:
        raise ValueError(f"unsupported method: {treatment.method}")

    run_name = (
        f"review_{treatment.method}_n{treatment.num_clients}"
        f"_a{tag(treatment.alpha)}_b{tag(treatment.beta)}"
        f"_l{tag(treatment.lambda_jsdn)}_r{args.rounds}_seed{seed}"
    )
    config.setdefault("logging", {})["output_dir"] = str(
        (args.output_root / "logs").relative_to(REPO_ROOT)
    ).replace("\\", "/")
    config["logging"]["checkpoint_dir"] = str(
        (args.output_root / "checkpoints").relative_to(REPO_ROOT)
    ).replace("\\", "/")
    config["logging"]["run_name"] = run_name
    return config, run_name


def write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    validate_args(args)
    if not args.base_config.is_absolute():
        args.base_config = REPO_ROOT / args.base_config
    if not args.output_root.is_absolute():
        args.output_root = REPO_ROOT / args.output_root
    try:
        args.output_root.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise ValueError("--output-root must remain inside the repository") from exc

    treatments = build_treatments(args)
    seeds = parse_number_list(args.seeds, int)
    if args.dry_run:
        print("Dry run: CUDA availability is not required.", flush=True)
    else:
        gpu_name, cuda_version = require_cuda()
        print(f"GPU: {gpu_name}; torch CUDA: {cuda_version}", flush=True)

    base = json.loads(args.base_config.read_text(encoding="utf-8"))
    resolved_dir = args.output_root / "resolved_configs"
    console_dir = args.output_root / "console"
    logs_dir = args.output_root / "logs"
    resolved_dir.mkdir(parents=True, exist_ok=True)
    console_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_root / f"manifest_{args.study}.csv"
    manifest: list[dict[str, object]] = []

    jobs: list[tuple[Treatment, int, dict, str, Path, Path, Path]] = []
    for treatment in treatments:
        for seed in seeds:
            config, run_name = build_config(base, treatment, seed=seed, args=args)
            config_path = resolved_dir / f"{run_name}.json"
            csv_path = logs_dir / f"{run_name}.csv"
            console_path = console_dir / f"{run_name}.log"
            jobs.append(
                (treatment, seed, config, run_name, config_path, csv_path, console_path)
            )
            manifest.append(
                {
                    "run_name": run_name,
                    "sources": ";".join(treatment.sources),
                    "method": treatment.method,
                    "num_clients": treatment.num_clients,
                    "alpha": treatment.alpha,
                    "beta": treatment.beta,
                    "lambda_jsdn": treatment.lambda_jsdn,
                    "seed": seed,
                    "rounds": args.rounds,
                    "local_epochs": args.local_epochs,
                    "batch_size": args.batch_size,
                    "plain_sgd": int(not args.keep_base_optimizer),
                    "status": "pending",
                    "completed_rounds": completed_rounds(csv_path),
                }
            )
    write_manifest(manifest_path, manifest)
    print(
        f"study={args.study}; unique treatments={len(treatments)}; jobs={len(jobs)}; "
        f"manifest={manifest_path}",
        flush=True,
    )

    failures: list[str] = []
    for index, job in enumerate(jobs):
        treatment, seed, config, run_name, config_path, csv_path, console_path = job
        config_path.write_text(
            json.dumps(config, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        done = completed_rounds(csv_path)
        if done >= args.rounds and has_final_test(console_path) and not args.force:
            manifest[index]["status"] = "skipped_complete"
            manifest[index]["completed_rounds"] = done
            write_manifest(manifest_path, manifest)
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
        manifest[index]["status"] = "running"
        write_manifest(manifest_path, manifest)
        print(f"START {run_name}; console={console_path}", flush=True)
        with console_path.open("w", encoding="utf-8") as console:
            result = subprocess.run(
                command,
                cwd=REPO_ROOT,
                stdout=console,
                stderr=subprocess.STDOUT,
                text=True,
            )
        done = completed_rounds(csv_path)
        manifest[index]["completed_rounds"] = done
        if result.returncode:
            manifest[index]["status"] = "failed"
            failures.append(run_name)
            write_manifest(manifest_path, manifest)
            print(f"FAILED {run_name}; inspect {console_path}", flush=True)
            if not args.continue_on_error:
                raise SystemExit(result.returncode)
        else:
            manifest[index]["status"] = "dry_run" if args.dry_run else "done"
            write_manifest(manifest_path, manifest)
            print(f"DONE {run_name}", flush=True)

    if failures:
        raise SystemExit(f"failed runs: {failures}")
    print("All requested reviewer-grid jobs completed.", flush=True)


if __name__ == "__main__":
    main()
