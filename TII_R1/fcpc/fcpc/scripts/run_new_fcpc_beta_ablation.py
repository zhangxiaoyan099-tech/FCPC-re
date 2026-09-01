"""Select New-FCPC beta on CIFAR-10 using validation accuracy only.

The experiment keeps the full-sample comparison protocol fixed and changes
only the initial value of beta.  Runs are sequential because the reference
server has a single GPU.
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
BASE_CONFIG = REPO_ROOT / "configs" / "cifar10_full_comparison_base.yaml"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "cifar10_beta_ablation"
DEFAULT_BETAS = "0.005,0.01,0.02,0.05,0.1,0.2"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sequential, GPU-required New-FCPC beta ablation. "
            "Choose beta from validation accuracy, not test accuracy."
        )
    )
    parser.add_argument(
        "--betas",
        default=DEFAULT_BETAS,
        help=f"comma-separated non-negative beta values (default: {DEFAULT_BETAS})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="development seed used for validation-based tuning",
    )
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def _parse_betas(value: str) -> list[float]:
    betas = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not betas:
        raise ValueError("at least one beta value is required")
    if any(beta < 0.0 for beta in betas):
        raise ValueError(f"beta values must be non-negative: {betas}")
    if len(set(betas)) != len(betas):
        raise ValueError(f"duplicate beta values are not allowed: {betas}")
    return betas


def _beta_tag(beta: float) -> str:
    value = format(beta, ".12g")
    return value.replace("-", "m").replace(".", "p").replace("+", "")


def _completed_rounds(csv_path: Path) -> int:
    if not csv_path.exists():
        return 0
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def _best_validation(csv_path: Path) -> tuple[int, float] | None:
    if not csv_path.exists():
        return None
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row.get("val_acc") not in (None, "")
        ]
    if not rows:
        return None
    best = max(rows, key=lambda row: float(row["val_acc"]))
    return int(best["round"]), float(best["val_acc"])


def _require_cuda() -> tuple[str, str]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "This ablation requires CUDA, but torch.cuda.is_available() is False"
        )
    return str(torch.cuda.get_device_name(0)), str(torch.version.cuda or "unknown")


def _build_config(base: dict, *, beta: float, seed: int, rounds: int | None) -> dict:
    config = copy.deepcopy(base)
    config["seed"] = seed
    config["evaluation"]["validation_seed"] = seed + 10_000
    if rounds is not None:
        config["federated"]["rounds"] = int(rounds)
    config["algorithm"] = {"name": "fedavg"}
    config["fcpc"] = {
        "enabled": True,
        "metric": "pair_complementarity",
        "reference_strategy": "pair_center",
        "beta": beta,
        "beta_schedule": "cosine_decay",
        "min_beta": 0.0,
        "epsilon": 1.0,
        "pairing_strategy": "optimal",
        "partner_weighting": "uniform",
    }
    config["logging"]["output_dir"] = str(
        OUTPUT_ROOT.relative_to(REPO_ROOT) / "logs"
    ).replace("\\", "/")
    config["logging"]["checkpoint_dir"] = str(
        OUTPUT_ROOT.relative_to(REPO_ROOT) / "checkpoints"
    ).replace("\\", "/")
    return config


def main() -> None:
    args = _parse_args()
    betas = _parse_betas(args.betas)
    if args.rounds is not None and args.rounds <= 0:
        raise ValueError("rounds must be positive")

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
    csv_by_beta: dict[float, Path] = {}
    for beta in betas:
        config = _build_config(
            base,
            beta=beta,
            seed=args.seed,
            rounds=args.rounds,
        )
        rounds = int(config["federated"]["rounds"])
        run_name = (
            f"cifar10_full_a0p1_cpr{config['federated']['clients_per_round']}"
            f"_r{rounds}_new_fcpc_beta{_beta_tag(beta)}_seed{args.seed}"
        )
        config["logging"]["run_name"] = run_name
        config_path = resolved_dir / f"{run_name}.json"
        config_path.write_text(
            json.dumps(config, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        csv_path = REPO_ROOT / config["logging"]["output_dir"] / f"{run_name}.csv"
        csv_by_beta[beta] = csv_path
        completed = _completed_rounds(csv_path)
        if completed >= rounds and not args.force:
            print(
                f"SKIP completed: beta={beta:g}; {run_name} "
                f"({completed}/{rounds} rounds)",
                flush=True,
            )
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
        print(
            f"START beta={beta:g}; {run_name}; console={console_path}",
            flush=True,
        )
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
            print(f"FAILED beta={beta:g}; inspect {console_path}", flush=True)
            if not args.continue_on_error:
                raise SystemExit(completed_process.returncode)
        else:
            print(f"DONE beta={beta:g}; {run_name}", flush=True)

    print("\nVALIDATION-ONLY SUMMARY", flush=True)
    print("beta\tbest_round\tbest_val_acc", flush=True)
    candidates: list[tuple[float, float, int]] = []
    for beta in betas:
        result = _best_validation(csv_by_beta[beta])
        if result is None:
            print(f"{beta:g}\tNA\tNA", flush=True)
            continue
        best_round, best_val_acc = result
        candidates.append((best_val_acc, beta, best_round))
        print(f"{beta:g}\t{best_round}\t{best_val_acc:.6f}", flush=True)
    if candidates:
        best_val_acc, selected_beta, best_round = max(
            candidates,
            key=lambda item: (item[0], -item[1]),
        )
        print(
            f"SELECTED_BY_VALIDATION beta={selected_beta:g}; "
            f"best_round={best_round}; best_val_acc={best_val_acc:.6f}",
            flush=True,
        )

    if failures:
        raise SystemExit(f"failed runs: {failures}")
    print("All requested beta-ablation runs completed.", flush=True)


if __name__ == "__main__":
    main()
