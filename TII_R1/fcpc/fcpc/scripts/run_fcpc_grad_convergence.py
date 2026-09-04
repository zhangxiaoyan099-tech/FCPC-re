"""Compare validation-selected FCPC-grad with fixed baselines on held-out seeds."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import subprocess
import sys
from pathlib import Path

from scripts.run_cifar10_full_comparison import METHOD_OVERRIDES, _deep_update


REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = REPO_ROOT / "configs" / "cifar10_full_comparison_base.yaml"
DEFAULT_SELECTION = REPO_ROOT / "outputs" / "fcpc_grad_tuning" / "selected_hparams.json"
OUTPUT_ROOT = REPO_ROOT / "outputs" / "fcpc_grad_convergence"
BASELINE_METHODS = ("fedavg", "original_fcpc", "new_fcpc")
METHODS = BASELINE_METHODS + ("fcpc_grad_mix0", "fcpc_grad")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected", default=str(DEFAULT_SELECTION))
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--seeds", default="43,44,45")
    parser.add_argument("--rounds", type=int, default=200)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def _completed_rounds(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(newline="", encoding="utf-8") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def _require_cuda() -> tuple[str, str]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("FCPC-grad convergence comparison requires CUDA")
    return str(torch.cuda.get_device_name(0)), str(torch.version.cuda or "unknown")


def _fcpc_grad_override(selection: dict, *, gradient_mix: float | None = None) -> dict:
    return {
        "algorithm": {"name": "fedavg"},
        "fcpc": {
            "enabled": True,
            "metric": "pair_complementarity",
            "reference_strategy": "pair_grad_center",
            "update_rule": "proximal",
            "beta": float(selection["beta"]),
            "beta_schedule": "cosine_decay",
            "min_beta": 0.0,
            "epsilon": 1.0,
            "pairing_strategy": "optimal",
            "partner_weighting": "uniform",
            "grad_center_mix": float(
                selection["grad_center_mix"] if gradient_mix is None else gradient_mix
            ),
            "grad_center_step_scale": float(selection["grad_center_step_scale"]),
            "center_max_relative_distance": 0.05,
        },
    }


def main() -> None:
    args = _parse_args()
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    unsupported = sorted(set(methods) - set(METHODS))
    if unsupported:
        raise ValueError(f"unsupported methods: {unsupported}; choices={METHODS}")
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    if not seeds or args.rounds <= 0:
        raise ValueError("at least one seed and positive rounds are required")
    selection_path = Path(args.selected)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    overrides = {
        name: copy.deepcopy(METHOD_OVERRIDES[name]) for name in BASELINE_METHODS
    }
    overrides["fcpc_grad_mix0"] = _fcpc_grad_override(selection, gradient_mix=0.0)
    overrides["fcpc_grad"] = _fcpc_grad_override(selection)

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

    failures: list[str] = []
    for seed in seeds:
        for method in methods:
            config = copy.deepcopy(base)
            _deep_update(config, overrides[method])
            config["seed"] = seed
            config["evaluation"]["validation_seed"] = seed + 10_000
            config["federated"]["rounds"] = args.rounds
            config["logging"]["output_dir"] = "outputs/fcpc_grad_convergence/logs"
            config["logging"]["checkpoint_dir"] = "outputs/fcpc_grad_convergence/checkpoints"
            run_name = f"cifar10_r{args.rounds}_{method}_seed{seed}"
            config["logging"]["run_name"] = run_name
            config_path = resolved_dir / f"{run_name}.json"
            config_path.write_text(
                json.dumps(config, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            csv_path = OUTPUT_ROOT / "logs" / f"{run_name}.csv"
            if _completed_rounds(csv_path) >= args.rounds and not args.force:
                print(f"SKIP completed {run_name}", flush=True)
                continue
            command = [sys.executable, "-u", "-m", "src.main", "--config", str(config_path)]
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
            else:
                print(f"DONE {run_name}", flush=True)
    if failures:
        raise SystemExit(f"failed runs: {failures}")
    print("Convergence comparison completed.")


if __name__ == "__main__":
    main()
