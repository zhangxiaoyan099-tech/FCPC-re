"""Run pre-registered causal ablations for FCPC-grad on CIFAR-10.

The variants change one mechanism at a time while inheriting the same model,
partition, optimizer, client sampling, and evaluation protocol.
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


OUTPUT_ROOT = REPO_ROOT / "outputs" / "fcpc_grad_causal_ablation"

FCPC_COMMON = {
    "enabled": True,
    "metric": "pair_complementarity",
    "update_rule": "proximal",
    "beta": 0.2,
    "beta_schedule": "cosine_decay",
    "min_beta": 0.0,
    "epsilon": 1.0,
    "pairing_strategy": "optimal",
    "partner_weighting": "uniform",
    "grad_center_mix": 1.0,
    "grad_center_step_scale": 0.5,
    "grad_center_direction": "real",
    "center_max_relative_distance": 0.05,
    "proximal_frequency": "batch",
}


def _fcpc(**updates) -> dict:
    config = copy.deepcopy(FCPC_COMMON)
    config.update(updates)
    return {"algorithm": {"name": "fedavg"}, "fcpc": config}


METHOD_OVERRIDES = {
    # Baseline: no pairing, center, or proximal operation.
    "fedavg": {
        "algorithm": {"name": "fedavg"},
        "fcpc": {"enabled": False, "beta": 0.0},
    },
    # Pure contraction around w^t; isolates proximal stabilization.
    "global_prox": _fcpc(
        reference_strategy="global_center", pairing_strategy="random"
    ),
    # Shared historical parameter center, but no update extrapolation.
    "pair_center": _fcpc(reference_strategy="pair_center", grad_center_mix=0.0),
    # Full deployed FCPC-grad treatment.
    "grad_real": _fcpc(reference_strategy="pair_grad_center"),
    # Same round-level multiset of proxy norms, assigned to the wrong pairs.
    "grad_shuffled": _fcpc(
        reference_strategy="pair_grad_center", grad_center_direction="shuffled"
    ),
    # Same proxy norm and pair, opposite historical direction.
    "grad_reversed": _fcpc(
        reference_strategy="pair_grad_center", grad_center_direction="reversed"
    ),
    # Keeps the real proxy/proximal mechanism but removes complementary matching.
    "grad_random_pairing": _fcpc(
        reference_strategy="pair_grad_center", pairing_strategy="random"
    ),
    # Tests whether early-strong/late-weak regularization is necessary.
    "grad_constant_beta": _fcpc(
        reference_strategy="pair_grad_center", beta_schedule="constant"
    ),
    # Matches rho^H but applies the pull once after local SGD, separating
    # interleaving and quantity-dependent repeated-prox effects.
    "grad_local_end": _fcpc(
        reference_strategy="pair_grad_center",
        proximal_frequency="local_end_matched",
    ),
    # Tests whether the 5%-of-model-norm trust region is itself responsible.
    "grad_no_clip": _fcpc(
        reference_strategy="pair_grad_center",
        center_max_relative_distance=None,
    ),
    # Auxiliary cells for difference-in-differences interaction tests.  They
    # are not part of the first-stage screen and are run only after the screen
    # identifies a potentially useful component.
    "pair_center_random": _fcpc(
        reference_strategy="pair_center",
        grad_center_mix=0.0,
        pairing_strategy="random",
    ),
    "pair_center_constant_beta": _fcpc(
        reference_strategy="pair_center",
        grad_center_mix=0.0,
        beta_schedule="constant",
    ),
    "pair_center_local_end": _fcpc(
        reference_strategy="pair_center",
        grad_center_mix=0.0,
        proximal_frequency="local_end_matched",
    ),
    # Constant-beta confirmation cells. These keep beta fixed at 0.2 so the
    # learning-rate schedule is the only source of weakening over rounds.
    "global_prox_constant_beta": _fcpc(
        reference_strategy="global_center",
        pairing_strategy="random",
        beta_schedule="constant",
    ),
    "grad_constant_beta_reversed": _fcpc(
        reference_strategy="pair_grad_center",
        grad_center_direction="reversed",
        beta_schedule="constant",
    ),
    "grad_constant_beta_random_pairing": _fcpc(
        reference_strategy="pair_grad_center",
        pairing_strategy="random",
        beta_schedule="constant",
    ),
    "grad_constant_beta_local_end": _fcpc(
        reference_strategy="pair_grad_center",
        proximal_frequency="local_end_matched",
        beta_schedule="constant",
    ),
}

SCREEN_METHODS = (
    "fedavg",
    "global_prox",
    "pair_center",
    "grad_real",
    "grad_shuffled",
    "grad_reversed",
    "grad_random_pairing",
    "grad_constant_beta",
    "grad_local_end",
    "grad_no_clip",
)
INTERACTION_METHODS = (
    "pair_center",
    "pair_center_random",
    "grad_real",
    "grad_random_pairing",
    "pair_center_constant_beta",
    "grad_constant_beta",
    "pair_center_local_end",
    "grad_local_end",
)
CONSTANT_BETA_METHODS = (
    "fedavg",
    "global_prox_constant_beta",
    "grad_constant_beta",
    "grad_constant_beta_reversed",
    "grad_constant_beta_random_pairing",
    "grad_constant_beta_local_end",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--methods",
        default="screen",
        help=(
            "screen, interactions, constant_beta, all, or comma-separated "
            "method names"
        ),
    )
    parser.add_argument("--seeds", default="45,46,47")
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def _resolve_methods(value: str) -> list[str]:
    preset = value.strip().lower()
    if preset == "screen":
        return list(SCREEN_METHODS)
    if preset in {"interaction", "interactions"}:
        return list(INTERACTION_METHODS)
    if preset in {"constant", "constant_beta"}:
        return list(CONSTANT_BETA_METHODS)
    if preset == "all":
        return list(METHOD_OVERRIDES)
    methods = [item.strip().lower() for item in value.split(",") if item.strip()]
    unsupported = [name for name in methods if name not in METHOD_OVERRIDES]
    if unsupported:
        raise ValueError(
            f"unsupported methods: {unsupported}; choices={list(METHOD_OVERRIDES)}"
        )
    return methods


def _completed_rounds(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    try:
        rounds = [int(row["round"]) for row in rows]
    except (KeyError, TypeError, ValueError):
        return 0
    return len(rounds) if rounds == list(range(1, len(rounds) + 1)) else 0


def _has_final_test_metrics(path: Path) -> bool:
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8", errors="replace")
    return any(line.startswith("test_acc:") for line in text.splitlines()) and any(
        line.startswith("last_test_acc:") for line in text.splitlines()
    )


def _require_cuda() -> tuple[str, str]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the causal ablation")
    return str(torch.cuda.get_device_name(0)), str(torch.version.cuda or "unknown")


def build_config(base: dict, method: str, seed: int, rounds: int) -> dict:
    config = copy.deepcopy(base)
    _deep_update(config, METHOD_OVERRIDES[method])
    config["seed"] = int(seed)
    config["evaluation"]["validation_seed"] = int(seed) + 10_000
    config["federated"]["rounds"] = int(rounds)
    config["logging"]["output_dir"] = "outputs/fcpc_grad_causal_ablation/logs"
    config["logging"]["checkpoint_dir"] = (
        "outputs/fcpc_grad_causal_ablation/checkpoints"
    )
    run_name = (
        f"cifar10_full_a0p1_cpr{config['federated']['clients_per_round']}"
        f"_r{rounds}_{method}_seed{seed}"
    )
    config["logging"]["run_name"] = run_name
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
    failures = []
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
    print("All requested causal-ablation runs completed.", flush=True)


if __name__ == "__main__":
    main()
