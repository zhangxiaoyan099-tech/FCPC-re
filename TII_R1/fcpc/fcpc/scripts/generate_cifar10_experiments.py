"""Generate the staged CIFAR-10 FedAvg/FCPC experiment configurations."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


def base_config(data_root: str, seed: int) -> dict:
    return {
        "seed": seed,
        "dataset": {
            "name": "cifar10",
            "num_classes": 10,
            "input_channels": 3,
            "root": data_root,
            "download": False,
            "augment": True,
            "normalize": True,
        },
        "model": {"name": "resnet18"},
        "federated": {
            "num_clients": 10,
            "clients_per_round": 10,
            "rounds": 40,
            "local_epochs": 1,
            "batch_size": 64,
            "num_workers": 2,
            "max_batches_per_client": None,
            "max_eval_batches": None,
            "aggregation": "weighted",
            "device": "cuda",
        },
        "partition": {
            "mode": "dual_skew",
            "alpha": 0.1,
        },
        "algorithm": {"name": "fedavg"},
        "fcpc": {
            "enabled": False,
            "lambda_jsdn": 0.3,
            "beta": 0.0,
            "epsilon": 1.0,
            "pairing_strategy": "fair_greedy_dissimilar",
        },
        "optimizer": {
            "name": "sgd",
            "lr": 0.03,
            "momentum": 0.9,
            "weight_decay": 0.0005,
            "nesterov": False,
        },
        "scheduler": {"name": "constant"},
        "evaluation": {
            "validation_fraction": 0.1,
            "validation_seed": seed + 10_000,
            "test_every_round": False,
        },
        "profiling": {"sample_interval_s": 0.2},
        "logging": {
            "output_dir": "outputs/cifar10_plan",
            "checkpoint_dir": "outputs/cifar10_plan",
            "run_name": "replace_me",
        },
    }


def write_config(output_dir: Path, name: str, config: dict) -> Path:
    config["logging"]["run_name"] = name
    path = output_dir / f"{name}.yaml"
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="configs/generated_cifar10")
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--selected-lr", type=float, default=0.03)
    parser.add_argument("--selected-beta", type=float, default=0.001)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base = base_config(args.data_root, args.seed)
    written: list[Path] = []

    iid = copy.deepcopy(base)
    iid["partition"]["mode"] = "iid"
    iid["federated"]["rounds"] = 10
    iid["optimizer"]["lr"] = args.selected_lr
    written.append(write_config(output_dir, "cifar10_iid_fedavg_r10", iid))

    for lr in (0.01, 0.03, 0.05):
        candidate = copy.deepcopy(base)
        candidate["optimizer"]["lr"] = lr
        suffix = str(lr).replace(".", "p")
        written.append(
            write_config(output_dir, f"cifar10_dual_a0p1_fedavg_lr{suffix}_r40", candidate)
        )

    fedavg_200 = copy.deepcopy(base)
    fedavg_200["federated"]["rounds"] = 200
    fedavg_200["optimizer"]["lr"] = args.selected_lr
    written.append(
        write_config(output_dir, "cifar10_dual_a0p1_fedavg_selected_r200", fedavg_200)
    )

    for beta in (1e-6, 1e-5, 1e-4, 1e-3, 1e-2):
        candidate = copy.deepcopy(base)
        candidate["federated"]["rounds"] = 30
        candidate["optimizer"]["lr"] = args.selected_lr
        candidate["fcpc"]["enabled"] = True
        candidate["fcpc"]["beta"] = beta
        suffix = f"{beta:.0e}".replace("-", "m")
        written.append(
            write_config(output_dir, f"cifar10_dual_a0p1_fcpc_beta{suffix}_r30", candidate)
        )

    fcpc_200 = copy.deepcopy(base)
    fcpc_200["federated"]["rounds"] = 200
    fcpc_200["optimizer"]["lr"] = args.selected_lr
    fcpc_200["fcpc"]["enabled"] = True
    fcpc_200["fcpc"]["beta"] = args.selected_beta
    written.append(
        write_config(output_dir, "cifar10_dual_a0p1_fcpc_selected_r200", fcpc_200)
    )

    print(f"generated {len(written)} configs under {output_dir}")
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
