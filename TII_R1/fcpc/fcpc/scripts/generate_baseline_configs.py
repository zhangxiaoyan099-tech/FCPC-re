"""Generate matched CIFAR-10 and synthetic-smoke baseline configurations."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


ALGORITHMS = {
    "fcpc": {"_algorithm": "fedavg"},
    "fedavg": {},
    "fedprox": {"mu": 0.01},
    "moon": {"mu": 1.0, "temperature": 0.5},
    "feddyn": {"alpha": 0.01, "adaptive_alpha": True},
    "fblg": {
        "candidate_ratio": 0.5,
        "epsilon": 0.01,
        "sigma": 0.1,
    },
    "fedcfa": {
        "topk": 24,
        "rates": "1:5:5",
        "mean_batch_size": 128,
    },
}


def base_config(smoke: bool, rounds: int, clients_per_round: int) -> dict:
    if smoke:
        dataset = {
            "name": "synthetic",
            "num_classes": 10,
            "input_channels": 1,
            "image_size": 28,
            "train_samples": 160,
            "test_samples": 40,
        }
        model = {"name": "simple_cnn"}
        num_clients = 4
        clients_per_round = min(clients_per_round, num_clients)
        batch_size = 8
        max_batches = 1
        device = "cpu"
        workers = 0
        output_dir = "outputs/baseline_smoke"
    else:
        dataset = {
            "name": "cifar10",
            "num_classes": 10,
            "input_channels": 3,
            "root": "./data",
            "download": False,
            "augment": True,
            "normalize": True,
        }
        model = {"name": "resnet18"}
        num_clients = 10
        clients_per_round = min(clients_per_round, num_clients)
        batch_size = 64
        max_batches = None
        device = "cuda"
        workers = 2
        output_dir = "outputs/cifar10_baselines"
    return {
        "seed": 42,
        "dataset": dataset,
        "model": model,
        "federated": {
            "num_clients": num_clients,
            "clients_per_round": clients_per_round,
            "rounds": rounds,
            "local_epochs": 1,
            "batch_size": batch_size,
            "num_workers": workers,
            "max_batches_per_client": max_batches,
            "max_eval_batches": 1 if smoke else None,
            "aggregation": "weighted",
            "device": device,
        },
        "partition": {"mode": "dual_skew", "alpha": 0.1},
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
            "validation_seed": 10042,
            "test_every_round": False,
        },
        "profiling": {"sample_interval_s": 0.2},
        "logging": {
            "output_dir": output_dir,
            "checkpoint_dir": output_dir,
            "run_name": "placeholder",
        },
    }


def write_suite(output_dir: Path, smoke: bool, rounds: int, clients_per_round: int):
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = "smoke" if smoke else "cifar10_dual_a0p1"
    for name, parameters in ALGORITHMS.items():
        config = copy.deepcopy(base_config(smoke, rounds, clients_per_round))
        parameters = dict(parameters)
        algorithm_name = parameters.pop("_algorithm", name)
        config["algorithm"] = {"name": algorithm_name, **parameters}
        if name == "fcpc":
            config["fcpc"].update(
                {
                    "enabled": True,
                    "beta": 0.01,
                    "beta_schedule": "cosine_decay",
                    "min_beta": 0.0,
                    "partner_weighting": "sample_ratio",
                }
            )
        run_name = f"{prefix}_{name}_cpr{config['federated']['clients_per_round']}_r{rounds}"
        config["logging"]["run_name"] = run_name
        path = output_dir / f"{run_name}.yaml"
        path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        print(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="configs/baselines")
    parser.add_argument("--rounds", type=int, default=200)
    parser.add_argument("--clients-per-round", type=int, default=2)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    write_suite(
        Path(args.output_dir),
        smoke=args.smoke,
        rounds=args.rounds,
        clients_per_round=args.clients_per_round,
    )


if __name__ == "__main__":
    main()
