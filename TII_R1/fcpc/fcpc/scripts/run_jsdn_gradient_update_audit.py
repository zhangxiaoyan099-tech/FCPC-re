"""Compare whole-JSDN with pairwise gradient and local-update differences.

The audit follows a neutral FedAvg trajectory.  At each frozen checkpoint it
computes every client's empirical gradient at the same global model and then
replays one ordinary, unregularized local epoch from that same model.  Thus the
gradient/update comparison contains no FCPC treatment and cannot be circular.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from scripts.run_at_m_audit import (
    build_loader,
    compute_client_gradients,
    iter_neutral_checkpoints,
    model_factory,
    prepare_data,
    prepare_metadata,
    select_device,
    spearman,
)
from src.algorithms import build_algorithm
from src.experiments.at_m import flatten_state_delta
from src.fcpc.jsdn import metric_matrix
from src.federated.client import Client
from src.utils.config import load_config
from src.utils.seed import set_seed


PAIR_FIELDS = [
    "model_seed",
    "checkpoint_round",
    "panel",
    "client_i",
    "client_j",
    "jsdn",
    "gradient_difference",
    "update_difference",
    "gradient_update_gap",
    "gradient_cosine",
    "update_cosine",
    "gradient_norm_i",
    "gradient_norm_j",
    "update_norm_i",
    "update_norm_j",
]

SUMMARY_FIELDS = [
    "model_seed",
    "checkpoint_round",
    "panel",
    "pair_count",
    "spearman_jsdn_gradient",
    "spearman_jsdn_update",
    "spearman_gradient_update",
    "mean_gradient_difference",
    "mean_update_difference",
    "mean_absolute_gradient_update_gap",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def normalized_difference(first, second, eps: float = 1e-12) -> tuple[float, float]:
    """Return normalized squared distance and cosine for two flat vectors."""
    first = first.detach().cpu().float().reshape(-1)
    second = second.detach().cpu().float().reshape(-1)
    first_sq = float(first.dot(first).item())
    second_sq = float(second.dot(second).item())
    inner = float(first.dot(second).item())
    distance = float((first_sq + second_sq - 2.0 * inner) / (first_sq + second_sq + eps))
    cosine = float(inner / ((first_sq * second_sq) ** 0.5 + eps))
    return max(distance, 0.0), cosine


def replay_unregularized_updates(
    config: Mapping[str, object],
    data: Mapping[str, object],
    global_state: Mapping[str, object],
    *,
    checkpoint_round: int,
    device: str,
):
    audit_cfg = config.get("audit", {})
    update_cfg = audit_cfg.get("update_probe", {})
    federated_cfg = config.get("federated", {})
    optimizer_cfg = update_cfg.get("optimizer", {})
    batch_size = int(update_cfg.get("batch_size", federated_cfg.get("batch_size", 128)))
    local_epochs = int(update_cfg.get("local_epochs", 1))
    max_batches = update_cfg.get("max_batches_per_client")
    max_batches = None if max_batches in (None, 0) else int(max_batches)
    num_workers = int(update_cfg.get("num_workers", 0))
    pin_memory = bool(federated_cfg.get("pin_memory", False))
    seed = int(config.get("seed", 42))
    algorithm = build_algorithm("fedavg")
    mean_count = float(np.mean(list(data["sample_counts"].values())))
    probe_model = model_factory(config, data)
    parameter_names = [name for name, _ in probe_model.named_parameters()]
    updates = {}

    for client_id in range(int(data["num_clients"])):
        loader_seed = seed * 1_000_003 + checkpoint_round * 10_007 + client_id
        set_seed(loader_seed)
        loader = build_loader(
            data["dataset"],
            data["client_indices"][client_id],
            batch_size=batch_size,
            shuffle=True,
            seed=loader_seed,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        client = Client(
            client_id=client_id,
            train_loader=loader,
            sample_count=int(data["sample_counts"][client_id]),
            label_histogram=data["histograms"][client_id],
        )
        local_model = model_factory(config, data)
        state, metrics = client.local_train(
            local_model,
            algorithm,
            global_state,
            use_fcpc=False,
            beta=0.0,
            lr=float(optimizer_cfg.get("lr", 0.05)),
            optimizer_name=str(optimizer_cfg.get("name", "sgd")),
            momentum=float(optimizer_cfg.get("momentum", 0.9)),
            weight_decay=float(optimizer_cfg.get("weight_decay", 0.0005)),
            nesterov=bool(optimizer_cfg.get("nesterov", False)),
            local_epochs=local_epochs,
            device=device,
            max_batches=max_batches,
            mean_sample_count=mean_count,
            return_metrics=True,
        )
        updates[client_id] = flatten_state_delta(state, global_state, parameter_names)
        print(
            f"update_probe: checkpoint={checkpoint_round}, client={client_id}, "
            f"batches={metrics['processed_batches']}, "
            f"norm={float(updates[client_id].norm().item()):.6e}",
            flush=True,
        )
        del local_model
    return updates


def run(config: Mapping[str, object]) -> tuple[Path, Path]:
    import torch

    requested_device = str(config.get("federated", {}).get("device", "cuda"))
    device = select_device(requested_device)
    if not device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the JSDN gradient/update audit")
    print(f"runtime_device: {device}; gpu_name: {torch.cuda.get_device_name(0)}", flush=True)
    data = prepare_data(config)
    audit_cfg = config.get("audit", {})
    metadata = prepare_metadata(
        data,
        epsilon=float(audit_cfg.get("epsilon", 1.0)),
        ldp_seed=int(audit_cfg.get("ldp_seed", int(config.get("seed", 42)) + 20_000)),
    )
    panels = [str(value).lower() for value in audit_cfg.get("panels", ["raw", "ldp"])]
    matrices = {
        panel: metric_matrix(
            "jsdn",
            [metadata[panel]["distributions"][i] for i in range(data["num_clients"])],
            [metadata[panel]["counts"][i] for i in range(data["num_clients"])],
            lambda_jsdn=float(audit_cfg.get("lambda_jsdn", 0.3)),
        )
        for panel in panels
    }
    pair_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    model_seed = int(config.get("seed", 42))

    for checkpoint_round, checkpoint in iter_neutral_checkpoints(config, data):
        gradients = compute_client_gradients(
            config,
            data,
            checkpoint["model_state"],
            device,
        )
        updates = replay_unregularized_updates(
            config,
            data,
            checkpoint["model_state"],
            checkpoint_round=checkpoint_round,
            device=device,
        )
        for panel in panels:
            rows_at_checkpoint = []
            for client_i in range(int(data["num_clients"])):
                for client_j in range(client_i + 1, int(data["num_clients"])):
                    grad_difference, grad_cosine = normalized_difference(
                        gradients[client_i], gradients[client_j]
                    )
                    update_difference, update_cosine = normalized_difference(
                        updates[client_i], updates[client_j]
                    )
                    row = {
                        "model_seed": model_seed,
                        "checkpoint_round": checkpoint_round,
                        "panel": panel,
                        "client_i": client_i,
                        "client_j": client_j,
                        "jsdn": float(matrices[panel][client_i, client_j]),
                        "gradient_difference": grad_difference,
                        "update_difference": update_difference,
                        "gradient_update_gap": update_difference - grad_difference,
                        "gradient_cosine": grad_cosine,
                        "update_cosine": update_cosine,
                        "gradient_norm_i": float(gradients[client_i].norm().item()),
                        "gradient_norm_j": float(gradients[client_j].norm().item()),
                        "update_norm_i": float(updates[client_i].norm().item()),
                        "update_norm_j": float(updates[client_j].norm().item()),
                    }
                    pair_rows.append(row)
                    rows_at_checkpoint.append(row)
            jsdn = [float(row["jsdn"]) for row in rows_at_checkpoint]
            grad = [float(row["gradient_difference"]) for row in rows_at_checkpoint]
            update = [float(row["update_difference"]) for row in rows_at_checkpoint]
            gaps = [abs(float(row["gradient_update_gap"])) for row in rows_at_checkpoint]
            summary_rows.append(
                {
                    "model_seed": model_seed,
                    "checkpoint_round": checkpoint_round,
                    "panel": panel,
                    "pair_count": len(rows_at_checkpoint),
                    "spearman_jsdn_gradient": spearman(jsdn, grad),
                    "spearman_jsdn_update": spearman(jsdn, update),
                    "spearman_gradient_update": spearman(grad, update),
                    "mean_gradient_difference": float(np.mean(grad)),
                    "mean_update_difference": float(np.mean(update)),
                    "mean_absolute_gradient_update_gap": float(np.mean(gaps)),
                }
            )
        del gradients, updates
        torch.cuda.empty_cache()

    output_dir = Path(str(audit_cfg.get("output_dir", "outputs/jsdn_gradient_update")))
    output_dir.mkdir(parents=True, exist_ok=True)
    pair_path = output_dir / "jsdn_gradient_update_pairs.csv"
    summary_path = output_dir / "jsdn_gradient_update_summary.csv"
    with pair_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PAIR_FIELDS)
        writer.writeheader()
        writer.writerows(pair_rows)
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"pair_path: {pair_path}", flush=True)
    print(f"summary_path: {summary_path}", flush=True)
    for row in summary_rows:
        print(
            "checkpoint={checkpoint_round} panel={panel} "
            "rho(JSDN,Dgrad)={spearman_jsdn_gradient:.4f} "
            "rho(JSDN,Dupdate)={spearman_jsdn_update:.4f} "
            "rho(Dgrad,Dupdate)={spearman_gradient_update:.4f}".format(**row),
            flush=True,
        )
    return pair_path, summary_path


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    print(json.dumps(config, indent=2, ensure_ascii=False), flush=True)
    run(config)


if __name__ == "__main__":
    main()
