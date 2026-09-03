"""Frozen-checkpoint counterfactual audit of FCPC pairing strategies.

The runner first creates neutral FedAvg checkpoints.  At every requested
checkpoint it computes client gradients once, then replays one FCPC round from
the exact same state and batch trace under several pairings.  This isolates
the current-round effect of matching from accumulated trajectory differences.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from src.algorithms import build_algorithm
from src.data.datasets import get_client_ids, get_dataset_spec, get_targets, load_dataset
from src.data.partition import (
    build_client_indices,
    client_label_histograms,
    natural_client_partition,
    validate_complete_partition,
)
from src.data.split import stratified_holdout_indices
from src.experiments.at_m import (
    compute_intervention_metrics,
    compute_matching_metrics,
    pair_mixture_kl_residual,
)
from src.fcpc.jsdn import metric_matrix
from src.fcpc.ldp import PrivacyBudget, perturb_client_metadata
from src.fcpc.pairing import PairingResult, pair_clients, pairing_weight
from src.fcpc.regularizer import (
    clip_state_center_to_global,
    state_l2_distance,
    state_l2_norm,
    weighted_state_center,
)
from src.federated.aggregation import fedavg_aggregate
from src.federated.client import Client
from src.models import build_model
from src.utils.config import load_config
from src.utils.seed import set_seed


MAIN_FIELDS = [
    "model_seed",
    "checkpoint_round",
    "panel",
    "pairing_strategy",
    "pairing_seed",
    "batch_seed",
    "pair_list",
    "selection_score_name",
    "selection_score_sum",
    "true_weighted_js_sum",
    "R_M",
    "R_observed_M",
    "H_t_M",
    "A_t_M",
    "A_t_normalized",
    "U_t_M",
    "U_t_normalized",
    "D_t_M",
    "D_t_normalized",
    "D_t_normalized_ideal",
    "D_client_rms",
    "D_client_normalized",
    "D_retention_ratio",
    "D_cancellation_fraction",
    "fedavg_global_update_norm",
    "fedavg_client_update_rms",
    "fedavg_U_t",
    "correction_inner_product",
    "correction_alignment_cosine",
    "correction_gain_U",
    "correction_gain_relative",
    "correction_gain_from_identity",
    "correction_identity_gap",
    "cancellation_ratio_U_over_A",
    "kappa_U_over_A",
    "U_le_A_gap",
    "residual_group_count",
    "residual_angle_count",
    "residual_cosine_mean",
    "residual_cosine_median",
    "residual_cosine_min",
    "residual_cosine_positive_fraction",
    "residual_inner_product_positive_fraction",
    "residual_inner_product_nonnegative_fraction",
    "residual_diagonal_term",
    "residual_cross_term",
    "U_from_residual_expansion",
    "b_min",
    "alignment_lower_bound_bmin_A",
    "alignment_assumption_holds",
    "U_minus_alignment_lower_bound",
    "global_gradient_norm",
    "global_update_norm",
    "gamma",
    "local_steps",
    "learning_rate",
    "beta",
    "partner_weighting",
    "update_rule",
    "mean_effective_beta",
    "mean_center_distance",
    "max_center_distance",
    "mean_center_clip_scale",
    "mean_pair_history_distance",
    "mean_pair_endpoint_distance",
    "mean_task_loss",
    "mean_fcpc_raw_loss",
    "mean_fcpc_weighted_loss",
    "mean_total_loss",
    "val_loss_before",
    "val_loss_after",
    "val_loss_change",
    "val_acc_before",
    "val_acc_after",
    "val_acc_change",
]

PAIR_FIELDS = [
    "model_seed",
    "checkpoint_round",
    "panel",
    "pairing_strategy",
    "pairing_seed",
    "batch_seed",
    "group_index",
    "client_i",
    "client_j",
    "group_mass",
    "execution_residual_sq",
    "A_contribution",
    "gradient_residual_sq",
    "H_contribution",
]

ANGLE_FIELDS = [
    "model_seed",
    "checkpoint_round",
    "panel",
    "pairing_strategy",
    "pairing_seed",
    "batch_seed",
    "group_index_a",
    "clients_a",
    "group_index_b",
    "clients_b",
    "weight_a",
    "weight_b",
    "inner_product",
    "cosine",
    "cross_contribution",
    "inner_product_positive",
    "inner_product_nonnegative",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit R(M), H_t(M), A_t(M), and U_t(M) from frozen checkpoints"
    )
    parser.add_argument("--config", required=True, help="Path to an audit YAML/JSON file")
    parser.add_argument(
        "--reuse-checkpoints",
        action="store_true",
        help="Skip neutral warmup and reuse compatible saved checkpoints",
    )
    return parser.parse_args()


def clone_state(state: Mapping[str, object]) -> dict[str, object]:
    result: dict[str, object] = {}
    for name, value in state.items():
        if hasattr(value, "detach"):
            result[name] = value.detach().cpu().clone()
        else:
            result[name] = value
    return result


def _dataset_kwargs(dataset_cfg: Mapping[str, Any]) -> dict[str, Any]:
    excluded = {
        "name",
        "root",
        "download",
        "num_classes",
        "input_channels",
        "dataset_grid",
    }
    return {key: value for key, value in dataset_cfg.items() if key not in excluded}


def prepare_data(config: Mapping[str, Any]):
    from torch.utils.data import DataLoader, Subset

    seed = int(config.get("seed", 42))
    dataset_cfg = config.get("dataset", {})
    federated_cfg = config.get("federated", {})
    partition_cfg = config.get("partition", {})
    evaluation_cfg = config.get("evaluation", {})
    dataset_name = str(dataset_cfg.get("name", "cifar10"))
    spec = get_dataset_spec(dataset_name)
    num_classes = int(dataset_cfg.get("num_classes", spec.num_classes))
    kwargs = _dataset_kwargs(dataset_cfg)
    if bool(kwargs.get("augment", False)):
        raise ValueError(
            "A_t(M) audit requires dataset.augment=false so every replay sees "
            "an identical deterministic example tensor"
        )
    train_dataset = load_dataset(
        dataset_name,
        root=dataset_cfg.get("root", "./data"),
        train=True,
        download=bool(dataset_cfg.get("download", False)),
        seed=seed,
        num_classes=num_classes,
        **kwargs,
    )
    validation_kwargs = dict(kwargs)
    validation_kwargs["augment"] = False
    validation_dataset = load_dataset(
        dataset_name,
        root=dataset_cfg.get("root", "./data"),
        train=True,
        download=False,
        seed=seed,
        num_classes=num_classes,
        **validation_kwargs,
    )
    labels = get_targets(train_dataset)
    validation_fraction = float(evaluation_cfg.get("validation_fraction", 0.1))
    train_indices, validation_indices = stratified_holdout_indices(
        labels,
        validation_fraction=validation_fraction,
        seed=int(evaluation_cfg.get("validation_seed", seed + 10_000)),
    )
    train_labels = [labels[index] for index in train_indices]
    partition_mode = str(partition_cfg.get("mode", "dual_skew")).lower()
    if partition_mode == "natural":
        source_ids = get_client_ids(train_dataset)
        relative_indices = natural_client_partition([source_ids[index] for index in train_indices])
        num_clients = len(relative_indices)
    else:
        num_clients = int(federated_cfg.get("num_clients", 10))
        relative_indices = build_client_indices(
            train_labels,
            num_clients=num_clients,
            partition=partition_mode,
            alpha=float(partition_cfg.get("alpha", 0.1)),
            seed=seed,
            quantity_min_fraction=float(partition_cfg.get("quantity_min_fraction", 0.25)),
        )
    stats = validate_complete_partition(relative_indices, len(train_indices))
    client_indices = {
        client_id: [train_indices[position] for position in positions]
        for client_id, positions in relative_indices.items()
    }
    histograms_array = client_label_histograms(
        train_labels,
        relative_indices,
        num_classes=num_classes,
    )
    histograms = {
        client_id: np.asarray(histograms_array[client_id], dtype=np.float64)
        for client_id in range(num_clients)
    }
    sample_counts = {
        client_id: len(client_indices[client_id])
        for client_id in range(num_clients)
    }
    batch_size = int(federated_cfg.get("batch_size", 64))
    validation_loader = DataLoader(
        Subset(validation_dataset, validation_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(federated_cfg.get("num_workers", 0)),
        pin_memory=bool(federated_cfg.get("pin_memory", False)),
        persistent_workers=bool(int(federated_cfg.get("num_workers", 0)) > 0),
    )
    print(
        "partition_coverage: "
        f"assigned={stats['assigned_examples']}, unique={stats['unique_examples']}, "
        f"expected={len(train_indices)}, client_min={stats['client_examples_min']}, "
        f"client_max={stats['client_examples_max']}",
        flush=True,
    )
    return {
        "dataset": train_dataset,
        "validation_loader": validation_loader,
        "client_indices": client_indices,
        "histograms": histograms,
        "sample_counts": sample_counts,
        "num_clients": num_clients,
        "num_classes": num_classes,
        "input_channels": int(dataset_cfg.get("input_channels", spec.input_channels)),
    }


def build_loader(
    dataset,
    indices: Iterable[int],
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
    pin_memory: bool,
):
    import torch
    from torch.utils.data import DataLoader, Subset

    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(
        Subset(dataset, list(indices)),
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        generator=generator,
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        # Ten persistent client loaders would keep num_clients*num_workers
        # processes alive.  The single validation loader is persistent, while
        # client/probe loaders are intentionally short-lived.
        persistent_workers=False,
    )


def model_factory(config: Mapping[str, Any], data: Mapping[str, Any]):
    model_cfg = config.get("model", {})
    return build_model(
        model_cfg.get("name", "resnet18"),
        num_classes=int(data["num_classes"]),
        input_channels=int(data["input_channels"]),
    )


def checkpoint_signature(config: Mapping[str, Any]) -> str:
    audit = config.get("audit", {})
    payload = {
        "seed": config.get("seed", 42),
        "dataset": config.get("dataset", {}),
        "model": config.get("model", {}),
        "federated": config.get("federated", {}),
        "partition": config.get("partition", {}),
        "evaluation": config.get("evaluation", {}),
        "warmup": audit.get("warmup", {}),
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def create_neutral_checkpoints(
    config: Mapping[str, Any],
    data: Mapping[str, Any],
    checkpoint_dir: Path,
) -> None:
    import torch

    seed = int(config.get("seed", 42))
    audit_cfg = config.get("audit", {})
    warmup_cfg = audit_cfg.get("warmup", {})
    federated_cfg = config.get("federated", {})
    checkpoints = sorted({int(value) for value in audit_cfg.get("checkpoints", [5, 10, 20, 50])})
    if not checkpoints or checkpoints[0] < 0:
        raise ValueError("audit.checkpoints must contain non-negative rounds")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    initial_model = model_factory(config, data)
    initial_model.to("cpu")
    global_state = clone_state(initial_model.state_dict())
    batch_size = int(warmup_cfg.get("batch_size", federated_cfg.get("batch_size", 64)))
    num_workers = int(warmup_cfg.get("num_workers", federated_cfg.get("num_workers", 0)))
    pin_memory = bool(federated_cfg.get("pin_memory", False))
    clients: list[Client] = []
    for client_id in range(int(data["num_clients"])):
        loader = build_loader(
            data["dataset"],
            data["client_indices"][client_id],
            batch_size=batch_size,
            shuffle=True,
            seed=seed * 1000 + client_id,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        clients.append(
            Client(
                client_id=client_id,
                train_loader=loader,
                sample_count=int(data["sample_counts"][client_id]),
                label_histogram=data["histograms"][client_id],
            )
        )

    signature = checkpoint_signature(config)

    def save(round_number: int) -> None:
        path = checkpoint_dir / f"fedavg_round_{round_number:03d}.pt"
        torch.save(
            {
                "round": round_number,
                "signature": signature,
                "model_state": clone_state(global_state),
                "client_previous_states": {
                    client.client_id: clone_state(client.previous_state or global_state)
                    for client in clients
                },
            },
            path,
        )
        print(f"saved_checkpoint: {path}", flush=True)

    if 0 in checkpoints:
        save(0)
    algorithm = build_algorithm("fedavg")
    device = select_device(str(federated_cfg.get("device", "auto")))
    local_epochs = int(warmup_cfg.get("local_epochs", 1))
    max_batches = warmup_cfg.get("max_batches_per_client")
    max_batches = None if max_batches in (None, 0) else int(max_batches)
    optimizer = warmup_cfg.get("optimizer", {})
    lr = float(optimizer.get("lr", 0.03))
    momentum = float(optimizer.get("momentum", 0.9))
    weight_decay = float(optimizer.get("weight_decay", 0.0005))
    nesterov = bool(optimizer.get("nesterov", False))
    mean_count = float(np.mean(list(data["sample_counts"].values())))
    for round_number in range(1, max(checkpoints, default=0) + 1):
        local_states = []
        for client in clients:
            local_model = model_factory(config, data)
            state = client.local_train(
                local_model,
                algorithm,
                global_state,
                use_fcpc=False,
                lr=lr,
                optimizer_name=str(optimizer.get("name", "sgd")),
                momentum=momentum,
                weight_decay=weight_decay,
                nesterov=nesterov,
                local_epochs=local_epochs,
                device=device,
                max_batches=max_batches,
                mean_sample_count=mean_count,
            )
            local_states.append(state)
        global_state = fedavg_aggregate(
            local_states,
            [data["sample_counts"][client_id] for client_id in range(data["num_clients"])],
            weighted=True,
        )
        print(f"warmup_round: {round_number}/{max(checkpoints)}", flush=True)
        if round_number in checkpoints:
            save(round_number)


def load_neutral_checkpoint(
    path: Path,
    expected_signature: str,
) -> dict[str, Any]:
    import torch

    if not path.exists():
        raise FileNotFoundError(
            f"neutral checkpoint not found: {path}; rerun without --reuse-checkpoints"
        )
    checkpoint = torch.load(path, map_location="cpu")
    if checkpoint.get("signature") != expected_signature:
        raise ValueError(
            f"checkpoint configuration mismatch: {path}; regenerate neutral checkpoints"
        )
    return checkpoint


def compute_client_gradients(
    config: Mapping[str, Any],
    data: Mapping[str, Any],
    global_state: Mapping[str, object],
    device: str,
) -> dict[int, object]:
    import torch
    from torch import nn

    audit_cfg = config.get("audit", {})
    batch_size = int(audit_cfg.get("gradient_batch_size", 128))
    max_batches = audit_cfg.get("gradient_max_batches")
    max_batches = None if max_batches in (None, 0) else int(max_batches)
    num_workers = int(audit_cfg.get("gradient_num_workers", 0))
    pin_memory = bool(config.get("federated", {}).get("pin_memory", False))
    parameter_names = [name for name, _ in model_factory(config, data).named_parameters()]
    gradients: dict[int, object] = {}
    for client_id in range(int(data["num_clients"])):
        indices = list(data["client_indices"][client_id])
        if max_batches is not None:
            indices = indices[: batch_size * max_batches]
        if not indices:
            raise ValueError(f"client {client_id} has no gradient-probe examples")
        loader = build_loader(
            data["dataset"],
            indices,
            batch_size=batch_size,
            shuffle=False,
            seed=0,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        model = model_factory(config, data)
        model.load_state_dict(global_state)
        model.to(device)
        # Evaluation mode freezes BatchNorm statistics so g_i(w^t) is
        # evaluated at one common state and is independent of probe batching.
        model.eval()
        model.zero_grad(set_to_none=True)
        criterion = nn.CrossEntropyLoss(reduction="sum")
        denominator = float(len(indices))
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=device.startswith("cuda"))
            targets = targets.to(device, non_blocking=device.startswith("cuda"))
            loss = criterion(model(inputs), targets) / denominator
            loss.backward()
        named_parameters = dict(model.named_parameters())
        pieces = []
        for name in parameter_names:
            parameter = named_parameters[name]
            if parameter.grad is None:
                pieces.append(torch.zeros_like(parameter).reshape(-1).cpu())
            else:
                pieces.append(parameter.grad.detach().float().reshape(-1).cpu().clone())
        gradients[client_id] = torch.cat(pieces)
        print(
            f"gradient_probe: client={client_id}, examples={len(indices)}, "
            f"norm={float(gradients[client_id].norm().item()):.6e}",
            flush=True,
        )
        del model
    return gradients


def prepare_metadata(
    data: Mapping[str, Any],
    *,
    epsilon: float,
    ldp_seed: int,
) -> dict[str, dict[str, Any]]:
    client_ids = range(int(data["num_clients"]))
    raw_distributions = {
        client_id: normalise_histogram(data["histograms"][client_id])
        for client_id in client_ids
    }
    raw_counts = {
        client_id: float(data["sample_counts"][client_id])
        for client_id in client_ids
    }
    rng = np.random.default_rng(int(ldp_seed))
    ldp_distributions = {}
    ldp_counts = {}
    for client_id in client_ids:
        distribution, count = perturb_client_metadata(
            data["histograms"][client_id],
            int(data["sample_counts"][client_id]),
            budget=PrivacyBudget(epsilon=float(epsilon)),
            rng=rng,
        )
        ldp_distributions[client_id] = distribution
        ldp_counts[client_id] = float(count)
    return {
        "raw": {"distributions": raw_distributions, "counts": raw_counts},
        "ldp": {"distributions": ldp_distributions, "counts": ldp_counts},
    }


def matrices_for_metadata(
    metadata: Mapping[str, Any],
    *,
    lambda_jsdn: float,
) -> dict[str, np.ndarray]:
    client_ids = sorted(metadata["counts"])
    distributions = [metadata["distributions"][client_id] for client_id in client_ids]
    counts = [metadata["counts"][client_id] for client_id in client_ids]
    return {
        "weighted_js": metric_matrix("pair_complementarity", distributions, counts),
        "jsdn": metric_matrix("jsdn", distributions, counts, lambda_jsdn=lambda_jsdn),
    }


def choose_pairing(
    strategy: str,
    matrices: Mapping[str, np.ndarray],
    *,
    seed: int,
) -> tuple[PairingResult, str, float]:
    strategy = str(strategy).lower()
    if strategy == "optimal":
        matrix = matrices["weighted_js"]
        pairing = pair_clients(matrix, strategy="optimal", seed=seed)
        score_name = "weighted_js"
    elif strategy == "similar":
        matrix = matrices["weighted_js"]
        pairing = pair_clients(matrix, strategy="similar", seed=seed)
        score_name = "weighted_js"
    elif strategy == "jsdn":
        matrix = matrices["jsdn"]
        pairing = pair_clients(matrix, strategy="greedy_dissimilar", seed=seed)
        score_name = "jsdn"
    elif strategy == "random":
        matrix = matrices["weighted_js"]
        pairing = pair_clients(matrix, strategy="random", seed=seed)
        score_name = "random"
    else:
        raise ValueError(f"unsupported audit pairing strategy: {strategy}")
    return pairing, score_name, pairing_weight(pairing, matrix)


def replay_one_round(
    config: Mapping[str, Any],
    data: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    pairing: PairingResult,
    client_gradients: Mapping[int, object],
    *,
    batch_seed: int,
    device: str,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    Mapping[str, object],
    Mapping[int, Mapping[str, object]],
]:
    replay_cfg = config.get("audit", {}).get("replay", {})
    global_state = checkpoint["model_state"]
    previous_states = checkpoint["client_previous_states"]
    probe_model = model_factory(config, data)
    parameter_names = [name for name, _ in probe_model.named_parameters()]
    parameter_name_set = set(parameter_names)
    global_norm = state_l2_norm(global_state, parameter_names=parameter_name_set)
    relative_limit = replay_cfg.get("center_max_relative_distance")
    absolute_limit = replay_cfg.get("center_max_distance")
    if absolute_limit not in (None, ""):
        center_limit = float(absolute_limit)
    elif relative_limit not in (None, ""):
        center_limit = float(relative_limit) * global_norm
    else:
        center_limit = None

    references: dict[int, Mapping[str, object]] = {}
    center_distances = []
    center_clip_scales = []
    history_distances = []
    for client_i, client_j in pairing.pairs:
        state_i = previous_states.get(client_i, global_state)
        state_j = previous_states.get(client_j, global_state)
        center = weighted_state_center(
            state_i,
            state_j,
            data["sample_counts"][client_i],
            data["sample_counts"][client_j],
            fallback_state=global_state,
        )
        center, distance, clip_scale = clip_state_center_to_global(
            center,
            global_state,
            max_distance=center_limit,
            parameter_names=parameter_name_set,
        )
        references[client_i] = center
        references[client_j] = center
        center_distances.append(float(distance))
        center_clip_scales.append(float(clip_scale))
        history_distances.append(
            state_l2_distance(state_i, state_j, parameter_names=parameter_name_set)
        )

    batch_size = int(replay_cfg.get("batch_size", config.get("federated", {}).get("batch_size", 64)))
    local_steps = int(replay_cfg.get("local_steps", 2))
    if local_steps <= 0:
        raise ValueError("audit.replay.local_steps must be positive")
    optimizer = replay_cfg.get("optimizer", {})
    if str(optimizer.get("name", "sgd")).lower() != "sgd":
        raise ValueError("the controlled A_t(M) replay currently requires SGD")
    lr = float(optimizer.get("lr", 0.01))
    momentum = float(optimizer.get("momentum", 0.0))
    weight_decay = float(optimizer.get("weight_decay", 0.0))
    if momentum != 0.0 or weight_decay != 0.0:
        raise ValueError(
            "the controlled A_t(M) replay requires momentum=0 and weight_decay=0; "
            "otherwise gamma=lr*local_steps is not the stated ideal reference"
        )
    beta = float(replay_cfg.get("beta", 0.01))
    partner_weighting = str(replay_cfg.get("partner_weighting", "sample_ratio"))
    update_rule = str(replay_cfg.get("update_rule", "proximal"))
    num_workers = int(replay_cfg.get("num_workers", 0))
    if num_workers != 0:
        raise ValueError("counterfactual replay requires replay.num_workers=0")
    pin_memory = bool(config.get("federated", {}).get("pin_memory", False))
    mean_count = float(np.mean(list(data["sample_counts"].values())))
    algorithm = build_algorithm("fedavg")
    local_states: dict[int, Mapping[str, object]] = {}
    local_metrics = []
    effective_betas = []
    for client_id in range(int(data["num_clients"])):
        loader = build_loader(
            data["dataset"],
            data["client_indices"][client_id],
            batch_size=batch_size,
            shuffle=True,
            seed=mixed_seed(batch_seed, client_id),
            num_workers=0,
            pin_memory=pin_memory,
        )
        client = Client(
            client_id=client_id,
            train_loader=loader,
            sample_count=int(data["sample_counts"][client_id]),
            label_histogram=data["histograms"][client_id],
        )
        client.previous_state = clone_state(previous_states.get(client_id, global_state))
        partner_id = pairing.pair_map.get(client_id)
        effective_beta = 0.0
        reference = None
        if partner_id is not None:
            reference = references[client_id]
            effective_beta = beta * partner_weight(
                data["sample_counts"][client_id],
                data["sample_counts"][partner_id],
                partner_weighting,
            )
            effective_betas.append(effective_beta)
        local_model = model_factory(config, data)
        set_seed(mixed_seed(batch_seed + 1_000_000, client_id))
        state, metrics = client.local_train(
            local_model,
            algorithm,
            global_state,
            paired_previous_state=reference,
            use_fcpc=partner_id is not None,
            beta=effective_beta,
            lr=lr,
            optimizer_name="sgd",
            momentum=0.0,
            weight_decay=0.0,
            nesterov=False,
            local_epochs=1,
            device=device,
            max_batches=local_steps,
            mean_sample_count=mean_count,
            fcpc_update_rule=update_rule,
            return_metrics=True,
        )
        if int(metrics["processed_batches"]) != local_steps:
            raise ValueError(
                f"client {client_id} produced {metrics['processed_batches']} batches; "
                f"the audit requires exactly {local_steps} equal local steps"
            )
        local_states[client_id] = state
        local_metrics.append(metrics)

    metric_values, pair_rows, residual_angle_rows = compute_matching_metrics(
        global_state=global_state,
        client_states=local_states,
        client_gradients=client_gradients,
        sample_counts=data["sample_counts"],
        pairing=pairing,
        gamma=lr * local_steps,
        parameter_names=parameter_names,
    )
    numerical_tolerance = max(1e-8, 1e-4 * float(metric_values["A_t_M"]))
    if float(metric_values["U_t_M"]) > float(metric_values["A_t_M"]) + numerical_tolerance:
        raise AssertionError(
            "U_t(M) exceeded A_t(M); check client coverage, aggregation weights, "
            "and the common gamma definition"
        )
    endpoint_distances = [
        state_l2_distance(
            local_states[client_i],
            local_states[client_j],
            parameter_names=parameter_name_set,
        )
        for client_i, client_j in pairing.pairs
    ]
    metric_values.update(
        {
            "gamma": lr * local_steps,
            "local_steps": local_steps,
            "learning_rate": lr,
            "beta": beta,
            "partner_weighting": partner_weighting,
            "update_rule": update_rule,
            "mean_effective_beta": mean_or_zero(effective_betas),
            "mean_center_distance": mean_or_zero(center_distances),
            "max_center_distance": max(center_distances, default=0.0),
            "mean_center_clip_scale": mean_or_default(center_clip_scales, 1.0),
            "mean_pair_history_distance": mean_or_zero(history_distances),
            "mean_pair_endpoint_distance": mean_or_zero(endpoint_distances),
            **aggregate_local_metrics(local_metrics),
        }
    )
    aggregated_state = fedavg_aggregate(
        [local_states[index] for index in range(int(data["num_clients"]))],
        [data["sample_counts"][index] for index in range(int(data["num_clients"]))],
        weighted=True,
    )
    return metric_values, pair_rows, residual_angle_rows, aggregated_state, local_states


def aggregate_local_metrics(metrics: list[Mapping[str, Any]]) -> dict[str, float]:
    total_examples = sum(int(item.get("processed_examples", 0)) for item in metrics)
    denominator = max(total_examples, 1)
    output = {}
    for name in ("task_loss", "fcpc_raw_loss", "fcpc_weighted_loss", "total_loss"):
        output[f"mean_{name}"] = sum(
            float(item.get(name, 0.0)) * int(item.get("processed_examples", 0))
            for item in metrics
        ) / denominator
    return output


def evaluate(
    config: Mapping[str, Any],
    data: Mapping[str, Any],
    state: Mapping[str, object],
    device: str,
) -> dict[str, float]:
    import torch
    from torch import nn

    max_batches = config.get("audit", {}).get("validation_max_batches")
    max_batches = None if max_batches in (None, 0) else int(max_batches)
    model = model_factory(config, data)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    with torch.no_grad():
        for batch_index, (inputs, targets) in enumerate(data["validation_loader"]):
            if max_batches is not None and batch_index >= max_batches:
                break
            inputs = inputs.to(device, non_blocking=device.startswith("cuda"))
            targets = targets.to(device, non_blocking=device.startswith("cuda"))
            logits = model(inputs)
            total_loss += float(criterion(logits, targets).item())
            total_correct += int((logits.argmax(dim=1) == targets).sum().item())
            total_examples += int(targets.numel())
    if total_examples == 0:
        return {"loss": float("nan"), "acc": float("nan")}
    return {
        "loss": total_loss / total_examples,
        "acc": total_correct / total_examples,
    }


def run_audit(config: Mapping[str, Any], *, reuse_checkpoints: bool) -> dict[str, str]:
    seed = int(config.get("seed", 42))
    audit_cfg = config.get("audit", {})
    output_dir = Path(audit_cfg.get("output_dir", "outputs/at_m_audit"))
    checkpoint_dir = Path(
        audit_cfg.get("checkpoint_dir", output_dir / "neutral_checkpoints")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    data = prepare_data(config)
    device = select_device(str(config.get("federated", {}).get("device", "auto")))
    print(f"runtime_device: {device}", flush=True)
    checkpoints = sorted({int(value) for value in audit_cfg.get("checkpoints", [5, 10, 20, 50])})
    if not reuse_checkpoints:
        create_neutral_checkpoints(config, data, checkpoint_dir)
    signature = checkpoint_signature(config)
    metadata_panels = prepare_metadata(
        data,
        epsilon=float(audit_cfg.get("epsilon", 1.0)),
        ldp_seed=int(audit_cfg.get("ldp_seed", seed + 20_000)),
    )
    panel_names = [str(value).lower() for value in audit_cfg.get("panels", ["raw", "ldp"])]
    strategies = [str(value).lower() for value in audit_cfg.get(
        "strategies", ["optimal", "random", "similar", "jsdn"]
    )]
    batch_seeds = [int(value) for value in audit_cfg.get("batch_seeds", [100])]
    random_seeds = [int(value) for value in audit_cfg.get("random_pairing_seeds", list(range(20)))]
    lambda_jsdn = float(audit_cfg.get("lambda_jsdn", 0.3))
    true_matrices = matrices_for_metadata(
        metadata_panels["raw"],
        lambda_jsdn=lambda_jsdn,
    )
    metrics_path = output_dir / "at_m_metrics.csv"
    pairs_path = output_dir / "at_m_pairs.csv"
    angles_path = output_dir / "at_m_residual_angles.csv"
    all_rows: list[dict[str, Any]] = []
    with (
        metrics_path.open("w", newline="", encoding="utf-8") as metrics_file,
        pairs_path.open("w", newline="", encoding="utf-8") as pairs_file,
        angles_path.open("w", newline="", encoding="utf-8") as angles_file,
    ):
        metrics_writer = csv.DictWriter(metrics_file, fieldnames=MAIN_FIELDS)
        pair_writer = csv.DictWriter(pairs_file, fieldnames=PAIR_FIELDS)
        angle_writer = csv.DictWriter(angles_file, fieldnames=ANGLE_FIELDS)
        metrics_writer.writeheader()
        pair_writer.writeheader()
        angle_writer.writeheader()
        for checkpoint_round in checkpoints:
            checkpoint_path = checkpoint_dir / f"fedavg_round_{checkpoint_round:03d}.pt"
            checkpoint = load_neutral_checkpoint(checkpoint_path, signature)
            print(f"audit_checkpoint: {checkpoint_round}", flush=True)
            gradients = compute_client_gradients(
                config,
                data,
                checkpoint["model_state"],
                device,
            )
            audit_parameter_names = [
                name for name, _ in model_factory(config, data).named_parameters()
            ]
            validation_before = evaluate(
                config,
                data,
                checkpoint["model_state"],
                device,
            )
            for panel in panel_names:
                if panel not in metadata_panels:
                    raise ValueError(f"unsupported audit panel: {panel}")
                observed_metadata = metadata_panels[panel]
                observed_matrices = matrices_for_metadata(
                    observed_metadata,
                    lambda_jsdn=lambda_jsdn,
                )
                for batch_seed in batch_seeds:
                    fedavg_pairing = PairingResult(
                        pairs=[],
                        pair_map={},
                        unpaired=list(range(int(data["num_clients"]))),
                    )
                    (
                        fedavg_metric_values,
                        _,
                        _,
                        _,
                        fedavg_local_states,
                    ) = replay_one_round(
                        config,
                        data,
                        checkpoint,
                        fedavg_pairing,
                        gradients,
                        batch_seed=batch_seed,
                        device=device,
                    )
                    jobs = []
                    for strategy in strategies:
                        seeds = random_seeds if strategy == "random" else [seed]
                        jobs.extend((strategy, pairing_seed) for pairing_seed in seeds)
                    for strategy, pairing_seed in jobs:
                        pairing, score_name, score_sum = choose_pairing(
                            strategy,
                            observed_matrices,
                            seed=pairing_seed,
                        )
                        (
                            metric_values,
                            pair_rows,
                            residual_angle_rows,
                            aggregated_state,
                            local_states,
                        ) = replay_one_round(
                            config,
                            data,
                            checkpoint,
                            pairing,
                            gradients,
                            batch_seed=batch_seed,
                            device=device,
                        )
                        intervention_values = compute_intervention_metrics(
                            global_state=checkpoint["model_state"],
                            client_states=local_states,
                            fedavg_client_states=fedavg_local_states,
                            client_gradients=gradients,
                            sample_counts=data["sample_counts"],
                            gamma=float(metric_values["gamma"]),
                            parameter_names=audit_parameter_names,
                        )
                        u_gap = abs(
                            float(metric_values["U_t_M"])
                            - float(intervention_values["fedavg_U_t"])
                            + float(intervention_values["correction_gain_U"])
                        )
                        u_tolerance = max(1e-8, 1e-4 * float(metric_values["U_t_M"]))
                        if u_gap > u_tolerance:
                            raise AssertionError(
                                "the intervention identity disagrees with U_t(M); "
                                "check the matched FedAvg replay"
                            )
                        fedavg_u_gap = abs(
                            float(fedavg_metric_values["U_t_M"])
                            - float(intervention_values["fedavg_U_t"])
                        )
                        if fedavg_u_gap > max(
                            1e-8, 1e-4 * float(intervention_values["fedavg_U_t"])
                        ):
                            raise AssertionError(
                                "the FedAvg replay residual disagrees with the D_t baseline"
                            )
                        metric_values.update(intervention_values)
                        validation_after = evaluate(
                            config,
                            data,
                            aggregated_state,
                            device,
                        )
                        row = {
                            "model_seed": seed,
                            "checkpoint_round": checkpoint_round,
                            "panel": panel,
                            "pairing_strategy": strategy,
                            "pairing_seed": pairing_seed if strategy == "random" else "",
                            "batch_seed": batch_seed,
                            "pair_list": json.dumps(pairing.pairs, separators=(",", ":")),
                            "selection_score_name": score_name,
                            "selection_score_sum": score_sum,
                            "true_weighted_js_sum": pairing_weight(
                                pairing,
                                true_matrices["weighted_js"],
                            ),
                            "R_M": pair_mixture_kl_residual(
                                data["histograms"],
                                data["sample_counts"],
                                pairing,
                            ),
                            "R_observed_M": pair_mixture_kl_residual(
                                observed_metadata["distributions"],
                                observed_metadata["counts"],
                                pairing,
                            ),
                            **metric_values,
                            "val_loss_before": validation_before["loss"],
                            "val_loss_after": validation_after["loss"],
                            "val_loss_change": validation_after["loss"] - validation_before["loss"],
                            "val_acc_before": validation_before["acc"],
                            "val_acc_after": validation_after["acc"],
                            "val_acc_change": validation_after["acc"] - validation_before["acc"],
                        }
                        metrics_writer.writerow({field: row.get(field, "") for field in MAIN_FIELDS})
                        metrics_file.flush()
                        all_rows.append(row)
                        for pair_row in pair_rows:
                            enriched = {
                                "model_seed": seed,
                                "checkpoint_round": checkpoint_round,
                                "panel": panel,
                                "pairing_strategy": strategy,
                                "pairing_seed": pairing_seed if strategy == "random" else "",
                                "batch_seed": batch_seed,
                                **pair_row,
                            }
                            pair_writer.writerow(
                                {field: enriched.get(field, "") for field in PAIR_FIELDS}
                            )
                        pairs_file.flush()
                        for angle_row in residual_angle_rows:
                            enriched_angle = {
                                "model_seed": seed,
                                "checkpoint_round": checkpoint_round,
                                "panel": panel,
                                "pairing_strategy": strategy,
                                "pairing_seed": pairing_seed if strategy == "random" else "",
                                "batch_seed": batch_seed,
                                **angle_row,
                            }
                            angle_writer.writerow(
                                {
                                    field: enriched_angle.get(field, "")
                                    for field in ANGLE_FIELDS
                                }
                            )
                        angles_file.flush()
                        print(
                            f"audit_result: t={checkpoint_round}, panel={panel}, "
                            f"strategy={strategy}, pairing_seed={pairing_seed}, "
                            f"batch_seed={batch_seed}, R={row['R_M']:.6e}, "
                            f"A={row['A_t_M']:.6e}, U={row['U_t_M']:.6e}, "
                            f"Dnorm={row['D_t_normalized']:.3%}, "
                            f"gain={row['correction_gain_U']:+.3e}",
                            flush=True,
                        )
            del gradients
    summary_path = output_dir / "at_m_summary.csv"
    write_summary(all_rows, summary_path)
    correlations_path = output_dir / "at_m_correlations.csv"
    write_correlations(all_rows, correlations_path)
    return {
        "metrics_path": str(metrics_path),
        "pairs_path": str(pairs_path),
        "residual_angles_path": str(angles_path),
        "summary_path": str(summary_path),
        "correlations_path": str(correlations_path),
        "checkpoint_dir": str(checkpoint_dir),
    }


def write_summary(rows: list[Mapping[str, Any]], path: Path) -> None:
    grouped: dict[tuple[int, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                int(row["checkpoint_round"]),
                str(row["panel"]),
                str(row["pairing_strategy"]),
            )
        ].append(row)
    fields = [
        "checkpoint_round",
        "panel",
        "pairing_strategy",
        "n",
        "R_mean",
        "H_mean",
        "A_mean",
        "A_std",
        "U_mean",
        "U_std",
        "D_mean",
        "D_normalized_mean",
        "D_normalized_std",
        "D_normalized_ideal_mean",
        "D_client_normalized_mean",
        "D_retention_ratio_mean",
        "D_cancellation_fraction_mean",
        "correction_alignment_cosine_mean",
        "correction_gain_U_mean",
        "correction_gain_relative_mean",
        "correction_gain_positive_fraction",
        "kappa_mean",
        "residual_cosine_mean",
        "residual_cosine_min",
        "residual_cosine_positive_fraction_mean",
        "residual_inner_product_nonnegative_fraction_mean",
        "residual_cross_term_mean",
        "alignment_assumption_hold_fraction",
        "b_min_mean",
        "val_loss_change_mean",
        "val_acc_change_mean",
        "R_delta_vs_random",
        "H_delta_vs_random",
        "A_delta_vs_random",
        "A_percent_vs_random",
        "U_delta_vs_random",
        "U_percent_vs_random",
        "val_acc_change_delta_vs_random",
    ]
    random_means: dict[tuple[int, str], dict[str, float]] = {}
    for (checkpoint_round, panel, strategy), values in grouped.items():
        if strategy != "random":
            continue
        random_means[(checkpoint_round, panel)] = {
            name: mean_or_zero([float(row[name]) for row in values])
            for name in (
                "R_M",
                "H_t_M",
                "A_t_M",
                "U_t_M",
                "val_acc_change",
            )
        }
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for key in sorted(grouped):
            values = grouped[key]
            output = {
                "checkpoint_round": key[0],
                "panel": key[1],
                "pairing_strategy": key[2],
                "n": len(values),
                "R_mean": mean_or_zero([float(row["R_M"]) for row in values]),
                "H_mean": mean_or_zero([float(row["H_t_M"]) for row in values]),
                "A_mean": mean_or_zero([float(row["A_t_M"]) for row in values]),
                "A_std": float(np.std([float(row["A_t_M"]) for row in values])),
                "U_mean": mean_or_zero([float(row["U_t_M"]) for row in values]),
                "U_std": float(np.std([float(row["U_t_M"]) for row in values])),
                "D_mean": mean_or_zero([float(row["D_t_M"]) for row in values]),
                "D_normalized_mean": mean_or_zero(
                    [float(row["D_t_normalized"]) for row in values]
                ),
                "D_normalized_std": float(
                    np.std([float(row["D_t_normalized"]) for row in values])
                ),
                "D_normalized_ideal_mean": mean_or_zero(
                    [float(row["D_t_normalized_ideal"]) for row in values]
                ),
                "D_client_normalized_mean": mean_or_zero(
                    [float(row["D_client_normalized"]) for row in values]
                ),
                "D_retention_ratio_mean": mean_or_zero(
                    [float(row["D_retention_ratio"]) for row in values]
                ),
                "D_cancellation_fraction_mean": mean_or_zero(
                    [float(row["D_cancellation_fraction"]) for row in values]
                ),
                "correction_alignment_cosine_mean": mean_finite_or_nan(
                    [float(row["correction_alignment_cosine"]) for row in values]
                ),
                "correction_gain_U_mean": mean_or_zero(
                    [float(row["correction_gain_U"]) for row in values]
                ),
                "correction_gain_relative_mean": mean_or_zero(
                    [float(row["correction_gain_relative"]) for row in values]
                ),
                "correction_gain_positive_fraction": mean_or_zero(
                    [float(row["correction_gain_U"] > 0.0) for row in values]
                ),
                "kappa_mean": mean_or_zero(
                    [float(row["kappa_U_over_A"]) for row in values]
                ),
                "residual_cosine_mean": mean_finite_or_nan(
                    [float(row["residual_cosine_mean"]) for row in values]
                ),
                "residual_cosine_min": min_finite_or_nan(
                    [float(row["residual_cosine_min"]) for row in values]
                ),
                "residual_cosine_positive_fraction_mean": mean_finite_or_nan(
                    [
                        float(row["residual_cosine_positive_fraction"])
                        for row in values
                    ]
                ),
                "residual_inner_product_nonnegative_fraction_mean": mean_finite_or_nan(
                    [
                        float(row["residual_inner_product_nonnegative_fraction"])
                        for row in values
                    ]
                ),
                "residual_cross_term_mean": mean_or_zero(
                    [float(row["residual_cross_term"]) for row in values]
                ),
                "alignment_assumption_hold_fraction": mean_or_zero(
                    [float(bool(row["alignment_assumption_holds"])) for row in values]
                ),
                "b_min_mean": mean_or_zero(
                    [float(row["b_min"]) for row in values]
                ),
                "val_loss_change_mean": mean_or_zero(
                    [float(row["val_loss_change"]) for row in values]
                ),
                "val_acc_change_mean": mean_or_zero(
                    [float(row["val_acc_change"]) for row in values]
                ),
            }
            random_reference = random_means.get((key[0], key[1]))
            if random_reference is not None:
                output.update(
                    {
                        "R_delta_vs_random": output["R_mean"] - random_reference["R_M"],
                        "H_delta_vs_random": output["H_mean"] - random_reference["H_t_M"],
                        "A_delta_vs_random": output["A_mean"] - random_reference["A_t_M"],
                        "A_percent_vs_random": percent_change(
                            output["A_mean"], random_reference["A_t_M"]
                        ),
                        "U_delta_vs_random": output["U_mean"] - random_reference["U_t_M"],
                        "U_percent_vs_random": percent_change(
                            output["U_mean"], random_reference["U_t_M"]
                        ),
                        "val_acc_change_delta_vs_random": (
                            output["val_acc_change_mean"]
                            - random_reference["val_acc_change"]
                        ),
                    }
                )
            writer.writerow(output)
            print(
                f"summary: t={key[0]}, panel={key[1]}, strategy={key[2]}, "
                f"R={output['R_mean']:.6e}, A={output['A_mean']:.6e}, "
                f"U={output['U_mean']:.6e}, Dnorm={output['D_normalized_mean']:.3%}, "
                f"gain={output['correction_gain_U_mean']:+.3e}, "
                f"dval={output['val_acc_change_mean']:+.4f}",
                flush=True,
            )


def write_correlations(rows: list[Mapping[str, Any]], path: Path) -> None:
    """Write per-checkpoint/batch rank correlations for the proposed chain."""
    grouped: dict[tuple[int, str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                int(row["checkpoint_round"]),
                str(row["panel"]),
                int(row["batch_seed"]),
            )
        ].append(row)
    fields = [
        "checkpoint_round",
        "panel",
        "batch_seed",
        "n",
        "spearman_R_H",
        "spearman_R_A",
        "spearman_A_U",
        "spearman_D_correction_gain",
        "spearman_U_val_loss_change",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for key in sorted(grouped):
            values = grouped[key]
            writer.writerow(
                {
                    "checkpoint_round": key[0],
                    "panel": key[1],
                    "batch_seed": key[2],
                    "n": len(values),
                    "spearman_R_H": spearman(
                        [float(row["R_M"]) for row in values],
                        [float(row["H_t_M"]) for row in values],
                    ),
                    "spearman_R_A": spearman(
                        [float(row["R_M"]) for row in values],
                        [float(row["A_t_M"]) for row in values],
                    ),
                    "spearman_A_U": spearman(
                        [float(row["A_t_M"]) for row in values],
                        [float(row["U_t_M"]) for row in values],
                    ),
                    "spearman_D_correction_gain": spearman(
                        [float(row["D_t_normalized"]) for row in values],
                        [float(row["correction_gain_U"]) for row in values],
                    ),
                    "spearman_U_val_loss_change": spearman(
                        [float(row["U_t_M"]) for row in values],
                        [float(row["val_loss_change"]) for row in values],
                    ),
                }
            )


def spearman(left: Iterable[float], right: Iterable[float]) -> float:
    left_values = np.asarray(list(left), dtype=np.float64)
    right_values = np.asarray(list(right), dtype=np.float64)
    if left_values.shape != right_values.shape or left_values.size < 2:
        return float("nan")
    left_rank = average_ranks(left_values)
    right_rank = average_ranks(right_values)
    left_centered = left_rank - float(left_rank.mean())
    right_centered = right_rank - float(right_rank.mean())
    left_sum_sq = float(np.sum(left_centered * left_centered))
    right_sum_sq = float(np.sum(right_centered * right_centered))
    if left_sum_sq == 0.0 or right_sum_sq == 0.0:
        return float("nan")
    numerator = float(np.sum(left_centered * right_centered))
    return float(numerator / np.sqrt(left_sum_sq * right_sum_sq))


def average_ranks(values: np.ndarray) -> np.ndarray:
    """Assign one-based average ranks, including exact ties."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and values[order[stop]] == values[order[start]]:
            stop += 1
        average = 0.5 * ((start + 1) + stop)
        ranks[order[start:stop]] = average
        start = stop
    return ranks


def percent_change(value: float, reference: float) -> float:
    if abs(float(reference)) <= 1e-12:
        return float("nan")
    return 100.0 * (float(value) - float(reference)) / float(reference)


def partner_weight(client_count: float, partner_count: float, strategy: str) -> float:
    strategy = str(strategy).lower()
    if strategy in {"uniform", "none"}:
        return 1.0
    if strategy in {"sample_ratio", "sample_ratio_x2"}:
        denominator = max(float(client_count), 0.0) + max(float(partner_count), 0.0)
        if denominator <= 0.0:
            return 0.0
        multiplier = 2.0 if strategy == "sample_ratio_x2" else 1.0
        return multiplier * max(float(partner_count), 0.0) / denominator
    raise ValueError(f"unsupported partner weighting: {strategy}")


def normalise_histogram(histogram: Iterable[float]) -> np.ndarray:
    values = np.maximum(np.asarray(list(histogram), dtype=np.float64), 0.0)
    total = float(values.sum())
    if total <= 0.0:
        return np.full(values.shape, 1.0 / max(values.size, 1), dtype=np.float64)
    return values / total


def mixed_seed(base_seed: int, client_id: int) -> int:
    return int((int(base_seed) * 1_000_003 + int(client_id) * 97) % (2**31 - 1))


def mean_or_zero(values: Iterable[float]) -> float:
    values = list(values)
    return float(np.mean(values)) if values else 0.0


def mean_or_default(values: Iterable[float], default: float) -> float:
    values = list(values)
    return float(np.mean(values)) if values else float(default)


def mean_finite_or_nan(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if np.isfinite(float(value))]
    return float(np.mean(finite)) if finite else float("nan")


def min_finite_or_nan(values: Iterable[float]) -> float:
    finite = [float(value) for value in values if np.isfinite(float(value))]
    return float(min(finite)) if finite else float("nan")


def select_device(requested: str) -> str:
    import torch

    requested = str(requested).lower()
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but unavailable: {requested}")
    return requested


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    result = run_audit(config, reuse_checkpoints=bool(args.reuse_checkpoints))
    for name, value in result.items():
        print(f"{name}: {value}")


if __name__ == "__main__":
    main()
