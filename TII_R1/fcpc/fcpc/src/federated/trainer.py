from __future__ import annotations

from dataclasses import dataclass
from math import cos, pi
from pathlib import Path
from time import perf_counter
from typing import Any, Dict

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
from src.fcpc.jsdn import metric_matrix
from src.fcpc.pairing import PairingResult, pair_clients
from src.fcpc.regularizer import (
    clip_state_center_to_global,
    state_l2_distance,
    state_l2_norm,
    weighted_state_center,
)
from src.federated.client import Client
from src.federated.server import Server
from src.models import build_model
from src.utils.checkpoint import save_checkpoint
from src.utils.logger import CSVLogger
from src.utils.profiler import ResourceMonitor, state_dict_nbytes
from src.utils.seed import set_seed


@dataclass
class Trainer:
    config: Dict[str, Any]

    def dry_run(self) -> Dict[str, Any]:
        """Validate config and core FCPC path without dataset download/training."""
        dataset = self.config.get("dataset", {})
        model_cfg = self.config.get("model", {})
        fcpc_cfg = self.config.get("fcpc", {})
        federated = self.config.get("federated", {})

        num_clients = int(federated.get("num_clients", 10))
        num_classes = int(dataset.get("num_classes", 10))
        rng = np.random.default_rng(int(self.config.get("seed", 42)))
        mock_hists = rng.dirichlet(np.ones(num_classes), size=num_clients)
        mock_counts = rng.integers(20, 200, size=num_clients)
        pairing_metric = str(fcpc_cfg.get("metric", "jsdn"))
        matrix = metric_matrix(
            pairing_metric,
            mock_hists,
            mock_counts,
            lambda_jsdn=float(fcpc_cfg.get("lambda_jsdn", 0.3)),
        )
        pairing_strategy = str(fcpc_cfg.get("pairing_strategy", "greedy_dissimilar"))
        dry_strategy = (
            "greedy_dissimilar"
            if pairing_strategy.lower() == "fair_greedy_dissimilar"
            else pairing_strategy
        )
        pairing = pair_clients(matrix, strategy=dry_strategy, seed=int(self.config.get("seed", 42)))
        algorithm = build_algorithm(self.config.get("algorithm", {}).get("name", "fedavg"))

        torch_available = True
        try:
            model = build_model(
                model_cfg.get("name", "simple_cnn"),
                num_classes=num_classes,
                input_channels=int(dataset.get("input_channels", 1)),
            )
            model_name = model.__class__.__name__
        except Exception as exc:
            torch_available = False
            model_name = f"skipped ({exc})"

        return {
            "num_clients": num_clients,
            "num_classes": num_classes,
            "algorithm": algorithm.name,
            "pair_count": len(pairing.pairs),
            "unpaired": pairing.unpaired,
            "pairing_metric": pairing_metric,
            "pairing_strategy": pairing_strategy,
            "reference_strategy": str(fcpc_cfg.get("reference_strategy", "partner")),
            "fcpc_update_rule": str(fcpc_cfg.get("update_rule", "penalty")),
            "pairing_matrix_shape": tuple(matrix.shape),
            "jsdn_shape": tuple(matrix.shape),
            "model": model_name,
            "torch_available": torch_available,
        }

    def train(self) -> Dict[str, Any]:
        """Run a real FedAvg/FedProx + optional FCPC training job."""
        try:
            import torch
            from torch.utils.data import DataLoader, Subset
        except Exception as exc:  # pragma: no cover
            raise ImportError("PyTorch is required for full training. Use --dry-run without PyTorch.") from exc

        seed = int(self.config.get("seed", 42))
        set_seed(seed)

        dataset_cfg = self.config.get("dataset", {})
        model_cfg = self.config.get("model", {})
        federated = self.config.get("federated", {})
        partition_cfg = self.config.get("partition", {})
        fcpc_cfg = self.config.get("fcpc", {})
        optimizer_cfg = self.config.get("optimizer", {})
        scheduler_cfg = self.config.get("scheduler", {})
        evaluation_cfg = self.config.get("evaluation", {})
        logging_cfg = self.config.get("logging", {})
        algorithm_cfg = self.config.get("algorithm", {})

        dataset_name = dataset_cfg.get("name", "mnist")
        spec = get_dataset_spec(dataset_name)
        num_classes = int(dataset_cfg.get("num_classes", spec.num_classes))
        input_channels = int(dataset_cfg.get("input_channels", spec.input_channels))
        dataset_kwargs = dict(dataset_cfg)
        dataset_kwargs.pop("name", None)
        dataset_kwargs.pop("root", None)
        dataset_kwargs.pop("download", None)
        dataset_kwargs.pop("num_classes", None)
        dataset_kwargs.pop("input_channels", None)
        dataset_kwargs.pop("dataset_grid", None)
        train_dataset = load_dataset(
            dataset_name,
            root=dataset_cfg.get("root", "./data"),
            train=True,
            download=bool(dataset_cfg.get("download", False)),
            seed=seed,
            num_classes=num_classes,
            **dataset_kwargs,
        )
        validation_fraction = float(evaluation_cfg.get("validation_fraction", 0.0))
        validation_dataset = None
        if validation_fraction > 0.0:
            validation_kwargs = dict(dataset_kwargs)
            # A validation view of the training split must be deterministic.
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
        test_dataset = load_dataset(
            dataset_name,
            root=dataset_cfg.get("root", "./data"),
            train=False,
            download=bool(dataset_cfg.get("download", False)),
            seed=seed,
            num_classes=num_classes,
            **dataset_kwargs,
        )

        all_labels = get_targets(train_dataset)
        train_indices, validation_indices = stratified_holdout_indices(
            all_labels,
            validation_fraction=validation_fraction,
            seed=int(evaluation_cfg.get("validation_seed", seed + 10_000)),
        )
        train_labels = [all_labels[index] for index in train_indices]
        partition_mode = str(partition_cfg.get("mode", "dual_skew")).lower()
        if partition_mode == "natural":
            all_client_ids = get_client_ids(train_dataset)
            train_client_ids = [all_client_ids[index] for index in train_indices]
            relative_client_indices = natural_client_partition(train_client_ids)
            num_clients = len(relative_client_indices)
        else:
            num_clients = int(federated.get("num_clients", 10))
            relative_client_indices = build_client_indices(
                train_labels,
                num_clients=num_clients,
                partition=partition_mode,
                alpha=float(partition_cfg.get("alpha", 0.1)),
                seed=seed,
                quantity_min_fraction=float(
                    partition_cfg.get("quantity_min_fraction", 0.25)
                ),
            )
        partition_stats = validate_complete_partition(
            relative_client_indices,
            len(train_indices),
        )
        print(
            "partition_coverage: "
            f"assigned={partition_stats['assigned_examples']}, "
            f"unique={partition_stats['unique_examples']}, "
            f"expected={len(train_indices)}, "
            f"client_min={partition_stats['client_examples_min']}, "
            f"client_max={partition_stats['client_examples_max']}",
            flush=True,
        )
        client_indices = {
            client_id: [train_indices[position] for position in positions]
            for client_id, positions in relative_client_indices.items()
        }
        histograms = client_label_histograms(
            train_labels,
            relative_client_indices,
            num_classes=num_classes,
        )
        batch_size = int(federated.get("batch_size", 64))
        num_workers = int(federated.get("num_workers", 0))
        pin_memory = bool(federated.get("pin_memory", False))
        persistent_workers = bool(
            federated.get("persistent_workers", False)
        ) and num_workers > 0
        clients: list[Client] = []
        for client_id in range(num_clients):
            subset = Subset(train_dataset, client_indices[client_id])
            loader = DataLoader(
                subset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
            )
            clients.append(
                Client(
                    client_id=client_id,
                    train_loader=loader,
                    sample_count=len(client_indices[client_id]),
                    label_histogram=histograms[client_id],
                )
            )

        device = self._select_device(str(federated.get("device", "auto")))
        gpu_name = ""
        cuda_version = ""
        if device.startswith("cuda"):
            device_index = torch.device(device).index
            if device_index is None:
                device_index = torch.cuda.current_device()
            gpu_name = str(torch.cuda.get_device_name(device_index))
            cuda_version = str(torch.version.cuda or "")
        print(
            f"runtime_device: {device}; gpu_name: {gpu_name or 'none'}; "
            f"torch_cuda: {cuda_version or 'none'}",
            flush=True,
        )
        model = build_model(model_cfg.get("name", "simple_cnn"), num_classes=num_classes, input_channels=input_channels)
        model.to(device)
        global_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        algorithm_kwargs = {k: v for k, v in algorithm_cfg.items() if k != "name"}
        if str(algorithm_cfg.get("name", "fedavg")).lower() == "fedcfa":
            algorithm_kwargs.setdefault("num_classes", num_classes)
        algorithm = build_algorithm(algorithm_cfg.get("name", "fedavg"), **algorithm_kwargs)
        use_fcpc = bool(fcpc_cfg.get("enabled", True))
        model_factory = lambda: build_model(
            model_cfg.get("name", "simple_cnn"),
            num_classes=num_classes,
            input_channels=input_channels,
        )
        server = Server(
            clients=clients,
            lambda_jsdn=float(fcpc_cfg.get("lambda_jsdn", 0.3)),
            pairing_metric=str(fcpc_cfg.get("metric", "jsdn")),
            aggregation_weighted=str(federated.get("aggregation", "weighted")).lower() != "equal",
        )
        pairing_strategy = str(
            fcpc_cfg.get("pairing_strategy", "fair_greedy_dissimilar")
        )
        reference_strategy = str(
            fcpc_cfg.get("reference_strategy", "partner")
        ).lower()
        if reference_strategy not in {"partner", "pair_center"}:
            raise ValueError(
                "fcpc.reference_strategy must be 'partner' or 'pair_center'"
            )
        fcpc_update_rule = str(fcpc_cfg.get("update_rule", "penalty")).lower()
        if fcpc_update_rule not in {"penalty", "proximal"}:
            raise ValueError("fcpc.update_rule must be 'penalty' or 'proximal'")
        if fcpc_update_rule == "proximal" and reference_strategy != "pair_center":
            raise ValueError(
                "fcpc.update_rule='proximal' requires reference_strategy='pair_center'"
            )
        if use_fcpc:
            server.build_pairing(
                epsilon=float(fcpc_cfg.get("epsilon", 1.0)),
                seed=seed,
                strategy=pairing_strategy,
            )

        output_dir = Path(logging_cfg.get("output_dir", "outputs/logs"))
        run_name = logging_cfg.get("run_name", f"{dataset_name}_{algorithm.name}")
        logger = CSVLogger(
            output_dir / f"{run_name}.csv",
            fieldnames=[
                "round",
                "dataset",
                "algorithm",
                "fcpc",
                "device",
                "gpu_name",
                "gpu_monitor_backend",
                "gpu_sample_count",
                "pairing_metric",
                "reference_strategy",
                "fcpc_update_rule",
                "center_clip_limit",
                "mean_center_distance",
                "mean_center_clip_scale",
                "safe_pairing_limit",
                "safe_pairing_feasible_edges",
                "train_pool_examples",
                "assigned_unique_examples",
                "client_examples_min",
                "client_examples_max",
                "alpha",
                "beta_base",
                "beta",
                "beta_schedule",
                "beta_min",
                "partner_weighting",
                "mean_partner_weight",
                "mean_effective_beta",
                "lambda_jsdn",
                "learning_rate",
                "train_clients",
                "train_examples",
                "train_task_loss",
                "train_algorithm_loss",
                "train_fcpc_raw_loss",
                "train_fcpc_weighted_loss",
                "train_total_loss",
                "pairing_strategy",
                "pair_count",
                "unpaired_count",
                "pairing_time_s",
                "round_time_s",
                "process_cpu_mean_pct",
                "process_cpu_peak_pct",
                "rss_peak_mib",
                "gpu_util_mean_pct",
                "gpu_util_peak_pct",
                "gpu_memory_peak_mib",
                "server_download_bytes",
                "server_upload_bytes",
                "peer_upload_bytes",
                "round_total_bytes",
                "cumulative_total_bytes",
                "val_loss",
                "val_acc",
                "test_loss",
                "test_acc",
            ],
        )

        validation_loader = None
        if validation_dataset is not None and validation_indices:
            validation_loader = DataLoader(
                Subset(validation_dataset, validation_indices),
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
            )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
        reference_inputs = None
        if validation_loader is not None:
            try:
                reference_inputs = next(iter(validation_loader))[0].detach().cpu()
            except StopIteration:
                reference_inputs = None

        global_mean_x = None
        global_mean_y = None
        rounds = int(federated.get("rounds", 100))
        clients_per_round = int(federated.get("clients_per_round", num_clients))
        local_epochs = int(federated.get("local_epochs", 1))
        base_lr = float(optimizer_cfg.get("lr", 0.01))
        optimizer_name = str(optimizer_cfg.get("name", "sgd"))
        momentum = float(optimizer_cfg.get("momentum", 0.0))
        weight_decay = float(optimizer_cfg.get("weight_decay", 0.0))
        nesterov = bool(optimizer_cfg.get("nesterov", False))
        beta_base = float(fcpc_cfg.get("beta", 0.2))
        beta_schedule = str(fcpc_cfg.get("beta_schedule", "constant")).lower()
        beta_min = float(fcpc_cfg.get("min_beta", 0.0))
        partner_weighting = str(fcpc_cfg.get("partner_weighting", "uniform")).lower()
        max_batches = federated.get("max_batches_per_client")
        max_batches = None if max_batches in (None, 0) else int(max_batches)
        checkpoint_dir = Path(logging_cfg.get("checkpoint_dir", "outputs/checkpoints"))
        checkpoint_path = checkpoint_dir / f"{run_name}.pt"
        best_checkpoint_path = checkpoint_dir / f"{run_name}_best.pt"
        profiling_cfg = self.config.get("profiling", {})
        profile_interval_s = float(profiling_cfg.get("sample_interval_s", 0.1))
        model_bytes = state_dict_nbytes(global_state)
        parameter_names = set(dict(model.named_parameters()))
        cumulative_total_bytes = 0

        max_eval_batches = federated.get("max_eval_batches")
        test_every_round = bool(
            evaluation_cfg.get("test_every_round", validation_loader is None)
        )
        best_val_acc = float("-inf")
        best_val_loss = float("inf")
        best_round = 0
        best_global_state = None
        final_test_metrics = {"test_loss": None, "test_acc": None}
        final_val_metrics = {"test_loss": None, "test_acc": None}
        mean_sample_count = float(np.mean([client.sample_count for client in clients]))
        for round_idx in range(rounds):
            round_started = perf_counter()
            beta = self._beta_for_round(
                beta_base,
                round_idx=round_idx,
                total_rounds=rounds,
                schedule=beta_schedule,
                min_beta=beta_min,
            )
            round_lr = self._learning_rate_for_round(
                base_lr,
                round_idx=round_idx,
                total_rounds=rounds,
                scheduler_cfg=scheduler_cfg,
            )
            algorithm.begin_round(
                round_idx=round_idx,
                global_state=global_state,
                clients=clients,
                global_mean_x=global_mean_x,
                global_mean_y=global_mean_y,
            )
            selected = algorithm.select_clients(
                server,
                clients_per_round,
                seed + round_idx,
                global_state=global_state,
                model_factory=model_factory,
                device=device,
                max_batches=max_batches,
            )
            # Freeze every selected client's pre-round state before pairing or
            # local training.  The same snapshots are used to construct safe
            # matching constraints and the common centers.
            previous_states = {
                client_id: clients[client_id].previous_state
                for client_id in selected
            }
            global_parameter_norm = state_l2_norm(
                global_state,
                parameter_names=parameter_names,
            )
            safe_pairing_limit = self._resolve_distance_limit(
                fcpc_cfg.get("pairing_max_center_distance"),
                fcpc_cfg.get("pairing_max_center_relative_distance"),
                global_parameter_norm,
            )
            feasible_mask = None
            feasible_edge_count = ""
            if use_fcpc and safe_pairing_limit is not None:
                if pairing_strategy.lower() == "random":
                    raise ValueError(
                        "safe center constraints are not supported with random pairing"
                    )
                feasible_mask, feasible_edge_count = self._safe_pairing_mask(
                    clients=clients,
                    selected=selected,
                    previous_states=previous_states,
                    global_state=global_state,
                    parameter_names=parameter_names,
                    max_center_distance=safe_pairing_limit,
                )
            if use_fcpc:
                pairing_started = perf_counter()
                pairing = server.pair_selected(
                    selected,
                    strategy=pairing_strategy,
                    seed=seed + round_idx,
                    feasible_mask=feasible_mask,
                )
                pairing_time_s = perf_counter() - pairing_started
            else:
                pairing = PairingResult(
                    pairs=[],
                    pair_map={},
                    unpaired=[int(client_id) for client_id in selected],
                )
                pairing_time_s = 0.0
            reference_states = {}
            center_distances = []
            center_clip_scales = []
            center_clip_limit = self._resolve_distance_limit(
                fcpc_cfg.get("center_max_distance"),
                fcpc_cfg.get("center_max_relative_distance"),
                global_parameter_norm,
            )
            if reference_strategy == "pair_center":
                for client_a, client_b in pairing.pairs:
                    center = weighted_state_center(
                        previous_states.get(client_a),
                        previous_states.get(client_b),
                        clients[client_a].sample_count,
                        clients[client_b].sample_count,
                        fallback_state=global_state,
                    )
                    center, center_distance, clip_scale = clip_state_center_to_global(
                        center,
                        global_state,
                        max_distance=center_clip_limit,
                        parameter_names=parameter_names,
                    )
                    center_distances.append(center_distance)
                    center_clip_scales.append(clip_scale)
                    # Both clients are constrained to exactly the same frozen
                    # pair center.  This is the model-space counterpart of the
                    # sample-weighted label mixture used in the edge score.
                    reference_states[client_a] = center
                    reference_states[client_b] = center
            monitor = ResourceMonitor(device=device, interval_s=profile_interval_s).start()
            try:
                local_states = []
                local_metrics = []
                active_partner_weights = []
                active_effective_betas = []
                for client_id in selected:
                    client = clients[client_id]
                    pair_id = pairing.pair_map.get(client_id)
                    if reference_strategy == "pair_center":
                        paired_previous = reference_states.get(client_id)
                    else:
                        paired_previous = previous_states.get(pair_id) if pair_id is not None else None
                    if pair_id is not None and paired_previous is None:
                        # Define the pre-round-0 partner model as the common
                        # global initialization so FCPC is defined in round 1.
                        paired_previous = global_state
                    partner_weight = 0.0
                    effective_beta = 0.0
                    if use_fcpc and pair_id is not None:
                        partner_weight = self._partner_weight_for_pair(
                            client.sample_count,
                            clients[pair_id].sample_count,
                            strategy=partner_weighting,
                        )
                        effective_beta = beta * partner_weight
                        active_partner_weights.append(partner_weight)
                        active_effective_betas.append(effective_beta)
                    local_model = build_model(
                        model_cfg.get("name", "simple_cnn"),
                        num_classes=num_classes,
                        input_channels=input_channels,
                    )
                    state, metrics = client.local_train(
                        local_model,
                        algorithm,
                        global_state,
                        paired_previous_state=paired_previous,
                        use_fcpc=use_fcpc,
                        beta=effective_beta,
                        lr=round_lr,
                        optimizer_name=optimizer_name,
                        momentum=momentum,
                        weight_decay=weight_decay,
                        nesterov=nesterov,
                        local_epochs=local_epochs,
                        device=device,
                        max_batches=max_batches,
                        mean_sample_count=mean_sample_count,
                        fcpc_update_rule=fcpc_update_rule,
                        return_metrics=True,
                    )
                    self._assert_finite_state(
                        state,
                        where=f"round {round_idx + 1}, client {client_id}",
                    )
                    local_states.append(state)
                    local_metrics.append(metrics)
                default_global_state = server.aggregate(selected, local_states)
                custom_global_state = algorithm.aggregate(
                    selected_client_ids=selected,
                    client_states=local_states,
                    default_state=default_global_state,
                    previous_global_state=global_state,
                    clients=clients,
                )
                global_state = (
                    custom_global_state
                    if custom_global_state is not None
                    else default_global_state
                )
                self._assert_finite_state(
                    global_state,
                    where=f"round {round_idx + 1}, server aggregation",
                )
                algorithm.after_round(
                    round_idx=round_idx,
                    selected_client_ids=selected,
                    client_states=local_states,
                    local_metrics=local_metrics,
                    global_state=global_state,
                    model_factory=model_factory,
                    reference_inputs=reference_inputs,
                    device=device,
                    clients=clients,
                    num_classes=num_classes,
                )
                if algorithm.name == "fedcfa":
                    global_mean_x, global_mean_y = self._build_global_mean_pool(
                        [clients[client_id] for client_id in selected],
                        num_classes=num_classes,
                        mean_batch_size=int(
                            getattr(algorithm, "mean_batch_size", batch_size)
                        ),
                    )
                model.load_state_dict(global_state)
                round_train_metrics = self._aggregate_local_metrics(local_metrics)
                if validation_loader is not None:
                    final_val_metrics = self.evaluate(
                        model,
                        validation_loader,
                        device=device,
                        max_batches=max_eval_batches,
                    )
                    if float(final_val_metrics["test_acc"]) > best_val_acc:
                        best_val_acc = float(final_val_metrics["test_acc"])
                        best_val_loss = float(final_val_metrics["test_loss"])
                        best_round = round_idx + 1
                        best_global_state = {
                            key: value.detach().cpu().clone()
                            for key, value in global_state.items()
                        }
                should_test = test_every_round or round_idx == rounds - 1
                round_test_metrics = {"test_loss": "", "test_acc": ""}
                if should_test:
                    final_test_metrics = self.evaluate(
                        model,
                        test_loader,
                        device=device,
                        max_batches=max_eval_batches,
                    )
                    round_test_metrics = final_test_metrics
            finally:
                resource_stats = monitor.stop()

            server_download_bytes = len(selected) * model_bytes
            server_upload_bytes = len(selected) * model_bytes
            peer_upload_bytes = 2 * len(pairing.pairs) * model_bytes if use_fcpc else 0
            round_total_bytes = (
                server_download_bytes + server_upload_bytes + peer_upload_bytes
            )
            cumulative_total_bytes += round_total_bytes
            round_time_s = perf_counter() - round_started
            logger.log(
                {
                    "round": round_idx + 1,
                    "dataset": dataset_name,
                    "algorithm": algorithm.name,
                    "fcpc": use_fcpc,
                    "device": device,
                    "gpu_name": gpu_name,
                    "gpu_monitor_backend": resource_stats.gpu_monitor_backend,
                    "gpu_sample_count": resource_stats.gpu_sample_count,
                    "pairing_metric": server.pairing_metric,
                    "reference_strategy": reference_strategy,
                    "fcpc_update_rule": fcpc_update_rule,
                    "center_clip_limit": (
                        "" if center_clip_limit is None else center_clip_limit
                    ),
                    "mean_center_distance": (
                        float(np.mean(center_distances)) if center_distances else 0.0
                    ),
                    "mean_center_clip_scale": (
                        float(np.mean(center_clip_scales)) if center_clip_scales else 1.0
                    ),
                    "safe_pairing_limit": (
                        "" if safe_pairing_limit is None else safe_pairing_limit
                    ),
                    "safe_pairing_feasible_edges": feasible_edge_count,
                    "train_pool_examples": len(train_indices),
                    "assigned_unique_examples": partition_stats["unique_examples"],
                    "client_examples_min": partition_stats["client_examples_min"],
                    "client_examples_max": partition_stats["client_examples_max"],
                    "alpha": partition_cfg.get("alpha", ""),
                    "beta_base": beta_base,
                    "beta": beta,
                    "beta_schedule": beta_schedule,
                    "beta_min": beta_min,
                    "partner_weighting": partner_weighting,
                    "mean_partner_weight": (
                        float(np.mean(active_partner_weights))
                        if active_partner_weights else 0.0
                    ),
                    "mean_effective_beta": (
                        float(np.mean(active_effective_betas))
                        if active_effective_betas else 0.0
                    ),
                    "lambda_jsdn": fcpc_cfg.get("lambda_jsdn", ""),
                    "learning_rate": round_lr,
                    "train_clients": len(selected),
                    "train_examples": round_train_metrics["processed_examples"],
                    "train_task_loss": round_train_metrics["task_loss"],
                    "train_algorithm_loss": round_train_metrics["algorithm_loss"],
                    "train_fcpc_raw_loss": round_train_metrics["fcpc_raw_loss"],
                    "train_fcpc_weighted_loss": round_train_metrics["fcpc_weighted_loss"],
                    "train_total_loss": round_train_metrics["total_loss"],
                    "pairing_strategy": pairing_strategy,
                    "pair_count": len(pairing.pairs),
                    "unpaired_count": len(pairing.unpaired),
                    "pairing_time_s": pairing_time_s,
                    "round_time_s": round_time_s,
                    "process_cpu_mean_pct": resource_stats.process_cpu_mean_pct,
                    "process_cpu_peak_pct": resource_stats.process_cpu_peak_pct,
                    "rss_peak_mib": resource_stats.rss_peak_mib,
                    "gpu_util_mean_pct": resource_stats.gpu_util_mean_pct,
                    "gpu_util_peak_pct": resource_stats.gpu_util_peak_pct,
                    "gpu_memory_peak_mib": resource_stats.gpu_memory_peak_mib,
                    "server_download_bytes": server_download_bytes,
                    "server_upload_bytes": server_upload_bytes,
                    "peer_upload_bytes": peer_upload_bytes,
                    "round_total_bytes": round_total_bytes,
                    "cumulative_total_bytes": cumulative_total_bytes,
                    "val_loss": final_val_metrics["test_loss"] if validation_loader is not None else "",
                    "val_acc": final_val_metrics["test_acc"] if validation_loader is not None else "",
                    "test_loss": round_test_metrics["test_loss"],
                    "test_acc": round_test_metrics["test_acc"],
                }
            )

        last_global_state = global_state
        last_test_metrics = final_test_metrics
        selected_test_metrics = last_test_metrics
        if best_global_state is not None:
            model.load_state_dict(best_global_state)
            selected_test_metrics = self.evaluate(
                model,
                test_loader,
                device=device,
                max_batches=max_eval_batches,
            )
            save_checkpoint(
                {
                    "model_state": best_global_state,
                    "config": self.config,
                    "metrics": {
                        "best_round": best_round,
                        "best_val_loss": best_val_loss,
                        "best_val_acc": best_val_acc,
                        **selected_test_metrics,
                    },
                },
                best_checkpoint_path,
            )

        save_checkpoint(
            {
                "model_state": last_global_state,
                "config": self.config,
                "metrics": {
                    "last_test_loss": last_test_metrics["test_loss"],
                    "last_test_acc": last_test_metrics["test_acc"],
                    "best_round": best_round or rounds,
                    "best_val_loss": None if best_global_state is None else best_val_loss,
                    "best_val_acc": None if best_global_state is None else best_val_acc,
                },
            },
            checkpoint_path,
        )
        return {
            "log_path": str(output_dir / f"{run_name}.csv"),
            "checkpoint_path": str(checkpoint_path),
            "best_checkpoint_path": str(best_checkpoint_path) if best_global_state is not None else "",
            "best_round": best_round or rounds,
            "best_val_loss": None if best_global_state is None else best_val_loss,
            "best_val_acc": None if best_global_state is None else best_val_acc,
            "test_loss": selected_test_metrics["test_loss"],
            "test_acc": selected_test_metrics["test_acc"],
            "last_test_loss": last_test_metrics["test_loss"],
            "last_test_acc": last_test_metrics["test_acc"],
            "device": device,
            "gpu_name": gpu_name,
            "train_pool_examples": len(train_indices),
            "assigned_unique_examples": partition_stats["unique_examples"],
        }

    @staticmethod
    def _assert_finite_state(state, where: str):
        """Fail at the first non-finite model tensor with useful context."""
        import torch

        invalid = [
            name
            for name, value in state.items()
            if value.is_floating_point() and not bool(torch.isfinite(value).all())
        ]
        if invalid:
            preview = ", ".join(invalid[:5])
            suffix = " ..." if len(invalid) > 5 else ""
            raise FloatingPointError(
                f"non-finite model state after {where}: {preview}{suffix}"
            )

    @staticmethod
    def _build_global_mean_pool(clients, num_classes: int, mean_batch_size: int):
        """Build FedCFA's server pool from client-side batch summaries.

        Each stored item is an average input and an average one-hot label for
        one local mini-batch.  Raw client examples are not retained by this
        helper, matching the released FedCFA code's mean-data interface.
        """
        import torch
        import torch.nn.functional as F

        mean_batch_size = max(int(mean_batch_size), 1)
        mean_inputs = []
        mean_labels = []
        for client in clients:
            buffered_inputs = []
            buffered_labels = []
            buffered_count = 0
            for inputs, targets in client.train_loader:
                offset = 0
                while offset < len(targets):
                    take = min(mean_batch_size - buffered_count, len(targets) - offset)
                    buffered_inputs.append(inputs[offset : offset + take].detach().cpu())
                    buffered_labels.append(targets[offset : offset + take].detach().cpu())
                    buffered_count += take
                    offset += take
                    if buffered_count == mean_batch_size:
                        group_inputs = torch.cat(buffered_inputs, dim=0)
                        group_targets = torch.cat(buffered_labels, dim=0)
                        mean_inputs.append(group_inputs.float().mean(dim=0))
                        mean_labels.append(
                            F.one_hot(group_targets.long(), num_classes=num_classes)
                            .float()
                            .mean(dim=0)
                        )
                        buffered_inputs = []
                        buffered_labels = []
                        buffered_count = 0
            if buffered_count:
                group_inputs = torch.cat(buffered_inputs, dim=0)
                group_targets = torch.cat(buffered_labels, dim=0)
                mean_inputs.append(group_inputs.float().mean(dim=0))
                mean_labels.append(
                    F.one_hot(group_targets.long(), num_classes=num_classes)
                    .float()
                    .mean(dim=0)
                )
        if not mean_inputs:
            raise ValueError("FedCFA requires at least one non-empty client")
        return torch.stack(mean_inputs), torch.stack(mean_labels)

    def evaluate(self, model, data_loader, device: str, max_batches: int | None = None) -> Dict[str, float]:
        import torch
        from torch import nn

        model.to(device)
        model.eval()
        criterion = nn.CrossEntropyLoss(reduction="sum")
        non_blocking = str(device).startswith("cuda")
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(data_loader):
                if max_batches is not None and batch_idx >= int(max_batches):
                    break
                x = x.to(device, non_blocking=non_blocking)
                y = y.to(device, non_blocking=non_blocking)
                logits = model(x)
                if not bool(torch.isfinite(logits).all()):
                    raise FloatingPointError(
                        f"non-finite logits during evaluation batch {batch_idx}"
                    )
                total_loss += float(criterion(logits, y).item())
                total_correct += int((logits.argmax(dim=1) == y).sum().item())
                total_samples += int(y.numel())
        total_samples = max(total_samples, 1)
        return {"test_loss": total_loss / total_samples, "test_acc": total_correct / total_samples}

    @staticmethod
    def _aggregate_local_metrics(local_metrics: list[Dict[str, Any]]) -> Dict[str, float]:
        """Compute example-weighted local-loss means for a communication round."""
        total_examples = sum(int(metrics.get("processed_examples", 0)) for metrics in local_metrics)
        denominator = max(total_examples, 1)
        result: Dict[str, float] = {
            "processed_examples": float(total_examples),
            "processed_batches": float(
                sum(int(metrics.get("processed_batches", 0)) for metrics in local_metrics)
            ),
        }
        for name in (
            "task_loss",
            "algorithm_loss",
            "fcpc_raw_loss",
            "fcpc_weighted_loss",
            "total_loss",
        ):
            weighted_sum = sum(
                float(metrics.get(name, 0.0)) * int(metrics.get("processed_examples", 0))
                for metrics in local_metrics
            )
            result[name] = weighted_sum / denominator
        return result

    @staticmethod
    def _learning_rate_for_round(
        base_lr: float,
        round_idx: int,
        total_rounds: int,
        scheduler_cfg: Dict[str, Any],
    ) -> float:
        """Resolve a round-level learning rate for stateless local optimizers."""
        name = str(scheduler_cfg.get("name", "constant")).lower()
        if name in {"constant", "none"}:
            return float(base_lr)
        if name == "cosine":
            min_lr = float(scheduler_cfg.get("min_lr", 0.0))
            progress = round_idx / max(total_rounds - 1, 1)
            return min_lr + 0.5 * (float(base_lr) - min_lr) * (1.0 + cos(pi * progress))
        if name == "step":
            step_size = max(int(scheduler_cfg.get("step_size", 50)), 1)
            gamma = float(scheduler_cfg.get("gamma", 0.1))
            return float(base_lr) * gamma ** (round_idx // step_size)
        raise ValueError(f"unsupported scheduler: {name}")

    @staticmethod
    def _beta_for_round(
        base_beta: float,
        round_idx: int,
        total_rounds: int,
        schedule: str = "constant",
        min_beta: float = 0.0,
    ) -> float:
        """Resolve the round-specific FCPC coefficient beta_t."""
        base_beta = float(base_beta)
        min_beta = float(min_beta)
        if base_beta < 0.0 or min_beta < 0.0:
            raise ValueError("FCPC beta values must be nonnegative")
        name = str(schedule).lower()
        if name in {"constant", "none"}:
            return base_beta
        if name in {"cosine", "cosine_decay"}:
            if min_beta > base_beta:
                raise ValueError("min_beta cannot exceed the initial beta")
            progress = round_idx / max(total_rounds - 1, 1)
            return min_beta + 0.5 * (base_beta - min_beta) * (
                1.0 + cos(pi * progress)
            )
        raise ValueError(f"unsupported FCPC beta schedule: {schedule}")

    @staticmethod
    def _partner_weight_for_pair(
        client_sample_count: int,
        partner_sample_count: int,
        strategy: str = "uniform",
    ) -> float:
        """Allocate beta by partner reliability with explicit x1/x2 ablations."""
        name = str(strategy).lower()
        if name in {"uniform", "none"}:
            return 1.0
        if name in {"sample_ratio", "sample_ratio_x2"}:
            client_count = max(int(client_sample_count), 0)
            partner_count = max(int(partner_sample_count), 0)
            denominator = client_count + partner_count
            if not denominator:
                return 0.0
            scale = 2.0 if name == "sample_ratio_x2" else 1.0
            return float(scale * partner_count / denominator)
        raise ValueError(f"unsupported FCPC partner weighting: {strategy}")

    @staticmethod
    def _resolve_distance_limit(
        absolute_value,
        relative_value,
        reference_norm: float,
    ) -> float | None:
        """Resolve an absolute or scale-aware model-space safety radius."""
        if absolute_value not in (None, ""):
            limit = float(absolute_value)
            if limit < 0.0:
                raise ValueError("absolute center distance limits must be non-negative")
            return limit
        if relative_value not in (None, ""):
            ratio = float(relative_value)
            if ratio < 0.0:
                raise ValueError("relative center distance limits must be non-negative")
            return ratio * float(reference_norm)
        return None

    @staticmethod
    def _safe_pairing_mask(
        *,
        clients,
        selected: list[int],
        previous_states,
        global_state,
        parameter_names: set[str],
        max_center_distance: float,
    ) -> tuple[np.ndarray, int]:
        """Allow only edges whose *unclipped* center is near the global model.

        An infeasible client is left unpaired rather than silently falling
        back to an unsafe edge.  It still performs ordinary local training and
        participates in server aggregation.
        """
        n_clients = len(clients)
        mask = np.zeros((n_clients, n_clients), dtype=bool)
        feasible_edges = 0
        selected = [int(client_id) for client_id in selected]
        for offset, client_a in enumerate(selected):
            for client_b in selected[offset + 1 :]:
                center = weighted_state_center(
                    previous_states.get(client_a),
                    previous_states.get(client_b),
                    clients[client_a].sample_count,
                    clients[client_b].sample_count,
                    fallback_state=global_state,
                )
                distance = state_l2_distance(
                    center,
                    global_state,
                    parameter_names=parameter_names,
                )
                if distance <= float(max_center_distance):
                    mask[client_a, client_b] = True
                    mask[client_b, client_a] = True
                    feasible_edges += 1
        return mask, feasible_edges

    @staticmethod
    def _select_device(requested: str) -> str:
        import torch

        requested = str(requested).lower()
        if requested == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA was explicitly requested ({requested}) but torch.cuda.is_available() is False"
            )
        return requested
