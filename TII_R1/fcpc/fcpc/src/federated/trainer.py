from __future__ import annotations

from dataclasses import dataclass
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
)
from src.fcpc.jsdn import build_jsdn_matrix
from src.fcpc.pairing import greedy_high_dissimilarity_pairing
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
        matrix = build_jsdn_matrix(mock_hists, mock_counts, lambda_jsdn=float(fcpc_cfg.get("lambda_jsdn", 0.3)))
        pairing = greedy_high_dissimilarity_pairing(matrix)
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
        test_dataset = load_dataset(
            dataset_name,
            root=dataset_cfg.get("root", "./data"),
            train=False,
            download=bool(dataset_cfg.get("download", False)),
            seed=seed,
            num_classes=num_classes,
            **dataset_kwargs,
        )

        labels = get_targets(train_dataset)
        partition_mode = str(partition_cfg.get("mode", "dual_skew")).lower()
        if partition_mode == "natural":
            client_indices = natural_client_partition(get_client_ids(train_dataset))
            num_clients = len(client_indices)
        else:
            num_clients = int(federated.get("num_clients", 10))
            client_indices = build_client_indices(
                labels,
                num_clients=num_clients,
                partition=partition_mode,
                alpha=float(partition_cfg.get("alpha", 0.1)),
                seed=seed,
            )
        histograms = client_label_histograms(labels, client_indices, num_classes=num_classes)
        batch_size = int(federated.get("batch_size", 64))
        clients: list[Client] = []
        for client_id in range(num_clients):
            subset = Subset(train_dataset, client_indices[client_id])
            loader = DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=int(federated.get("num_workers", 0)))
            clients.append(
                Client(
                    client_id=client_id,
                    train_loader=loader,
                    sample_count=len(client_indices[client_id]),
                    label_histogram=histograms[client_id],
                )
            )

        device = self._select_device(str(federated.get("device", "auto")))
        model = build_model(model_cfg.get("name", "simple_cnn"), num_classes=num_classes, input_channels=input_channels)
        model.to(device)
        global_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        algorithm_kwargs = {k: v for k, v in algorithm_cfg.items() if k != "name"}
        algorithm = build_algorithm(algorithm_cfg.get("name", "fedavg"), **algorithm_kwargs)
        server = Server(
            clients=clients,
            lambda_jsdn=float(fcpc_cfg.get("lambda_jsdn", 0.3)),
            aggregation_weighted=str(federated.get("aggregation", "weighted")).lower() != "equal",
        )
        pairing_strategy = str(
            fcpc_cfg.get("pairing_strategy", "fair_greedy_dissimilar")
        )
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
                "alpha",
                "beta",
                "lambda_jsdn",
                "train_clients",
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
                "test_loss",
                "test_acc",
            ],
        )

        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=int(federated.get("num_workers", 0)))
        rounds = int(federated.get("rounds", 100))
        clients_per_round = int(federated.get("clients_per_round", num_clients))
        local_epochs = int(federated.get("local_epochs", 1))
        lr = float(optimizer_cfg.get("lr", 0.01))
        use_fcpc = bool(fcpc_cfg.get("enabled", True))
        beta = float(fcpc_cfg.get("beta", 0.2))
        max_batches = federated.get("max_batches_per_client")
        max_batches = None if max_batches in (None, 0) else int(max_batches)
        checkpoint_dir = Path(logging_cfg.get("checkpoint_dir", "outputs/checkpoints"))
        profiling_cfg = self.config.get("profiling", {})
        profile_interval_s = float(profiling_cfg.get("sample_interval_s", 0.1))
        model_bytes = state_dict_nbytes(global_state)
        cumulative_total_bytes = 0

        final_metrics = {"test_loss": None, "test_acc": None}
        for round_idx in range(rounds):
            round_started = perf_counter()
            selected = server.sample_clients(clients_per_round, seed + round_idx)
            pairing_started = perf_counter()
            pairing = server.pair_selected(
                selected,
                strategy=pairing_strategy,
                seed=seed + round_idx,
            )
            pairing_time_s = perf_counter() - pairing_started
            # Freeze all partner references at the start of the round. Without
            # this snapshot, sequential simulation would make the constraint
            # asymmetric: the second client in a pair could observe the first
            # client's already-updated current-round model.
            previous_states = {
                client_id: clients[client_id].previous_state
                for client_id in selected
            }
            monitor = ResourceMonitor(device=device, interval_s=profile_interval_s).start()
            try:
                local_states = []
                for client_id in selected:
                    client = clients[client_id]
                    pair_id = pairing.pair_map.get(client_id)
                    paired_previous = previous_states.get(pair_id) if pair_id is not None else None
                    if pair_id is not None and paired_previous is None:
                        # Define the pre-round-0 partner model as the common
                        # global initialization so FCPC is defined in round 1.
                        paired_previous = global_state
                    local_model = build_model(
                        model_cfg.get("name", "simple_cnn"),
                        num_classes=num_classes,
                        input_channels=input_channels,
                    )
                    state = client.local_train(
                        local_model,
                        algorithm,
                        global_state,
                        paired_previous_state=paired_previous,
                        use_fcpc=use_fcpc,
                        beta=beta,
                        lr=lr,
                        local_epochs=local_epochs,
                        device=device,
                        max_batches=max_batches,
                    )
                    local_states.append(state)
                global_state = server.aggregate(selected, local_states)
                model.load_state_dict(global_state)
                final_metrics = self.evaluate(
                    model,
                    test_loader,
                    device=device,
                    max_batches=federated.get("max_eval_batches"),
                )
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
                    "alpha": partition_cfg.get("alpha", ""),
                    "beta": beta,
                    "lambda_jsdn": fcpc_cfg.get("lambda_jsdn", ""),
                    "train_clients": len(selected),
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
                    "test_loss": final_metrics["test_loss"],
                    "test_acc": final_metrics["test_acc"],
                }
            )

        checkpoint_path = checkpoint_dir / f"{run_name}.pt"
        save_checkpoint({"model_state": global_state, "config": self.config, "metrics": final_metrics}, checkpoint_path)
        return {"log_path": str(output_dir / f"{run_name}.csv"), "checkpoint_path": str(checkpoint_path), **final_metrics}

    def evaluate(self, model, data_loader, device: str, max_batches: int | None = None) -> Dict[str, float]:
        import torch
        from torch import nn

        model.to(device)
        model.eval()
        criterion = nn.CrossEntropyLoss(reduction="sum")
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(data_loader):
                if max_batches is not None and batch_idx >= int(max_batches):
                    break
                x = x.to(device)
                y = y.to(device)
                logits = model(x)
                total_loss += float(criterion(logits, y).item())
                total_correct += int((logits.argmax(dim=1) == y).sum().item())
                total_samples += int(y.numel())
        total_samples = max(total_samples, 1)
        return {"test_loss": total_loss / total_samples, "test_acc": total_correct / total_samples}

    @staticmethod
    def _select_device(requested: str) -> str:
        import torch

        if requested == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return requested
