from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from src.fcpc.regularizer import fcpc_regularization


@dataclass
class Client:
    client_id: int
    train_loader: object | None = None
    sample_count: int = 0
    label_histogram: np.ndarray | None = None
    previous_state: Mapping[str, object] | None = field(default=None, init=False)

    def local_train(
        self,
        model,
        algorithm,
        global_state: Mapping[str, object],
        paired_previous_state: Mapping[str, object] | None = None,
        use_fcpc: bool = False,
        beta: float = 0.2,
        lr: float = 0.01,
        optimizer_name: str = "sgd",
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
        local_epochs: int = 1,
        device: str = "cpu",
        max_batches: int | None = None,
        return_metrics: bool = False,
    ):
        """Run local training. Requires PyTorch and a DataLoader."""
        if self.train_loader is None:
            raise ValueError("train_loader is required for local training")
        import torch
        from torch import nn

        model.load_state_dict(global_state)
        model.to(device)
        model.train()
        optimizer = self._build_optimizer(
            torch,
            model,
            name=optimizer_name,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )
        criterion = nn.CrossEntropyLoss()

        metric_sums = {
            "task_loss": 0.0,
            "algorithm_loss": 0.0,
            "fcpc_raw_loss": 0.0,
            "fcpc_weighted_loss": 0.0,
            "total_loss": 0.0,
        }
        processed_examples = 0
        processed_batches = 0

        for _ in range(local_epochs):
            for batch_idx, (x, y) in enumerate(self.train_loader):
                if max_batches is not None and batch_idx >= max_batches:
                    break
                x = x.to(device)
                y = y.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(x)
                task_loss = criterion(logits, y)
                context = {
                    "global_state": global_state,
                    "current_logits": logits,
                }
                algorithm_loss = algorithm.extra_loss(model, (x, y), task_loss, context)
                fcpc_raw_loss = task_loss.new_tensor(0.0)
                if use_fcpc and paired_previous_state is not None:
                    fcpc_raw_loss = fcpc_regularization(
                        dict(model.named_parameters()),
                        paired_previous_state,
                        beta=1.0,
                    )
                fcpc_weighted_loss = float(beta) * fcpc_raw_loss
                loss = task_loss + algorithm_loss + fcpc_weighted_loss
                loss.backward()
                optimizer.step()

                batch_examples = int(y.numel())
                processed_examples += batch_examples
                processed_batches += 1
                values = {
                    "task_loss": task_loss,
                    "algorithm_loss": algorithm_loss,
                    "fcpc_raw_loss": fcpc_raw_loss,
                    "fcpc_weighted_loss": fcpc_weighted_loss,
                    "total_loss": loss,
                }
                for name, value in values.items():
                    metric_sums[name] += float(value.detach().item()) * batch_examples

        self.previous_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if not return_metrics:
            return self.previous_state

        denominator = max(processed_examples, 1)
        metrics: dict[str, Any] = {
            name: value / denominator for name, value in metric_sums.items()
        }
        metrics.update(
            {
                "processed_examples": processed_examples,
                "processed_batches": processed_batches,
            }
        )
        return self.previous_state, metrics

    @staticmethod
    def _build_optimizer(
        torch,
        model,
        name: str,
        lr: float,
        momentum: float,
        weight_decay: float,
        nesterov: bool,
    ):
        optimizer_name = str(name).lower()
        if optimizer_name == "sgd":
            if nesterov and momentum <= 0:
                raise ValueError("Nesterov SGD requires momentum > 0")
            return torch.optim.SGD(
                model.parameters(),
                lr=float(lr),
                momentum=float(momentum),
                weight_decay=float(weight_decay),
                nesterov=bool(nesterov),
            )
        if optimizer_name == "adam":
            return torch.optim.Adam(
                model.parameters(),
                lr=float(lr),
                weight_decay=float(weight_decay),
            )
        if optimizer_name == "adamw":
            return torch.optim.AdamW(
                model.parameters(),
                lr=float(lr),
                weight_decay=float(weight_decay),
            )
        raise ValueError(f"unsupported optimizer: {name}")
