from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from src.fcpc.regularizer import fcpc_regularization, proximal_center_step


@dataclass
class Client:
    client_id: int
    train_loader: object | None = None
    sample_count: int = 0
    label_histogram: np.ndarray | None = None
    previous_state: Mapping[str, object] | None = field(default=None, init=False)
    previous_global_state: Mapping[str, object] | None = field(default=None, init=False)

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
        mean_sample_count: float = 1.0,
        fcpc_update_rule: str = "penalty",
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
        fcpc_update_rule = str(fcpc_update_rule).lower()
        if fcpc_update_rule not in {"penalty", "proximal"}:
            raise ValueError("fcpc_update_rule must be 'penalty' or 'proximal'")
        algorithm.begin_local_train(
            model=model,
            client_id=self.client_id,
            global_state=global_state,
            previous_local_state=self.previous_state,
            sample_count=self.sample_count,
            mean_sample_count=mean_sample_count,
            device=device,
        )

        metric_sums = {
            "task_loss": 0.0,
            "algorithm_loss": 0.0,
            "fcpc_raw_loss": 0.0,
            "fcpc_weighted_loss": 0.0,
            "total_loss": 0.0,
        }
        processed_examples = 0
        processed_batches = 0
        non_blocking = str(device).startswith("cuda")

        try:
            for _ in range(local_epochs):
                for batch_idx, (x, y) in enumerate(self.train_loader):
                    if max_batches is not None and batch_idx >= max_batches:
                        break
                    x = x.to(device, non_blocking=non_blocking)
                    y = y.to(device, non_blocking=non_blocking)
                    optimizer.zero_grad(set_to_none=True)
                    logits, forward_context = algorithm.forward(model, x)
                    task_loss = criterion(logits, y)
                    context = {
                        "client_id": self.client_id,
                        "global_state": global_state,
                        "current_logits": logits,
                        **forward_context,
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
                    reported_total_loss = task_loss + algorithm_loss + fcpc_weighted_loss
                    optimization_loss = task_loss + algorithm_loss
                    if fcpc_update_rule == "penalty":
                        optimization_loss = optimization_loss + fcpc_weighted_loss
                    optimization_loss.backward()
                    algorithm.after_backward(
                        model=model,
                        client_id=self.client_id,
                        batch=(x, y),
                    )
                    optimizer.step()
                    if (
                        fcpc_update_rule == "proximal"
                        and use_fcpc
                        and paired_previous_state is not None
                    ):
                        proximal_center_step(
                            dict(model.named_parameters()),
                            paired_previous_state,
                            beta=float(beta),
                            learning_rate=float(lr),
                        )

                    batch_examples = int(y.numel())
                    processed_examples += batch_examples
                    processed_batches += 1
                    values = {
                        "task_loss": task_loss,
                        "algorithm_loss": algorithm_loss,
                        "fcpc_raw_loss": fcpc_raw_loss,
                        "fcpc_weighted_loss": fcpc_weighted_loss,
                        # In proximal mode this is the pre-step composite
                        # objective used for monitoring; the actual update is
                        # task/algorithm optimizer step followed by prox.
                        "total_loss": reported_total_loss,
                    }
                    for name, value in values.items():
                        metric_sums[name] += float(value.detach().item()) * batch_examples

            self.previous_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            # The server already knows the model broadcast to this client.
            # Retaining it lets FCPC-grad reconstruct the client's previous
            # update without an additional gradient/model upload.
            self.previous_global_state = {
                k: v.detach().cpu().clone()
                for k, v in global_state.items()
                if hasattr(v, "detach")
            }
            algorithm.after_local_train(
                client_id=self.client_id,
                local_state=self.previous_state,
                global_state=global_state,
                sample_count=self.sample_count,
            )
        finally:
            algorithm.end_local_train()
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
