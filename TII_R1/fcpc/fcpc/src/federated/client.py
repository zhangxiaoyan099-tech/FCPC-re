from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

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
        local_epochs: int = 1,
        device: str = "cpu",
        max_batches: int | None = None,
    ):
        """Run local training. Requires PyTorch and a DataLoader."""
        if self.train_loader is None:
            raise ValueError("train_loader is required for local training")
        import torch
        from torch import nn

        model.load_state_dict(global_state)
        model.to(device)
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        for _ in range(local_epochs):
            for batch_idx, (x, y) in enumerate(self.train_loader):
                if max_batches is not None and batch_idx >= max_batches:
                    break
                x = x.to(device)
                y = y.to(device)
                optimizer.zero_grad()
                logits = model(x)
                task_loss = criterion(logits, y)
                context = {
                    "global_state": global_state,
                    "current_logits": logits,
                }
                loss = task_loss + algorithm.extra_loss(model, (x, y), task_loss, context)
                if use_fcpc and paired_previous_state is not None:
                    loss = loss + fcpc_regularization(
                        dict(model.named_parameters()),
                        paired_previous_state,
                        beta=beta,
                    )
                loss.backward()
                optimizer.step()

        self.previous_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        return self.previous_state
