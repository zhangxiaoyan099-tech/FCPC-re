from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass
class AlgorithmAdapter:
    """Base local-objective adapter.

    FCPC is attached outside the adapter so every algorithm can be combined with
    the same paired regularizer.
    """

    name: str = "base"

    def begin_round(self, **_context):
        """Receive server-side context before client selection/training."""

    def begin_local_train(self, **_context):
        """Prepare client-specific references before local mini-batches."""

    def forward(self, model, inputs):
        """Return logits plus algorithm-specific forward context."""
        return model(inputs), {}

    def extra_loss(self, model, batch, task_loss, context: Mapping[str, object]):
        return _zero_like(task_loss)

    def after_local_train(self, **_context):
        """Update per-client algorithm state after local optimization."""

    def end_local_train(self):
        """Release temporary client-specific models or tensors."""

    def select_clients(self, server, clients_per_round: int, seed: int, **_context):
        return server.sample_clients(clients_per_round, seed)

    def aggregate(self, **_context):
        """Return a custom global state, or ``None`` for FedAvg aggregation."""
        return None

    def after_round(self, **_context):
        """Observe completed client updates and update server-side state."""


def _zero_like(value):
    if hasattr(value, "new_tensor"):
        return value.new_tensor(0.0)
    return 0.0

