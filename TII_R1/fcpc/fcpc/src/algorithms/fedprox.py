from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .base import AlgorithmAdapter, _zero_like


@dataclass
class FedProxAdapter(AlgorithmAdapter):
    """FedProx proximal local objective.

    Adds mu/2 * ||w_local - w_global||^2. This readable implementation is
    enough for reconstruction and can be combined with FCPC.
    """

    name: str = "fedprox"
    mu: float = 0.01

    def extra_loss(self, model, batch, task_loss, context: Mapping[str, object]):
        global_state = context.get("global_state")
        if global_state is None:
            return _zero_like(task_loss)
        total = None
        # Use live Parameters here. ``model.state_dict()`` returns detached
        # tensors, which would make the proximal term produce no gradient.
        for key, value in model.named_parameters():
            base = global_state.get(key)
            if base is None or tuple(value.shape) != tuple(base.shape):
                continue
            if hasattr(value, "is_floating_point") and not value.is_floating_point():
                continue
            diff = value - base.to(value.device)
            term = (diff * diff).sum()
            total = term if total is None else total + term
        if total is None:
            return _zero_like(task_loss)
        return 0.5 * float(self.mu) * total
