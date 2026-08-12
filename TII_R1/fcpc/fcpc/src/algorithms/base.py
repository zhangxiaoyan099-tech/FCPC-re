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

    def extra_loss(self, model, batch, task_loss, context: Mapping[str, object]):
        return _zero_like(task_loss)


def _zero_like(value):
    if hasattr(value, "new_tensor"):
        return value.new_tensor(0.0)
    return 0.0

