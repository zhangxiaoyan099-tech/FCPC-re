from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Metrics:
    loss: float = 0.0
    accuracy: float = 0.0
    samples: int = 0


def accuracy_from_logits(logits, labels) -> float:
    preds = logits.argmax(dim=1)
    return float((preds == labels).float().mean().item())

