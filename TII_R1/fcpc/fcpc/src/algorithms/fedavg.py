from __future__ import annotations

from dataclasses import dataclass

from .base import AlgorithmAdapter


@dataclass
class FedAvgAdapter(AlgorithmAdapter):
    """Classic FedAvg: no additional local loss beyond task loss."""

    name: str = "fedavg"

