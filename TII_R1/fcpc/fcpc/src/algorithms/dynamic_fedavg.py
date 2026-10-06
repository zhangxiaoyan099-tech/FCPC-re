from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .fedavg import FedAvgAdapter


@dataclass
class DynamicFedAvgAdapter(FedAvgAdapter):
    """FedAvg with a deterministic staged client-availability schedule.

    Client IDs ``[0, initial_clients)`` are active before ``join_round``
    (zero-based). IDs up to ``final_clients`` become active from that round
    onward. Only client availability changes; local objectives and aggregation
    remain the ordinary FedAvg/optional-FCPC paths.
    """

    initial_clients: int = 2
    join_round: int = 10
    final_clients: int | None = None
    name: str = "dynamic_fedavg"
    _round_idx: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if int(self.initial_clients) < 1:
            raise ValueError("initial_clients must be positive")
        if int(self.join_round) < 0:
            raise ValueError("join_round must be non-negative")
        if self.final_clients is not None and int(self.final_clients) < int(
            self.initial_clients
        ):
            raise ValueError("final_clients cannot be smaller than initial_clients")

    def begin_round(self, **context):
        self._round_idx = int(context.get("round_idx", 0))

    def select_clients(self, server, clients_per_round: int, seed: int, **_context):
        total_clients = len(server.clients)
        final_clients = (
            total_clients if self.final_clients is None else int(self.final_clients)
        )
        if final_clients > total_clients:
            raise ValueError(
                f"final_clients={final_clients} exceeds available clients={total_clients}"
            )
        active_count = (
            int(self.initial_clients)
            if self._round_idx < int(self.join_round)
            else final_clients
        )
        active_count = min(active_count, total_clients)
        requested = min(int(clients_per_round), active_count)
        if requested >= active_count:
            return list(range(active_count))
        rng = np.random.default_rng(int(seed))
        return rng.choice(
            np.arange(active_count),
            size=requested,
            replace=False,
        ).astype(int).tolist()
