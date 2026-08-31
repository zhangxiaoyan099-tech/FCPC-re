from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from src.fcpc.jsdn import metric_matrix
from src.fcpc.ldp import PrivacyBudget, perturb_client_metadata
from src.fcpc.pairing import PairingResult, pair_clients
from src.federated.aggregation import fedavg_aggregate


@dataclass
class Server:
    clients: list
    lambda_jsdn: float = 0.3
    pairing_metric: str = "jsdn"
    aggregation_weighted: bool = True
    pairing_result: PairingResult | None = field(default=None, init=False)
    pairing_matrix: np.ndarray | None = field(default=None, init=False)
    unpaired_counts: Dict[int, int] = field(default_factory=dict, init=False)

    def build_pairing(
        self,
        epsilon: float = 1.0,
        seed: int = 42,
        strategy: str = "greedy_dissimilar",
    ) -> PairingResult:
        rng = np.random.default_rng(seed)
        label_distributions = []
        sample_counts = []
        for client in self.clients:
            if client.label_histogram is None:
                raise ValueError(f"client {client.client_id} has no label_histogram")
            perturbed_hist, perturbed_count = perturb_client_metadata(
                client.label_histogram,
                client.sample_count,
                budget=PrivacyBudget(epsilon=epsilon),
                rng=rng,
            )
            label_distributions.append(perturbed_hist)
            sample_counts.append(perturbed_count)
        self.pairing_matrix = metric_matrix(
            self.pairing_metric,
            label_distributions,
            sample_counts,
            lambda_jsdn=self.lambda_jsdn,
        )
        base_strategy = (
            "greedy_dissimilar"
            if strategy.lower() == "fair_greedy_dissimilar"
            else strategy
        )
        self.pairing_result = pair_clients(
            self.pairing_matrix,
            strategy=base_strategy,
            seed=seed,
        )
        return self.pairing_result

    def pair_selected(
        self,
        selected_client_ids: List[int],
        strategy: str = "greedy_dissimilar",
        seed: int = 42,
    ) -> PairingResult:
        """Pair only clients participating in the current round."""
        if self.pairing_matrix is None:
            raise RuntimeError("build_pairing must be called before pair_selected")
        selected = [int(client_id) for client_id in selected_client_ids]
        pairing_pool = selected
        forced_unpaired: list[int] = []
        base_strategy = strategy
        if strategy.lower() == "fair_greedy_dissimilar":
            base_strategy = "greedy_dissimilar"
            if len(selected) % 2:
                # Lexicographically minimize prior bye count and client id.
                # Thus every always-active client receives a bye at most once
                # more than any other client.
                bye = min(
                    selected,
                    key=lambda client_id: (
                        self.unpaired_counts.get(client_id, 0),
                        client_id,
                    ),
                )
                self.unpaired_counts[bye] = self.unpaired_counts.get(bye, 0) + 1
                forced_unpaired = [bye]
                pairing_pool = [client_id for client_id in selected if client_id != bye]

        submatrix = self.pairing_matrix[np.ix_(pairing_pool, pairing_pool)]
        local = pair_clients(submatrix, strategy=base_strategy, seed=seed)
        pairs = [(pairing_pool[i], pairing_pool[j]) for i, j in local.pairs]
        pair_map = {}
        for i, j in pairs:
            pair_map[i] = j
            pair_map[j] = i
        result = PairingResult(
            pairs=pairs,
            pair_map=pair_map,
            unpaired=forced_unpaired + [pairing_pool[i] for i in local.unpaired],
        )
        self.pairing_result = result
        return result

    def sample_clients(self, clients_per_round: int, seed: int) -> List[int]:
        rng = np.random.default_rng(seed)
        count = min(clients_per_round, len(self.clients))
        return rng.choice(np.arange(len(self.clients)), size=count, replace=False).astype(int).tolist()

    def aggregate(self, selected_client_ids: List[int], client_states: list):
        sample_counts = [self.clients[i].sample_count for i in selected_client_ids]
        return fedavg_aggregate(client_states, sample_counts, weighted=self.aggregation_weighted)
