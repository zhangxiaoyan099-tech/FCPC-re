"""Client-pairing strategies used by FCPC and its ablations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass(frozen=True)
class PairingResult:
    pairs: List[Tuple[int, int]]
    pair_map: Dict[int, int]
    unpaired: List[int]


def greedy_high_dissimilarity_pairing(jsdn_matrix: np.ndarray) -> PairingResult:
    """Greedily pair clients by repeatedly selecting the largest matrix entry.

    If the number of clients is odd, the final remaining client is returned in
    `unpaired` and receives no FCPC regularization in the current round.
    """
    matrix = _validated_matrix(jsdn_matrix)
    n_clients = matrix.shape[0]
    available = set(range(n_clients))
    pairs: List[Tuple[int, int]] = []
    edges = [
        (float(matrix[i, j]), i, j)
        for i in range(n_clients)
        for j in range(i + 1, n_clients)
    ]
    # Deterministic descending weight order with client-id tie breaking.
    edges.sort(key=lambda edge: (-edge[0], edge[1], edge[2]))
    for _, i, j in edges:
        if i in available and j in available:
            pairs.append((i, j))
            available.remove(i)
            available.remove(j)
            if len(available) < 2:
                break

    return _result(pairs, sorted(available))


def greedy_similarity_pairing(jsdn_matrix: np.ndarray) -> PairingResult:
    """Greedily pair the most similar remaining clients.

    This is an ablation baseline, not the proposed FCPC strategy.
    """
    matrix = _validated_matrix(jsdn_matrix)
    n_clients = matrix.shape[0]
    available = set(range(n_clients))
    pairs: List[Tuple[int, int]] = []
    edges = [
        (float(matrix[i, j]), i, j)
        for i in range(n_clients)
        for j in range(i + 1, n_clients)
    ]
    edges.sort(key=lambda edge: (edge[0], edge[1], edge[2]))
    for _, i, j in edges:
        if i in available and j in available:
            pairs.append((i, j))
            available.remove(i)
            available.remove(j)
            if len(available) < 2:
                break
    return _result(pairs, sorted(available))


def random_pairing(n_clients: int, seed: int = 42) -> PairingResult:
    """Random-pairing ablation with at most one unpaired client."""
    if n_clients < 0:
        raise ValueError("n_clients must be non-negative")
    order = np.random.default_rng(seed).permutation(n_clients).astype(int).tolist()
    pairs = [(order[i], order[i + 1]) for i in range(0, n_clients - 1, 2)]
    unpaired = order[-1:] if n_clients % 2 else []
    return _result(pairs, unpaired)


def optimal_high_dissimilarity_pairing(jsdn_matrix: np.ndarray) -> PairingResult:
    """Compute an exact maximum-weight, maximum-cardinality matching.

    This optimization-based strategy measures the quality/speed trade-off of
    the proposed greedy matcher. It requires NetworkX.
    """
    matrix = _validated_matrix(jsdn_matrix)
    try:
        import networkx as nx
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError("NetworkX is required for optimal pairing") from exc

    n_clients = matrix.shape[0]
    graph = nx.Graph()
    graph.add_nodes_from(range(n_clients))
    for i in range(n_clients):
        for j in range(i + 1, n_clients):
            graph.add_edge(i, j, weight=float(matrix[i, j]))
    matching = nx.algorithms.matching.max_weight_matching(
        graph,
        maxcardinality=True,
        weight="weight",
    )
    pairs = sorted((min(int(i), int(j)), max(int(i), int(j))) for i, j in matching)
    paired_clients = {client_id for pair in pairs for client_id in pair}
    unpaired = sorted(set(range(n_clients)) - paired_clients)
    return _result(pairs, unpaired)


def pair_clients(
    jsdn_matrix: np.ndarray,
    strategy: str = "greedy_dissimilar",
    seed: int = 42,
) -> PairingResult:
    """Dispatch a proposed or ablation pairing strategy."""
    strategy = strategy.lower()
    if strategy in {"greedy_dissimilar", "greedy", "fcpc"}:
        return greedy_high_dissimilarity_pairing(jsdn_matrix)
    if strategy in {"similar", "greedy_similar"}:
        return greedy_similarity_pairing(jsdn_matrix)
    if strategy == "random":
        matrix = _validated_matrix(jsdn_matrix)
        return random_pairing(matrix.shape[0], seed=seed)
    if strategy in {"optimal", "max_weight"}:
        return optimal_high_dissimilarity_pairing(jsdn_matrix)
    raise ValueError(f"unsupported pairing strategy: {strategy}")


def pairing_weight(result: PairingResult, matrix: np.ndarray) -> float:
    """Return the total JSDN edge weight of a pairing."""
    matrix = np.asarray(matrix, dtype=float)
    return float(sum(matrix[i, j] for i, j in result.pairs))


def _validated_matrix(jsdn_matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(jsdn_matrix, dtype=float).copy()
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("jsdn_matrix must be square")
    if np.any(~np.isfinite(matrix)):
        raise ValueError("jsdn_matrix must contain finite values")
    if np.any(matrix < 0):
        raise ValueError("jsdn_matrix must contain non-negative weights")
    np.fill_diagonal(matrix, 0.0)
    return matrix


def _result(pairs: List[Tuple[int, int]], unpaired: List[int]) -> PairingResult:
    pair_map: Dict[int, int] = {}
    for i, j in pairs:
        pair_map[i] = j
        pair_map[j] = i
    return PairingResult(pairs=pairs, pair_map=pair_map, unpaired=sorted(unpaired))
