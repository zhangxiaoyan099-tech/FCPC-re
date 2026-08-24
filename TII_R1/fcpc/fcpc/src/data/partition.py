"""Non-IID partition utilities for thesis-style experiments."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Sequence

import numpy as np


def _as_numpy_labels(labels: Iterable[int]) -> np.ndarray:
    return np.asarray(list(labels), dtype=int)


def dirichlet_label_skew_partition(
    labels: Sequence[int],
    num_clients: int,
    alpha: float,
    seed: int = 42,
) -> Dict[int, List[int]]:
    """Partition indices by class using Dirichlet proportions."""
    rng = np.random.default_rng(seed)
    labels = _as_numpy_labels(labels)
    classes = np.unique(labels)
    client_indices = {i: [] for i in range(num_clients)}

    for cls in classes:
        cls_indices = np.where(labels == cls)[0]
        rng.shuffle(cls_indices)
        proportions = rng.dirichlet(np.full(num_clients, alpha))
        split_points = (np.cumsum(proportions)[:-1] * len(cls_indices)).astype(int)
        splits = np.split(cls_indices, split_points)
        for client_id, split in enumerate(splits):
            client_indices[client_id].extend(split.tolist())

    for indices in client_indices.values():
        rng.shuffle(indices)
    return client_indices


def quantity_skew_resample(
    client_indices: Dict[int, List[int]],
    min_fraction: float = 0.25,
    seed: int = 42,
) -> Dict[int, List[int]]:
    """Apply sample quantity skew by trimming each client with exponential weights."""
    rng = np.random.default_rng(seed)
    raw = rng.exponential(scale=1.0, size=len(client_indices))
    raw = min_fraction + (1.0 - min_fraction) * raw / max(raw.max(), 1e-12)
    skewed: Dict[int, List[int]] = {}
    for client_id, indices in client_indices.items():
        if not indices:
            skewed[client_id] = []
            continue
        keep = max(1, int(len(indices) * raw[client_id]))
        chosen = rng.choice(np.asarray(indices), size=min(keep, len(indices)), replace=False)
        skewed[client_id] = chosen.tolist()
    return skewed


def iid_partition(labels: Sequence[int], num_clients: int, seed: int = 42) -> Dict[int, List[int]]:
    rng = np.random.default_rng(seed)
    indices = np.arange(len(labels))
    rng.shuffle(indices)
    splits = np.array_split(indices, num_clients)
    return {i: split.tolist() for i, split in enumerate(splits)}


def natural_client_partition(client_ids: Sequence[int]) -> Dict[int, List[int]]:
    """Use dataset-provided device/subject identifiers as FL clients."""
    ids = np.asarray(list(client_ids), dtype=int)
    unique_ids = sorted(np.unique(ids).astype(int).tolist())
    return {
        client_index: np.where(ids == source_id)[0].astype(int).tolist()
        for client_index, source_id in enumerate(unique_ids)
    }


def build_client_indices(
    labels: Sequence[int],
    num_clients: int,
    partition: str = "dual_skew",
    alpha: float = 0.1,
    seed: int = 42,
) -> Dict[int, List[int]]:
    """Build label-skew, quantity-skew, dual-skew, or iid client partitions."""
    partition = partition.lower()
    if partition == "iid":
        return iid_partition(labels, num_clients, seed)
    if partition == "label_skew":
        return dirichlet_label_skew_partition(labels, num_clients, alpha, seed)
    if partition == "quantity_skew":
        base = iid_partition(labels, num_clients, seed)
        return quantity_skew_resample(base, seed=seed + 1)
    if partition == "dual_skew":
        base = dirichlet_label_skew_partition(labels, num_clients, alpha, seed)
        return quantity_skew_resample(base, seed=seed + 1)
    raise ValueError(f"unsupported partition mode: {partition}")


def client_label_histograms(
    labels: Sequence[int],
    client_indices: Dict[int, List[int]],
    num_classes: int,
) -> Dict[int, np.ndarray]:
    labels = _as_numpy_labels(labels)
    histograms: Dict[int, np.ndarray] = {}
    for client_id, indices in client_indices.items():
        hist = np.zeros(num_classes, dtype=float)
        for label in labels[np.asarray(indices, dtype=int)]:
            hist[int(label)] += 1.0
        histograms[client_id] = hist
    return histograms
