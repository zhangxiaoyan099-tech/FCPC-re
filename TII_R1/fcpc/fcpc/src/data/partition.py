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
    """Redistribute all supplied indices with unequal client quantities.

    The previous implementation trimmed every client's existing subset and
    silently discarded the remainder.  That made method comparisons use only
    part of the CIFAR training split.  This implementation pools the indices
    and repartitions them without replacement, so every input example appears
    in exactly one output client.
    """
    rng = np.random.default_rng(seed)
    client_ids = sorted(client_indices)
    pooled = np.asarray(
        [index for client_id in client_ids for index in client_indices[client_id]],
        dtype=int,
    )
    if len(np.unique(pooled)) != len(pooled):
        raise ValueError("client_indices must not contain duplicate sample indices")
    rng.shuffle(pooled)
    profile = _quantity_profile(len(client_ids), min_fraction, rng)
    counts = _integer_counts(len(pooled), profile)
    split_points = np.cumsum(counts)[:-1]
    splits = np.split(pooled, split_points)
    result = {
        client_id: split.astype(int).tolist()
        for client_id, split in zip(client_ids, splits)
    }
    _ensure_nonempty_clients(result, rng)
    return result


def dirichlet_dual_skew_partition(
    labels: Sequence[int],
    num_clients: int,
    alpha: float,
    min_fraction: float = 0.25,
    seed: int = 42,
) -> Dict[int, List[int]]:
    """Create label and quantity skew while assigning every sample once.

    A fixed exponential client-size profile is multiplied into an independent
    Dirichlet label allocation for every class.  Per-class integer allocation
    uses largest remainders, so no example is duplicated or discarded.
    """
    if num_clients <= 0:
        raise ValueError("num_clients must be positive")
    if alpha <= 0.0:
        raise ValueError("alpha must be positive")
    rng = np.random.default_rng(seed)
    labels = _as_numpy_labels(labels)
    client_indices = {i: [] for i in range(num_clients)}
    quantity_profile = _quantity_profile(num_clients, min_fraction, rng)

    for cls in np.unique(labels):
        cls_indices = np.where(labels == cls)[0]
        rng.shuffle(cls_indices)
        label_profile = rng.dirichlet(np.full(num_clients, alpha))
        joint_profile = label_profile * quantity_profile
        counts = _integer_counts(len(cls_indices), joint_profile)
        split_points = np.cumsum(counts)[:-1]
        for client_id, split in enumerate(np.split(cls_indices, split_points)):
            client_indices[client_id].extend(split.astype(int).tolist())

    _ensure_nonempty_clients(client_indices, rng)
    for indices in client_indices.values():
        rng.shuffle(indices)
    return client_indices


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
    quantity_min_fraction: float = 0.25,
) -> Dict[int, List[int]]:
    """Build label-skew, quantity-skew, dual-skew, or iid client partitions."""
    partition = partition.lower()
    if partition == "iid":
        return iid_partition(labels, num_clients, seed)
    if partition == "label_skew":
        return dirichlet_label_skew_partition(labels, num_clients, alpha, seed)
    if partition == "quantity_skew":
        base = iid_partition(labels, num_clients, seed)
        return quantity_skew_resample(
            base,
            min_fraction=quantity_min_fraction,
            seed=seed + 1,
        )
    if partition == "dual_skew":
        return dirichlet_dual_skew_partition(
            labels,
            num_clients,
            alpha,
            min_fraction=quantity_min_fraction,
            seed=seed,
        )
    raise ValueError(f"unsupported partition mode: {partition}")


def validate_complete_partition(
    client_indices: Dict[int, List[int]],
    num_samples: int,
) -> Dict[str, int]:
    """Validate exact, non-overlapping coverage and return size statistics."""
    flattened = [index for indices in client_indices.values() for index in indices]
    if any(index < 0 or index >= num_samples for index in flattened):
        raise ValueError("partition contains an out-of-range sample index")
    unique = set(flattened)
    if len(unique) != len(flattened):
        raise ValueError("partition contains duplicate sample assignments")
    if unique != set(range(num_samples)):
        missing = num_samples - len(unique)
        raise ValueError(f"partition does not cover the training split: missing={missing}")
    sizes = [len(indices) for indices in client_indices.values()]
    return {
        "assigned_examples": len(flattened),
        "unique_examples": len(unique),
        "client_examples_min": min(sizes, default=0),
        "client_examples_max": max(sizes, default=0),
    }


def _quantity_profile(
    num_clients: int,
    min_fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if num_clients <= 0:
        raise ValueError("num_clients must be positive")
    if not 0.0 <= float(min_fraction) <= 1.0:
        raise ValueError("min_fraction must be in [0, 1]")
    raw = rng.exponential(scale=1.0, size=num_clients)
    scaled = float(min_fraction) + (1.0 - float(min_fraction)) * raw / max(
        float(raw.max()),
        1e-12,
    )
    return scaled / scaled.sum()


def _integer_counts(total: int, proportions: np.ndarray) -> np.ndarray:
    proportions = np.maximum(np.asarray(proportions, dtype=float), 0.0)
    if not np.any(proportions):
        proportions = np.ones_like(proportions)
    proportions /= proportions.sum()
    exact = int(total) * proportions
    counts = np.floor(exact).astype(int)
    remainder = int(total) - int(counts.sum())
    if remainder:
        order = np.argsort(-(exact - counts), kind="stable")
        counts[order[:remainder]] += 1
    return counts


def _ensure_nonempty_clients(
    client_indices: Dict[int, List[int]],
    rng: np.random.Generator,
) -> None:
    """Move samples from the largest clients if rounding creates an empty one."""
    for empty_id in [client_id for client_id, values in client_indices.items() if not values]:
        donor_id = max(client_indices, key=lambda client_id: len(client_indices[client_id]))
        if len(client_indices[donor_id]) <= 1:
            raise ValueError("not enough samples to give every client one example")
        position = int(rng.integers(0, len(client_indices[donor_id])))
        client_indices[empty_id].append(client_indices[donor_id].pop(position))


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
