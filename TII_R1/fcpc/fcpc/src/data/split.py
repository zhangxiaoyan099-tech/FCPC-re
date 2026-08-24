"""Deterministic train/validation splitting utilities."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def stratified_holdout_indices(
    labels: Sequence[int],
    validation_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Return disjoint train/validation indices while preserving all samples.

    Classes with at least two examples contribute at least one validation
    example when ``validation_fraction`` is positive. Singleton classes remain
    in training so a holdout cannot remove a class entirely.
    """
    fraction = float(validation_fraction)
    if not 0.0 <= fraction < 1.0:
        raise ValueError("validation_fraction must be in [0, 1)")

    labels_array = np.asarray(list(labels), dtype=int)
    if fraction == 0.0 or labels_array.size == 0:
        return list(range(int(labels_array.size))), []

    rng = np.random.default_rng(int(seed))
    train_indices: list[int] = []
    validation_indices: list[int] = []
    for label in np.unique(labels_array):
        class_indices = np.flatnonzero(labels_array == label)
        rng.shuffle(class_indices)
        if len(class_indices) < 2:
            train_indices.extend(class_indices.tolist())
            continue
        validation_count = int(round(len(class_indices) * fraction))
        validation_count = min(max(validation_count, 1), len(class_indices) - 1)
        validation_indices.extend(class_indices[:validation_count].tolist())
        train_indices.extend(class_indices[validation_count:].tolist())

    rng.shuffle(train_indices)
    rng.shuffle(validation_indices)
    return train_indices, validation_indices
