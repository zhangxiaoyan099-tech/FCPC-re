"""Local differential privacy utilities for FCPC client metadata upload."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np


@dataclass(frozen=True)
class PrivacyBudget:
    epsilon: float = 1.0
    epsilon_h: float | None = None
    epsilon_n: float | None = None


def split_privacy_budget(total_epsilon: float, num_classes: int, sample_count: int) -> Tuple[float, float]:
    """Thesis-style split between label histogram and sample-count budgets."""
    sample_count = max(int(sample_count), 1)
    numerator = np.sqrt(float(num_classes) / float(sample_count))
    epsilon_h = float(total_epsilon * numerator / (numerator + 1.0))
    epsilon_n = float(total_epsilon - epsilon_h)
    return max(epsilon_h, 1e-6), max(epsilon_n, 1e-6)


def perturb_label_distribution(
    label_histogram: Iterable[float],
    epsilon_h: float = 1.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Add Laplace noise with scale 2 / epsilon_h to a label histogram."""
    rng = rng or np.random.default_rng()
    hist = np.asarray(list(label_histogram), dtype=float)
    noise = rng.laplace(loc=0.0, scale=2.0 / max(float(epsilon_h), 1e-6), size=hist.shape)
    perturbed = np.maximum(hist + noise, 0.0)
    total = perturbed.sum()
    if total <= 1e-12:
        return np.full_like(perturbed, 1.0 / max(len(perturbed), 1))
    return perturbed / total


def perturb_sample_count(
    sample_count: float,
    epsilon_n: float = 1.0,
    rng: np.random.Generator | None = None,
) -> float:
    """Add Laplace noise with scale 1 / epsilon_n to the sample count."""
    rng = rng or np.random.default_rng()
    noisy = float(sample_count) + float(rng.laplace(loc=0.0, scale=1.0 / max(float(epsilon_n), 1e-6)))
    return max(noisy, 1.0)


def perturb_client_metadata(
    label_histogram: Iterable[float],
    sample_count: int,
    budget: PrivacyBudget | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, float]:
    """Perturb both uploaded client metadata fields."""
    budget = budget or PrivacyBudget()
    hist = np.asarray(list(label_histogram), dtype=float)
    if budget.epsilon_h is None or budget.epsilon_n is None:
        epsilon_h, epsilon_n = split_privacy_budget(budget.epsilon, len(hist), sample_count)
    else:
        epsilon_h, epsilon_n = budget.epsilon_h, budget.epsilon_n
    return (
        perturb_label_distribution(hist, epsilon_h=epsilon_h, rng=rng),
        perturb_sample_count(sample_count, epsilon_n=epsilon_n, rng=rng),
    )

