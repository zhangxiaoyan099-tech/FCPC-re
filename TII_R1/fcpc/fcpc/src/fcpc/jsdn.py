"""JSDN metric from the thesis.

JSDN combines label-distribution skew and quantity skew:

    JSDN = (1 - lambda) * JSD + lambda * N

where N = |Na - Nb| / (Na + Nb).
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np


def _normalize_distribution(values: Iterable[float], eps: float = 1e-12) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    arr = np.maximum(arr, 0.0)
    total = float(arr.sum())
    if total <= eps:
        return np.full_like(arr, 1.0 / max(len(arr), 1), dtype=float)
    return arr / total


def _kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return float(np.sum(p * np.log2(p / q)))


def js_divergence(label_dist_a: Iterable[float], label_dist_b: Iterable[float]) -> float:
    """Compute Jensen-Shannon divergence in [0, 1] for base-2 logs."""
    p = _normalize_distribution(label_dist_a)
    q = _normalize_distribution(label_dist_b)
    if p.shape != q.shape:
        raise ValueError(f"label distribution shape mismatch: {p.shape} vs {q.shape}")
    m = 0.5 * (p + q)
    value = 0.5 * _kl_divergence(p, m) + 0.5 * _kl_divergence(q, m)
    return float(min(max(value, 0.0), 1.0))


def weighted_js_divergence(
    label_dist_a: Iterable[float],
    label_dist_b: Iterable[float],
    count_a: float,
    count_b: float,
    eps: float = 1e-12,
) -> float:
    """Generalized JS divergence with sample-count mixture weights.

    For ``theta = Na / (Na + Nb)`` and ``q = theta p + (1-theta) r``:

        JS_theta(p, r) = theta KL(p || q) + (1-theta) KL(r || q).

    Unlike the original JSDN interpolation, this quantity is exactly the
    reduction in label-distribution KL obtained by replacing two clients with
    their sample-weighted pair mixture.
    """
    p = _normalize_distribution(label_dist_a)
    r = _normalize_distribution(label_dist_b)
    if p.shape != r.shape:
        raise ValueError(f"label distribution shape mismatch: {p.shape} vs {r.shape}")
    count_a = max(float(count_a), 0.0)
    count_b = max(float(count_b), 0.0)
    denominator = count_a + count_b
    theta = 0.5 if denominator <= eps else count_a / denominator
    mixture = theta * p + (1.0 - theta) * r
    value = theta * _kl_divergence(p, mixture)
    value += (1.0 - theta) * _kl_divergence(r, mixture)
    return float(min(max(value, 0.0), 1.0))


def pair_complementarity_score(
    label_dist_a: Iterable[float],
    label_dist_b: Iterable[float],
    count_a: float,
    count_b: float,
    total_count: float,
    eps: float = 1e-12,
) -> float:
    """Pair benefit used by the theory-aligned FCPC variant.

    The outer mass ``(Na + Nb) / N`` makes edge scores comparable under
    quantity skew.  Across a complete matching, maximizing the sum of these
    scores is equivalent to minimizing the weighted KL distance between pair
    mixtures and the global label distribution.
    """
    pair_mass = max(float(count_a), 0.0) + max(float(count_b), 0.0)
    mass_weight = pair_mass / max(float(total_count), eps)
    return float(
        mass_weight
        * weighted_js_divergence(
            label_dist_a,
            label_dist_b,
            count_a,
            count_b,
            eps=eps,
        )
    )


def quantity_skew_indicator(count_a: float, count_b: float, eps: float = 1e-12) -> float:
    """Compute normalized sample-count skew N = |Na - Nb| / (Na + Nb)."""
    denom = max(float(count_a) + float(count_b), eps)
    value = abs(float(count_a) - float(count_b)) / denom
    return float(min(max(value, 0.0), 1.0))


def jsdn_score(
    label_dist_a: Iterable[float],
    label_dist_b: Iterable[float],
    count_a: float,
    count_b: float,
    lambda_jsdn: float = 0.3,
) -> float:
    """Compute the thesis JSDN score for a client pair."""
    if not 0.0 <= lambda_jsdn <= 1.0:
        raise ValueError("lambda_jsdn must be in [0, 1]")
    jsd = js_divergence(label_dist_a, label_dist_b)
    n_skew = quantity_skew_indicator(count_a, count_b)
    return float((1.0 - lambda_jsdn) * jsd + lambda_jsdn * n_skew)


def build_jsdn_matrix(
    label_distributions: Sequence[Iterable[float]],
    sample_counts: Sequence[float],
    lambda_jsdn: float = 0.3,
    block_size: int = 128,
) -> np.ndarray:
    """Build the symmetric n x n JSDN matrix used by FCPC pairing.

    Label distributions are normalized once, and pairwise values are computed
    in blocks. This preserves O(n^2 C) arithmetic while avoiding the repeated
    Python-level normalization in a nested client-pair loop.
    """
    n_clients = len(label_distributions)
    if len(sample_counts) != n_clients:
        raise ValueError("sample_counts length must match label_distributions length")
    if not 0.0 <= lambda_jsdn <= 1.0:
        raise ValueError("lambda_jsdn must be in [0, 1]")
    if n_clients == 0:
        return np.zeros((0, 0), dtype=float)

    distributions = np.stack(
        [_normalize_distribution(values) for values in label_distributions],
        axis=0,
    )
    counts = np.asarray(sample_counts, dtype=float)
    block_size = max(int(block_size), 1)
    matrix = np.zeros((n_clients, n_clients), dtype=float)
    eps = 1e-12

    for start in range(0, n_clients, block_size):
        stop = min(start + block_size, n_clients)
        p = distributions[start:stop, None, :]
        q = distributions[None, :, :]
        midpoint = np.clip(0.5 * (p + q), eps, 1.0)
        p_safe = np.clip(p, eps, 1.0)
        q_safe = np.clip(q, eps, 1.0)
        jsd = 0.5 * np.sum(p_safe * np.log2(p_safe / midpoint), axis=2)
        jsd += 0.5 * np.sum(q_safe * np.log2(q_safe / midpoint), axis=2)

        count_a = counts[start:stop, None]
        count_b = counts[None, :]
        quantity = np.abs(count_a - count_b) / np.maximum(count_a + count_b, eps)
        matrix[start:stop] = (1.0 - lambda_jsdn) * jsd + lambda_jsdn * quantity

    matrix = np.clip(0.5 * (matrix + matrix.T), 0.0, 1.0)
    np.fill_diagonal(matrix, 0.0)
    if np.any(~np.isfinite(matrix)):
        raise ValueError("JSDN matrix contains non-finite values")
    return matrix


def build_pair_complementarity_matrix(
    label_distributions: Sequence[Iterable[float]],
    sample_counts: Sequence[float],
    block_size: int = 128,
) -> np.ndarray:
    """Build the weighted-JS pair-complementarity edge matrix.

    Entry ``(i, j)`` is

        ((Ni + Nj) / N) * JS_{Ni/(Ni+Nj)}(p_i, p_j).

    The block implementation keeps the same O(n^2 C) complexity as JSDN and
    avoids a Python loop over every client pair.
    """
    n_clients = len(label_distributions)
    if len(sample_counts) != n_clients:
        raise ValueError("sample_counts length must match label_distributions length")
    if n_clients == 0:
        return np.zeros((0, 0), dtype=float)

    distributions = np.stack(
        [_normalize_distribution(values) for values in label_distributions],
        axis=0,
    )
    counts = np.maximum(np.asarray(sample_counts, dtype=float), 0.0)
    total_count = max(float(counts.sum()), 1e-12)
    block_size = max(int(block_size), 1)
    matrix = np.zeros((n_clients, n_clients), dtype=float)
    eps = 1e-12

    for start in range(0, n_clients, block_size):
        stop = min(start + block_size, n_clients)
        p = distributions[start:stop, None, :]
        r = distributions[None, :, :]
        count_a = counts[start:stop, None]
        count_b = counts[None, :]
        pair_count = count_a + count_b
        theta = np.divide(
            count_a,
            pair_count,
            out=np.full_like(pair_count, 0.5, dtype=float),
            where=pair_count > eps,
        )
        mixture = np.clip(
            theta[:, :, None] * p + (1.0 - theta[:, :, None]) * r,
            eps,
            1.0,
        )
        p_safe = np.clip(p, eps, 1.0)
        r_safe = np.clip(r, eps, 1.0)
        kl_p = np.sum(p_safe * np.log2(p_safe / mixture), axis=2)
        kl_r = np.sum(r_safe * np.log2(r_safe / mixture), axis=2)
        weighted_js = theta * kl_p + (1.0 - theta) * kl_r
        matrix[start:stop] = (pair_count / total_count) * weighted_js

    matrix = np.maximum(0.5 * (matrix + matrix.T), 0.0)
    np.fill_diagonal(matrix, 0.0)
    if np.any(~np.isfinite(matrix)):
        raise ValueError("pair-complementarity matrix contains non-finite values")
    return matrix


def metric_matrix(
    metric: str,
    label_distributions: Sequence[Iterable[float]],
    sample_counts: Sequence[float],
    lambda_jsdn: float = 0.3,
) -> np.ndarray:
    """Build a matrix for pairing metric ablations."""
    metric = metric.lower()
    if metric == "jsdn":
        return build_jsdn_matrix(label_distributions, sample_counts, lambda_jsdn)
    if metric in {
        "pair_complementarity",
        "weighted_js",
        "weighted_js_complementarity",
    }:
        return build_pair_complementarity_matrix(label_distributions, sample_counts)
    n_clients = len(label_distributions)
    dists = [_normalize_distribution(d) for d in label_distributions]
    matrix = np.zeros((n_clients, n_clients), dtype=float)
    for i in range(n_clients):
        for j in range(i + 1, n_clients):
            if metric == "jsd":
                score = js_divergence(dists[i], dists[j])
            elif metric == "euclidean":
                score = float(np.linalg.norm(dists[i] - dists[j]))
            elif metric == "cosine":
                denom = max(float(np.linalg.norm(dists[i]) * np.linalg.norm(dists[j])), 1e-12)
                score = 1.0 - float(np.dot(dists[i], dists[j]) / denom)
            else:
                raise ValueError(f"unsupported metric: {metric}")
            matrix[i, j] = score
            matrix[j, i] = score
    return matrix
