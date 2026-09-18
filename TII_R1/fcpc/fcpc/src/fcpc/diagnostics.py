from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np

from src.fcpc.pairing import PairingResult


def _distribution(histogram: Sequence[float]) -> np.ndarray:
    values = np.asarray(histogram, dtype=float)
    if values.ndim != 1:
        raise ValueError("label histograms must be one-dimensional")
    if np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("label histograms must be finite and non-negative")
    total = float(values.sum())
    if total <= 0.0:
        raise ValueError("label histograms must contain positive mass")
    return values / total


def _kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    positive = p > 0.0
    if np.any(q[positive] <= 0.0):
        return float("inf")
    return float(np.sum(p[positive] * np.log(p[positive] / q[positive])))


def pairing_distribution_metrics(
    pairing: PairingResult,
    label_histograms: Sequence[Sequence[float]],
    sample_counts: Sequence[float],
    *,
    pairing_matrix: np.ndarray | None = None,
) -> dict[str, float]:
    """Measure what pairing changes in label-distribution space.

    Metrics use the unperturbed client histograms.  The matching itself may
    still be constructed from LDP metadata.  Consequently these values audit
    the downstream quality of the pairing without changing or informing it.
    """

    if len(label_histograms) != len(sample_counts):
        raise ValueError("label_histograms and sample_counts must have equal length")
    if not label_histograms:
        raise ValueError("at least one client is required")
    distributions = [_distribution(histogram) for histogram in label_histograms]
    num_classes = distributions[0].size
    if any(distribution.size != num_classes for distribution in distributions):
        raise ValueError("all label histograms must use the same number of classes")
    counts = np.asarray(sample_counts, dtype=float)
    if np.any(~np.isfinite(counts)) or np.any(counts < 0.0):
        raise ValueError("sample_counts must be finite and non-negative")
    total_count = float(counts.sum())
    if total_count <= 0.0:
        raise ValueError("sample_counts must contain positive mass")

    global_histogram = np.sum(
        [np.asarray(histogram, dtype=float) for histogram in label_histograms],
        axis=0,
    )
    global_distribution = _distribution(global_histogram)
    matrix = None if pairing_matrix is None else np.asarray(pairing_matrix, dtype=float)
    if matrix is not None and matrix.shape != (len(distributions), len(distributions)):
        raise ValueError("pairing_matrix shape must match the number of clients")

    paired_mass = 0.0
    individual_residual = 0.0
    mixture_residual = 0.0
    complementarity_gain = 0.0
    entropies: list[float] = []
    coverages: list[float] = []
    edge_scores: list[float] = []

    for client_i, client_j in pairing.pairs:
        if client_i == client_j:
            raise ValueError("a client cannot be paired with itself")
        if not 0 <= client_i < len(distributions) or not 0 <= client_j < len(distributions):
            raise IndexError("pair contains an unknown client")
        count_i = float(counts[client_i])
        count_j = float(counts[client_j])
        pair_count = count_i + count_j
        if pair_count <= 0.0:
            continue
        weight_i = count_i / total_count
        weight_j = count_j / total_count
        pair_weight = weight_i + weight_j
        distribution_i = distributions[client_i]
        distribution_j = distributions[client_j]
        mixture = (count_i * distribution_i + count_j * distribution_j) / pair_count

        before = weight_i * _kl_divergence(distribution_i, global_distribution)
        before += weight_j * _kl_divergence(distribution_j, global_distribution)
        residual = pair_weight * _kl_divergence(mixture, global_distribution)
        gain = weight_i * _kl_divergence(distribution_i, mixture)
        gain += weight_j * _kl_divergence(distribution_j, mixture)

        paired_mass += pair_weight
        individual_residual += before
        mixture_residual += residual
        complementarity_gain += gain
        positive = mixture > 0.0
        entropy = -float(np.sum(mixture[positive] * np.log(mixture[positive])))
        if num_classes > 1:
            entropy /= float(np.log(num_classes))
        entropies.append(entropy)
        coverages.append(float(np.count_nonzero(positive)) / float(num_classes))
        if matrix is not None:
            edge_scores.append(float(matrix[client_i, client_j]))

    denominator = paired_mass if paired_mass > 0.0 else 1.0
    return {
        "paired_global_mass": paired_mass,
        "pair_individual_kl_residual": individual_residual,
        "pair_mixture_kl_residual": mixture_residual,
        "pair_mixture_kl_residual_normalized": mixture_residual / denominator,
        "pair_complementarity_gain": complementarity_gain,
        "pair_complementarity_gain_normalized": complementarity_gain / denominator,
        "pair_kl_identity_error": individual_residual
        - mixture_residual
        - complementarity_gain,
        "mean_pair_label_entropy": float(np.mean(entropies)) if entropies else 0.0,
        "min_pair_label_entropy": float(np.min(entropies)) if entropies else 0.0,
        "mean_pair_class_coverage": float(np.mean(coverages)) if coverages else 0.0,
        "min_pair_class_coverage": float(np.min(coverages)) if coverages else 0.0,
        "mean_selected_pairing_score": float(np.mean(edge_scores)) if edge_scores else 0.0,
        "total_selected_pairing_score": float(np.sum(edge_scores)) if edge_scores else 0.0,
    }


def classification_fairness_metrics(
    confusion_matrix: Sequence[Sequence[float]],
    client_label_histograms: Iterable[Sequence[float]],
) -> dict[str, float]:
    """Summarize class performance and label-shift client fairness.

    ``client_accuracy_proxy`` is not a local test accuracy.  It is the expected
    accuracy under a label-shift-only model: each client's training-label
    distribution weights the recalls measured on the common test set.
    """

    confusion = np.asarray(confusion_matrix, dtype=float)
    if confusion.ndim != 2 or confusion.shape[0] != confusion.shape[1]:
        raise ValueError("confusion_matrix must be square")
    if np.any(~np.isfinite(confusion)) or np.any(confusion < 0.0):
        raise ValueError("confusion_matrix must be finite and non-negative")
    support = confusion.sum(axis=1)
    predicted = confusion.sum(axis=0)
    true_positive = np.diag(confusion)
    recall = np.divide(
        true_positive,
        support,
        out=np.zeros_like(true_positive),
        where=support > 0.0,
    )
    precision = np.divide(
        true_positive,
        predicted,
        out=np.zeros_like(true_positive),
        where=predicted > 0.0,
    )
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(true_positive),
        where=(precision + recall) > 0.0,
    )
    supported = support > 0.0
    supported_recall = recall[supported]
    supported_f1 = f1[supported]

    client_scores: list[float] = []
    client_minority_scores: list[float] = []
    for histogram in client_label_histograms:
        distribution = _distribution(histogram)
        if distribution.size != confusion.shape[0]:
            raise ValueError("client histogram class count must match confusion_matrix")
        client_scores.append(float(np.dot(distribution, recall)))
        positive_probabilities = distribution[distribution > 0.0]
        if positive_probabilities.size:
            threshold = float(np.quantile(positive_probabilities, 0.25))
            minority = (distribution > 0.0) & (distribution <= threshold)
            client_minority_scores.append(float(np.mean(recall[minority])))

    scores = np.asarray(client_scores, dtype=float)
    minority_scores = np.asarray(client_minority_scores, dtype=float)
    squared_sum = float(np.sum(scores * scores))
    jain = (
        float(scores.sum() ** 2) / (float(scores.size) * squared_sum)
        if scores.size and squared_sum > 0.0
        else 0.0
    )
    return {
        "test_macro_recall": float(np.mean(supported_recall)) if supported_recall.size else 0.0,
        "test_macro_f1": float(np.mean(supported_f1)) if supported_f1.size else 0.0,
        "test_worst_class_recall": float(np.min(supported_recall)) if supported_recall.size else 0.0,
        "client_accuracy_proxy_mean": float(np.mean(scores)) if scores.size else 0.0,
        "client_accuracy_proxy_min": float(np.min(scores)) if scores.size else 0.0,
        "client_accuracy_proxy_p10": float(np.quantile(scores, 0.10)) if scores.size else 0.0,
        "client_accuracy_proxy_std": float(np.std(scores)) if scores.size else 0.0,
        "client_accuracy_proxy_jain": jain,
        "client_minority_recall_proxy_mean": (
            float(np.mean(minority_scores)) if minority_scores.size else 0.0
        ),
        "client_minority_recall_proxy_min": (
            float(np.min(minority_scores)) if minority_scores.size else 0.0
        ),
    }
