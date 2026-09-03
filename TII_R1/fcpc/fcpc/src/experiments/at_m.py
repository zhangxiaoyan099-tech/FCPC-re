"""Metrics for the frozen-checkpoint :math:`A_t(M)` pairing audit.

Only trainable parameters enter these diagnostics.  Batch-normalization
buffers and other state-dict buffers are deliberately excluded because the
theoretical update and gradient are vectors in parameter space.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np

from src.fcpc.pairing import PairingResult


def _normalise(values: Iterable[float], eps: float = 1e-12) -> np.ndarray:
    vector = np.maximum(np.asarray(list(values), dtype=np.float64), 0.0)
    total = float(vector.sum())
    if total <= eps:
        return np.full(vector.shape, 1.0 / max(vector.size, 1), dtype=np.float64)
    return vector / total


def _kl_bits(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = _normalise(p, eps=eps)
    q = _normalise(q, eps=eps)
    positive = p > 0.0
    q_safe = np.clip(q, eps, None)
    return float(np.sum(p[positive] * np.log2(p[positive] / q_safe[positive])))


def matching_groups(
    pairing: PairingResult,
    client_ids: Sequence[int],
) -> list[tuple[int, ...]]:
    """Return disjoint pair/singleton groups and validate exact coverage."""
    expected = {int(client_id) for client_id in client_ids}
    groups: list[tuple[int, ...]] = []
    seen: set[int] = set()
    for left, right in pairing.pairs:
        group = (int(left), int(right))
        if group[0] == group[1] or any(client_id in seen for client_id in group):
            raise ValueError("pairing contains a duplicate client")
        groups.append(group)
        seen.update(group)
    for client_id in pairing.unpaired:
        client_id = int(client_id)
        if client_id in seen:
            raise ValueError("an unpaired client also appears in a pair")
        groups.append((client_id,))
        seen.add(client_id)
    if seen != expected:
        missing = sorted(expected - seen)
        extra = sorted(seen - expected)
        raise ValueError(f"pairing does not cover the client set: missing={missing}, extra={extra}")
    return groups


def pair_mixture_kl_residual(
    label_histograms: Mapping[int, Iterable[float]] | Sequence[Iterable[float]],
    sample_counts: Mapping[int, float] | Sequence[float],
    pairing: PairingResult,
) -> float:
    """Compute ``R(M)`` with base-2 KL, including any singleton groups."""
    client_ids = _client_ids(sample_counts)
    counts = {client_id: _value(sample_counts, client_id) for client_id in client_ids}
    total_count = float(sum(max(count, 0.0) for count in counts.values()))
    if total_count <= 0.0:
        raise ValueError("sample counts must have positive total mass")
    distributions = {
        client_id: _normalise(_value(label_histograms, client_id))
        for client_id in client_ids
    }
    shapes = {distribution.shape for distribution in distributions.values()}
    if len(shapes) != 1:
        raise ValueError("all label histograms must have the same shape")

    global_distribution = sum(
        (max(counts[client_id], 0.0) / total_count) * distributions[client_id]
        for client_id in client_ids
    )
    residual = 0.0
    for group in matching_groups(pairing, client_ids):
        group_count = sum(max(counts[client_id], 0.0) for client_id in group)
        if group_count <= 0.0:
            continue
        mixture = sum(
            (max(counts[client_id], 0.0) / group_count) * distributions[client_id]
            for client_id in group
        )
        residual += (group_count / total_count) * _kl_bits(
            mixture,
            global_distribution,
        )
    return float(residual)


def compute_matching_metrics(
    *,
    global_state: Mapping[str, object],
    client_states: Mapping[int, Mapping[str, object]],
    client_gradients: Mapping[int, object],
    sample_counts: Mapping[int, float] | Sequence[float],
    pairing: PairingResult,
    gamma: float,
    parameter_names: Sequence[str],
    eps: float = 1e-12,
) -> tuple[dict[str, float], list[dict[str, float | int | str]]]:
    """Compute ``A_t(M)``, ``H_t(M)`` and ``U_t(M)``.

    ``client_gradients[i]`` must be the flattened mean empirical gradient of
    client ``i`` evaluated at the same frozen global model.  ``client_states``
    are the counterfactual local endpoints produced from that model.
    """
    import torch

    client_ids = _client_ids(sample_counts)
    if set(client_states) != set(client_ids):
        raise ValueError("client_states must contain every audited client exactly once")
    if set(client_gradients) != set(client_ids):
        raise ValueError("client_gradients must contain every audited client exactly once")
    counts = {
        client_id: max(float(_value(sample_counts, client_id)), 0.0)
        for client_id in client_ids
    }
    total_count = float(sum(counts.values()))
    if total_count <= 0.0:
        raise ValueError("sample counts must have positive total mass")
    weights = {client_id: counts[client_id] / total_count for client_id in client_ids}

    first_gradient = client_gradients[client_ids[0]].detach().cpu().float().reshape(-1)
    global_gradient = torch.zeros_like(first_gradient)
    global_delta = torch.zeros_like(first_gradient)
    for client_id in client_ids:
        gradient = client_gradients[client_id].detach().cpu().float().reshape(-1)
        if gradient.shape != first_gradient.shape:
            raise ValueError("all flattened client gradients must have the same shape")
        global_gradient.add_(gradient, alpha=weights[client_id])
        delta = flatten_state_delta(
            client_states[client_id],
            global_state,
            parameter_names,
        )
        if delta.shape != first_gradient.shape:
            raise ValueError("state and gradient parameter vectors have different shapes")
        global_delta.add_(delta, alpha=weights[client_id])

    a_value = 0.0
    h_value = 0.0
    pair_rows: list[dict[str, float | int | str]] = []
    for group_index, group in enumerate(matching_groups(pairing, client_ids)):
        group_mass = float(sum(weights[client_id] for client_id in group))
        if group_mass <= 0.0:
            continue
        group_gradient = torch.zeros_like(first_gradient)
        group_delta = torch.zeros_like(first_gradient)
        for client_id in group:
            conditional_weight = weights[client_id] / group_mass
            group_gradient.add_(
                client_gradients[client_id].detach().cpu().float().reshape(-1),
                alpha=conditional_weight,
            )
            group_delta.add_(
                flatten_state_delta(
                    client_states[client_id],
                    global_state,
                    parameter_names,
                ),
                alpha=conditional_weight,
            )
        execution_residual = group_delta + float(gamma) * group_gradient
        gradient_residual = group_gradient - global_gradient
        execution_squared = squared_l2(execution_residual)
        gradient_squared = squared_l2(gradient_residual)
        a_contribution = group_mass * execution_squared
        h_contribution = group_mass * gradient_squared
        a_value += a_contribution
        h_value += h_contribution
        pair_rows.append(
            {
                "group_index": group_index,
                "client_i": group[0],
                "client_j": "" if len(group) == 1 else group[1],
                "group_mass": group_mass,
                "execution_residual_sq": execution_squared,
                "A_contribution": a_contribution,
                "gradient_residual_sq": gradient_squared,
                "H_contribution": h_contribution,
            }
        )

    global_update_residual = global_delta + float(gamma) * global_gradient
    u_value = squared_l2(global_update_residual)
    gradient_norm_sq = squared_l2(global_gradient)
    denominator = float(gamma) ** 2 * gradient_norm_sq + float(eps)
    result = {
        "A_t_M": float(a_value),
        "A_t_normalized": float(a_value / denominator),
        "H_t_M": float(h_value),
        "U_t_M": float(u_value),
        "U_t_normalized": float(u_value / denominator),
        "global_gradient_norm": float(gradient_norm_sq**0.5),
        "global_update_norm": float(squared_l2(global_delta) ** 0.5),
        "cancellation_ratio_U_over_A": float(u_value / max(a_value, eps)),
        "U_le_A_gap": float(a_value - u_value),
    }
    return result, pair_rows


def flatten_state_delta(
    state: Mapping[str, object],
    reference_state: Mapping[str, object],
    parameter_names: Sequence[str],
):
    """Flatten ``state - reference_state`` in model parameter order."""
    import torch

    pieces = []
    for name in parameter_names:
        if name not in state or name not in reference_state:
            raise KeyError(f"missing model parameter in state dict: {name}")
        value = state[name]
        reference = reference_state[name]
        if tuple(value.shape) != tuple(reference.shape):
            raise ValueError(f"state shape mismatch for parameter {name}")
        pieces.append(
            (value.detach().cpu().float() - reference.detach().cpu().float()).reshape(-1)
        )
    if not pieces:
        return torch.empty(0, dtype=torch.float32)
    return torch.cat(pieces)


def squared_l2(vector) -> float:
    """Return a stable-enough squared Euclidean norm without a float64 copy."""
    vector = vector.detach().cpu().float().reshape(-1)
    return float(vector.dot(vector).item())


def _client_ids(values: Mapping[int, object] | Sequence[object]) -> list[int]:
    if isinstance(values, Mapping):
        return sorted(int(client_id) for client_id in values)
    return list(range(len(values)))


def _value(values: Mapping[int, object] | Sequence[object], client_id: int):
    return values[client_id]
