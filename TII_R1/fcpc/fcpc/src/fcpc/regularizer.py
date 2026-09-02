"""FCPC paired-model regularization.

The current model mapping must contain live ``Parameter`` objects (normally
``dict(model.named_parameters())``).  A ``state_dict`` is suitable for the
frozen partner reference, but not for the current model: state-dict tensors do
not participate in autograd and would silently turn FCPC into a no-op.
"""

from __future__ import annotations

from math import sqrt
from typing import Mapping


def weighted_state_center(
    state_a: Mapping[str, object] | None,
    state_b: Mapping[str, object] | None,
    count_a: float,
    count_b: float,
    fallback_state: Mapping[str, object],
) -> dict[str, object]:
    """Return the frozen sample-weighted model center of a client pair.

    Missing pre-round states use the common global model.  Floating tensors
    are averaged with ``theta = Na/(Na+Nb)``; non-floating buffers are cloned
    from the first compatible state because they never enter the parameter
    regularizer.
    """
    left = fallback_state if state_a is None else state_a
    right = fallback_state if state_b is None else state_b
    count_a = max(float(count_a), 0.0)
    count_b = max(float(count_b), 0.0)
    denominator = count_a + count_b
    theta = 0.5 if denominator <= 0.0 else count_a / denominator
    center: dict[str, object] = {}
    for name, fallback_value in fallback_state.items():
        value_a = left.get(name, fallback_value)
        value_b = right.get(name, fallback_value)
        if (
            hasattr(value_a, "shape")
            and hasattr(value_b, "shape")
            and tuple(value_a.shape) == tuple(value_b.shape)
            and hasattr(value_a, "is_floating_point")
            and value_a.is_floating_point()
        ):
            center[name] = (theta * value_a + (1.0 - theta) * value_b).detach().cpu().clone()
        elif hasattr(value_a, "detach"):
            center[name] = value_a.detach().cpu().clone()
        else:
            center[name] = value_a
    return center


def fcpc_regularization(
    current_parameters: Mapping[str, object],
    paired_previous_state: Mapping[str, object] | None,
    beta: float = 0.2,
):
    """Compute beta * sum(||current - paired_previous||^2).

    ``current_parameters`` must be a mapping of live trainable parameters.
    The partner state is treated as a fixed reference from the previous round.
    Keys absent from the reference and incompatible tensor shapes are skipped.
    """
    if paired_previous_state is None:
        return _zero_like_state(current_parameters)

    total = None
    for name, current_value in current_parameters.items():
        paired_value = paired_previous_state.get(name)
        if paired_value is None:
            continue
        if not hasattr(current_value, "shape") or not hasattr(paired_value, "shape"):
            continue
        if hasattr(current_value, "is_floating_point") and not current_value.is_floating_point():
            continue
        if tuple(current_value.shape) != tuple(paired_value.shape):
            continue
        if hasattr(paired_value, "to") and hasattr(current_value, "device"):
            paired_value = paired_value.to(current_value.device)
        diff = current_value - paired_value
        term = (diff * diff).sum()
        total = term if total is None else total + term
    if total is None:
        return _zero_like_state(current_parameters)
    return float(beta) * total


def proximal_center_step(
    current_parameters: Mapping[str, object],
    center_state: Mapping[str, object] | None,
    *,
    beta: float,
    learning_rate: float,
) -> float:
    """Apply the exact proximal map of ``beta * ||w - center||^2``.

    This function is called *after* the task/algorithm optimizer step.  For a
    provisional parameter ``u`` it performs

    ``w_next = (u + 2 * lr * beta * center) / (1 + 2 * lr * beta)``.

    If two clients use the same center, beta, and learning rate, their
    provisional disagreement is therefore multiplied by the returned factor
    ``1 / (1 + 2 * lr * beta)``.  The operation is deliberately separate from
    the task optimizer so the contraction does not rely on a small explicit
    regularizer step or on the absence of momentum.
    """
    beta = float(beta)
    learning_rate = float(learning_rate)
    if beta < 0.0:
        raise ValueError("beta must be non-negative")
    if learning_rate < 0.0:
        raise ValueError("learning_rate must be non-negative")
    denominator = 1.0 + 2.0 * learning_rate * beta
    contraction = 1.0 / denominator
    if center_state is None or beta == 0.0 or learning_rate == 0.0:
        return contraction

    import torch

    center_weight = 1.0 - contraction
    with torch.no_grad():
        for name, current_value in current_parameters.items():
            center_value = center_state.get(name)
            if center_value is None:
                continue
            if not hasattr(current_value, "shape") or not hasattr(center_value, "shape"):
                continue
            if tuple(current_value.shape) != tuple(center_value.shape):
                continue
            if hasattr(current_value, "is_floating_point") and not current_value.is_floating_point():
                continue
            if hasattr(center_value, "to"):
                center_value = center_value.to(
                    device=current_value.device,
                    dtype=current_value.dtype,
                )
            current_value.mul_(contraction).add_(center_value, alpha=center_weight)
    return contraction


def state_l2_norm(
    state: Mapping[str, object],
    parameter_names: set[str] | None = None,
) -> float:
    """Return the Euclidean norm of compatible floating tensors in a state."""
    total_squared = 0.0
    for name, value in state.items():
        if parameter_names is not None and name not in parameter_names:
            continue
        if not hasattr(value, "is_floating_point") or not value.is_floating_point():
            continue
        total_squared += float(value.detach().double().pow(2).sum().item())
    return sqrt(total_squared)


def state_l2_distance(
    state_a: Mapping[str, object],
    state_b: Mapping[str, object],
    parameter_names: set[str] | None = None,
) -> float:
    """Return the Euclidean distance between compatible floating tensors."""
    total_squared = 0.0
    for name, value_a in state_a.items():
        if parameter_names is not None and name not in parameter_names:
            continue
        value_b = state_b.get(name)
        if value_b is None:
            continue
        if not hasattr(value_a, "shape") or not hasattr(value_b, "shape"):
            continue
        if tuple(value_a.shape) != tuple(value_b.shape):
            continue
        if not hasattr(value_a, "is_floating_point") or not value_a.is_floating_point():
            continue
        if hasattr(value_b, "to"):
            value_b = value_b.to(device=value_a.device, dtype=value_a.dtype)
        total_squared += float(
            (value_a.detach() - value_b.detach()).double().pow(2).sum().item()
        )
    return sqrt(total_squared)


def clip_state_center_to_global(
    center_state: Mapping[str, object],
    global_state: Mapping[str, object],
    *,
    max_distance: float | None,
    parameter_names: set[str] | None = None,
) -> tuple[dict[str, object], float, float]:
    """Project a frozen pair center into an L2 ball around the global model.

    Returns ``(safe_center, original_distance, scale)``.  When clipping is
    active, floating parameters satisfy

    ``||safe_center - global_state|| <= max_distance``.

    Non-parameter buffers are cloned but do not affect the distance or FCPC
    regularizer.  ``max_distance=None`` disables clipping.
    """
    if max_distance is not None and float(max_distance) < 0.0:
        raise ValueError("max_distance must be non-negative or None")
    distance = state_l2_distance(
        center_state,
        global_state,
        parameter_names=parameter_names,
    )
    scale = 1.0
    if max_distance is not None and distance > float(max_distance):
        scale = 0.0 if distance == 0.0 else float(max_distance) / distance

    safe_center: dict[str, object] = {}
    for name, center_value in center_state.items():
        global_value = global_state.get(name)
        is_parameter = parameter_names is None or name in parameter_names
        compatible_float = (
            is_parameter
            and global_value is not None
            and hasattr(center_value, "shape")
            and hasattr(global_value, "shape")
            and tuple(center_value.shape) == tuple(global_value.shape)
            and hasattr(center_value, "is_floating_point")
            and center_value.is_floating_point()
        )
        if compatible_float:
            if hasattr(global_value, "to"):
                global_value = global_value.to(
                    device=center_value.device,
                    dtype=center_value.dtype,
                )
            safe_center[name] = (
                global_value + scale * (center_value - global_value)
            ).detach().cpu().clone()
        elif hasattr(center_value, "detach"):
            safe_center[name] = center_value.detach().cpu().clone()
        else:
            safe_center[name] = center_value
    return safe_center, distance, scale


def _zero_like_state(state: Mapping[str, object]):
    for value in state.values():
        if hasattr(value, "new_tensor"):
            return value.new_tensor(0.0)
    return 0.0
