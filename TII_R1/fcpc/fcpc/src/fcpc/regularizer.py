"""FCPC paired-model regularization.

The current model mapping must contain live ``Parameter`` objects (normally
``dict(model.named_parameters())``).  A ``state_dict`` is suitable for the
frozen partner reference, but not for the current model: state-dict tensors do
not participate in autograd and would silently turn FCPC into a no-op.
"""

from __future__ import annotations

from typing import Mapping


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


def _zero_like_state(state: Mapping[str, object]):
    for value in state.values():
        if hasattr(value, "new_tensor"):
            return value.new_tensor(0.0)
    return 0.0
