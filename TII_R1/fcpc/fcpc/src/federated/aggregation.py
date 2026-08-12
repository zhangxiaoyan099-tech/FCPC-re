from __future__ import annotations

from typing import Iterable, List, Mapping, Sequence


def fedavg_aggregate(
    client_states: Sequence[Mapping[str, object]],
    sample_counts: Sequence[int] | None = None,
    weighted: bool = True,
):
    """Aggregate client state_dicts with weighted or equal FedAvg."""
    if not client_states:
        raise ValueError("client_states cannot be empty")
    if sample_counts is None or not weighted:
        weights = [1.0 / len(client_states)] * len(client_states)
    else:
        total = max(float(sum(sample_counts)), 1.0)
        weights = [float(count) / total for count in sample_counts]

    aggregated = {}
    for key in client_states[0].keys():
        first = client_states[0][key]
        if hasattr(first, "is_floating_point") and not first.is_floating_point():
            aggregated[key] = first.detach().clone() if hasattr(first, "detach") else first
            continue
        value = None
        for state, weight in zip(client_states, weights):
            tensor = state[key]
            contribution = tensor.detach().clone() * weight if hasattr(tensor, "detach") else tensor * weight
            value = contribution if value is None else value + contribution
        aggregated[key] = value
    return aggregated
