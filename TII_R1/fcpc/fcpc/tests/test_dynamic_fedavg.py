from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.algorithms.dynamic_fedavg import DynamicFedAvgAdapter


def server_with_clients(count: int):
    return SimpleNamespace(clients=[object() for _ in range(count)])


def test_dynamic_fedavg_activates_new_clients_at_join_round():
    adapter = DynamicFedAvgAdapter(
        initial_clients=14,
        join_round=10,
        final_clients=21,
    )
    server = server_with_clients(21)

    adapter.begin_round(round_idx=9)
    assert adapter.select_clients(server, clients_per_round=21, seed=42) == list(
        range(14)
    )

    adapter.begin_round(round_idx=10)
    assert adapter.select_clients(server, clients_per_round=21, seed=42) == list(
        range(21)
    )


def test_dynamic_fedavg_partial_sampling_stays_inside_active_pool():
    adapter = DynamicFedAvgAdapter(
        initial_clients=14,
        join_round=10,
        final_clients=21,
    )
    server = server_with_clients(21)
    adapter.begin_round(round_idx=0)
    first = adapter.select_clients(server, clients_per_round=5, seed=7)
    second = adapter.select_clients(server, clients_per_round=5, seed=7)
    assert first == second
    assert len(first) == 5
    assert len(set(first)) == 5
    assert all(0 <= client_id < 14 for client_id in first)


def test_dynamic_fedavg_rejects_invalid_schedule():
    with pytest.raises(ValueError):
        DynamicFedAvgAdapter(initial_clients=0)
    with pytest.raises(ValueError):
        DynamicFedAvgAdapter(initial_clients=5, final_clients=4)
    with pytest.raises(ValueError):
        DynamicFedAvgAdapter(initial_clients=5, join_round=-1)


def test_dynamic_fedavg_rejects_too_many_final_clients():
    adapter = DynamicFedAvgAdapter(
        initial_clients=5,
        join_round=1,
        final_clients=11,
    )
    adapter.begin_round(round_idx=1)
    with pytest.raises(ValueError):
        adapter.select_clients(server_with_clients(10), clients_per_round=10, seed=0)
