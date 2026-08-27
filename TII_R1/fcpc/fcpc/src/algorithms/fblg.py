"""License-clean FBLG server-side selection implementation.

The algorithm follows IJCAI 2024 and the author's ``algorithm/FBLG.py`` path,
but does not copy the unlicensed upstream source.  It removes the proprietary
Gurobi dependency by exactly enumerating small candidate sets and using a
deterministic greedy fallback for large sets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from math import comb

import numpy as np

from src.models.representations import forward_with_representation

from .base import AlgorithmAdapter


@dataclass
class FBLGAdapter(AlgorithmAdapter):
    name: str = "fblg"
    candidate_ratio: float = 0.5
    epsilon: float = 0.01
    sigma: float = 0.1
    exact_combination_limit: int = 100_000
    _distance_matrix: np.ndarray | None = field(default=None, init=False, repr=False)

    def aggregate(self, default_state, previous_global_state=None, **_context):
        if self._distance_matrix is None and previous_global_state is not None:
            # The author's round zero is graph initialization only; its local
            # models build the graph but are not aggregated into the server.
            return {
                key: value.detach().cpu().clone()
                for key, value in previous_global_state.items()
            }
        return default_state

    def select_clients(
        self,
        server,
        clients_per_round,
        seed,
        global_state=None,
        model_factory=None,
        device="cpu",
        max_batches=None,
        **_context,
    ):
        if self._distance_matrix is None:
            # The author's round zero trains every client to construct a fixed
            # local graph from the resulting model embeddings.
            return list(range(len(server.clients)))
        losses = self._client_losses(
            server.clients,
            global_state,
            model_factory,
            device,
            max_batches,
        )
        total_clients = len(server.clients)
        candidate_count = max(
            min(int(np.ceil(self.candidate_ratio * total_clients)), total_clients),
            min(int(clients_per_round), total_clients),
        )
        candidates = np.argsort(np.asarray(losses))[::-1][:candidate_count].tolist()
        select_count = min(int(clients_per_round), len(candidates))
        return self._optimize_subset(candidates, select_count, server.clients)

    def after_round(
        self,
        selected_client_ids,
        client_states,
        model_factory,
        reference_inputs=None,
        device="cpu",
        **_context,
    ):
        if self._distance_matrix is not None:
            return
        if reference_inputs is None:
            raise ValueError("FBLG requires a server validation batch")
        if len(selected_client_ids) != len(client_states):
            raise ValueError("FBLG client states are incomplete")
        state_by_client = dict(zip(selected_client_ids, client_states))
        if len(state_by_client) != len(selected_client_ids):
            raise ValueError("duplicate client ids in FBLG graph construction")
        embeddings = []
        mean = reference_inputs.float().mean()
        std = reference_inputs.float().std().clamp_min(1e-6)
        import torch

        generator = torch.Generator(device="cpu").manual_seed(0)
        noise = torch.normal(
            mean=float(mean),
            std=float(std),
            size=tuple(reference_inputs.shape),
            generator=generator,
        ).to(device)
        for client_id in range(len(state_by_client)):
            model = model_factory()
            model.load_state_dict(state_by_client[client_id])
            model.to(device)
            model.eval()
            with torch.no_grad():
                _logits, representation = forward_with_representation(model, noise)
            vector = representation.flatten(start_dim=1).mean(dim=0)
            # The released code feeds the non-negative post-ReLU mean
            # embedding directly to scipy.stats.entropy.  Normalization is
            # therefore performed inside the JSD helper, not with softmax.
            embeddings.append(vector.clamp_min(0).cpu().numpy())
        self._distance_matrix = self._build_shortest_path_matrix(embeddings)

    def _build_shortest_path_matrix(self, embeddings):
        count = len(embeddings)
        similarity = np.zeros((count, count), dtype=np.float64)
        for i in range(count):
            for j in range(i + 1, count):
                js = _js_divergence(embeddings[i], embeddings[j])
                similarity[i, j] = similarity[j, i] = 1.0 / (1.0 + js)
        finite = similarity[np.isfinite(similarity)]
        minimum = float(finite.min()) if finite.size else 0.0
        maximum = float(finite.max()) if finite.size else 1.0
        scale = max(maximum - minimum, 1e-12)
        normalized = (similarity - minimum) / scale
        adjacency = np.full((count, count), np.inf, dtype=np.float64)
        np.fill_diagonal(adjacency, 0.0)
        for i in range(count):
            for j in range(i + 1, count):
                if normalized[i, j] > float(self.epsilon):
                    edge = np.exp(-(normalized[i, j] ** 2) / float(self.sigma))
                    adjacency[i, j] = adjacency[j, i] = edge
        distance = adjacency.copy()
        for k in range(count):
            distance = np.minimum(distance, distance[:, k, None] + distance[None, k, :])
        finite = distance[np.isfinite(distance)]
        fill = float(finite.max()) if finite.size else 1.0
        distance[~np.isfinite(distance)] = fill
        maximum = max(float(distance.max()), 1e-12)
        return distance / maximum

    def _optimize_subset(self, candidates, select_count, clients):
        if select_count >= len(candidates):
            return sorted(int(value) for value in candidates)
        counts = np.asarray([clients[index].sample_count for index in candidates], dtype=float)
        counts = counts / max(float(counts.sum()), 1.0)

        def score(subset):
            positions = [candidates.index(client_id) for client_id in subset]
            graph_score = 0.0
            for left, right in combinations(subset, 2):
                graph_score += float(self._distance_matrix[left, right])
            graph_score /= max(len(candidates) * (len(candidates) - 1), 1)
            sample_score = float(counts[positions].sum())
            return graph_score + sample_score

        if comb(len(candidates), select_count) <= int(self.exact_combination_limit):
            best = max(combinations(candidates, select_count), key=score)
            return sorted(int(value) for value in best)
        selected = []
        remaining = list(candidates)
        while len(selected) < select_count:
            chosen = max(remaining, key=lambda value: score(selected + [value]))
            selected.append(chosen)
            remaining.remove(chosen)
        return sorted(int(value) for value in selected)

    @staticmethod
    def _client_losses(clients, global_state, model_factory, device, max_batches):
        import torch
        from torch import nn

        model = model_factory()
        model.load_state_dict(global_state)
        model.to(device)
        model.eval()
        criterion = nn.CrossEntropyLoss(reduction="sum")
        losses = []
        with torch.no_grad():
            for client in clients:
                total = 0.0
                examples = 0
                for batch_idx, (inputs, targets) in enumerate(client.train_loader):
                    if max_batches is not None and batch_idx >= int(max_batches):
                        break
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    total += float(criterion(model(inputs), targets).item())
                    examples += int(targets.numel())
                losses.append(total / max(examples, 1))
        return losses


def _js_divergence(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left = np.clip(left, 1e-12, None)
    right = np.clip(right, 1e-12, None)
    left /= left.sum()
    right /= right.sum()
    middle = 0.5 * (left + right)
    return float(
        0.5 * np.sum(left * np.log(left / middle))
        + 0.5 * np.sum(right * np.log(right / middle))
    )
