"""FedDyn dynamic regularization adapted from alpemreacar/FedDyn."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from .base import AlgorithmAdapter, _zero_like


@dataclass
class FedDynAdapter(AlgorithmAdapter):
    name: str = "feddyn"
    alpha: float = 0.01
    adaptive_alpha: bool = True
    _client_history: dict[int, dict[str, object]] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _active_client_id: int | None = field(default=None, init=False, repr=False)
    _active_alpha: float = field(default=0.01, init=False, repr=False)
    _global_state: Mapping[str, object] | None = field(default=None, init=False, repr=False)
    _num_clients: int = field(default=1, init=False, repr=False)

    def begin_round(self, clients=None, **_context):
        if clients is not None:
            self._num_clients = max(len(clients), 1)

    def begin_local_train(
        self,
        client_id,
        global_state,
        sample_count=1,
        mean_sample_count=1.0,
        **_context,
    ):
        self._active_client_id = int(client_id)
        self._global_state = global_state
        scale = float(mean_sample_count) / max(float(sample_count), 1.0)
        self._active_alpha = float(self.alpha) * (scale if self.adaptive_alpha else 1.0)

    def extra_loss(self, model, batch, task_loss, context: Mapping[str, object]):
        if self._active_client_id is None or self._global_state is None:
            return _zero_like(task_loss)
        history = self._client_history.get(self._active_client_id, {})
        total = None
        for name, parameter in model.named_parameters():
            global_value = self._global_state.get(name)
            if global_value is None:
                continue
            history_value = history.get(name)
            if history_value is None:
                history_value = global_value.new_zeros(global_value.shape)
            target = -global_value.to(parameter.device) + history_value.to(parameter.device)
            term = 0.5 * (parameter * parameter).sum() + (parameter * target).sum()
            total = term if total is None else total + term
        if total is None:
            return _zero_like(task_loss)
        return self._active_alpha * total

    def after_local_train(self, client_id, local_state, global_state, **_context):
        history = self._client_history.setdefault(int(client_id), {})
        for name, local_value in local_state.items():
            global_value = global_state.get(name)
            if global_value is None or not local_value.is_floating_point():
                continue
            difference = local_value.detach().cpu() - global_value.detach().cpu()
            if name not in history:
                history[name] = difference.clone()
            else:
                history[name] = history[name] + difference

    def aggregate(self, selected_client_ids, client_states, default_state, **_context):
        if not client_states:
            return default_state
        import torch

        result = {}
        client_count = self._num_clients
        for name, default_value in default_state.items():
            if not default_value.is_floating_point():
                result[name] = default_value.detach().cpu().clone()
                continue
            selected_mean = torch.stack(
                [state[name].detach().cpu() for state in client_states],
                dim=0,
            ).mean(dim=0)
            histories = [
                values[name]
                for values in self._client_history.values()
                if name in values
            ]
            if histories:
                history_sum = torch.stack(histories, dim=0).sum(dim=0)
                selected_mean = selected_mean + history_sum / float(client_count)
            result[name] = selected_mean
        return result

    def end_local_train(self):
        self._active_client_id = None
        self._global_state = None
