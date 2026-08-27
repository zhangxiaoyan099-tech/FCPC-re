"""MOON adapter following Xtra-Computing/MOON ``train_net_fedcon``."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Mapping

from src.models.representations import forward_with_representation

from .base import AlgorithmAdapter, _zero_like


@dataclass
class MOONAdapter(AlgorithmAdapter):
    name: str = "moon"
    mu: float = 1.0
    temperature: float = 0.5
    _global_model: object | None = field(default=None, init=False, repr=False)
    _previous_model: object | None = field(default=None, init=False, repr=False)

    def begin_local_train(
        self,
        model,
        global_state,
        previous_local_state=None,
        device="cpu",
        **_context,
    ):
        self._global_model = self._frozen_copy(model, global_state, device)
        self._previous_model = None
        if previous_local_state is not None:
            self._previous_model = self._frozen_copy(
                model,
                previous_local_state,
                device,
            )

    def forward(self, model, inputs):
        logits, representation = forward_with_representation(model, inputs)
        return logits, {"current_representation": representation}

    def extra_loss(self, model, batch, task_loss, context: Mapping[str, object]):
        if self._global_model is None or self._previous_model is None:
            return _zero_like(task_loss)
        import torch
        import torch.nn.functional as F

        inputs, _targets = batch
        current = context["current_representation"]
        with torch.no_grad():
            _global_logits, global_rep = forward_with_representation(
                self._global_model,
                inputs,
            )
            _previous_logits, previous_rep = forward_with_representation(
                self._previous_model,
                inputs,
            )
        positive = F.cosine_similarity(current, global_rep, dim=-1)
        negative = F.cosine_similarity(current, previous_rep, dim=-1)
        contrastive_logits = torch.stack((positive, negative), dim=1)
        contrastive_logits = contrastive_logits / float(self.temperature)
        labels = torch.zeros(
            contrastive_logits.shape[0],
            dtype=torch.long,
            device=contrastive_logits.device,
        )
        return float(self.mu) * F.cross_entropy(contrastive_logits, labels)

    def end_local_train(self):
        self._global_model = None
        self._previous_model = None

    @staticmethod
    def _frozen_copy(model, state, device):
        reference = copy.deepcopy(model)
        reference.load_state_dict(state)
        reference.to(device)
        reference.eval()
        for parameter in reference.parameters():
            parameter.requires_grad_(False)
        return reference
