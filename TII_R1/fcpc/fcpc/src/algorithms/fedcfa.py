"""FedCFA counterfactual local objective adapted from hua-zi/FedCFA.

The upstream entry is ``alg/fedcfa.py``.  The public code does not apply the
paper's FDC term, so this adapter follows the released executable objective:
classification plus positive and negative counterfactual losses.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from src.models.representations import fedcfa_decode, fedcfa_encode

from .base import AlgorithmAdapter, _zero_like


@dataclass
class FedCFAAdapter(AlgorithmAdapter):
    name: str = "fedcfa"
    topk: int = 24
    rates: Sequence[float] | str = (1.0, 5.0, 5.0)
    mean_batch_size: int = 128
    num_classes: int = 10
    _global_model: object | None = field(default=None, init=False, repr=False)
    _global_mean_x: object | None = field(default=None, init=False, repr=False)
    _global_mean_y: object | None = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if isinstance(self.rates, str):
            self.rates = tuple(float(value) for value in self.rates.split(":"))
        else:
            self.rates = tuple(float(value) for value in self.rates)
        if len(self.rates) != 3:
            raise ValueError("FedCFA rates must contain cls:positive:negative")

    def begin_round(self, global_mean_x=None, global_mean_y=None, **_context):
        self._global_mean_x = global_mean_x
        self._global_mean_y = global_mean_y

    def begin_local_train(self, model, global_state, device="cpu", **_context):
        self._global_model = copy.deepcopy(model)
        self._global_model.load_state_dict(global_state)
        self._global_model.to(device)
        self._global_model.eval()
        for parameter in self._global_model.parameters():
            parameter.requires_grad_(False)

    def forward(self, model, inputs):
        latent = fedcfa_encode(model, inputs)
        return fedcfa_decode(model, latent), {"current_latent": latent}

    def extra_loss(self, model, batch, task_loss, context: Mapping[str, object]):
        if (
            self._global_model is None
            or self._global_mean_x is None
            or len(self._global_mean_x) == 0
        ):
            return _zero_like(task_loss)
        import torch
        import torch.nn.functional as F

        inputs, targets = batch
        latent = context["current_latent"]
        detached = latent.detach().requires_grad_(True)
        one_hot = F.one_hot(targets, num_classes=int(self.num_classes)).float()
        global_scores = fedcfa_decode(self._global_model, detached)
        selected_score = (global_scores * one_hot).sum() / max(len(targets), 1)
        gradients = torch.autograd.grad(selected_score, detached)[0]
        importance = gradients
        if importance.ndim > 2:
            importance = importance.mean(dim=tuple(range(2, importance.ndim)))
        feature_count = int(importance.shape[1])
        topk = min(max(int(self.topk), 1), feature_count)

        positive_mask = torch.ones_like(importance)
        positive_mask.scatter_(
            1,
            importance.topk(k=topk, dim=1, largest=False).indices,
            0,
        )
        negative_mask = torch.ones_like(importance)
        negative_mask.scatter_(
            1,
            importance.topk(k=topk, dim=1, largest=True).indices,
            0,
        )
        while positive_mask.ndim < latent.ndim:
            positive_mask = positive_mask.unsqueeze(-1)
            negative_mask = negative_mask.unsqueeze(-1)

        pool_size = int(len(self._global_mean_x))
        random_ids = torch.randint(0, pool_size, (len(inputs),))
        mean_inputs = self._global_mean_x[random_ids].to(inputs.device)
        mean_labels = self._global_mean_y[random_ids].to(inputs.device)
        with torch.no_grad():
            global_latent = fedcfa_encode(model, mean_inputs)

        local_latent = latent.detach()
        positive_latent = (
            positive_mask * local_latent
            + (1.0 - positive_mask) * global_latent
        )
        negative_latent = (
            negative_mask * local_latent
            + (1.0 - negative_mask) * global_latent
        )
        positive_logits = fedcfa_decode(model, positive_latent)
        negative_logits = fedcfa_decode(model, negative_latent)
        positive_loss = _soft_cross_entropy(positive_logits, one_hot)
        mask_fraction = negative_mask.float().mean()
        negative_target = mask_fraction * one_hot + (1.0 - mask_fraction) * mean_labels
        negative_loss = _soft_cross_entropy(negative_logits, negative_target)

        cls_rate, positive_rate, negative_rate = self.rates
        # task_loss is already added by Client.local_train. Return only the
        # difference needed to reproduce rate[0] * classification loss.
        return (
            (cls_rate - 1.0) * task_loss
            + positive_rate * positive_loss
            + negative_rate * negative_loss
        )

    def end_local_train(self):
        self._global_model = None


def _soft_cross_entropy(logits, target_probabilities):
    import torch.nn.functional as F

    return -(target_probabilities * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
