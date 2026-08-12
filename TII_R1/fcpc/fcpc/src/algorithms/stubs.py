from __future__ import annotations

from dataclasses import dataclass

from .base import AlgorithmAdapter


class NotIncludedAlgorithmMixin:
    reason: str = "not included in this reconstruction yet"

    def extra_loss(self, *_args, **_kwargs):
        raise NotImplementedError(f"{self.name} is {self.reason}.")


@dataclass
class MOONAdapter(NotIncludedAlgorithmMixin, AlgorithmAdapter):
    """Extension point for exact MOON.

    MOON requires current/global/previous local representation contrastive
    learning. This reconstruction does not expose a partial implementation as a
    formal baseline.
    """

    name: str = "moon"
    reason: str = "an extension point; exact contrastive MOON is not included in this reconstruction yet"


@dataclass
class FedDynAdapter(NotIncludedAlgorithmMixin, AlgorithmAdapter):
    """Interface stub for FedDyn.

    The exact dynamic regularization terms are intentionally not fabricated.
    """

    name: str = "feddyn"
    reason: str = "an extension point; exact FedDyn dynamic regularization is not included in this reconstruction yet"


@dataclass
class FBLGAdapter(NotIncludedAlgorithmMixin, AlgorithmAdapter):
    """Interface stub for FBLG graph-based dual-skew optimization."""

    name: str = "fblg"
    reason: str = "an extension point; exact FBLG graph logic is not included in this reconstruction yet"


@dataclass
class FedCFAAdapter(NotIncludedAlgorithmMixin, AlgorithmAdapter):
    """Interface stub for FedCFA causal/aggregation correction."""

    name: str = "fedcfa"
    reason: str = "an extension point; exact FedCFA causal correction is not included in this reconstruction yet"
