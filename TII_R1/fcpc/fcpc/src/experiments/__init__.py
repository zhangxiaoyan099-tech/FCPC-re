"""Experiment-only diagnostics that do not alter the training pipeline."""

from .at_m import (
    compute_matching_metrics,
    pair_mixture_kl_residual,
)

__all__ = [
    "compute_matching_metrics",
    "pair_mixture_kl_residual",
]
