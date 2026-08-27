"""Shared representation interfaces for MOON and FedCFA."""

from __future__ import annotations


def forward_with_representation(model, inputs):
    """Return ``(logits, representation)`` in one differentiable forward pass."""
    method = getattr(model, "forward_with_representation", None)
    if callable(method):
        return method(inputs)

    last_linear = _last_linear(model)
    if last_linear is None:
        logits = model(inputs)
        return logits, logits

    captured = []

    def capture(_module, args):
        captured.append(args[0])

    handle = last_linear.register_forward_pre_hook(capture)
    try:
        logits = model(inputs)
    finally:
        handle.remove()
    if not captured:
        raise RuntimeError("failed to capture the model representation")
    representation = captured[-1]
    if representation.ndim > 2:
        representation = representation.flatten(start_dim=1)
    return logits, representation


def fedcfa_encode(model, inputs):
    method = getattr(model, "fedcfa_encode", None)
    if not callable(method):
        raise TypeError(
            f"{model.__class__.__name__} does not expose fedcfa_encode; "
            "use simple_cnn, resnet18, mobilenetv2, or sensor_cnn"
        )
    return method(inputs)


def fedcfa_decode(model, latent):
    method = getattr(model, "fedcfa_decode", None)
    if not callable(method):
        raise TypeError(f"{model.__class__.__name__} does not expose fedcfa_decode")
    return method(latent)


def _last_linear(model):
    try:
        from torch import nn
    except ImportError:  # pragma: no cover
        return None
    result = None
    for module in model.modules():
        if isinstance(module, nn.Linear):
            result = module
    return result
