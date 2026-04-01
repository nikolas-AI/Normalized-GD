from __future__ import annotations

import torch


def compute_grad_W(model, loss_scalar: torch.Tensor) -> torch.Tensor:
    """Backpropagate through ``loss_scalar`` and return the gradient w.r.t. ``model.W``.

    Clears any existing gradients on all parameters before calling ``backward()``.

    Args:
        model: Network module with a trainable ``W`` parameter.
        loss_scalar: Scalar loss tensor (0-D) to differentiate.

    Returns:
        Detached gradient tensor with the same shape as ``model.W``.

    Raises:
        ValueError: If ``loss_scalar`` is not a scalar tensor.
        AttributeError: If ``model`` does not have a ``W`` attribute.
        RuntimeError: If ``model.W.grad`` is ``None`` after backward.
    """
    if loss_scalar.ndim != 0:
        raise ValueError("loss_scalar must be a scalar tensor")

    for p in model.parameters():
        if p.grad is not None:
            p.grad = None

    loss_scalar.backward()

    if not hasattr(model, "W"):
        raise AttributeError("model must have attribute W")
    if model.W.grad is None:
        raise RuntimeError("model.W.grad is None after backward()")
    return model.W.grad.detach().clone()

