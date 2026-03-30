from __future__ import annotations

import torch


def compute_grad_W(model, loss_scalar: torch.Tensor) -> torch.Tensor:
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

