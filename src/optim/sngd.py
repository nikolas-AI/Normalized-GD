from __future__ import annotations

import torch


def sngd_step(
    W: torch.Tensor,
    grad_batch: torch.Tensor,
    eta_base: float,
    *,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Stochastic normalized GD: W = W - eta * grad_batch / ||grad_batch||_2."""
    if W.shape != grad_batch.shape:
        raise ValueError("W and grad_batch must have same shape")
    grad_norm = grad_batch.norm()
    eta_t = float(eta_base) / (grad_norm.item() + float(eps))
    return W - eta_t * grad_batch

