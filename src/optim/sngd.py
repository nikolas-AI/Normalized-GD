from __future__ import annotations

import torch

from src.optim.ngd import ngd_stepsize


def sngd_step(
    W: torch.Tensor,
    grad_batch: torch.Tensor,
    eta_base: float,
    F_full: float | torch.Tensor,
    *,
    eps: float = 0.0,
) -> torch.Tensor:
    if W.shape != grad_batch.shape:
        raise ValueError("W and grad_batch must have same shape")
    eta_t = ngd_stepsize(eta_base=eta_base, F_scalar=F_full, eps=eps)
    return W - eta_t * grad_batch

