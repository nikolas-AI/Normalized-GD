from __future__ import annotations

import torch

from src.optim.gd import gd_step


def ngd_stepsize(eta_base: float, F_scalar: float | torch.Tensor, eps: float = 0.0) -> float:
    if eta_base <= 0:
        raise ValueError("eta_base must be positive")
    f = float(F_scalar.item() if isinstance(F_scalar, torch.Tensor) else F_scalar)
    if f <= 0 and eps <= 0:
        raise ValueError("F_scalar must be positive, or use eps > 0")
    denom = f + float(eps)
    if denom <= 0:
        raise ValueError("effective denominator must be positive")
    return float(eta_base) / denom


def ngd_step(
    W: torch.Tensor,
    gradW: torch.Tensor,
    eta_base: float,
    F_scalar: float | torch.Tensor,
    *,
    eps: float = 0.0,
) -> torch.Tensor:
    eta_t = ngd_stepsize(eta_base=eta_base, F_scalar=F_scalar, eps=eps)
    return gd_step(W, gradW, eta_t)

