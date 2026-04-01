from __future__ import annotations

import torch

from src.optim.gd import gd_step


def ngd_stepsize(eta_base: float, F_scalar: float | torch.Tensor, eps: float = 0.0) -> float:
    """Compute the effective step size for Normalised GD: ``eta_base / (F_scalar + eps)``.

    Args:
        eta_base: Base learning rate.
        F_scalar: Normalisation factor (e.g., current loss or gradient norm).
        eps: Small regularisation constant added to the denominator.

    Returns:
        Effective step size as a Python float.

    Raises:
        ValueError: If ``eta_base`` is not positive or the effective denominator is non-positive.
    """
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
    """Perform one Normalised GD step: ``W - (eta_base / F_scalar) * gradW``.

    Args:
        W: Current weight tensor.
        gradW: Gradient tensor with the same shape as ``W``.
        eta_base: Base learning rate.
        F_scalar: Normalisation factor used to scale the step size.
        eps: Small regularisation constant added to the denominator.

    Returns:
        Updated weight tensor of the same shape as ``W``.
    """
    eta_t = ngd_stepsize(eta_base=eta_base, F_scalar=F_scalar, eps=eps)
    return gd_step(W, gradW, eta_t)

