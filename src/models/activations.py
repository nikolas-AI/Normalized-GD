from __future__ import annotations

import torch


def leaky_relu(x: torch.Tensor, alpha: float = 0.2, ell: float = 1.0) -> torch.Tensor:
    """Element-wise leaky ReLU: ``ell * x`` for ``x >= 0``, ``alpha * x`` otherwise.

    Args:
        x: Input tensor.
        alpha: Slope for the negative region.
        ell: Slope for the non-negative region.

    Returns:
        Tensor of the same shape as ``x``.
    """
    return torch.where(x >= 0, ell * x, alpha * x)


def leaky_relu_prime(x: torch.Tensor, alpha: float = 0.2, ell: float = 1.0) -> torch.Tensor:
    """Element-wise derivative of leaky ReLU: ``ell`` for ``x >= 0``, ``alpha`` otherwise.

    Args:
        x: Input tensor (used only to determine the sign at each position).
        alpha: Slope for the negative region.
        ell: Slope for the non-negative region.

    Returns:
        Tensor of the same shape as ``x`` containing the pointwise derivative values.
    """
    ell_t = torch.tensor(ell, device=x.device, dtype=x.dtype)
    alpha_t = torch.tensor(alpha, device=x.device, dtype=x.dtype)
    return torch.where(x >= 0, ell_t, alpha_t)

