from __future__ import annotations

import torch

from src.models.activations import leaky_relu


def phi(
    W: torch.Tensor,
    a: torch.Tensor,
    x: torch.Tensor,
    *,
    alpha: float = 0.2,
    ell: float = 1.0,
) -> torch.Tensor:
    if W.ndim != 2:
        raise ValueError("W must have shape (m,d)")
    if a.ndim != 1:
        raise ValueError("a must have shape (m,)")
    if x.ndim != 2:
        raise ValueError("x must have shape (b,d)")
    if W.shape[0] != a.shape[0]:
        raise ValueError("W and a must agree on m")
    if W.shape[1] != x.shape[1]:
        raise ValueError("W and x must agree on d")

    z = x @ W.T
    h = leaky_relu(z, alpha=alpha, ell=ell)
    return h @ a

