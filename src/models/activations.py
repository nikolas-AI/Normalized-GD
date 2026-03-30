from __future__ import annotations

import torch


def leaky_relu(x: torch.Tensor, alpha: float = 0.2, ell: float = 1.0) -> torch.Tensor:
    return torch.where(x >= 0, ell * x, alpha * x)


def leaky_relu_prime(x: torch.Tensor, alpha: float = 0.2, ell: float = 1.0) -> torch.Tensor:
    ell_t = torch.tensor(ell, device=x.device, dtype=x.dtype)
    alpha_t = torch.tensor(alpha, device=x.device, dtype=x.dtype)
    return torch.where(x >= 0, ell_t, alpha_t)

