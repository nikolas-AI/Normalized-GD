from __future__ import annotations

import torch


def classification_error(phi: torch.Tensor, y: torch.Tensor) -> float:
    if phi.ndim != 1 or y.ndim != 1:
        raise ValueError("phi and y must be 1D tensors")
    if phi.shape[0] != y.shape[0]:
        raise ValueError("phi and y must have same length")
    # Convention: sign(0) = +1
    y_hat = torch.where(phi >= 0, torch.tensor(1, device=phi.device, dtype=y.dtype), torch.tensor(-1, device=phi.device, dtype=y.dtype))
    err = (y_hat != y).to(torch.float32).mean().item()
    return float(err)


def weight_norm(W: torch.Tensor) -> float:
    return float(W.flatten().norm().item())

