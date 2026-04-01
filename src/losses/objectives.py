from __future__ import annotations

import torch

from src.losses.exp_loss import exp_loss


def training_loss_exp(phi: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute the mean exponential loss: ``mean(exp(-y * phi))``.

    Args:
        phi: Model predictions of shape ``(n,)``.
        y: Labels in ``{+1, -1}`` of shape ``(n,)``.

    Returns:
        Scalar loss tensor.

    Raises:
        ValueError: If ``phi`` or ``y`` are not 1-D or have different lengths.
    """
    if phi.ndim != 1 or y.ndim != 1:
        raise ValueError("phi and y must be 1D tensors")
    if phi.shape[0] != y.shape[0]:
        raise ValueError("phi and y must have same length")
    t = y * phi
    return exp_loss(t).mean()

