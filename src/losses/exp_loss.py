from __future__ import annotations

import torch


def exp_loss(t: torch.Tensor) -> torch.Tensor:
    """Element-wise exponential loss: ``exp(-t)``.

    Args:
        t: Margin values (typically ``y * f(x)``).

    Returns:
        Tensor of the same shape as ``t``.
    """
    return torch.exp(-t)

