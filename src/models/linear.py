from __future__ import annotations

import torch
from torch import nn


class LinearBinary(nn.Module):
    """Linear classifier ``f(x) = x @ W`` for binary classification with labels in ``{+1, -1}``."""

    def __init__(self, d: int, *, init_zero: bool = True) -> None:
        """Initialise the linear model.

        Args:
            d: Input dimension.
            init_zero: If True, initialise weights to zero; otherwise use a random Gaussian initialisation.

        Raises:
            ValueError: If ``d`` is not positive.
        """
        super().__init__()
        if d <= 0:
            raise ValueError("d must be positive")
        w0 = torch.zeros(d) if init_zero else torch.randn(d)
        self.W = nn.Parameter(w0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the linear prediction scores.

        Args:
            x: Input batch of shape ``(batch, d)``.

        Returns:
            Score tensor of shape ``(batch,)``.

        Raises:
            ValueError: If ``x`` does not have the expected shape.
        """
        if x.ndim != 2 or x.shape[1] != self.W.shape[0]:
            raise ValueError(f"x must have shape (batch,{self.W.shape[0]})")
        return x @ self.W

