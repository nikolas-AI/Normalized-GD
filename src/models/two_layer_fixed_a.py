from __future__ import annotations

import torch
from torch import nn

from src.models.activations import leaky_relu


class TwoLayerFixedA(nn.Module):
    """Two-layer neural network with a fixed (non-trainable) second layer.

    The network computes ``f(x) = leaky_relu(x @ W^T) @ a`` where ``a`` is
    frozen at initialisation and only ``W`` is optimised.
    """

    def __init__(
        self,
        W_init: torch.Tensor,
        a_fixed: torch.Tensor,
        *,
        alpha: float = 0.2,
        ell: float = 1.0,
    ) -> None:
        """Initialise the network with given first-layer weights and fixed second-layer weights.

        Args:
            W_init: Initial first-layer weight matrix of shape ``(m, d)``.
            a_fixed: Fixed second-layer weights of shape ``(m,)``.
            alpha: Leaky ReLU negative slope.
            ell: Leaky ReLU positive slope.

        Raises:
            ValueError: If tensor shapes are inconsistent.
        """
        super().__init__()
        if W_init.ndim != 2:
            raise ValueError("W_init must have shape (m, d)")
        if a_fixed.ndim != 1:
            raise ValueError("a_fixed must have shape (m,)")
        if W_init.shape[0] != a_fixed.shape[0]:
            raise ValueError("W_init and a_fixed must agree on m")

        self.W = nn.Parameter(W_init.clone())
        self.register_buffer("a", a_fixed.clone())
        self.alpha = float(alpha)
        self.ell = float(ell)

    @property
    def m(self) -> int:
        """Number of neurons (hidden units)."""
        return int(self.W.shape[0])

    @property
    def d(self) -> int:
        """Input dimension."""
        return int(self.W.shape[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the network output for a batch of inputs.

        Args:
            x: Input batch of shape ``(batch, d)``.

        Returns:
            Output tensor of shape ``(batch,)``.

        Raises:
            ValueError: If ``x`` does not have the expected shape.
        """
        if x.ndim != 2 or x.shape[1] != self.d:
            raise ValueError(f"x must have shape (batch,{self.d})")
        z = x @ self.W.T  # (b,m)
        h = leaky_relu(z, alpha=self.alpha, ell=self.ell)
        return h @ self.a  # (b,)

