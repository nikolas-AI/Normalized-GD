from __future__ import annotations

import torch
from torch import nn

from src.models.activations import leaky_relu


class TwoLayerFixedA(nn.Module):
    def __init__(
        self,
        W_init: torch.Tensor,
        a_fixed: torch.Tensor,
        *,
        alpha: float = 0.2,
        ell: float = 1.0,
    ) -> None:
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
        return int(self.W.shape[0])

    @property
    def d(self) -> int:
        return int(self.W.shape[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.d:
            raise ValueError(f"x must have shape (batch,{self.d})")
        z = x @ self.W.T  # (b,m)
        h = leaky_relu(z, alpha=self.alpha, ell=self.ell)
        return h @ self.a  # (b,)

