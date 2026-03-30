from __future__ import annotations

import torch
from torch import nn


class LinearBinary(nn.Module):
    def __init__(self, d: int, *, init_zero: bool = True) -> None:
        super().__init__()
        if d <= 0:
            raise ValueError("d must be positive")
        w0 = torch.zeros(d) if init_zero else torch.randn(d)
        self.W = nn.Parameter(w0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.W.shape[0]:
            raise ValueError(f"x must have shape (batch,{self.W.shape[0]})")
        return x @ self.W

