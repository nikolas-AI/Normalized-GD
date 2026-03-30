from __future__ import annotations

import torch


def exp_loss(t: torch.Tensor) -> torch.Tensor:
    return torch.exp(-t)

