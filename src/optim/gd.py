from __future__ import annotations

import torch


def gd_step(W: torch.Tensor, gradW: torch.Tensor, eta: float) -> torch.Tensor:
    if eta <= 0:
        raise ValueError("eta must be positive")
    if W.shape != gradW.shape:
        raise ValueError("W and gradW must have same shape")
    return W - float(eta) * gradW


def apply_update_(param: torch.nn.Parameter, new_value: torch.Tensor) -> None:
    if param.shape != new_value.shape:
        raise ValueError("param and new_value must have same shape")
    with torch.no_grad():
        param.copy_(new_value)
    if param.grad is not None:
        param.grad = None

