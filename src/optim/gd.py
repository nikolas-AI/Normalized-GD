from __future__ import annotations

import torch


def gd_step(W: torch.Tensor, gradW: torch.Tensor, eta: float) -> torch.Tensor:
    """Return the updated weights after one gradient-descent step: ``W - eta * gradW``.

    Args:
        W: Current weight tensor.
        gradW: Gradient tensor with the same shape as ``W``.
        eta: Learning rate (must be positive).

    Returns:
        New weight tensor of the same shape as ``W``.

    Raises:
        ValueError: If ``eta`` is not positive or shapes disagree.
    """
    if eta <= 0:
        raise ValueError("eta must be positive")
    if W.shape != gradW.shape:
        raise ValueError("W and gradW must have same shape")
    return W - float(eta) * gradW


def apply_update_(param: torch.nn.Parameter, new_value: torch.Tensor) -> None:
    """In-place update of an ``nn.Parameter`` to ``new_value``, also clearing any stored gradient.

    Args:
        param: The parameter to update.
        new_value: Tensor with the desired new values; must have the same shape as ``param``.

    Raises:
        ValueError: If shapes of ``param`` and ``new_value`` differ.
    """
    if param.shape != new_value.shape:
        raise ValueError("param and new_value must have same shape")
    with torch.no_grad():
        param.copy_(new_value)
    if param.grad is not None:
        param.grad = None

