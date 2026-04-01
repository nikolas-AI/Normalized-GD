from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from src.losses.objectives import training_loss_exp
from src.optim.gd import apply_update_, gd_step
from src.optim.ngd import ngd_step, ngd_stepsize


@dataclass(frozen=True)
class DescentCheckResult:
    """Result of a one-step descent check.

    Attributes:
        passed: True if the loss did not increase beyond tolerance.
        F_current: Training loss before the step.
        F_next: Training loss after the step.
        eta_t: Effective step size used for the update.
    """

    passed: bool
    F_current: float
    F_next: float
    eta_t: float


def one_step_descent_check(
    model,
    X: torch.Tensor,
    y: torch.Tensor,
    *,
    optimizer_type: Literal["gd", "ngd"],
    eta_base: float,
    tol: float = 1e-8,
) -> DescentCheckResult:
    """Verify that a single optimiser step does not increase the training loss.

    Applies one GD or NGD update in-place to ``model.W`` and compares the loss
    before and after.

    Args:
        model: Neural network module with a trainable ``W`` parameter.
        X: Training input tensor of shape ``(n, d)``.
        y: Training labels of shape ``(n,)`` in ``{+1, -1}``.
        optimizer_type: Either ``"gd"`` or ``"ngd"``.
        eta_base: Base learning rate.
        tol: Relative tolerance; the check passes if ``F_next <= F_current * (1 + tol)``.

    Returns:
        :class:`DescentCheckResult` summarising whether the check passed.

    Raises:
        ValueError: If ``optimizer_type`` is not recognised.
    """
    model.W.grad = None
    phi = model(X)
    F_current_t = training_loss_exp(phi, y)
    F_current_t.backward()
    grad = model.W.grad.detach().clone()

    W_current = model.W.detach().clone()
    if optimizer_type == "gd":
        eta_t = float(eta_base)
        W_next = gd_step(W_current, grad, eta_t)
    elif optimizer_type == "ngd":
        eta_t = ngd_stepsize(eta_base, F_current_t)
        W_next = ngd_step(W_current, grad, eta_base, F_current_t)
    else:
        raise ValueError(f"Unknown optimizer_type={optimizer_type!r}")

    apply_update_(model.W, W_next)
    with torch.no_grad():
        F_next_t = training_loss_exp(model(X), y)

    passed = float(F_next_t.item()) <= float(F_current_t.item()) * (1.0 + tol)
    return DescentCheckResult(
        passed=passed,
        F_current=float(F_current_t.item()),
        F_next=float(F_next_t.item()),
        eta_t=float(eta_t),
    )

