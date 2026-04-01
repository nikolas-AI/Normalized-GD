from __future__ import annotations

from dataclasses import dataclass

import torch

from src.losses.objectives import training_loss_exp
from src.models.functional import phi as phi_fn


@dataclass(frozen=True)
class GradcheckResult:
    """Gradient check outcome comparing autograd against finite differences.

    Attributes:
        max_abs_err: Maximum absolute element-wise error between the two gradient estimates.
        max_rel_err: Maximum relative element-wise error between the two gradient estimates.
    """

    max_abs_err: float
    max_rel_err: float


def finite_difference_grad_W(
    W: torch.Tensor,
    a: torch.Tensor,
    X: torch.Tensor,
    y: torch.Tensor,
    *,
    alpha: float = 0.2,
    ell: float = 1.0,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Estimate ``∂L/∂W`` via central finite differences over every element of ``W``.

    Computations are performed in float64 for numerical accuracy then cast back
    to the original dtype of ``W``.

    Args:
        W: First-layer weights of shape ``(m, d)``.
        a: Fixed second-layer weights of shape ``(m,)``.
        X: Input batch of shape ``(n, d)``.
        y: Labels of shape ``(n,)`` in ``{+1, -1}``.
        alpha: Leaky ReLU negative slope.
        ell: Leaky ReLU positive slope.
        eps: Finite-difference step size.

    Returns:
        Gradient estimate tensor of the same shape as ``W``.
    """
    g = torch.zeros_like(W)
    base_dtype = W.dtype
    W = W.detach().clone().to(dtype=torch.float64)
    a = a.detach().clone().to(dtype=torch.float64)
    X = X.detach().clone().to(dtype=torch.float64)
    y = y.detach().clone().to(dtype=torch.float64)

    for j in range(W.shape[0]):
        for k in range(W.shape[1]):
            Wp = W.clone()
            Wm = W.clone()
            Wp[j, k] += eps
            Wm[j, k] -= eps
            fp = training_loss_exp(phi_fn(Wp, a, X, alpha=alpha, ell=ell), y)
            fm = training_loss_exp(phi_fn(Wm, a, X, alpha=alpha, ell=ell), y)
            g[j, k] = ((fp - fm) / (2 * eps)).to(dtype=base_dtype)
    return g


def gradcheck_autograd_vs_fd(
    W: torch.Tensor,
    a: torch.Tensor,
    X: torch.Tensor,
    y: torch.Tensor,
    *,
    alpha: float = 0.2,
    ell: float = 1.0,
    eps: float = 1e-5,
    atol: float = 1e-3,
    rtol: float = 1e-3,
) -> GradcheckResult:
    """Assert that PyTorch autograd and finite-difference gradients agree for ``W``.

    Args:
        W: First-layer weights of shape ``(m, d)``.
        a: Fixed second-layer weights of shape ``(m,)``.
        X: Input batch of shape ``(n, d)``.
        y: Labels of shape ``(n,)`` in ``{+1, -1}``.
        alpha: Leaky ReLU negative slope.
        ell: Leaky ReLU positive slope.
        eps: Finite-difference step size.
        atol: Absolute tolerance for :func:`torch.allclose`.
        rtol: Relative tolerance for :func:`torch.allclose`.

    Returns:
        :class:`GradcheckResult` with the maximum absolute and relative errors.

    Raises:
        AssertionError: If the two gradients do not agree within ``atol``/``rtol``.
    """
    W_ag = W.detach().clone().requires_grad_(True)
    phi = phi_fn(W_ag, a, X, alpha=alpha, ell=ell)
    loss = training_loss_exp(phi, y)
    loss.backward()
    g_ag = W_ag.grad.detach()

    g_fd = finite_difference_grad_W(W, a, X, y, alpha=alpha, ell=ell, eps=eps)

    abs_err = (g_ag - g_fd).abs()
    denom = torch.maximum(g_ag.abs(), g_fd.abs()).clamp_min(1e-12)
    rel_err = abs_err / denom
    res = GradcheckResult(max_abs_err=float(abs_err.max().item()), max_rel_err=float(rel_err.max().item()))

    if not torch.allclose(g_ag, g_fd, atol=atol, rtol=rtol):
        raise AssertionError(f"gradcheck failed: {res}")
    return res

