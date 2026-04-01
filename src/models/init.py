from __future__ import annotations

import torch


def init_second_layer(m: int, *, device: torch.device | None = None, dtype: torch.dtype | None = None) -> torch.Tensor:
    """Initialise the second-layer weights as random ±1/m signs.

    Args:
        m: Number of neurons; determines output length.
        device: Target device for the returned tensor.
        dtype: Data type for the returned tensor (defaults to ``float32``).

    Returns:
        1-D tensor of shape ``(m,)`` with entries in ``{+1/m, -1/m}``.

    Raises:
        ValueError: If ``m`` is not positive.
    """
    if m <= 0:
        raise ValueError("m must be positive")
    signs = torch.randint(0, 2, (m,), device=device)
    signs = torch.where(signs == 0, torch.tensor(-1, device=device), torch.tensor(1, device=device))
    return (signs.to(dtype=dtype or torch.float32) / float(m))


def init_first_layer(
    m: int,
    d: int,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    normalize: bool = True,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Initialise the first-layer weight matrix from a standard Gaussian, optionally unit-normalised per neuron.

    Args:
        m: Number of neurons (rows of ``W``).
        d: Input dimension (columns of ``W``).
        device: Target device for the returned tensor.
        dtype: Data type for the returned tensor (defaults to ``float32``).
        normalize: If True, normalise each row to unit L2 norm.
        eps: Small constant added to norms before division to prevent division by zero.

    Returns:
        Weight matrix of shape ``(m, d)``.

    Raises:
        ValueError: If ``m`` or ``d`` is not positive.
    """
    if m <= 0 or d <= 0:
        raise ValueError("m and d must be positive")
    W = torch.randn((m, d), device=device, dtype=dtype or torch.float32)
    if normalize:
        norms = W.norm(dim=1, keepdim=True).clamp_min(eps)
        W = W / norms
    return W

