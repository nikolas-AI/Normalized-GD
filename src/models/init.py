from __future__ import annotations

import torch


def init_second_layer(m: int, *, device: torch.device | None = None, dtype: torch.dtype | None = None) -> torch.Tensor:
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
    if m <= 0 or d <= 0:
        raise ValueError("m and d must be positive")
    W = torch.randn((m, d), device=device, dtype=dtype or torch.float32)
    if normalize:
        norms = W.norm(dim=1, keepdim=True).clamp_min(eps)
        W = W / norms
    return W

