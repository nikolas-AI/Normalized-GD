from __future__ import annotations

from collections.abc import Iterator

import torch


def iterate_minibatches(
    X: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
    *,
    shuffle: bool = True,
    generator: torch.Generator | None = None,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    if X.ndim != 2:
        raise ValueError("X must be 2D")
    if y.ndim != 1:
        raise ValueError("y must be 1D")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have same number of rows")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    n = X.shape[0]
    if shuffle:
        idx = torch.randperm(n, generator=generator, device=X.device)
    else:
        idx = torch.arange(n, device=X.device)

    for s in range(0, n, batch_size):
        bidx = idx[s : s + batch_size]
        yield X[bidx], y[bidx]

