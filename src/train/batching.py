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
    """Yield successive ``(X_batch, y_batch)`` mini-batches from a dataset.

    Args:
        X: Feature matrix of shape ``(n, d)``.
        y: Label vector of shape ``(n,)``.
        batch_size: Number of samples per batch.
        shuffle: If True, samples are randomly permuted before batching.
        generator: Optional :class:`torch.Generator` for reproducible shuffling.

    Yields:
        Tuples of ``(X_batch, y_batch)`` with ``X_batch`` of shape ``(batch_size, d)``
        and ``y_batch`` of shape ``(batch_size,)`` (last batch may be smaller).

    Raises:
        ValueError: If shapes are inconsistent or ``batch_size`` is not positive.
    """
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

