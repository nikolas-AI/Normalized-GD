from __future__ import annotations

import torch


def classification_error(phi: torch.Tensor, y: torch.Tensor) -> float:
    """Compute the fraction of misclassified examples.

    The predicted label is ``+1`` if ``phi >= 0``, else ``-1`` (``sign(0) = +1``).

    Args:
        phi: Model prediction scores of shape ``(n,)``.
        y: Ground-truth labels in ``{+1, -1}`` of shape ``(n,)``.

    Returns:
        Misclassification rate in ``[0, 1]``.

    Raises:
        ValueError: If tensors are not 1-D or have different lengths.
    """
    if phi.ndim != 1 or y.ndim != 1:
        raise ValueError("phi and y must be 1D tensors")
    if phi.shape[0] != y.shape[0]:
        raise ValueError("phi and y must have same length")
    # Convention: sign(0) = +1
    y_hat = torch.where(phi >= 0, torch.tensor(1, device=phi.device, dtype=y.dtype), torch.tensor(-1, device=phi.device, dtype=y.dtype))
    err = (y_hat != y).to(torch.float32).mean().item()
    return float(err)


def weight_norm(W: torch.Tensor) -> float:
    """Compute the Frobenius (L2) norm of a weight tensor.

    Args:
        W: Weight tensor of any shape.

    Returns:
        Scalar L2 norm as a Python float.
    """
    return float(W.flatten().norm().item())

