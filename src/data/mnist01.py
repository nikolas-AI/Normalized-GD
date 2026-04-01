from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torchvision import datasets, transforms


@dataclass(frozen=True)
class Mnist01Splits:
    """Train/test split for the binary MNIST 0-vs-1 task.

    Attributes:
        X_train: Flattened training images of shape ``(n_train, 784)``, float32.
        y_train: Training labels in ``{+1, -1}`` of shape ``(n_train,)``, float32.
        X_test: Flattened test images of shape ``(n_test, 784)``, float32.
        y_test: Test labels in ``{+1, -1}`` of shape ``(n_test,)``, float32.
    """

    X_train: torch.Tensor  # (n_train, 784) float32
    y_train: torch.Tensor  # (n_train,) float32 in {+1,-1}
    X_test: torch.Tensor  # (n_test, 784) float32
    y_test: torch.Tensor  # (n_test,) float32 in {+1,-1}


def load_mnist(root: str | Path, *, train: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """Download (if needed) and load the full MNIST split into tensors.

    Args:
        root: Directory where MNIST data is stored or will be downloaded.
        train: If True, load the training set; otherwise load the test set.

    Returns:
        Tuple ``(X, y_digit)`` where ``X`` has shape ``(N, 1, 28, 28)`` float32
        and ``y_digit`` has shape ``(N,)`` int64 with digit class labels 0–9.
    """
    root_p = Path(root)
    tfm = transforms.ToTensor()  # returns float in [0,1], shape (1,28,28)
    ds = datasets.MNIST(root=str(root_p), train=train, download=True, transform=tfm)

    X_list = []
    y_list = []
    for x, y in ds:
        X_list.append(x)  # (1,28,28)
        y_list.append(y)

    X = torch.stack(X_list, dim=0).to(torch.float32)  # (N,1,28,28)
    y_digit = torch.tensor(y_list, dtype=torch.int64)  # (N,)
    return X, y_digit


def filter_mnist_01(X: torch.Tensor, y_digit: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep only digit-0 and digit-1 examples and remap labels to ``{-1, +1}``.

    Args:
        X: Full MNIST images of shape ``(N, 1, 28, 28)``.
        y_digit: Digit labels of shape ``(N,)``.

    Returns:
        Tuple ``(X01, y_pm1)`` containing only class-0/1 images and their
        ``±1`` labels (digit 1 → +1, digit 0 → −1).

    Raises:
        ValueError: If tensor shapes are not as expected.
    """
    if X.ndim != 4 or X.shape[1:] != (1, 28, 28):
        raise ValueError("X must have shape (N,1,28,28)")
    if y_digit.ndim != 1 or y_digit.shape[0] != X.shape[0]:
        raise ValueError("y_digit must have shape (N,)")

    mask = (y_digit == 0) | (y_digit == 1)
    X01 = X[mask]
    y01 = y_digit[mask]
    # Map: digit 1 -> +1, digit 0 -> -1
    y_pm1 = torch.where(y01 == 1, torch.tensor(1.0), torch.tensor(-1.0))
    return X01, y_pm1


def subsample(X: torch.Tensor, y: torch.Tensor, n: int, *, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Randomly subsample ``n`` examples from ``(X, y)`` without replacement.

    Args:
        X: Feature tensor of shape ``(N, ...)``.
        y: Label tensor of shape ``(N,)``.
        n: Number of examples to select.
        generator: :class:`torch.Generator` for reproducible sampling.

    Returns:
        Tuple ``(X_sub, y_sub)`` each with first dimension ``n``.

    Raises:
        ValueError: If ``n`` is not positive, shapes mismatch, or ``n > N``.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have matching first dimension")
    if n > X.shape[0]:
        raise ValueError("n cannot exceed dataset size")
    idx = torch.randperm(X.shape[0], generator=generator)[:n]
    return X[idx], y[idx]


def get_mnist01_splits(
    *,
    root: str | Path = "data",
    n_train: int = 1000,
    seed: int = 0,
) -> Mnist01Splits:
    """Load, filter, subsample, and flatten MNIST to produce a ready-to-use binary split.

    Args:
        root: Directory for MNIST data (downloaded if absent).
        n_train: Number of training examples to subsample from the filtered set.
        seed: Random seed for subsampling.

    Returns:
        :class:`Mnist01Splits` with flattened (784-D) images and ``±1`` labels.
    """
    g = torch.Generator().manual_seed(int(seed))

    Xtr, ytr_d = load_mnist(root, train=True)
    Xte, yte_d = load_mnist(root, train=False)

    Xtr01, ytr01 = filter_mnist_01(Xtr, ytr_d)
    Xte01, yte01 = filter_mnist_01(Xte, yte_d)

    Xtr01, ytr01 = subsample(Xtr01, ytr01, n_train, generator=g)

    # Flatten to (N,784) as used by the paper (d=784)
    Xtr_flat = Xtr01.view(Xtr01.shape[0], -1).contiguous()
    Xte_flat = Xte01.view(Xte01.shape[0], -1).contiguous()

    return Mnist01Splits(
        X_train=Xtr_flat,
        y_train=ytr01.to(torch.float32),
        X_test=Xte_flat,
        y_test=yte01.to(torch.float32),
    )

