from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SyntheticDataset:
    X: torch.Tensor  # (n,d) float32
    y: torch.Tensor  # (n,) float32 in {+1,-1}


def _labels_pm1(n0: int, n1: int, *, device=None) -> torch.Tensor:
    y0 = -torch.ones(n0, device=device)
    y1 = torch.ones(n1, device=device)
    return torch.cat([y0, y1], dim=0).to(torch.float32)


def gaussian_mixture_zero_mean(
    *,
    n: int = 40,
    d: int = 2,
    Sigma0: torch.Tensor | None = None,
    Sigma1: torch.Tensor | None = None,
    seed: int = 0,
    device: torch.device | None = None,
) -> SyntheticDataset:
    """
    Paper Fig. 2 describes zero-mean GMM datasets.
    - Top: d=2, n=40, two classes with different covariance matrices (unspecified)
    - Bottom: d=5, n=40, Sigma0=I, Sigma1=1/4 I
    """
    if n <= 1 or n % 2 != 0:
        raise ValueError("n must be even and >= 2")
    if d <= 0:
        raise ValueError("d must be positive")

    g = torch.Generator(device="cpu").manual_seed(int(seed))
    n0 = n // 2
    n1 = n - n0

    if Sigma0 is None:
        Sigma0 = torch.eye(d)
    if Sigma1 is None:
        # Default chosen to create visibly different class geometry in d=2
        Sigma1 = torch.diag(torch.tensor([0.25] + [1.0] * (d - 1)))

    Sigma0 = Sigma0.to(torch.float64)
    Sigma1 = Sigma1.to(torch.float64)
    if Sigma0.shape != (d, d) or Sigma1.shape != (d, d):
        raise ValueError("Sigma matrices must have shape (d,d)")

    # Manual, generator-controlled MVN sampling: if Z~N(0,I), then X=Z L^T has cov LL^T = Sigma.
    L0 = torch.linalg.cholesky(Sigma0)
    L1 = torch.linalg.cholesky(Sigma1)
    Z0 = torch.randn((n0, d), generator=g, dtype=torch.float64)
    Z1 = torch.randn((n1, d), generator=g, dtype=torch.float64)
    X0 = Z0 @ L0.T
    X1 = Z1 @ L1.T
    X = torch.cat([X0, X1], dim=0).to(torch.float32)
    y = _labels_pm1(n0, n1)

    if device is not None:
        X = X.to(device)
        y = y.to(device)

    return SyntheticDataset(X=X, y=y)


def gaussian_mixture_d2_fig2_top(*, n: int = 40, seed: int = 0, device: torch.device | None = None) -> SyntheticDataset:
    d = 2
    # Different covariances (paper doesn't specify exact values)
    Sigma0 = torch.tensor([[1.0, 0.8], [0.8, 1.0]])
    Sigma1 = torch.tensor([[1.0, -0.6], [-0.6, 1.5]])
    return gaussian_mixture_zero_mean(n=n, d=d, Sigma0=Sigma0, Sigma1=Sigma1, seed=seed, device=device)


def gaussian_mixture_d5_fig2_bottom(*, n: int = 40, seed: int = 0, device: torch.device | None = None) -> SyntheticDataset:
    d = 5
    Sigma0 = torch.eye(d)
    Sigma1 = 0.25 * torch.eye(d)
    return gaussian_mixture_zero_mean(n=n, d=d, Sigma0=Sigma0, Sigma1=Sigma1, seed=seed, device=device)


def signed_linear_measurements(
    *,
    n: int = 100,
    d: int = 50,
    seed: int = 0,
    device: torch.device | None = None,
) -> tuple[SyntheticDataset, torch.Tensor]:
    """
    Paper Fig. 3 (Top): y = sign(x^T w*), d=50, n=100.
    Returns dataset and the ground-truth w*.
    """
    if n <= 0 or d <= 0:
        raise ValueError("n and d must be positive")
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    X = torch.randn((n, d), generator=g).to(torch.float32)
    w_star = torch.randn((d,), generator=g).to(torch.float32)
    scores = X @ w_star
    y = torch.where(scores >= 0, torch.tensor(1.0), torch.tensor(-1.0))
    if device is not None:
        X = X.to(device)
        y = y.to(device)
        w_star = w_star.to(device)
    return SyntheticDataset(X=X, y=y), w_star

