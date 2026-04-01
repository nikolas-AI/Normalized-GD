from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SyntheticDataset:
    """Container for a synthetic binary classification dataset.

    Attributes:
        X: Feature matrix of shape ``(n, d)``, float32.
        y: Labels in ``{+1, -1}`` of shape ``(n,)``, float32.
    """

    X: torch.Tensor  # (n,d) float32
    y: torch.Tensor  # (n,) float32 in {+1,-1}


def _labels_pm1(n0: int, n1: int, *, device=None) -> torch.Tensor:
    """Build a label vector with ``n0`` entries of ``-1`` followed by ``n1`` entries of ``+1``."""
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
    """Generate the d=2 GMM dataset used in Fig. 2 (top) of the paper.

    Each class is a zero-mean Gaussian elongated along one diagonal direction,
    producing an X-shaped scatter plot. Class −1 is elongated along [1, −1]
    and class +1 along [1, 1].

    Args:
        n: Total number of samples (must be even).
        seed: Random seed.
        device: Target device for the returned tensors.

    Returns:
        :class:`SyntheticDataset` with ``X`` of shape ``(n, 2)``.
    """
    d = 2
    # X-shaped GMM: each class is a zero-mean Gaussian elongated along one diagonal.
    # Sigma = lambda_large * u*u^T + lambda_small * v*v^T
    # u = [1,1]/sqrt(2), v = [1,-1]/sqrt(2), lambda_large=4, lambda_small=0.1
    # Class -1: elongated along [1,-1]  -> Sigma0 = [[2.05, -1.95], [-1.95, 2.05]]
    # Class +1: elongated along [1, 1]  -> Sigma1 = [[2.05,  1.95], [ 1.95, 2.05]]
    Sigma0 = torch.tensor([[2.05, -1.95], [-1.95, 2.05]])
    Sigma1 = torch.tensor([[2.05,  1.95], [ 1.95, 2.05]])
    return gaussian_mixture_zero_mean(n=n, d=d, Sigma0=Sigma0, Sigma1=Sigma1, seed=seed, device=device)


def gaussian_mixture_d5_fig2_bottom(*, n: int = 40, seed: int = 0, device: torch.device | None = None) -> SyntheticDataset:
    """Generate the d=5 GMM dataset used in Fig. 2 (bottom) of the paper.

    Class −1 has covariance ``I`` and class +1 has covariance ``¼I``.

    Args:
        n: Total number of samples (must be even).
        seed: Random seed.
        device: Target device for the returned tensors.

    Returns:
        :class:`SyntheticDataset` with ``X`` of shape ``(n, 5)``.
    """
    d = 5
    Sigma0 = torch.eye(d)
    Sigma1 = 0.25 * torch.eye(d)
    return gaussian_mixture_zero_mean(n=n, d=d, Sigma0=Sigma0, Sigma1=Sigma1, seed=seed, device=device)


def x_shaped_d2_fig2_top(
    *,
    n: int = 40,
    noise_std: float = 0.2,
    t_scale: float = 4.0,
    seed: int = 0,
    device: torch.device | None = None,
) -> SyntheticDataset:
    """
    X-shaped binary dataset for Fig 2 (top, d=2).
    Class +1 (+1): x = t*[1,1] + noise, class -1 (-1): x = t*[1,-1] + noise.
    """
    if n % 2 != 0:
        raise ValueError("n must be even")
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    n_half = n // 2

    t0 = (torch.rand(n_half, generator=g, dtype=torch.float64) * 2 - 1) * t_scale
    t1 = (torch.rand(n_half, generator=g, dtype=torch.float64) * 2 - 1) * t_scale
    noise0 = torch.randn(n_half, 2, generator=g, dtype=torch.float64) * noise_std
    noise1 = torch.randn(n_half, 2, generator=g, dtype=torch.float64) * noise_std

    dir0 = torch.tensor([1.0, -1.0], dtype=torch.float64)
    dir1 = torch.tensor([1.0, 1.0], dtype=torch.float64)

    X0 = t0.unsqueeze(1) * dir0 + noise0
    X1 = t1.unsqueeze(1) * dir1 + noise1

    X = torch.cat([X0, X1], dim=0).to(torch.float32)
    y = _labels_pm1(n_half, n_half)

    if device is not None:
        X = X.to(device)
        y = y.to(device)
    return SyntheticDataset(X=X, y=y)


def x_shaped_d5_fig2_bottom(
    *,
    n: int = 40,
    noise_std: float = 0.1,
    t_scale: float = 1.5,
    seed: int = 0,
    device: torch.device | None = None,
) -> SyntheticDataset:
    """
    X-shaped dataset embedded in d=5 for Fig 2 (bottom).
    First 2 dims are the X-shape; last 3 dims are small Gaussian noise.
    """
    ds2 = x_shaped_d2_fig2_top(n=n, noise_std=noise_std, t_scale=t_scale, seed=seed)
    g = torch.Generator(device="cpu").manual_seed(int(seed) + 9999)
    extra = torch.randn(n, 3, generator=g, dtype=torch.float32) * noise_std
    X = torch.cat([ds2.X, extra], dim=1)
    if device is not None:
        X = X.to(device)
    return SyntheticDataset(X=X, y=ds2.y.to(device) if device is not None else ds2.y)


def xor_d2_fig3_bottom(
    *,
    n: int = 40,
    scale: float = 2.5,
    noise_std: float = 0.35,
    seed: int = 0,
    device: torch.device | None = None,
) -> SyntheticDataset:
    """
    XOR-style binary dataset for Fig 3 (bottom-left, d=2, n=40).
    Class +1: quadrants 1 (+,+) and 3 (-,-).
    Class -1: quadrants 2 (-,+) and 4 (+,-).
    Points are uniformly sampled in each quadrant then Gaussian noise is added.
    """
    if n % 4 != 0:
        raise ValueError("n must be divisible by 4")
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    q = n // 4  # points per quadrant

    def _quad(sx: float, sy: float) -> torch.Tensor:
        pts = torch.rand(q, 2, generator=g, dtype=torch.float64) * scale + 0.3
        return pts * torch.tensor([sx, sy], dtype=torch.float64)

    # Class -1 (label −1): Q2 (-,+) and Q4 (+,-)
    X_neg = torch.cat([_quad(-1, 1), _quad(1, -1)], dim=0)
    noise_neg = torch.randn(2 * q, 2, generator=g, dtype=torch.float64) * noise_std
    X_neg = X_neg + noise_neg

    # Class +1 (label +1): Q1 (+,+) and Q3 (-,-)
    X_pos = torch.cat([_quad(1, 1), _quad(-1, -1)], dim=0)
    noise_pos = torch.randn(2 * q, 2, generator=g, dtype=torch.float64) * noise_std
    X_pos = X_pos + noise_pos

    X = torch.cat([X_neg, X_pos], dim=0).to(torch.float32)
    y = _labels_pm1(2 * q, 2 * q)

    if device is not None:
        X = X.to(device)
        y = y.to(device)
    return SyntheticDataset(X=X, y=y)


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

