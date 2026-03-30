from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import torch

from src.data.mnist01 import get_mnist01_splits
from src.data.synthetic import (
    gaussian_mixture_d2_fig2_top,
    gaussian_mixture_d5_fig2_bottom,
    signed_linear_measurements,
)
from src.utils.io import make_run_dir, save_json
from src.utils.seed import set_seed


def _scatter2(X: torch.Tensor, y: torch.Tensor, *, title: str, path: str) -> None:
    X = X.detach().cpu()
    y = y.detach().cpu()
    plt.figure(figsize=(4, 4))
    mask_pos = y > 0
    plt.scatter(X[~mask_pos, 0], X[~mask_pos, 1], s=30, label="-1")
    plt.scatter(X[mask_pos, 0], X[mask_pos, 1], s=30, label="+1")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mnist_root", type=str, default="data")
    args = ap.parse_args()

    set_seed(args.seed, deterministic=True)
    run_dir = make_run_dir(name=f"phase3_data_seed{args.seed}")

    # MNIST 0/1 splits
    mn = get_mnist01_splits(root=args.mnist_root, n_train=1000, seed=args.seed)
    mn_stats = {
        "mnist01": {
            "X_train_shape": list(mn.X_train.shape),
            "y_train_unique": sorted(set(mn.y_train.cpu().tolist())),
            "X_test_shape": list(mn.X_test.shape),
            "y_test_unique": sorted(set(mn.y_test.cpu().tolist())),
        }
    }

    # Synthetic datasets for Fig.2 and Fig.3 setup
    ds2_top = gaussian_mixture_d2_fig2_top(seed=args.seed)
    ds2_bot = gaussian_mixture_d5_fig2_bottom(seed=args.seed)
    ds3_lin, w_star = signed_linear_measurements(seed=args.seed)

    _scatter2(ds2_top.X[:, :2], ds2_top.y, title="Synthetic GMM d=2 (Fig2 top)", path=str(run_dir / "fig2_top_scatter.png"))
    _scatter2(ds2_bot.X[:, :2], ds2_bot.y, title="Synthetic GMM d=5 (first 2 dims) (Fig2 bottom)", path=str(run_dir / "fig2_bottom_scatter.png"))

    stats = {
        **mn_stats,
        "synthetic": {
            "fig2_top": {"X_shape": list(ds2_top.X.shape), "y_unique": sorted(set(ds2_top.y.cpu().tolist()))},
            "fig2_bottom": {"X_shape": list(ds2_bot.X.shape), "y_unique": sorted(set(ds2_bot.y.cpu().tolist()))},
            "fig3_linear": {"X_shape": list(ds3_lin.X.shape), "y_unique": sorted(set(ds3_lin.y.cpu().tolist())), "w_star_shape": list(w_star.shape)},
        },
    }
    save_json(run_dir / "data_stats.json", stats)
    print(f"wrote {run_dir}")


if __name__ == "__main__":
    main()

