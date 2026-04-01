from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import torch

from src.data.synthetic import signed_linear_measurements, x_shaped_d2_fig2_top
from src.losses.metrics import classification_error
from src.models.init import init_first_layer, init_second_layer
from src.models.linear import LinearBinary
from src.models.two_layer_fixed_a import TwoLayerFixedA
from src.train.engine import TrainParams, run_training
from src.utils.io import make_run_dir, save_json
from src.utils.seed import set_seed

# Per-batch-size learning rates from the paper legend
_ETA_LINEAR = {1: 0.35, 4: 0.2, 10: 0.2, 50: 0.15, 100: 0.1}
_ETA_NN = {1: 9, 4: 6, 10: 5, 20: 5, 40: 4}


def _linear_dataset(seed: int):
    """Build training and test tensors for the signed linear measurement problem (Fig. 3 top).

    Generates a 100-sample training set with ``d=50`` via :func:`signed_linear_measurements`
    and an independent 3 000-sample test set from the same ground-truth ``w*``.

    Args:
        seed: Random seed for both training and test generation.

    Returns:
        Tuple ``(X_train, y_train, X_test, y_test)`` as float32 tensors.
    """
    tr, w_star = signed_linear_measurements(n=100, d=50, seed=seed)
    g = torch.Generator(device="cpu").manual_seed(seed + 1)
    Xte = torch.randn((3000, 50), generator=g)
    yte = torch.where((Xte @ w_star) >= 0, torch.tensor(1.0), torch.tensor(-1.0))
    return tr.X, tr.y, Xte, yte


def main() -> None:
    """Run the Fig. 3 SNGD experiment across batch sizes and save a 2×2 summary plot."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps_linear", type=int, default=500)
    ap.add_argument("--steps_nn", type=int, default=1000)
    args = ap.parse_args()
    set_seed(args.seed, deterministic=True)
    run_dir = make_run_dir(name=f"fig3_sngd_seed{args.seed}")

    # Top: linear model (d=50, n=100, w0=0), SNGD with per-b eta
    Xtr, ytr, Xte, yte = _linear_dataset(args.seed)
    b_list_linear = [1, 4, 10, 50, 100]
    lin_curves = {}
    for b in b_list_linear:
        model = LinearBinary(d=50, init_zero=True)
        hist = run_training(
            model,
            Xtr,
            ytr,
            X_test=Xte,
            y_test=yte,
            params=TrainParams(optim="sngd", eta=_ETA_LINEAR[b], batch_size=b, steps=args.steps_linear, seed=args.seed),
            run_dir=run_dir / f"linear_b{b}",
            metadata={"figure": 3, "model": "linear"},
        )
        lin_curves[b] = hist

    # Bottom: two-layer NN on XOR dataset (d=2, n=40), SNGD with per-b eta
    ds = x_shaped_d2_fig2_top(seed=args.seed)
    b_list_nn = [1, 4, 10, 20, 40]
    nn_curves = {}
    for b in b_list_nn:
        set_seed(args.seed, deterministic=True)
        W = init_first_layer(50, 2, normalize=True)
        a = init_second_layer(50)
        model = TwoLayerFixedA(W, a, alpha=0.2, ell=1.0)
        hist = run_training(
            model,
            ds.X,
            ds.y,
            params=TrainParams(optim="sngd", eta=_ETA_NN[b], batch_size=b, steps=args.steps_nn, seed=args.seed),
            run_dir=run_dir / f"nn_b{b}",
            metadata={"figure": 3, "model": "two_layer_nn"},
        )
        nn_curves[b] = hist

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Top-left: linear training loss (log scale)
    ax = axes[0, 0]
    for b in b_list_linear:
        h = lin_curves[b]
        ax.semilogy([r["iter"] for r in h], [r["train_loss"] for r in h], label=f"b={b}, η={_ETA_LINEAR[b]}")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Training Loss")
    ax.legend(fontsize=8)

    # Top-right: test error (fraction, not percent, matching paper y-axis ~0.1–0.6)
    ax = axes[0, 1]
    for b in b_list_linear:
        h = lin_curves[b]
        ax.plot([r["iter"] for r in h], [r.get("test_error", float("nan")) for r in h], label=f"b={b}")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Test Error")
    ax.legend(fontsize=8)

    # Bottom-left: scatter of XOR dataset
    ax = axes[1, 0]
    mask = ds.y > 0
    ax.scatter(ds.X[~mask, 0].numpy(), ds.X[~mask, 1].numpy(),
               marker="x", color="tab:blue", s=60, linewidths=1.5)
    ax.scatter(ds.X[mask, 0].numpy(), ds.X[mask, 1].numpy(),
               marker="+", color="tab:green", s=80, linewidths=1.5)

    # Bottom-right: NN training loss (log scale)
    ax = axes[1, 1]
    for b in b_list_nn:
        h = nn_curves[b]
        ax.semilogy([r["iter"] for r in h], [r["train_loss"] for r in h], label=f"b={b}, η={_ETA_NN[b]}")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Training Loss")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(run_dir / "figure3_like.png", dpi=160)
    plt.close()

    summary = {
        "linear_final_test_error": {str(b): lin_curves[b][-1].get("test_error", None) for b in b_list_linear},
        "linear_final_train_loss": {str(b): lin_curves[b][-1]["train_loss"] for b in b_list_linear},
        "nn_final_train_loss": {str(b): nn_curves[b][-1]["train_loss"] for b in b_list_nn},
    }
    save_json(run_dir / "summary.json", summary)
    print(f"wrote {run_dir}")


if __name__ == "__main__":
    main()
