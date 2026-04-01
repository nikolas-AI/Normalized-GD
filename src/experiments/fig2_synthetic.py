from __future__ import annotations

import argparse

import matplotlib.pyplot as plt

from src.data.synthetic import x_shaped_d2_fig2_top, gaussian_mixture_d5_fig2_bottom
from src.models.init import init_first_layer, init_second_layer
from src.models.two_layer_fixed_a import TwoLayerFixedA
from src.train.engine import TrainParams, run_training
from src.utils.io import make_run_dir, save_json
from src.utils.seed import set_seed


def _run_case(X, y, *, m: int, eta_gd: float, eta_ngd: float, steps: int, seed: int):
    """Train both GD and NGD on a single dataset case and return their metric histories.

    Args:
        X: Feature matrix of shape ``(n, d)``.
        y: Labels of shape ``(n,)`` in ``{+1, -1}``.
        m: Number of hidden neurons.
        eta_gd: Learning rate for GD.
        eta_ngd: Learning rate for Normalised GD.
        steps: Number of gradient steps for each optimiser.
        seed: Random seed for weight initialisation.

    Returns:
        Tuple ``(hist_gd, hist_ngd)`` of metric log lists from :func:`run_training`.
    """
    d = X.shape[1]
    set_seed(seed, deterministic=True)
    W0 = init_first_layer(m, d, normalize=True)
    a0 = init_second_layer(m)
    model_gd = TwoLayerFixedA(W0, a0, alpha=0.2, ell=1.0)
    model_ngd = TwoLayerFixedA(W0.clone(), a0.clone(), alpha=0.2, ell=1.0)
    hist_gd = run_training(model_gd, X, y, params=TrainParams(optim="gd", eta=eta_gd, steps=steps, seed=seed))
    hist_ngd = run_training(model_ngd, X, y, params=TrainParams(optim="ngd", eta=eta_ngd, steps=steps, seed=seed))
    return hist_gd, hist_ngd


def main() -> None:
    """Run the Fig. 2 experiment on both synthetic datasets and save a 2×2 comparison plot."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps_top", type=int, default=1000)
    ap.add_argument("--steps_bottom", type=int, default=2000)
    args = ap.parse_args()
    set_seed(args.seed, deterministic=True)
    run_dir = make_run_dir(name=f"fig2_synth_seed{args.seed}")

    top = x_shaped_d2_fig2_top(seed=args.seed)
    bot = gaussian_mixture_d5_fig2_bottom(seed=args.seed)

    top_gd, top_ngd = _run_case(top.X, top.y, m=50, eta_gd=8, eta_ngd=3, steps=args.steps_top, seed=args.seed)
    bot_gd, bot_ngd = _run_case(bot.X, bot.y, m=100, eta_gd=35, eta_ngd=2, steps=args.steps_bottom, seed=args.seed)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Scatter plots — match paper marker style (+/x)
    for ax, ds, ttl in [
        (axes[0, 0], top, " d=2, n=40"),
        (axes[1, 0], bot, "GMM d=5, n=40, Σ₁=I, Σ₂=¼I"),
    ]:
        mask = ds.y > 0
        ax.scatter(ds.X[~mask, 0].numpy(), ds.X[~mask, 1].numpy(),
                   marker="x", color="tab:blue", s=60, linewidths=1.5)
        ax.scatter(ds.X[mask, 0].numpy(), ds.X[mask, 1].numpy(),
                   marker="+", color="tab:green", s=80, linewidths=1.5)
        ax.set_title(ttl)

    # Loss plots — log scale, both optimizers
    for ax, gd_hist, ngd_hist, steps in [
        (axes[0, 1], top_gd, top_ngd, args.steps_top),
        (axes[1, 1], bot_gd, bot_ngd, args.steps_bottom),
    ]:
        ax.semilogy([r["iter"] for r in gd_hist], [r["train_loss"] for r in gd_hist],
                    color="tab:blue", label="GD")
        ax.semilogy([r["iter"] for r in ngd_hist], [r["train_loss"] for r in ngd_hist],
                    color="tab:orange", label="Normalized GD")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Training Loss")
        ax.legend()
    axes[0, 1].set_ylim(1e-4, 1e0)

    plt.tight_layout()
    plt.savefig(run_dir / "figure2_like.png", dpi=160)
    plt.close()
    save_json(
        run_dir / "summary.json",
        {"top_final": {"gd": top_gd[-1], "ngd": top_ngd[-1]}, "bottom_final": {"gd": bot_gd[-1], "ngd": bot_ngd[-1]}},
    )
    print(f"wrote {run_dir}")


if __name__ == "__main__":
    main()
