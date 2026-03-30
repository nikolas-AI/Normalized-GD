from __future__ import annotations

import argparse

import matplotlib.pyplot as plt

from src.data.synthetic import gaussian_mixture_d2_fig2_top, gaussian_mixture_d5_fig2_bottom
from src.models.init import init_first_layer, init_second_layer
from src.models.two_layer_fixed_a import TwoLayerFixedA
from src.train.engine import TrainParams, run_training
from src.utils.io import make_run_dir, save_json
from src.utils.seed import set_seed


def _run_case(X, y, *, m: int, eta_gd: float, eta_ngd: float, steps: int, seed: int):
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps_top", type=int, default=1000)
    ap.add_argument("--steps_bottom", type=int, default=2500)
    args = ap.parse_args()
    set_seed(args.seed, deterministic=True)
    run_dir = make_run_dir(name=f"fig2_synth_seed{args.seed}")

    top = gaussian_mixture_d2_fig2_top(seed=args.seed)
    bot = gaussian_mixture_d5_fig2_bottom(seed=args.seed)

    top_gd, top_ngd = _run_case(top.X, top.y, m=50, eta_gd=80.0, eta_ngd=30.0, steps=args.steps_top, seed=args.seed)
    bot_gd, bot_ngd = _run_case(bot.X, bot.y, m=100, eta_gd=350.0, eta_ngd=20.0, steps=args.steps_bottom, seed=args.seed)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, ds, ttl in [
        (axes[0, 0], top, "Synthetic d=2"),
        (axes[1, 0], bot, "Synthetic d=5 (first 2 dims)"),
    ]:
        mask = ds.y > 0
        ax.scatter(ds.X[~mask, 0], ds.X[~mask, 1], label="-1")
        ax.scatter(ds.X[mask, 0], ds.X[mask, 1], label="+1")
        ax.set_title(ttl)
        ax.legend()

    axes[0, 1].semilogy([r["iter"] for r in top_gd], [r["train_loss"] for r in top_gd], label="GD")
    axes[0, 1].semilogy([r["iter"] for r in top_ngd], [r["train_loss"] for r in top_ngd], label="Normalized GD")
    axes[0, 1].set_title("Top: Training Loss")
    axes[0, 1].legend()

    axes[1, 1].semilogy([r["iter"] for r in bot_gd], [r["train_loss"] for r in bot_gd], label="GD")
    axes[1, 1].semilogy([r["iter"] for r in bot_ngd], [r["train_loss"] for r in bot_ngd], label="Normalized GD")
    axes[1, 1].set_title("Bottom: Training Loss")
    axes[1, 1].legend()

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

