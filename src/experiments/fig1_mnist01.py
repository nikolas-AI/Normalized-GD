from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import torch

from src.data.mnist01 import get_mnist01_splits
from src.models.init import init_first_layer, init_second_layer
from src.models.two_layer_fixed_a import TwoLayerFixedA
from src.train.engine import TrainParams, run_training
from src.utils.io import make_run_dir, save_json
from src.utils.seed import set_seed


def _build_model(m: int, d: int, seed: int) -> TwoLayerFixedA:
    set_seed(seed, deterministic=True)
    W = init_first_layer(m, d, normalize=True)
    a = init_second_layer(m)
    return TwoLayerFixedA(W, a, alpha=0.2, ell=1.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mnist_root", type=str, default="data")
    ap.add_argument("--steps", type=int, default=100)
    args = ap.parse_args()

    set_seed(args.seed, deterministic=True)
    run_dir = make_run_dir(name=f"fig1_mnist01_seed{args.seed}")
    ds = get_mnist01_splits(root=args.mnist_root, n_train=1000, seed=args.seed)

    m = 50
    d = ds.X_train.shape[1]
    model_gd = _build_model(m, d, seed=args.seed)
    model_ngd = _build_model(m, d, seed=args.seed)

    gd_hist = run_training(
        model_gd,
        ds.X_train,
        ds.y_train,
        X_test=ds.X_test,
        y_test=ds.y_test,
        params=TrainParams(optim="gd", eta=30.0, steps=args.steps, eval_every=1, seed=args.seed),
        run_dir=run_dir / "gd",
        metadata={"figure": 1},
    )
    ngd_hist = run_training(
        model_ngd,
        ds.X_train,
        ds.y_train,
        X_test=ds.X_test,
        y_test=ds.y_test,
        params=TrainParams(optim="ngd", eta=5.0, steps=args.steps, eval_every=1, seed=args.seed),
        run_dir=run_dir / "ngd",
        metadata={"figure": 1},
    )

    it_gd = [int(r["iter"]) for r in gd_hist]
    it_ngd = [int(r["iter"]) for r in ngd_hist]

    plt.figure(figsize=(12, 3.8))
    plt.subplot(1, 3, 1)
    plt.semilogy(it_gd, [r["train_loss"] for r in gd_hist], label="GD")
    plt.semilogy(it_ngd, [r["train_loss"] for r in ngd_hist], label="Normalized GD")
    plt.title("Training Loss")
    plt.xlabel("Iteration")
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(it_gd, [100.0 * r.get("test_error", 0.0) for r in gd_hist], label="GD")
    plt.plot(it_ngd, [100.0 * r.get("test_error", 0.0) for r in ngd_hist], label="Normalized GD")
    plt.title("Test Error (%)")
    plt.xlabel("Iteration")
    plt.ylim(0, 1.5)
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(it_gd, [r["weight_norm"] for r in gd_hist], label="GD")
    plt.plot(it_ngd, [r["weight_norm"] for r in ngd_hist], label="Normalized GD")
    plt.title("Weight Norm")
    plt.xlabel("Iteration")
    plt.legend()

    plt.tight_layout()
    plt.savefig(run_dir / "figure1_like.png", dpi=160)
    plt.close()

    save_json(run_dir / "summary.json", {"gd_final": gd_hist[-1], "ngd_final": ngd_hist[-1]})
    print(f"wrote {run_dir}")


if __name__ == "__main__":
    main()

