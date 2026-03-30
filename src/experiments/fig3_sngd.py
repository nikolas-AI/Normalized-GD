from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import torch

from src.data.synthetic import gaussian_mixture_d2_fig2_top, signed_linear_measurements
from src.losses.metrics import classification_error
from src.models.init import init_first_layer, init_second_layer
from src.models.linear import LinearBinary
from src.models.two_layer_fixed_a import TwoLayerFixedA
from src.train.engine import TrainParams, run_training
from src.utils.io import make_run_dir, save_json
from src.utils.seed import set_seed


def _linear_dataset(seed: int):
    tr, w_star = signed_linear_measurements(n=100, d=50, seed=seed)
    # Large synthetic test set from same w*
    g = torch.Generator(device="cpu").manual_seed(seed + 1)
    Xte = torch.randn((3000, 50), generator=g)
    yte = torch.where((Xte @ w_star) >= 0, torch.tensor(1.0), torch.tensor(-1.0))
    return tr.X, tr.y, Xte, yte


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps_linear", type=int, default=500)
    ap.add_argument("--steps_nn", type=int, default=1000)
    args = ap.parse_args()
    set_seed(args.seed, deterministic=True)
    run_dir = make_run_dir(name=f"fig3_sngd_seed{args.seed}")

    # Top: linear model (d=50,n=100,w0=0), batch sizes [1,4,10,50,100]
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
            params=TrainParams(optim="sngd", eta=0.5, batch_size=b, steps=args.steps_linear, seed=args.seed),
            run_dir=run_dir / f"linear_b{b}",
            metadata={"figure": 3, "model": "linear"},
        )
        lin_curves[b] = hist

    # Bottom: neural net on d=2 synthetic from Fig2 top, batch sizes [1,4,10,20,40]
    ds = gaussian_mixture_d2_fig2_top(seed=args.seed)
    b_list_nn = [1, 4, 10, 20, 40]
    nn_curves = {}
    for b in b_list_nn:
        W = init_first_layer(50, 2, normalize=True)
        a = init_second_layer(50)
        model = TwoLayerFixedA(W, a, alpha=0.2, ell=1.0)
        hist = run_training(
            model,
            ds.X,
            ds.y,
            params=TrainParams(optim="sngd", eta=0.5, batch_size=b, steps=args.steps_nn, seed=args.seed),
            run_dir=run_dir / f"nn_b{b}",
            metadata={"figure": 3, "model": "two_layer_nn"},
        )
        nn_curves[b] = hist

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    for b in b_list_linear:
        h = lin_curves[b]
        plt.semilogy([r["iter"] for r in h], [r["train_loss"] for r in h], label=f"b={b}")
    plt.title("Linear: Training Loss")
    plt.xlabel("Iteration")
    plt.legend()

    plt.subplot(1, 3, 2)
    for b in b_list_linear:
        h = lin_curves[b]
        plt.plot([r["iter"] for r in h], [100.0 * r.get("test_error", 0.0) for r in h], label=f"b={b}")
    plt.title("Linear: Test Error (%)")
    plt.xlabel("Iteration")
    plt.legend()

    plt.subplot(1, 3, 3)
    for b in b_list_nn:
        h = nn_curves[b]
        plt.semilogy([r["iter"] for r in h], [r["train_loss"] for r in h], label=f"b={b}")
    plt.title("NN(d=2): Training Loss")
    plt.xlabel("Iteration")
    plt.legend()

    plt.tight_layout()
    plt.savefig(run_dir / "figure3_like.png", dpi=160)
    plt.close()

    summary = {
        "linear_final_test_error_pct": {str(b): 100.0 * lin_curves[b][-1].get("test_error", 0.0) for b in b_list_linear},
        "linear_final_train_loss": {str(b): lin_curves[b][-1]["train_loss"] for b in b_list_linear},
        "nn_final_train_loss": {str(b): nn_curves[b][-1]["train_loss"] for b in b_list_nn},
    }
    save_json(run_dir / "summary.json", summary)
    print(f"wrote {run_dir}")


if __name__ == "__main__":
    main()

