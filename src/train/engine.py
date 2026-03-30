from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import torch

from src.losses.metrics import classification_error, weight_norm
from src.losses.objectives import training_loss_exp
from src.optim.gd import apply_update_, gd_step
from src.optim.ngd import ngd_step
from src.optim.sngd import sngd_step
from src.train.batching import iterate_minibatches
from src.utils.io import save_csv, save_json


OptimType = Literal["gd", "ngd", "sngd"]


@dataclass(frozen=True)
class TrainParams:
    optim: OptimType
    eta: float
    steps: int
    eval_every: int = 1
    batch_size: int | None = None
    shuffle: bool = True
    seed: int = 0


def _full_loss(model, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return training_loss_exp(model(X), y)


def run_training(
    model,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    *,
    params: TrainParams,
    X_test: torch.Tensor | None = None,
    y_test: torch.Tensor | None = None,
    run_dir: str | Path | None = None,
    metadata: dict[str, Any] | None = None,
) -> list[dict[str, float]]:
    if params.steps <= 0:
        raise ValueError("steps must be positive")
    if params.optim == "sngd" and (params.batch_size is None or params.batch_size <= 0):
        raise ValueError("batch_size must be set and positive for sngd")

    g = torch.Generator(device="cpu").manual_seed(int(params.seed))
    metrics: list[dict[str, float]] = []
    batch_iter = None

    for t in range(params.steps + 1):
        with torch.no_grad():
            if t % params.eval_every == 0 or t == params.steps:
                tr_phi = model(X_train)
                tr_loss = training_loss_exp(tr_phi, y_train).item()
                row = {
                    "iter": float(t),
                    "train_loss": float(tr_loss),
                    "weight_norm": float(weight_norm(model.W.detach())),
                }
                if X_test is not None and y_test is not None:
                    te_phi = model(X_test)
                    row["test_error"] = float(classification_error(te_phi, y_test))
                metrics.append(row)

        if t == params.steps:
            break

        model.W.grad = None
        if params.optim == "gd":
            loss = _full_loss(model, X_train, y_train)
            loss.backward()
            grad = model.W.grad.detach().clone()
            W_next = gd_step(model.W.detach(), grad, eta=params.eta)
            apply_update_(model.W, W_next)
        elif params.optim == "ngd":
            loss = _full_loss(model, X_train, y_train)
            loss.backward()
            grad = model.W.grad.detach().clone()
            grad_norm = grad.norm()
            W_next = ngd_step(model.W.detach(), grad, eta_base=params.eta, F_scalar=grad_norm, eps=1e-12)
            apply_update_(model.W, W_next)
        elif params.optim == "sngd":
            full_loss = _full_loss(model, X_train, y_train).detach()
            if batch_iter is None:
                batch_iter = iter(
                    iterate_minibatches(
                        X_train,
                        y_train,
                        batch_size=int(params.batch_size),
                        shuffle=params.shuffle,
                        generator=g,
                    )
                )
            try:
                xb, yb = next(batch_iter)
            except StopIteration:
                batch_iter = iter(
                    iterate_minibatches(
                        X_train,
                        y_train,
                        batch_size=int(params.batch_size),
                        shuffle=params.shuffle,
                        generator=g,
                    )
                )
                xb, yb = next(batch_iter)
            batch_loss = _full_loss(model, xb, yb)
            batch_loss.backward()
            grad_b = model.W.grad.detach().clone()
            W_next = sngd_step(model.W.detach(), grad_b, eta_base=params.eta, F_full=full_loss)
            apply_update_(model.W, W_next)
        else:
            raise ValueError(f"Unknown optimizer: {params.optim!r}")

    if run_dir is not None:
        run_path = Path(run_dir)
        run_path.mkdir(parents=True, exist_ok=True)
        save_csv(run_path / "metrics.csv", metrics)
        save_json(
            run_path / "meta.json",
            {
                "params": asdict(params),
                "metadata": metadata or {},
                "n_train": int(X_train.shape[0]),
                "d": int(X_train.shape[1]),
            },
        )
    return metrics

