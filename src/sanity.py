from __future__ import annotations

import json
from dataclasses import asdict

import torch

from src.config.schema import (
    DataConfig,
    ExperimentConfig,
    LossConfig,
    ModelConfig,
    OptimizerConfig,
    TrainConfig,
)
from src.utils.io import make_run_dir, save_json
from src.utils.seed import set_seed


def _pick_device(choice: str) -> str:
    """Resolve a device string to a concrete ``"cpu"`` or ``"cuda"`` string.

    Args:
        choice: One of ``"cpu"``, ``"cuda"``, or ``"auto"``. ``"auto"`` selects
            CUDA when available, otherwise CPU.

    Returns:
        Resolved device string.

    Raises:
        ValueError: If ``choice`` is not a recognised option.
    """
    if choice == "cpu":
        return "cpu"
    if choice == "cuda":
        return "cuda"
    if choice == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    raise ValueError(f"Unknown device choice: {choice!r}")


def main() -> None:
    """Run a minimal sanity check: validate config, set seed, and verify determinism."""
    cfg = ExperimentConfig(
        model=ModelConfig(d=3, m=4, alpha=0.2, ell=1.0),
        loss=LossConfig(type="exp"),
        optim=OptimizerConfig(type="ngd", eta=5.0, batch_size=None),
        data=DataConfig(dataset="synthetic_linear", n_train=5, seed=0),
        train=TrainConfig(steps=1, eval_every=1, device="auto"),
        seed=0,
    )
    cfg.validate()

    set_seed(cfg.seed, deterministic=True)
    device = _pick_device(cfg.train.device)

    run_dir = make_run_dir()
    save_json(run_dir / "config.json", asdict(cfg) | {"resolved_device": device})

    # Determinism smoke test: same seed -> same tensor.
    set_seed(cfg.seed, deterministic=True)
    a = torch.randn(5)
    set_seed(cfg.seed, deterministic=True)
    b = torch.randn(5)
    print(json.dumps({"device": device, "deterministic_match": bool(torch.allclose(a, b))}))
    print(f"run_dir={run_dir}")


if __name__ == "__main__":
    main()

