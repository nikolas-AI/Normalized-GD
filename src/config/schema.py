from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


OptimizerType = Literal["gd", "ngd", "sngd"]
LossType = Literal["exp"]
DatasetType = Literal["mnist01", "synthetic_gmm_d2", "synthetic_gmm_d5", "synthetic_linear"]


@dataclass(frozen=True)
class ModelConfig:
    d: int
    m: int
    alpha: float = 0.2
    ell: float = 1.0
    init_type: Literal["gaussian"] = "gaussian"
    init_norm: Literal["unit_per_neuron"] = "unit_per_neuron"


@dataclass(frozen=True)
class LossConfig:
    type: LossType = "exp"


@dataclass(frozen=True)
class OptimizerConfig:
    type: OptimizerType
    eta: float
    batch_size: int | None = None


@dataclass(frozen=True)
class DataConfig:
    dataset: DatasetType
    root: str | None = None
    n_train: int | None = None
    seed: int = 0


@dataclass(frozen=True)
class TrainConfig:
    steps: int
    eval_every: int = 1
    device: Literal["cpu", "cuda", "auto"] = "auto"


@dataclass(frozen=True)
class ExperimentConfig:
    model: ModelConfig
    loss: LossConfig
    optim: OptimizerConfig
    data: DataConfig
    train: TrainConfig
    seed: int = 0

    def validate(self) -> None:
        if self.model.m <= 0:
            raise ValueError("model.m must be positive")
        if self.model.d <= 0:
            raise ValueError("model.d must be positive")
        if self.optim.eta <= 0:
            raise ValueError("optim.eta must be positive")
        if self.train.steps <= 0:
            raise ValueError("train.steps must be positive")
        if self.optim.type not in ("gd", "ngd", "sngd"):
            raise ValueError(f"Unknown optim.type={self.optim.type!r}")
        if self.data.dataset not in (
            "mnist01",
            "synthetic_gmm_d2",
            "synthetic_gmm_d5",
            "synthetic_linear",
        ):
            raise ValueError(f"Unknown data.dataset={self.data.dataset!r}")
        if self.optim.type == "sngd":
            if self.optim.batch_size is None or self.optim.batch_size <= 0:
                raise ValueError("optim.batch_size must be set and positive for sngd")

