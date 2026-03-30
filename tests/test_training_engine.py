import torch

from src.models.init import init_first_layer, init_second_layer
from src.models.two_layer_fixed_a import TwoLayerFixedA
from src.train.engine import TrainParams, run_training


def test_training_engine_runs_end_to_end_for_gd():
    torch.manual_seed(0)
    X = torch.randn(20, 3)
    y = torch.where(torch.randn(20) >= 0, torch.tensor(1.0), torch.tensor(-1.0))
    model = TwoLayerFixedA(init_first_layer(6, 3), init_second_layer(6))
    hist = run_training(model, X, y, params=TrainParams(optim="gd", eta=0.01, steps=5, eval_every=1))
    assert len(hist) == 6
    assert "train_loss" in hist[0]


def test_training_engine_runs_end_to_end_for_sngd():
    torch.manual_seed(0)
    X = torch.randn(20, 3)
    y = torch.where(torch.randn(20) >= 0, torch.tensor(1.0), torch.tensor(-1.0))
    model = TwoLayerFixedA(init_first_layer(6, 3), init_second_layer(6))
    hist = run_training(model, X, y, params=TrainParams(optim="sngd", eta=0.01, steps=5, batch_size=4, eval_every=1))
    assert len(hist) == 6
    assert "weight_norm" in hist[-1]

