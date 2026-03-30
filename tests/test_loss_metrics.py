import math

import torch

from src.losses.exp_loss import exp_loss
from src.losses.metrics import classification_error, weight_norm
from src.losses.objectives import training_loss_exp


def test_exp_loss_scalar():
    assert torch.allclose(exp_loss(torch.tensor(0.0)), torch.tensor(1.0))
    assert torch.allclose(exp_loss(torch.tensor(math.log(2.0))), torch.tensor(0.5), atol=1e-6)


def test_training_loss_exp_basic():
    phi = torch.tensor([0.0, 0.0])
    y = torch.tensor([1.0, -1.0])
    loss = training_loss_exp(phi, y)
    assert torch.allclose(loss, torch.tensor(1.0))


def test_classification_error_convention_sign0_plus1():
    phi = torch.tensor([0.0, -1.0, 2.0])
    y = torch.tensor([1, -1, 1])
    assert classification_error(phi, y) == 0.0


def test_weight_norm_matches_flatten_norm():
    W = torch.randn(4, 3)
    assert math.isclose(weight_norm(W), float(W.flatten().norm().item()), rel_tol=0, abs_tol=1e-12)

