import torch

from src.optim.gd import gd_step
from src.optim.ngd import ngd_step, ngd_stepsize
from src.optim.sngd import sngd_step


def test_gd_step_formula():
    W = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    G = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
    eta = 0.5
    out = gd_step(W, G, eta)
    assert torch.allclose(out, W - eta * G)


def test_ngd_stepsize_inverse_scaling():
    eta = 5.0
    s1 = ngd_stepsize(eta, 2.0)
    s2 = ngd_stepsize(eta, 1.0)
    assert s2 == 2.0 * s1


def test_ngd_step_formula():
    W = torch.tensor([[1.0, -1.0]])
    G = torch.tensor([[0.5, -0.5]])
    eta_base = 4.0
    F = 2.0
    out = ngd_step(W, G, eta_base, F)
    assert torch.allclose(out, W - (eta_base / F) * G)


def test_sngd_normalizes_by_batch_grad_norm():
    W = torch.randn(3, 4)
    grad = torch.randn(3, 4)
    eta_base = 3.0
    out = sngd_step(W, grad, eta_base)
    grad_norm = grad.norm()
    expected = W - (eta_base / grad_norm) * grad
    assert torch.allclose(out, expected)

