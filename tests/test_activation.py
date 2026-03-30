import torch

from src.models.activations import leaky_relu, leaky_relu_prime


def test_leaky_relu_values():
    x = torch.tensor([-2.0, 0.0, 3.0])
    y = leaky_relu(x, alpha=0.2, ell=1.0)
    assert torch.allclose(y, torch.tensor([-0.4, 0.0, 3.0]))


def test_leaky_relu_prime_values():
    x = torch.tensor([-2.0, 0.0, 3.0])
    yp = leaky_relu_prime(x, alpha=0.2, ell=1.0)
    assert torch.allclose(yp, torch.tensor([0.2, 1.0, 1.0]))

