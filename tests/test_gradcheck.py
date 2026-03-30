import torch

from src.models.init import init_first_layer, init_second_layer
from src.tests.gradcheck_fd import gradcheck_autograd_vs_fd


def test_gradcheck_autograd_vs_fd_small():
    torch.manual_seed(0)
    m, d, n = 4, 3, 6
    W = init_first_layer(m, d, normalize=False)
    a = init_second_layer(m)
    X = torch.randn(n, d)
    y = torch.where(torch.randn(n) >= 0, torch.tensor(1.0), torch.tensor(-1.0))

    # Avoid exact zeros in pre-activations with high probability by random init;
    # gradcheck tolerance is modest due to nondifferentiability at 0.
    gradcheck_autograd_vs_fd(W, a, X, y, eps=1e-5, atol=1e-3, rtol=1e-3)

