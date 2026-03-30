import torch

from src.models.functional import phi as phi_fn
from src.models.two_layer_fixed_a import TwoLayerFixedA


def test_forward_shapes_and_requires_grad():
    m, d, b = 4, 3, 5
    W = torch.randn(m, d)
    a = torch.ones(m) / m
    model = TwoLayerFixedA(W, a, alpha=0.2, ell=1.0)

    x = torch.randn(b, d)
    out = model(x)
    assert out.shape == (b,)
    assert model.W.requires_grad is True
    assert model.a.requires_grad is False


def test_forward_matches_functional():
    torch.manual_seed(0)
    m, d, b = 6, 4, 7
    W = torch.randn(m, d)
    a = torch.randn(m)
    x = torch.randn(b, d)

    model = TwoLayerFixedA(W, a, alpha=0.2, ell=1.0)
    out_m = model(x)
    out_f = phi_fn(W, a, x, alpha=0.2, ell=1.0)
    assert torch.allclose(out_m, out_f)

