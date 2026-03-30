import torch

from src.models.init import init_first_layer, init_second_layer
from src.models.two_layer_fixed_a import TwoLayerFixedA
from src.optim.checks import one_step_descent_check


def test_one_step_descent_check_runs_for_ngd():
    torch.manual_seed(0)
    m, d, n = 5, 3, 20
    W = init_first_layer(m, d, normalize=False)
    a = init_second_layer(m)
    model = TwoLayerFixedA(W, a)
    X = torch.randn(n, d)
    y = torch.where(torch.randn(n) >= 0, torch.tensor(1.0), torch.tensor(-1.0))
    out = one_step_descent_check(model, X, y, optimizer_type="ngd", eta_base=0.1)
    assert out.eta_t > 0.0
    assert out.F_current > 0.0
    assert out.F_next > 0.0

