import torch

from src.train.batching import iterate_minibatches


def test_iterate_minibatches_cover_all_samples_no_shuffle():
    X = torch.arange(20).view(10, 2).to(torch.float32)
    y = torch.arange(10)
    seen = []
    for xb, yb in iterate_minibatches(X, y, batch_size=3, shuffle=False):
        seen.extend(yb.tolist())
        assert xb.shape[0] == yb.shape[0]
    assert seen == list(range(10))


def test_iterate_minibatches_reproducible_shuffle():
    X = torch.randn(10, 2)
    y = torch.arange(10)
    g1 = torch.Generator().manual_seed(123)
    g2 = torch.Generator().manual_seed(123)
    order1 = torch.cat([yb for _, yb in iterate_minibatches(X, y, 4, shuffle=True, generator=g1)])
    order2 = torch.cat([yb for _, yb in iterate_minibatches(X, y, 4, shuffle=True, generator=g2)])
    assert torch.equal(order1, order2)

