import torch

from src.data.mnist01 import filter_mnist_01


def test_filter_mnist_01_maps_labels_to_pm1():
    # Fake MNIST-like tensors
    X = torch.zeros(5, 1, 28, 28)
    y = torch.tensor([0, 1, 2, 1, 0])
    X01, ypm1 = filter_mnist_01(X, y)
    assert X01.shape[0] == 4
    assert ypm1.shape == (4,)
    assert set(ypm1.tolist()) == {-1.0, 1.0}
    # Expected mapping: 1 -> +1, 0 -> -1
    assert ypm1.tolist() == [-1.0, 1.0, 1.0, -1.0]

