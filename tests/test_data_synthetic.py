import torch

from src.data.synthetic import (
    gaussian_mixture_d2_fig2_top,
    gaussian_mixture_d5_fig2_bottom,
    signed_linear_measurements,
)


def test_gaussian_mixture_d2_shapes_labels():
    ds = gaussian_mixture_d2_fig2_top(seed=0)
    assert ds.X.shape == (40, 2)
    assert ds.y.shape == (40,)
    assert set(ds.y.tolist()) == {-1.0, 1.0}


def test_gaussian_mixture_d5_shapes_labels():
    ds = gaussian_mixture_d5_fig2_bottom(seed=0)
    assert ds.X.shape == (40, 5)
    assert ds.y.shape == (40,)
    assert set(ds.y.tolist()) == {-1.0, 1.0}


def test_signed_linear_shapes_labels():
    ds, w_star = signed_linear_measurements(seed=0)
    assert ds.X.shape == (100, 50)
    assert ds.y.shape == (100,)
    assert w_star.shape == (50,)
    assert set(ds.y.tolist()) == {-1.0, 1.0}

