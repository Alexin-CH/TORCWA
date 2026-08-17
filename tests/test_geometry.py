import torch

import src as torcwa


def test_geometry_circle():
    geo = torcwa.geometry(Lx=1.0, Ly=1.0, nx=64, ny=64)
    m = geo.circle(R=0.3, Cx=0.5, Cy=0.5)
    assert m.shape == (64, 64)
    assert torch.isfinite(m).all()
    assert m.min() >= 0.0
    assert m.max() <= 1.0


def test_geometry_square():
    geo = torcwa.geometry(Lx=1.0, Ly=1.0, nx=32, ny=32)
    m = geo.square(W=0.4, Cx=0.5, Cy=0.5)
    assert m.shape == (32, 32)
    assert torch.isfinite(m).all()


def test_geometry_boolean_ops():
    geo = torcwa.geometry(Lx=1.0, Ly=1.0, nx=64, ny=64)
    a = geo.circle(R=0.3, Cx=0.5, Cy=0.5)
    b = geo.circle(R=0.2, Cx=0.5, Cy=0.5)
    assert torch.equal(geo.union(a, b), torch.maximum(a, b))
    assert torch.equal(geo.intersection(a, b), torch.minimum(a, b))
    assert torch.equal(geo.difference(a, b), torch.minimum(a, 1 - b))


def test_rcwa_geo_class_level():
    torcwa.rcwa_geo.Lx = 1.0
    torcwa.rcwa_geo.Ly = 1.0
    torcwa.rcwa_geo.nx = 32
    torcwa.rcwa_geo.ny = 32
    m = torcwa.rcwa_geo.circle(R=0.3, Cx=0.5, Cy=0.5)
    assert m.shape == (32, 32)
    assert torch.isfinite(m).all()