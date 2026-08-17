import math

import pytest
import torch


def test_lattice_is_preserved(build_sim):
    # Regression: self.L used to be overwritten from a complex tensor back to
    # the raw lattice; it must keep the input value and type.
    lattice = [1000.0, 1000.0]
    sim = build_sim(lattice=lattice)
    assert sim.L == lattice
    assert not isinstance(sim.L, torch.Tensor)


def test_freq_and_omega(build_sim):
    sim = build_sim(freq=1.0 / 800.0)
    assert torch.isclose(sim.freq, torch.tensor(1.0 / 800.0, dtype=sim._dtype))
    assert math.isclose(sim.omega, 2 * math.pi / 800.0, rel_tol=1e-6)


def test_order_N(build_sim):
    sim = build_sim(order=(1, 2))
    assert sim.order_N == (2 * 1 + 1) * (2 * 2 + 1) == 15


def test_order_axes(build_sim):
    sim = build_sim(order=(1, 2))
    assert len(sim.order_x) == 3
    assert len(sim.order_y) == 5
    assert sim.order_x.tolist() == [-1, 0, 1]
    assert sim.order_y.tolist() == [-2, -1, 0, 1, 2]


def test_default_layers_free_space(build_sim):
    sim = build_sim()
    assert torch.isclose(sim.eps_in, torch.tensor(1.0, dtype=sim._dtype))
    assert torch.isclose(sim.mu_in, torch.tensor(1.0, dtype=sim._dtype))
    assert torch.isclose(sim.eps_out, torch.tensor(1.0, dtype=sim._dtype))
    assert torch.isclose(sim.mu_out, torch.tensor(1.0, dtype=sim._dtype))


def test_invalid_dtype_raises(build_sim):
    with pytest.raises(ValueError):
        build_sim(dtype=torch.float32)
    with pytest.raises(ValueError):
        build_sim(dtype=torch.complex32)


def test_instability_flags(build_sim):
    sim = build_sim(avoid_pinv_instability=True, max_pinv_instability=0.01)
    assert sim.avoid_Pinv_instability is True
    assert sim.max_Pinv_instability == 0.01
    sim2 = build_sim()
    assert sim2.avoid_Pinv_instability is False