import math

import pytest
import torch

import torcwa


def _solve(freq, order, layer_eps, thickness, dtype=torch.complex64):
    """Solve a single lossless slab and return the reflected power |R|^2."""
    sim = torcwa.rcwa(freq=freq, order=list(order), lattice=[1000.0, 1000.0],
                      dtype=dtype, device="cpu")
    sim.add_input_layer(eps=1.0)
    sim.add_output_layer(eps=1.0)
    sim.set_incident_angle(inc_ang=0.0, azi_ang=0.0)
    sim.add_layer(thickness=thickness, eps=layer_eps, mu=1.0)
    sim.source_planewave(amplitude=[1.0, 0.0], direction="f")
    sim.solve_global_smatrix()
    s = sim.s_parameters(orders=[0, 0], direction="forward", port="reflection",
                         polarization="xx", power_norm=True)
    return (s.abs() ** 2).real


def _central_difference(f, t0, delta):
    return (f(t0 + delta) - f(t0 - delta)) / (2.0 * delta)


def test_gradient_finite_difference_homogeneous():
    # Plain-operation path (analytic eigenmodes, no custom Eig autograd).
    def loss(t):
        return _solve(1.0 / 800.0, [0, 0], 2.25, t)

    t0 = torch.tensor(120.0, dtype=torch.float32, requires_grad=True)
    analytic = loss(t0)
    analytic.backward()
    grad = t0.grad.item()
    numeric = _central_difference(loss, 120.0, 1e-2)
    assert abs(grad - numeric) <= 5e-2 * abs(numeric)


def test_gradient_finite_difference_inhomogeneous():
    # Path through the custom Eig autograd (inhomogeneous layer).
    nx = ny = 32
    xg = torch.linspace(-500, 500, nx).reshape(-1, 1) * torch.ones(1, ny)
    mask = (torch.sin(2 * torch.pi * xg / 1000.0) > 0).to(torch.complex64)
    layer_eps = mask * (-10 + 1j) + (1 - mask) * 1.0

    def loss(t):
        return _solve(1.0 / 800.0, [1, 1], layer_eps, t)

    t0 = torch.tensor(120.0, dtype=torch.float32, requires_grad=True)
    analytic = loss(t0)
    analytic.backward()
    grad = t0.grad.item()
    numeric = _central_difference(loss, 120.0, 1e-2)
    assert abs(grad - numeric) <= 5e-2 * abs(numeric)


def test_gradient_finite_difference_material_parameter():
    # Differentiate through the inhomogeneous eigenproblem itself, exercising
    # the custom Eig autograd backward pass (eigenvalue AND eigenvector grads).
    nx = ny = 32
    xg = torch.linspace(-500, 500, nx).reshape(-1, 1) * torch.ones(1, ny)
    pattern = (torch.sin(2 * torch.pi * xg / 1000.0) > 0).to(torch.complex64)

    def loss(alpha):
        layer_eps = (1.0 + alpha * pattern).to(torch.complex64)
        return _solve(1.0 / 800.0, [1, 1], layer_eps, torch.tensor(120.0))

    alpha0 = torch.tensor(5.0, dtype=torch.float32, requires_grad=True)
    analytic = loss(alpha0)
    analytic.backward()
    grad = alpha0.grad.item()
    numeric = _central_difference(loss, 5.0, 1e-2)
    assert abs(grad - numeric) <= 5e-2 * abs(numeric)


def test_complex128_end_to_end():
    sim = torcwa.rcwa(freq=1.0 / 800.0, order=[1, 1], lattice=[1000.0, 1000.0],
                      dtype=torch.complex128, device="cpu")
    sim.add_input_layer(eps=1.0)
    sim.add_output_layer(eps=1.0)
    sim.set_incident_angle(inc_ang=30.0 * torch.pi / 180, azi_ang=0.0)
    sim.add_layer(thickness=100.0, eps=2.25, mu=1.0)
    sim.source_planewave(amplitude=[1.0, 0.0], direction="f")
    sim.solve_global_smatrix()
    for pol in ["xx", "yx", "xy", "yy", "pp", "sp", "ps", "ss"]:
        s = sim.s_parameters(orders=[0, 0], direction="forward", port="transmission",
                             polarization=pol, power_norm=True)
        assert s.dtype == torch.complex128
        assert torch.isfinite(s)


def test_complex128_matches_complex64():
    r64 = _solve(1.0 / 800.0, [0, 0], 2.25, torch.tensor(120.0)).item()
    r128 = _solve(1.0 / 800.0, [0, 0], 2.25, torch.tensor(120.0),
                  dtype=torch.complex128).item()
    assert abs(r64 - r128) <= 1e-4