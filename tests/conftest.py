import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src as torcwa


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def dtype():
    return torch.complex64


@pytest.fixture
def build_sim():
    """Factory that returns a fresh (unsolved) RCWA simulation."""

    def _build(
        freq=1.0 / 800.0,
        order=(1, 1),
        lattice=(1000.0, 1000.0),
        dtype=torch.complex64,
        device=torch.device("cpu"),
        **kwargs,
    ):
        return torcwa.rcwa(
            freq=freq,
            order=list(order),
            lattice=list(lattice),
            dtype=dtype,
            device=device,
            **kwargs,
        )

    return _build


@pytest.fixture
def solved_sim(build_sim):
    """Simulation with an input layer, one homogeneous layer, angle, source and solved S-matrix."""

    def _solve(order=(1, 1), freq=1.0 / 800.0, inc_ang=30.0, eps_out=1.0):
        sim = build_sim(freq=freq, order=order)
        sim.add_input_layer(eps=1.0)
        sim.add_output_layer(eps=eps_out)
        sim.set_incident_angle(inc_ang=inc_ang * torch.pi / 180, azi_ang=0.0)
        sim.add_layer(thickness=100.0, eps=2.25, mu=1.0)
        sim.source_planewave(amplitude=[1.0, 0.0], direction="f")
        sim.solve()
        return sim

    return _solve