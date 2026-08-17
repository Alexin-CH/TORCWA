import pytest
import torch


def test_solve_returns_self_for_chaining(solved_sim):
    sim = solved_sim()
    ret = sim.solve()
    assert ret is sim


def test_solve_matches_solve_global_smatrix(build_sim):
    def setup():
        sim = build_sim(order=(1, 1), freq=1.0 / 800.0)
        sim.add_input_layer(eps=1.0)
        sim.add_output_layer(eps=1.0)
        sim.set_incident_angle(inc_ang=30.0 * torch.pi / 180, azi_ang=0.0)
        sim.add_layer(thickness=100.0, eps=2.25, mu=1.0)
        sim.source_planewave(amplitude=[1.0, 0.0], direction="f")
        return sim

    via_solve = setup()
    via_solve.solve()
    via_direct = setup()
    via_direct.solve_global_smatrix()

    for a, b in zip(via_solve.S, via_direct.S):
        assert torch.equal(a, b)


def test_solve_requires_incident_angle(build_sim):
    sim = build_sim()
    sim.source_planewave()
    with pytest.raises(RuntimeError, match="set_incident_angle"):
        sim.solve()


def test_solve_requires_source(build_sim):
    sim = build_sim()
    sim.set_incident_angle(inc_ang=30.0 * torch.pi / 180, azi_ang=0.0)
    sim.add_layer(thickness=100.0, eps=2.25)
    with pytest.raises(RuntimeError, match="source"):
        sim.solve()


def test_solve_idempotent(solved_sim):
    sim = solved_sim()
    sim.solve()
    first = [s.clone() for s in sim.S]
    sim.solve()
    for a, b in zip(first, sim.S):
        assert torch.equal(a, b)


def test_solve_then_s_parameters(solved_sim):
    sim = solved_sim()
    s = sim.solve().s_parameters(
        orders=[0, 0],
        direction="forward",
        port="transmission",
        polarization="xx",
    )
    assert s.shape == (1,)
    assert torch.isfinite(s).all()