import pytest
import torch


def test_matching_indices(build_sim):
    sim = build_sim(order=(1, 1))
    orders = torch.tensor([[-1, -1], [0, 0], [1, 1]], dtype=torch.int64)
    idx = sim._matching_indices(orders.clone())
    assert idx.tolist() == [0, 4, 8]


def test_matching_indices_clamps_out_of_range(build_sim):
    sim = build_sim(order=(1, 1))
    orders = torch.tensor([[5, -5], [-3, 3]], dtype=torch.int64)
    idx = sim._matching_indices(orders.clone())
    # clamped to [-1, 1]^2 before indexing: (1,-1) -> 6, (-1,1) -> 2
    assert idx.tolist() == [6, 2]


def test_matching_indices_does_not_mutate_input(build_sim):
    # Regression: _matching_indices used to clamp orders in place, corrupting
    # the caller's tensor (torch.as_tensor does not copy).
    sim = build_sim(order=(1, 1))
    orders = torch.tensor([[5, -5]], dtype=torch.int64)
    orders_orig = orders.clone()
    sim._matching_indices(orders)
    assert torch.equal(orders, orders_orig)


def test_matching_indices_accepts_python_list(build_sim):
    sim = build_sim(order=(1, 1))
    idx = sim._matching_indices([[0, 0]])
    assert idx.tolist() == [4]


def test_diffraction_angle_degree(build_sim, solved_sim):
    # Regression: unit="degree" used to raise NameError because `pi` was not imported.
    sim = solved_sim(order=(1, 1), inc_ang=30.0)
    inc, azi = sim.diffraction_angle(orders=[[0, 0]], layer="output", unit="degree")
    assert abs(inc.item() - 30.0) < 0.5
    assert abs(azi.item()) < 0.5


def test_diffraction_angle_radian(build_sim, solved_sim):
    sim = solved_sim(order=(1, 1), inc_ang=30.0)
    inc, azi = sim.diffraction_angle(orders=[[0, 0]], layer="output", unit="radian")
    assert abs(inc.item() - 30.0 * torch.pi / 180) < 1e-3


def test_diffraction_angle_invalid_unit_raises(build_sim, solved_sim):
    sim = solved_sim(order=(1, 1))
    with pytest.raises(ValueError):
        sim.diffraction_angle(orders=[[0, 0]], layer="output", unit="foo")


def test_diffraction_angle_invalid_layer_raises(build_sim, solved_sim):
    sim = solved_sim(order=(1, 1))
    with pytest.raises(ValueError):
        sim.diffraction_angle(orders=[[0, 0]], layer="middle")


def test_set_incident_angle_kvectors(build_sim):
    sim = build_sim()
    sim.set_incident_angle(inc_ang=30.0 * torch.pi / 180, azi_ang=0.0)
    # kx0 = n_in * sin(theta) with n_in = 1
    assert abs(sim.kx0_norm.item() - 0.5) < 1e-6
    assert abs(sim.ky0_norm.item()) < 1e-6
    assert sim.Vf.shape == (2 * sim.order_N, 2 * sim.order_N)