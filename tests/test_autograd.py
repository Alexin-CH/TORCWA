import torch

import src as torcwa


def test_gradient_flows_wrt_wavelength(build_sim):
    wl = torch.tensor(800.0, requires_grad=True)
    sim = build_sim(freq=1.0 / wl, order=(1, 1))
    sim.add_input_layer(eps=1.0)
    sim.add_output_layer(eps=1.0)
    sim.set_incident_angle(inc_ang=30.0 * torch.pi / 180, azi_ang=0.0)
    sim.add_layer(thickness=100.0, eps=2.25, mu=1.0)
    sim.source_planewave(amplitude=[1.0, 0.0], direction="f")
    sim.solve_global_smatrix()

    s = sim.s_parameters(orders=[0, 0], direction="forward", port="transmission",
                         polarization="xx", power_norm=True)
    loss = (s.abs() ** 2).real
    loss.backward()

    assert wl.grad is not None
    assert torch.isfinite(wl.grad)
    assert abs(wl.grad.item()) > 0.0


def test_gradient_flows_wrt_thickness(build_sim):
    thickness = torch.tensor(100.0, requires_grad=True)
    sim = build_sim(order=(1, 1))
    sim.add_input_layer(eps=1.0)
    sim.add_output_layer(eps=1.0)
    sim.set_incident_angle(inc_ang=0.0, azi_ang=0.0)
    sim.add_layer(thickness=thickness, eps=2.25, mu=1.0)
    sim.source_planewave(amplitude=[1.0, 0.0], direction="f")
    sim.solve_global_smatrix()

    s = sim.s_parameters(orders=[0, 0], direction="forward", port="reflection",
                         polarization="xx", power_norm=True)
    loss = (s.abs() ** 2).real
    loss.backward()

    assert thickness.grad is not None
    assert torch.isfinite(thickness.grad)


def test_stable_eig_grad_flag(build_sim):
    sim = build_sim(order=(1, 1), stable_eig_grad=False)
    sim.set_incident_angle(inc_ang=0.0, azi_ang=0.0)
    sim.add_layer(thickness=100.0, eps=2.25, mu=1.0)
    assert len(sim.kz_norm) == 1
    assert len(sim.E_eigvec) == 1