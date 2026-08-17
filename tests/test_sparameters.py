import pytest
import torch

POLARIZATIONS = ["xx", "yx", "xy", "yy", "pp", "sp", "ps", "ss"]


def _pows(sim, orders, pol, ref_order=(0, 0)):
    r = sim.s_parameters(orders=orders, direction="forward", port="reflection",
                         polarization=pol, ref_order=ref_order, power_norm=True)
    t = sim.s_parameters(orders=orders, direction="forward", port="transmission",
                         polarization=pol, ref_order=ref_order, power_norm=True)
    return abs(r) ** 2, abs(t) ** 2


def test_all_polarizations_finite(solved_sim):
    sim = solved_sim(order=(1, 1), inc_ang=30.0)
    for pol in POLARIZATIONS:
        for port in ["reflection", "transmission"]:
            s = sim.s_parameters(orders=[0, 0], direction="forward", port=port,
                                 polarization=pol, ref_order=[0, 0])
            assert s.numel() == 1
            assert torch.isfinite(s), f"{pol}/{port} not finite: {s}"


def test_backward_direction_finite(solved_sim):
    sim = solved_sim(order=(1, 1), inc_ang=30.0)
    for pol in POLARIZATIONS:
        s = sim.s_parameters(orders=[0, 0], direction="backward", port="transmission",
                             polarization=pol, ref_order=[0, 0])
        assert torch.isfinite(s), f"{pol} not finite: {s}"


def test_invalid_polarization_raises(solved_sim):
    sim = solved_sim(order=(1, 1))
    with pytest.raises(ValueError):
        sim.s_parameters(orders=[0, 0], polarization="zx")


def test_invalid_direction_raises(solved_sim):
    sim = solved_sim(order=(1, 1))
    with pytest.raises(ValueError):
        sim.s_parameters(orders=[0, 0], direction="sideways")


def test_invalid_port_raises(solved_sim):
    sim = solved_sim(order=(1, 1))
    with pytest.raises(ValueError):
        sim.s_parameters(orders=[0, 0], port="sideways")


def test_single_interface_fresnel(build_sim):
    # Air -> n=1.5 interface at normal incidence: R = |(1-n)/(1+n)|^2 = 0.04
    sim = build_sim(order=(0, 0))
    sim.add_input_layer(eps=1.0)
    sim.add_output_layer(eps=2.25)
    sim.set_incident_angle(inc_ang=0.0, azi_ang=0.0)
    sim.source_planewave(amplitude=[1.0, 0.0], direction="f")
    sim.solve_global_smatrix()

    for pol in ["xx", "pp"]:
        r, t = _pows(sim, [0, 0], pol)
        assert abs(r.item() - 0.04) < 1e-3, pol
        assert abs(t.item() - 0.96) < 1e-3, pol
        assert abs((r + t).item() - 1.0) < 1e-3, pol


def test_lossless_slab_conservation(build_sim):
    # Lossless homogeneous slab: R + T must equal 1 (power conservation).
    sim = build_sim(order=(0, 0))
    sim.add_input_layer(eps=1.0)
    sim.add_output_layer(eps=1.0)
    sim.set_incident_angle(inc_ang=0.0, azi_ang=0.0)
    sim.add_layer(thickness=200.0, eps=2.25, mu=1.0)
    sim.source_planewave(amplitude=[1.0, 0.0], direction="f")
    sim.solve_global_smatrix()

    for pol in ["xx", "yy", "pp", "ss"]:
        r, t = _pows(sim, [0, 0], pol)
        assert abs((r + t).item() - 1.0) < 1e-2, pol


def test_evanescent_order_returns_zero(solved_sim):
    # Order (1,1) at 30 deg incidence, wl=800nm, period=1000nm is evanescent.
    sim = solved_sim(order=(1, 1), inc_ang=30.0)
    s = sim.s_parameters(orders=[1, 1], direction="forward", port="transmission",
                         polarization="xx", power_norm=True)
    assert s.item() == 0.0


def test_power_norm_false_returns_raw_amplitude(solved_sim):
    sim = solved_sim(order=(1, 1), inc_ang=30.0)
    raw = sim.s_parameters(orders=[0, 0], direction="forward", port="transmission",
                           polarization="xx", power_norm=False)
    normed = sim.s_parameters(orders=[0, 0], direction="forward", port="transmission",
                              polarization="xx", power_norm=True)
    assert torch.isfinite(raw)
    assert torch.isfinite(normed)


def test_grazing_diffraction_order_no_nan(build_sim):
    # Regression: wl=1500nm, period=1000nm, 30deg incidence puts order (-1,0)
    # exactly at grazing (kx^2+ky^2 == n^2, kz == 0), which made the E->H
    # transformation matrices singular (0/0 -> NaN).
    sim = build_sim(freq=1.0 / 1500.0, order=(1, 1))
    sim.add_input_layer(eps=1.0)
    sim.add_output_layer(eps=1.0)
    sim.set_incident_angle(inc_ang=30.0 * torch.pi / 180, azi_ang=0.0)
    assert torch.isfinite(sim.Vf).all(), "free-space E->H matrix is NaN"
    assert torch.isfinite(sim.Vi).all(), "input-layer E->H matrix is NaN"
    assert torch.isfinite(sim.Sin[0]).all(), "input-layer S-matrix is NaN"

    sim.add_layer(thickness=100.0, eps=2.25, mu=1.0)
    sim.source_planewave(amplitude=[1.0, 0.0], direction="f")
    sim.solve_global_smatrix()
    s = sim.s_parameters(orders=[0, 0], direction="forward", port="transmission",
                         polarization="xx", power_norm=True)
    assert torch.isfinite(s)