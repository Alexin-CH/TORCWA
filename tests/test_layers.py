import torch


def test_add_homogeneous_float_eps(build_sim):
    sim = build_sim()
    sim.set_incident_angle(inc_ang=0.0, azi_ang=0.0)
    sim.add_layer(thickness=50.0, eps=2.25, mu=1.0)
    assert sim.layer_N == 1
    assert sim.thickness == [50.0]
    assert sim.eps_conv[0].shape == (sim.order_N, sim.order_N)
    assert torch.allclose(sim.eps_conv[0], 2.25 * torch.eye(sim.order_N, dtype=sim._dtype))


def test_add_homogeneous_complex_mu(build_sim):
    # Regression: the homogeneous check used `isinstance(mu, float)` twice
    # instead of accepting a complex mu, causing mu.dim() to be called on a
    # python complex and raising AttributeError.
    sim = build_sim()
    sim.set_incident_angle(inc_ang=0.0, azi_ang=0.0)
    sim.add_layer(thickness=50.0, eps=2.25, mu=1.0 + 0j)
    assert sim.layer_N == 1
    assert torch.allclose(sim.mu_conv[0], (1.0 + 0j) * torch.eye(sim.order_N, dtype=sim._dtype))


def test_add_patterned_layer(build_sim):
    sim = build_sim(order=(1, 1))
    nx = ny = 16
    xg = torch.linspace(-500, 500, nx).reshape(-1, 1) * torch.ones(1, ny)
    yg = torch.linspace(-500, 500, ny).reshape(1, -1) * torch.ones(nx, 1)
    mask = (torch.sin(2 * torch.pi * xg / 1000.0) > 0).to(sim._dtype)
    layer_eps = mask * (-10 + 1j) + (1 - mask) * 1.0
    sim.set_incident_angle(inc_ang=0.0, azi_ang=0.0)
    sim.add_layer(thickness=40.0, eps=layer_eps, mu=1.0)
    assert sim.layer_N == 1
    assert sim.eps_conv[0].shape == (sim.order_N, sim.order_N)


def test_add_homogeneous_int_eps_mu(build_sim):
    # Regression: int eps/mu used to crash because .dim() was called on a
    # python int that was neither float nor complex.
    sim = build_sim()
    sim.set_incident_angle(inc_ang=0.0, azi_ang=0.0)
    sim.add_layer(thickness=50.0, eps=1, mu=1)
    assert sim.layer_N == 1
    assert torch.allclose(sim.eps_conv[0], torch.eye(sim.order_N, dtype=sim._dtype))
    assert torch.allclose(sim.mu_conv[0], torch.eye(sim.order_N, dtype=sim._dtype))


def test_add_input_output_layer(build_sim):
    sim = build_sim()
    sim.add_input_layer(eps=2.25)
    sim.add_output_layer(eps=4.0)
    assert torch.isclose(sim.eps_in, torch.tensor(2.25, dtype=sim._dtype))
    assert torch.isclose(sim.eps_out, torch.tensor(4.0, dtype=sim._dtype))


def test_return_layer_shape(build_sim, solved_sim):
    sim = solved_sim(order=(1, 1))
    eps_rec, mu_rec = sim.return_layer(0, nx=16, ny=16)
    assert eps_rec.shape == (16, 16)
    assert mu_rec.shape == (16, 16)
    assert torch.isfinite(eps_rec).all()


def test_material_conv_size(build_sim):
    sim = build_sim(order=(1, 1))
    material = torch.ones((16, 16), dtype=sim._dtype)
    conv = sim._material_conv(material)
    assert conv.shape == (sim.order_N, sim.order_N)