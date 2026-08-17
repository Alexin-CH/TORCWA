import pytest
import torch


def test_field_xy_input_internal_output_shapes(solved_sim):
    sim = solved_sim(order=(1, 1), inc_ang=30.0)
    x = torch.linspace(-500, 500, 16)
    y = torch.linspace(-500, 500, 16)
    for layer in [-1, 0, sim.layer_N]:
        (Ex, Ey, Ez), (Hx, Hy, Hz) = sim.field_xy(layer, x, y, z_prop=0.0)
        for f in (Ex, Ey, Ez, Hx, Hy, Hz):
            assert f.shape == (16, 16)
            assert torch.isfinite(f).all(), f"layer {layer}"


def test_field_xy_invalid_layer_raises(build_sim):
    sim = build_sim(order=(1, 1))
    x = torch.linspace(-500, 500, 8)
    with pytest.raises(IndexError):
        sim.field_xy(99, x, x)
    with pytest.raises(IndexError):
        sim.field_xy(-2, x, x)


def test_field_xy_invalid_axis_type_raises(solved_sim):
    sim = solved_sim(order=(1, 1))
    with pytest.raises(TypeError):
        sim.field_xy(0, [-500, 500], [-500, 500])


def test_field_xz_shape(solved_sim):
    sim = solved_sim(order=(1, 1), inc_ang=30.0)
    x = torch.linspace(-500, 500, 16)
    z = torch.linspace(-200, 200, 20)
    (Ex, Ey, Ez), (Hx, Hy, Hz) = sim.field_xz(x, z, y=0.0)
    for f in (Ex, Ey, Ez, Hx, Hy, Hz):
        assert f.shape == (16, 20)
        assert torch.isfinite(f).all()


def test_field_yz_shape(solved_sim):
    sim = solved_sim(order=(1, 1), inc_ang=30.0)
    y = torch.linspace(-500, 500, 16)
    z = torch.linspace(-200, 200, 20)
    (Ex, Ey, Ez), (Hx, Hy, Hz) = sim.field_yz(y, z, x=0.0)
    for f in (Ex, Ey, Ez, Hx, Hy, Hz):
        assert f.shape == (16, 20)
        assert torch.isfinite(f).all()


def test_field_xz_region_batching_invariant(solved_sim):
    # Regression: computing the field for a z_axis split into chunks must give
    # identical columns to a single call (vectorized per-region computation).
    # Tolerance allows for float32 rounding differences across batch sizes;
    # the pre-fix column-interleaving bug produced O(1) errors.
    sim = solved_sim(order=(1, 1), inc_ang=30.0)
    x = torch.linspace(-500, 500, 8)
    z_full = torch.linspace(-200, 220, 30)
    split = 15
    (E, H) = sim.field_xz(x, z_full, y=0.0)
    (Ea, Ha) = sim.field_xz(x, z_full[:split], y=0.0)
    (Eb, Hb) = sim.field_xz(x, z_full[split:], y=0.0)
    for f, a, b in zip(E + H, Ea + Ha, Eb + Hb):
        assert torch.allclose(f[:, :split], a, rtol=1e-5, atol=1e-5)
        assert torch.allclose(f[:, split:], b, rtol=1e-5, atol=1e-5)


def test_field_yz_region_batching_invariant(solved_sim):
    sim = solved_sim(order=(1, 1), inc_ang=30.0)
    y = torch.linspace(-500, 500, 8)
    z_full = torch.linspace(-200, 220, 30)
    split = 15
    (E, H) = sim.field_yz(y, z_full, x=0.0)
    (Ea, Ha) = sim.field_yz(y, z_full[:split], x=0.0)
    (Eb, Hb) = sim.field_yz(y, z_full[split:], x=0.0)
    for f, a, b in zip(E + H, Ea + Ha, Eb + Hb):
        assert torch.allclose(f[:, :split], a, rtol=1e-5, atol=1e-5)
        assert torch.allclose(f[:, split:], b, rtol=1e-5, atol=1e-5)


def test_field_xz_unsorted_z_preserves_order(solved_sim):
    # The field at each z is well-defined; reordering the input z_axis must only
    # reorder the output columns.
    sim = solved_sim(order=(1, 1), inc_ang=30.0)
    x = torch.linspace(-500, 500, 8)
    z = torch.tensor([150.0, -100.0, 50.0, -50.0, 120.0])
    order = torch.argsort(z)
    (E, H) = sim.field_xz(x, z, y=0.0)
    (Es, Hs) = sim.field_xz(x, torch.sort(z).values, y=0.0)
    for f, fs in zip(E + H, Es + Hs):
        reordered = torch.empty_like(f)
        reordered[:, order] = fs
        assert torch.equal(f, reordered)


def test_field_xz_backward_source(solved_sim):
    sim = solved_sim(order=(1, 1), inc_ang=30.0)
    sim.source_fourier(amplitude=[1.0, 0.0], orders=[[0, 0]], direction="b", notation="xy")
    sim.solve_global_smatrix()
    x = torch.linspace(-500, 500, 16)
    z = torch.linspace(-200, 200, 20)
    (Ex, Ey, Ez), (Hx, Hy, Hz) = sim.field_xz(x, z, y=0.0)
    assert Ex.shape == (16, 20)
    assert torch.isfinite(Ex).all()


def test_source_planewave_sets_E_i(solved_sim):
    sim = solved_sim(order=(1, 1))
    assert sim.E_i.shape == (2 * sim.order_N, 1)
    assert torch.isfinite(sim.E_i).all()


def test_source_fourier_ps_backward(solved_sim):
    sim = solved_sim(order=(1, 1))
    sim.source_fourier(amplitude=[1.0, 0.0], orders=[[0, 0]], direction="b", notation="ps")
    assert sim.E_i.shape == (2 * sim.order_N, 1)
    assert torch.isfinite(sim.E_i).all()


def test_source_fourier_invalid_notation_raises(solved_sim):
    sim = solved_sim(order=(1, 1))
    with pytest.raises(ValueError):
        sim.source_fourier(amplitude=[1.0, 0.0], orders=[[0, 0]], notation="foo")