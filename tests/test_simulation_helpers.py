import torch

from simulations.sin_tin.sintin_simulation import RCWAArgs, get_s_parameters


def test_rcwa_args_accept_plain_floats():
    # Regression: RCWAArgs used to call .requires_grad_() directly on the
    # inputs, crashing when passed plain floats.
    args = RCWAArgs(
        wl=800.0, ang=30.0, nh=1, discretization=32,
        sin_amplitude=55.0, sin_period=1000.0, uni_layer_h=0.0,
    )
    assert torch.isclose(args.wl, torch.tensor(800.0, dtype=torch.float64))
    assert torch.isclose(args.ang, torch.tensor(30.0, dtype=torch.float64))
    assert args.wl.requires_grad


def test_s_parameters_method_name(solved_sim):
    # Regression: the public method is the lowercase `s_parameters`;
    # the old scripts called a nonexistent `S_parameters`.
    sim = solved_sim(order=(1, 1))
    assert hasattr(sim, "s_parameters")
    assert not hasattr(sim, "S_parameters")


def test_get_s_parameters_returns_all_combinations(solved_sim):
    # Regression: the loop variables were ignored, returning the same
    # forward/transmission/xx value 64 times under distinct keys.
    sim = solved_sim(order=(1, 1))
    params = get_s_parameters(sim)
    assert len(params) == 32
    assert len(set(params.keys())) == 32
    assert all(torch.isfinite(torch.tensor(v)) for v in params.values())