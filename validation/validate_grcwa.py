#!/usr/bin/env python
"""
External reference validation of TORCWA against grcwa (Green's function RCWA).

Both solvers share the same conventions:
  - time harmonic exp(-i*omega*t), speed of light = 1
  - geometry units in nm, freq = 1/lambda
  - incidence angles in radians, phi = 0 (x-z plane)

Excitation / normalization mapping (phi = 0):
  grcwa p_amp = 1   <->  TORCWA source_planewave(amplitude=[cos(theta), 0])
  grcwa s_amp = 1   <->  TORCWA source_planewave(amplitude=[0, 1])
  grcwa RT_Solve(normalize=1)  <->  TORCWA s_parameters(..., power_norm=True), |S|^2

Run:  python validation/validate_grcwa.py
Exit code 0 if every case passes, 1 otherwise.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import torch

import grcwa
import torcwa

L = 1000.0  # nm, lattice constant for all cases
TOL = 1e-4  # PASS tolerance (TORCWA runs in complex64)


# ---------------------------------------------------------------------------
# Shared builders
# ---------------------------------------------------------------------------
def _torcwa_case(order, eps_layers, d_list, *, eps_in=1.0, eps_out=1.0,
                 theta_deg=0.0, amp=(1.0, 0.0), lam=800.0):
    """Run TORCWA, return {(m, n): (R, T)} with power-normalized efficiencies."""
    torcwa.rcwa_geo.dtype = torch.float32
    torcwa.rcwa_geo.device = "cpu"
    torcwa.rcwa_geo.Lx = L
    torcwa.rcwa_geo.Ly = L
    torcwa.rcwa_geo.grid()
    sim = torcwa.rcwa(freq=1.0 / lam, order=order, lattice=(L, L))
    sim.add_input_layer(eps=eps_in)
    sim.add_output_layer(eps=eps_out)
    sim.set_incident_angle(
        inc_ang=torch.deg2rad(torch.tensor(theta_deg, dtype=torch.float32)),
        azi_ang=0.0,
    )
    for eps, d in zip(eps_layers, d_list):
        sim.add_layer(thickness=d, eps=torch.as_tensor(eps, dtype=torch.complex64))
    sim.source_planewave(amplitude=list(amp), direction="f")
    sim.solve()

    in_pols = []
    if amp[0] != 0.0:
        in_pols += ["xx", "yx"]
    if amp[1] != 0.0:
        in_pols += ["xy", "yy"]
    mx, my = order
    res = {}
    for m in range(-mx, mx + 1):
        for n in range(-my, my + 1):
            R = sum(
                abs(
                    sim.s_parameters(
                        orders=[m, n], direction="forward", port="reflection",
                        polarization=p, ref_order=[0, 0], power_norm=True,
                    ).cpu()
                ) ** 2
                for p in in_pols
            )
            T = sum(
                abs(
                    sim.s_parameters(
                        orders=[m, n], direction="forward", port="transmission",
                        polarization=p, ref_order=[0, 0], power_norm=True,
                    ).cpu()
                ) ** 2
                for p in in_pols
            )
            res[(m, n)] = (float(R), float(T))
    return res


def _grcwa_case(order, eps_layers, d_list, *, eps_in=1.0, eps_out=1.0,
                theta_deg=0.0, p_amp=1.0, s_amp=0.0, lam=800.0, nx=128, ny=128):
    """Run grcwa, return ((R_per_order, T_per_order), G)."""
    nG = (2 * order[0] + 1) * (2 * order[1] + 1)
    obj = grcwa.obj(nG=nG, L1=[L, 0.0], L2=[0.0, L], freq=1.0 / lam,
                    theta=np.deg2rad(theta_deg), phi=0.0)
    obj.Add_LayerUniform(thickness=0.0, epsilon=eps_in)
    for eps, d in zip(eps_layers, d_list):
        if np.ndim(eps) == 0:
            obj.Add_LayerUniform(thickness=d, epsilon=complex(eps))
        else:
            obj.Add_LayerGrid(thickness=d, Nx=nx, Ny=ny)
    obj.Add_LayerUniform(thickness=1e6, epsilon=eps_out)
    obj.Init_Setup(Pscale=1.0, Gmethod=1)
    ep_all = [np.asarray(eps, dtype=np.complex128).flatten()
              for eps in eps_layers if np.ndim(eps) == 2]
    if ep_all:
        obj.GridLayer_geteps(np.concatenate(ep_all))
    obj.MakeExcitationPlanewave(p_amp=p_amp, p_phase=0.0, s_amp=s_amp, s_phase=0.0,
                                order=0, direction="forward")
    R, T = obj.RT_Solve(normalize=1, byorder=1)
    return R, T, obj.G


def _pattern_grid(nx, ny, fn):
    """Sample fn(x, y) on the same cell-center grid used by TORCWA's rcwa_geo."""
    x = (L / nx) * (np.arange(nx) + 0.5)
    y = (L / ny) * (np.arange(ny) + 0.5)
    return fn(*np.meshgrid(x, y, indexing="ij"))


def _compare_orders(name, torcwa_res, Rg, Tg, G, tol=TOL):
    """Print a per-order table and return the maximum absolute difference."""
    rows = []
    for i, (m, n) in enumerate(G):
        Rt, Tt = torcwa_res[(int(m), int(n))]
        rows.append((int(m), int(n), Rt, Rg[i], Tt, Tg[i]))
    maxdiff = max(max(abs(r[2] - r[3]), abs(r[4] - r[5])) for r in rows)
    print(f"\n=== {name} (max per-order diff {maxdiff:.2e}) ===")
    print(f"{'order':>8} | {'torcwa R':>10} {'grcwa R':>10} {'diff':>10} | "
          f"{'torcwa T':>10} {'grcwa T':>10} {'diff':>10}")
    for m, n, Rt, Rg_, Tt, Tg_ in rows:
        print(f"({m:>3},{n:>3}) | {Rt:>10.6f} {Rg_:>10.6f} {abs(Rt-Rg_):>10.2e} | "
              f"{Tt:>10.6f} {Tg_:>10.6f} {abs(Tt-Tg_):>10.2e}")
    Rt_total = sum(v[0] for v in torcwa_res.values())
    Tt_total = sum(v[1] for v in torcwa_res.values())
    print(f"totals  | {Rt_total:>10.6f} {Rg.sum():>10.6f} | "
          f"{Tt_total:>10.6f} {Tg.sum():>10.6f} (R+T={Rg.sum()+Tg.sum():.6f})")
    return maxdiff


def _check(name, maxdiff, tol=TOL):
    ok = maxdiff <= tol
    print(f"    -> {'PASS' if ok else 'FAIL'} ({maxdiff:.2e} <= {tol:.0e})")
    return ok


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------
def case_slab(theta_deg, pol, name):
    """Bare slab vs analytic Fresnel vs grcwa."""
    n1, n2, lam, d = 1.0, 1.5, 800.0, 200.0
    th = np.deg2rad(theta_deg)
    c1 = np.cos(th)
    ct = np.sqrt(n2 ** 2 - n1 ** 2 * np.sin(th) ** 2) / n2
    if pol == "s":
        r12 = (n1 * c1 - n2 * ct) / (n1 * c1 + n2 * ct)
    else:
        r12 = (n2 * c1 - n1 * ct) / (n2 * c1 + n1 * ct)
    beta = 2 * np.pi / lam * n2 * ct * d
    r = r12 * (1 - np.exp(2j * beta)) / (1 - r12 ** 2 * np.exp(2j * beta))
    R_analytic = abs(r) ** 2

    if pol == "p":
        amp, pa, sa = (np.cos(th), 0.0), 1.0, 0.0
    else:
        amp, pa, sa = (0.0, 1.0), 0.0, 1.0
    order = (1, 1)
    rt = _torcwa_case(order, [n2 ** 2], [d], eps_in=1.0, eps_out=1.0,
                      theta_deg=theta_deg, amp=amp, lam=lam)
    Rg, Tg, G = _grcwa_case(order, [n2 ** 2], [d], eps_in=1.0, eps_out=1.0,
                            theta_deg=theta_deg, p_amp=pa, s_amp=sa, lam=lam)
    Rt_total = sum(v[0] for v in rt.values())
    maxdiff = max(abs(Rt_total - R_analytic), abs(Rg.sum() - R_analytic),
                  abs(Tg.sum() - (1 - R_analytic)))
    print(f"\n=== {name} (slab, theta={theta_deg} deg, {pol}-pol) ===")
    print(f"analytic Fresnel R={R_analytic:.6f} | torcwa R={Rt_total:.6f} | "
          f"grcwa R={Rg.sum():.6f} T={Tg.sum():.6f}")
    return _check(name, maxdiff)


def case_grating(theta_deg, name):
    """1D binary grating, per-order diffraction."""
    nx = ny = 128
    eps_lo, eps_hi, w, d, lam = 1.0, 4.0, 500.0, 200.0, 800.0
    order = (1, 1)
    grid = _pattern_grid(nx, ny, lambda x, y: np.where(x < w, eps_hi, eps_lo))
    eps_t = torch.from_numpy(grid.astype(np.complex64))
    rt = _torcwa_case(order, [eps_t], [d], theta_deg=theta_deg, lam=lam)
    Rg, Tg, G = _grcwa_case(order, [grid.astype(np.complex128)], [d],
                            theta_deg=theta_deg, lam=lam, nx=nx, ny=ny)
    md = _compare_orders(name, rt, Rg, Tg, G)
    return _check(name, md)


def case_pillar_2d(name):
    """2D circular pillar array, per-order diffraction."""
    nx = ny = 128
    eps_lo, eps_hi, radius, d, lam = 1.0, 6.0, 250.0, 200.0, 800.0
    order = (1, 1)
    grid = _pattern_grid(
        nx, ny, lambda x, y: np.where((x - L / 2) ** 2 + (y - L / 2) ** 2 < radius ** 2,
                                      eps_hi, eps_lo)
    )
    eps_t = torch.from_numpy(grid.astype(np.complex64))
    rt = _torcwa_case(order, [eps_t], [d], lam=lam)
    Rg, Tg, G = _grcwa_case(order, [grid.astype(np.complex128)], [d],
                            lam=lam, nx=nx, ny=ny)
    md = _compare_orders(name, rt, Rg, Tg, G)
    return _check(name, md)


def case_metasurface(name):
    """sin_tin-style 40-slice sinusoidal corrugation + lossy TiN base at 1500 nm."""
    eps_metal = complex(-13.807142, 6.250260)  # TiN n,k interpolated at 1500 nm
    eps_air, eps_sub = 1.0, 2.25
    lam, num_layers = 1500.0, 40
    amp_h, period = 55.0, 1000.0
    dz = 2 * amp_h / num_layers
    order = (3, 3)
    nx = ny = 256
    base = amp_h

    def height(x):
        return amp_h * np.sin(2 * np.pi * x / period) + amp_h

    def slice_mask(x, z_mid):
        return (height(x) >= (z_mid - base)).astype(float)

    eps_layers, d_list = [], []
    for i in range(num_layers):
        z_mid = (i + 0.5) * dz
        grid = _pattern_grid(
            nx, ny, lambda x, y: slice_mask(x, z_mid) * eps_metal + (1 - slice_mask(x, z_mid)) * eps_air
        )
        eps_layers.append(grid.astype(np.complex128))
        d_list.append(dz)
    eps_layers.append(np.full((nx, ny), eps_metal, dtype=np.complex128))
    d_list.append(100.0)

    eps_t = [torch.from_numpy(e.astype(np.complex64)) for e in eps_layers]
    rt = _torcwa_case(order, eps_t, d_list, eps_in=eps_air, eps_out=eps_sub, lam=lam)
    Rg, Tg, G = _grcwa_case(order, eps_layers, d_list, eps_in=eps_air,
                            eps_out=eps_sub, lam=lam, nx=nx, ny=ny)

    i00 = G.tolist().index([0, 0])
    Rt, Tt = rt[(0, 0)]
    print(f"\n=== {name} (40-slice corrugation, lossy metal, nG={len(G)}) ===")
    print(f"(0,0): torcwa R={Rt:.6f} T={Tt:.6f} | grcwa R={Rg[i00]:.6f} T={Tg[i00]:.6f}")
    print(f"totals: torcwa R={sum(v[0] for v in rt.values()):.6f} "
          f"T={sum(v[1] for v in rt.values()):.6f} | "
          f"grcwa R={Rg.sum():.6f} T={Tg.sum():.6f} (R+T={Rg.sum()+Tg.sum():.6f})")
    md = max(max(abs(rt[(int(m), int(n))][0] - Rg[i]),
                 abs(rt[(int(m), int(n))][1] - Tg[i])) for i, (m, n) in enumerate(G))
    return _check(name, md)


def main():
    results = []
    results.append(case_slab(0.0, "p", "slab-normal"))
    results.append(case_slab(30.0, "p", "slab-oblique-p"))
    results.append(case_slab(30.0, "s", "slab-oblique-s"))
    results.append(case_grating(0.0, "grating-normal"))
    results.append(case_grating(30.0, "grating-oblique-p"))
    results.append(case_pillar_2d("pillar-2d"))
    results.append(case_metasurface("sin_tin-style"))
    print(f"\n{'=' * 60}\nSummary: {sum(results)}/{len(results)} cases passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())