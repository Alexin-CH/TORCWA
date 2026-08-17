# External reference validation

`validate_grcwa.py` validates the TORCWA RCWA implementation against an
independent external solver, **grcwa** (Green's function RCWA by Minsook Lee,
MIT license, <https://github.com/MinsooLee/grcwa>).

grcwa is a separate, pure-Python RCWA implementation. Agreement between the two
solvers therefore catches formulation, convention, and sign errors that a
second implementation sharing TORCWA's code could not.

## Install grcwa

```bash
pip install grcwa
```

## Run

```bash
python validation/validate_grcwa.py
```

Exit code is 0 if every case passes, 1 otherwise. TORCWA runs in
`complex64`, so the PASS tolerance is `1e-4`; typical per-order differences are
`~1e-6`.

## Convention mapping

Both solvers use the same physical conventions (verified numerically):

- time harmonic `exp(-i*omega*t)`, speed of light `c = 1`
- geometry units in nm, `freq = 1/lambda`
- incidence angles in radians, `phi = 0` (incidence in the x-z plane)

Excitation and normalization:

| grcwa                                  | TORCWA                                        |
|----------------------------------------|-----------------------------------------------|
| `p_amp = 1` (phi = 0)                  | `source_planewave(amplitude=[cos(theta), 0])` |
| `s_amp = 1` (phi = 0)                  | `source_planewave(amplitude=[0, 1])`          |
| `RT_Solve(normalize=1)` efficiencies   | `s_parameters(power_norm=True)`, sum of `|S|^2` over output polarizations |

To make the Fourier truncations identical, both solvers use the square
truncation `order=(m, m)` in TORCWA matching `nG = (2m+1)^2` with
`Gmethod=1` (parallelogramic) in grcwa.

## Cases

| Case                    | Description                                                         |
|-------------------------|---------------------------------------------------------------------|
| `slab-normal`           | bare slab, normal incidence, vs analytic Fresnel                    |
| `slab-oblique-p`        | bare slab, 30 deg, p-polarized, vs analytic Fresnel                 |
| `slab-oblique-s`        | bare slab, 30 deg, s-polarized, vs analytic Fresnel                 |
| `grating-normal`        | 1D binary grating, normal incidence, per-order efficiencies         |
| `grating-oblique-p`     | 1D binary grating, 30 deg, p-polarized, per-order efficiencies      |
| `pillar-2d`             | 2D circular pillar array, per-order efficiencies                    |
| `sin_tin-style`         | 40-slice sinusoidal corrugation + lossy TiN base at 1500 nm, nG=49  |

The `sin_tin-style` case replicates the metasurface geometry used in
`simulations/sin_tin/sintin_simulation.py` (same 40 slices, period 1000 nm,
amplitude 55 nm, TiN `n, k` interpolated at 1500 nm) but with a square
`(3, 3)` truncation so that both solvers use identical Fourier sets. Note the
lossy metal absorbs ~20% of the incident power in this case, which the two
solvers agree on.

## Notes

- S4 (Stanford Stratified Structure Solver) was considered as an alternative
  reference but requires a source build (its PyPI name is occupied by an
  unrelated tool), so grcwa was chosen for the low-friction pure-Python setup.
- The `tests/test_validation_grcwa.py` pytest wrapper runs a fast subset
  (slab + 1D grating) and is skipped automatically when grcwa is not installed.
