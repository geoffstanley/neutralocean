# neutralocean

<a href="https://neutralocean.readthedocs.org">
    <img alt="latest docs" src="https://img.shields.io/badge/documentation-latest-blue">
</a>

<a href="https://badge.fury.io/py/neutralocean">
    <img alt="pypi package" src="https://badge.fury.io/py/neutralocean.svg">
</a>

<a href="https://anaconda.org/conda-forge/neutralocean">
    <img alt="conda-forge package" src="https://img.shields.io/conda/vn/conda-forge/neutralocean.svg">
</a>

<a href="https://github.com/geoffstanley/neutralocean/actions/workflows/python-package.yml">
    <img alt="tests" src="https://github.com/geoffstanley/neutralocean/actions/workflows/python-package.yml/badge.svg">
</a>

`neutralocean` computes approximately neutral surfaces in the ocean, including
omega-surfaces, potential density surfaces, specific volume anomaly surfaces,
neutral trajectories, and neutrality diagnostics.

## What this package is for

The central workflow is omega-surface calculation following
[Stanley et al. (2021)](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2020MS002436),
with support routines for neutral analysis and trajectory calculations.

Topobaric surface software remains available in the original MATLAB toolbox:
[neutral-surfaces](https://github.com/geoffstanley/neutral-surfaces).

## Python compatibility

This project targets modern CPython versions:

- Python 3.12+

## Installation

### pip

```bash
pip install neutralocean
```

### conda-forge

```bash
conda install -c conda-forge neutralocean
```

### uv (development workflow)

```bash
uv sync
```

## Equation of state options

The package supports multiple EOS backends through `neutralocean.load_eos`.

- `"gsw_official"`: official Gibbs SeaWater (`gsw`) toolbox wrapper (in-situ density form).
- `"gswc"`: backward-compatible alias for `"gsw_official"`.
- `"gsw"`: bundled, numba-accelerated TEOS-10 75-term specific-volume polynomial.
- `"jmd95"`, `"jmdfwg06"`, `"polyTEOS10bsq"`: additional legacy or Boussinesq forms.

Examples:

```python
import neutralocean as no

# Fast bundled TEOS-10 polynomial (default used in most examples)
eos_fast = no.load_eos("gsw")

# Official GSW toolbox backend
eos_gsw = no.load_eos("gsw_official")
```

## Quickstart

```python
import neutralocean as no

# Create synthetic hydrography
S, T, Z, _ = no.data.synthocean((16, 32, 50), wrap=(False, False))
grid = no.grid.rectilinear.build_grid((16, 32), wrap=(False, False))

# Choose equation of state (default backend used by high-level routines)
eos = no.load_eos("gsw_official")
eos_s_t = no.load_eos("gsw_official", "_s_t")

# Compute omega-surface initialized from a pinned depth
s, t, z, diags = no.omega_surf(
    S,
    T,
    Z,
    grid,
    pin_cast=(8, 16),
    pin_p=2000.0,
    eos=eos,
    eos_s_t=eos_s_t,
    diags=True,
)
```

## Documentation

Full user and API docs:
<https://neutralocean.readthedocs.org>

## Citation

If you use this package, cite:

- Stanley, G. J., Barker, P. M., and McDougall, T. J. (2021),
  *Neutral surface topology and algorithms*,
  Journal of Advances in Modeling Earth Systems.
  <https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2020MS002436>
