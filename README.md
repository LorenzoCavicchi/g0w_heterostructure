# g0w_heterostructure

Layer-resolved one-shot **G0W** self-energy, electronic spectral function and **quantum twisting microscope
(QTM)** differential conductance of van der Waals heterostructures.

The package computes, for an arbitrary stack of 2D sheets in layered (possibly gated and dispersive)
dielectrics:

1. the layer-resolved Coulomb interaction V(q, ω) of the stack;
2. the RPA dielectric matrix and screened interaction W(q, ω);
3. the full-frequency G0W self-energy Σ(k, ε) of a sample sheet, as a band matrix;
4. the spectral function A(k, ε) from the matrix Dyson equation;
5. the QTM conductance dI/dV(V, θ) and d²I/dV² (Wei *et al.*, PRB **112**, 155157 (2025), Eq. 33).

Two samples are worked out completely: doped **monolayer graphene** (SLG tip | hBN | SLG) with analytic Dirac
bands, and **twisted bilayer graphene** (SLG tip | hBN | TBG) with the Bistritzer–MacDonald/Koshino continuum
model. Theory, derivations, assumptions, results and validation are in **`docs/notes.pdf`**.

---

## 1. Installation

Requirements: Python ≥ 3.10 and internet access for `pip`. Everything, including `pybandstructure` (needed
only for the twisted-bilayer part), is installed from PyPI; `git` is not needed.

Tested on Linux with:

| Python | pybandstructure | NumPy | SciPy | Matplotlib |
|---|---|---|---|---|
| 3.13 | 2.1 (PyPI) | 2.5 | 1.18 | 3.11 |
| 3.10 | 2.1 (PyPI) | 2.2 | 1.15 | 3.10 |

With these versions all 43 tests pass, and all notebooks and scripts run and reproduce the stored results.

### Option A — pip and a virtual environment

```bash
cd g0w_heterostructure
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

Equivalently, in one step: `pip install -e ".[tbg,notebooks,test]"` (the same dependencies, declared in
`pyproject.toml`). Minimal installs:

| what you need | command |
|---|---|
| graphene (SLG) part, electrostatics, self-energy core | `pip install -e .` (NumPy, SciPy, Matplotlib only) |
| + twisted bilayer graphene | `pip install -e ".[tbg]"` (adds `pybandstructure`, `tqdm`, `h5py`) |
| + notebooks | `pip install -e ".[tbg,notebooks]"` |

### Option B — conda

```bash
cd g0w_heterostructure
conda env create -f environment.yml
conda activate g0w
```

This creates the environment `g0w` (Python 3.13), installs `pybandstructure` from PyPI with `pip` and this
package in editable mode.

### Jupyter kernel

If Jupyter is not run from the environment above (e.g. a system-wide JupyterLab), register the environment as a
kernel and select it in the notebook:

```bash
python -m ipykernel install --user --name g0w --display-name "Python (g0w)"
```

### About `pybandstructure`

The twisted-bilayer model is built with [`pybandstructure`](https://gitlab.com/itorre/bandstructure-calculation)
(I. Torre, P. Novelli; LGPL-3.0 according to its PyPI metadata), installed separately, not bundled:

```bash
pip install "pybandstructure>=2.1"
```

* `g0w.compat` restores the NumPy aliases (`np.int`, `np.float`, ...) and the `scipy.linalg.eigh(eigvals=...)`
  keyword that `pybandstructure` still uses, so current NumPy (≥ 2) and SciPy (≥ 1.14) work. The fixes are applied
  automatically by the `g0w` TBG modules; in your own scripts that call `pybandstructure` directly, get it with
  `pb = g0w.compat.import_pybandstructure()` instead of `import pybandstructure`.
* On Python ≥ 3.12, importing `pybandstructure` prints a few `SyntaxWarning: invalid escape sequence` lines and
  its band-structure routines show `tqdm` progress bars. Both come from `pybandstructure` and are harmless.
* **Optional — Hartree self-consistency of doped TBG** (`g0w.models.tbg_model.apply_hartree`). This single
  function needs the option `Band_Structure.from_hamiltonian(regularize_occupations=...)`, which is not in the
  PyPI release; no notebook, script or test uses it. To enable it, replace the PyPI version with the development
  commit the package was originally written against (needs `git`):
  ```bash
  pip install --force-reinstall --no-deps \
      "pybandstructure @ git+https://gitlab.com/itorre/bandstructure-calculation.git@3a785c659a0700c39febfce1000214229547f66a"
  ```
  or, without `git`, from the source archive of the same commit:
  `pip install --force-reinstall --no-deps https://gitlab.com/itorre/bandstructure-calculation/-/archive/3a785c659a0700c39febfce1000214229547f66a/bandstructure-calculation-3a785c659a0700c39febfce1000214229547f66a.tar.gz`
* To use a source checkout without installing it: `export PYBANDSTRUCTURE_PATH=/path/to/bandstructure-calculation`
  (the folder that contains the `pybandstructure` package).

### Check the installation

```bash
python -c "import g0w; from g0w.compat import import_pybandstructure as ip; print(g0w.__version__, ip().__file__)"
python -m pytest tests            # 43 tests, about 1.5 min
# or, without pytest:
python tests/run_tests.py
```

All tests should pass. Without `pybandstructure` the 16 TBG tests (`tests/test_tbg.py`) are skipped and the
other 27 pass.

---

## 2. Running the examples

### Notebooks

```bash
cd notebooks
jupyter lab
```

| notebook | content | time (1 core) |
|---|---|---|
| `01_geometry_and_coulomb.ipynb` | electrostatics of the junction, closed-form checks, gates, dispersive hBN | < 1 min |
| `02_SLG_G0W.ipynb` | graphene: χ⁰, plasmons, full-frequency Σ, spectral function, layer channels | ~ 12 min |
| `03_TBG_G0W.ipynb` | TBG: bands, DOS, form factors, loss function, band-matrix Σ, A(k, ε) | ~ 2–4 min |
| `04_QTM_SLG.ipynb` | QTM on graphene: twist kinematics, dI/dV(V, θ), d²I/dV², tip back-action | ~ 14 min |
| `05_QTM_TBG.ipynb` | QTM on TBG: tip trajectory, decoupled-layer check, dI/dV, d²I/dV², interband terms | ~ 6 min |
| `06_satellite_interpretation.ipynb` | where the satellite energies come from (emission edge, energy budget) | ~ 4 min |

The notebooks are stored executed, so the results can be read without running them. `notebooks/src/` holds
the same notebooks as plain Python scripts (`# %%` cells); run them from the package folder, e.g.
`python notebooks/src/03_TBG_G0W.py`. All figures of `docs/notes.pdf` are produced by these
notebooks (copies in `docs/figures/`).

### Command-line scripts

```bash
python scripts/run_slg_qtm.py --out slg_qtm           # graphene QTM map, ~7 min
python scripts/run_tbg_qtm.py --out tbg_qtm           # TBG QTM map, ~3 min at an 18x18 mesh
python scripts/run_tbg_qtm.py --help                  # all options (twist angle, mesh, broadenings, bias window, ...)
```

Each writes `<out>.npz` (bias, angle, dI/dV, d²I/dV², spectral data) and `<out>.png`.

### Performance

* Memory is dominated by the TBG form factors, which grow as N_s⁴ for an N_s × N_s mesh: 0.2 GB at N_s = 18,
  0.7 GB at 24, 1.7 GB at 30.
* When several notebooks or scripts run at once, set `OPENBLAS_NUM_THREADS=1` (or `OMP_NUM_THREADS=1`):
  competing BLAS threads slowed the TBG pipeline down by an order of magnitude in our tests.
* `joblib`, if installed, parallelises χ⁰ in `layer_helpers_G`.

---

## 3. Package layout

| path | content |
|---|---|
| `g0w/layer_coulomb.py` | layer-resolved Coulomb kernel (admittance matrix), gates, capped slab, self-tests |
| `g0w/layer_helpers.py` | `layer_helpers_G` and the fast `LayerHelpers`: form factors, χ⁰, ε, loss from `pybandstructure` |
| `g0w/dielectrics.py` | hBN, SiO₂, Al₂O₃ dielectric functions |
| `g0w/screening.py` | block-diagonal χ⁰ of a stack, ε, W, loss, layer-channel split |
| `g0w/self_energy_full_frequency.py` | full-frequency Σ, band-matrix Σ_nn′, tabulated resolvent kernels |
| `g0w/self_energy_ppa.py` | plasmon poles and residues, plasmon-pole Σ (for comparison), radial q weights |
| `g0w/spectral.py` | scalar and matrix Dyson equation, spectral function, quasiparticle solution |
| `g0w/qtm.py` | QTM kinematics, tunneling amplitudes, dI/dV and d²I/dV² for SLG and TBG samples |
| `g0w/satellites.py` | satellite analysis: plasmon branch, emission edge, energy-loss decomposition of −Im Σ |
| `g0w/models/dirac_chi.py` | closed-form χ⁰ of doped graphene |
| `g0w/models/slg_stack.py`, `slg_self_energy.py` | SLG\|hBN\|SLG junction, Dirac form factors, Σ, A |
| `g0w/models/tbg_model.py` | TBG continuum model, lab/model frames, meshes, Hartree solver, valley-summed χ⁰ |
| `g0w/models/tbg_stack.py`, `tbg_self_energy.py` | SLG\|hBN\|TBG stack, TBG band-matrix Σ |
| `g0w/compat.py` | NumPy/SciPy compatibility layer for `pybandstructure` |
| `notebooks/` | executed examples 01–06 and their script versions in `notebooks/src/` |
| `scripts/` | command-line drivers for the QTM maps |
| `tests/` | 43 regression and physics tests |
| `docs/notes.pdf`, `docs/notes.tex`, `docs/sections/`, `docs/figures/` | the notes and their sources |
| `docs/reference/` | development record of the corrections made while building the package; it refers to earlier internal scripts that are not part of this distribution |
| `pyproject.toml`, `requirements.txt`, `environment.yml` | installation |

---

## 4. Units and conventions

Energies in eV, lengths in nm, momenta in nm⁻¹, densities in nm⁻² (the scripts take cm⁻²), Gaussian units
with e² = 1.4399764 eV nm, angles in degrees at the interface. χ⁰ < 0 at ω = 0, ε = 1 − Vχ⁰, W = ε⁻¹V, all
evaluated at ω + iη. The spin–valley degeneracy g = 4 enters χ⁰, not the q weights of the self-energy. The
self-energy is the correlation part only (no Fock term); see the notes, Sec. 5.5, and the list of limitations
in Sec. 10.4.

## 5. Troubleshooting

| symptom | fix |
|---|---|
| `pybandstructure could not be imported` | `pip install "pybandstructure>=2.1"` in the active environment (Section 1) |
| `ImportError` for `tqdm` or `h5py` | `pip install tqdm h5py` (dependencies of `pybandstructure`) |
| `AttributeError: module 'numpy' has no attribute 'float'`, or `eigh() got an unexpected keyword argument 'eigvals'` | `pybandstructure` used directly without the compatibility fixes: use `pb = g0w.compat.import_pybandstructure()` (Section 1) |
| `ImportError: ... regularize_occupations` from `apply_hartree` | optional Hartree step: install the development commit (Section 1, *About `pybandstructure`*) |
| TBG cells run very slowly | another process is using the BLAS threads: set `OPENBLAS_NUM_THREADS=1` |
| `MemoryError` at large meshes | reduce `n_sites` (memory grows as N_s⁴) or `n_bands` |
| notebook cannot import `g0w` or `pybandstructure` | the notebook kernel is not the environment where you installed the package: select it, or register it (Section 1, *Jupyter kernel*) |

## 6. References the code builds on

* N. Wei, F. Guinea, F. von Oppen, L. I. Glazman, *Theory of plasmon spectroscopy with the quantum twisting
  microscope*, Phys. Rev. B **112**, 155157 (2025) — QTM conductance.
* R. Bistritzer, A. H. MacDonald, PNAS **108**, 12233 (2011); M. Koshino *et al.*, Phys. Rev. X **8**, 031087 (2018)
  — continuum model of TBG.
* P. Novelli, I. Torre, F. H. L. Koppens, F. Taddei, M. Polini, Phys. Rev. B **102**, 125403 (2020) —
  `pybandstructure`.

The complete bibliography is in `docs/notes.pdf`.

## 7. Author

Lorenzo Cavicchi (SISSA, Trieste) — lcavicch@sissa.it
