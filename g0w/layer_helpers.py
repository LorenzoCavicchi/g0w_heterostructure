#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Layer-resolved RPA dielectric matrix of a multilayer crystal (pybandstructure).

(Source: ``layer_resolved/test_v2_7/Layer_Helpers_plasmons_v2_7.py`` v2.7.1; the
fast chi0 kernel ``LayerHelpers`` is from ``tbg_hbn_encapsulated_loss``.)

The class :class:`layer_helpers_G` couples a pybandstructure ``Crystal`` with
the layer-resolved Coulomb kernel :class:`~g0w.layer_coulomb.LayerCoulomb`:

* ``densityL[l, n, m, k, q, G] = <k n | rho_l(q + G) | k+q m>`` -- matrix
  elements of the density operator projected on sheet l (``compute_density_layer``);
* ``chi0_{l l'}(q+G, q+G'; w)`` -- independent-particle polarizability
  (``polarization_layered``/``polarization_tot``)::

      chi0 = C sum_{n m k} w_k (f_{nk} - f_{m,k+q}) rho^l_{nm} (rho^{l'}_{nm})^* / (E_{nk} - E_{m,k+q} + w + i eta)

* ``eps_{(l G),(l' G')}(q, w) = delta delta - sum_m V_{l m}(q+G, w) chi0_{m l'}(q+G, q+G'; w)``
  (``dielectric_function_layer``), i.e. eps = I - V chi0, and the loss
  L = -Im Tr[V_00 eps^-1_00] (``loss_function_trace``).

Conventions: retarded response at w + i eta; W = eps^-1 V (not V eps^-1);
the flattened (layer, G) index is ``l * n_G + G`` everywhere.

Fixes with respect to v2.7.1 (``docs/notes.pdf``, App. "Changes"):

* ``dielectric_function_layer(retarded=False)`` now really evaluates chi0 at
  w - i eta.  (In v2.7.1 it was a no-op, i.e. identical to ``retarded=True``:
  pass ``retarded=True`` to reproduce old results.)
* ``dielectric_function_layer(contour=...)`` serial branch: chi0 was evaluated
  at the local index 0..n-1 instead of the contour q indices, so chi0 and V came
  from different q points.  Fixed; ``legacy_contour_indexing=True`` restores the
  old behaviour for regression.
* ``LayerHelpers``: bounded-memory, vectorised chi0 (same formula).
"""

from typing import Callable, Optional, Sequence, Tuple, Union, Any
import warnings
import numpy as np

# Optional parallel + progress bar
try:
    from joblib import Parallel, delayed
    HAS_JOBLIB = True
except Exception:
    HAS_JOBLIB = False
    Parallel = None
    delayed = None

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x

from .layer_coulomb import (
    LayerCoulomb,
    TWOPI_E2_EV_NM,
    compute_kappa_Y,
    compute_kappa_Y_tracked,
)

# ---------------------------------------------------------------------------
# Built-in dispersive dielectric presets (v2.6)
# ---------------------------------------------------------------------------

# Unit conversion: wavenumber (cm^-1) -> photon energy (eV)
_CM1_TO_EV = 1.239841984e-4


def _lorentz_eps_from_eps0(
    omega,
    eps0: float,
    eps_inf: float,
    omega_T: float,
    gamma: float,
):
    """
    Single-oscillator Lorentz model in the same form used in your hBN notebook.

        ε(ω) = ε_inf + (ε0 - ε_inf) / ( 1 - (ω/ω_T)^2 - i γ ω / ω_T^2 )

    All frequencies are in eV.
    """
    omega = np.asarray(omega, dtype=np.complex128)
    omega = np.real(omega)
    num = eps0 - eps_inf
    den = 1.0 - (omega / omega_T) ** 2 - 1j * gamma * omega / (omega_T ** 2)
    return eps_inf + num / den


def _factorized_lo_to_eps(
    omega,
    eps_inf: float,
    poles_cm1,
):
    """
    Factorized LO-TO dielectric function for polar crystals:

      ε(ω) = ε_inf ∏_j ( ω_LO,j^2 - ω^2 - i ω γ_LO,j ) / ( ω_TO,j^2 - ω^2 - i ω γ_TO,j )

    where poles_cm1 is an iterable of tuples:
      (omega_TO_cm1, omega_LO_cm1, gamma_TO_cm1, gamma_LO_cm1)

    ω is photon energy in eV; we convert cm^-1 -> eV using _CM1_TO_EV.
    """
    omega = np.asarray(omega, dtype=np.complex128)
    omega = np.real(omega)
    w = omega
    eps = np.full_like(w, complex(eps_inf), dtype=np.complex128)
    for (wTO_cm1, wLO_cm1, gTO_cm1, gLO_cm1) in poles_cm1:
        wTO = wTO_cm1 * _CM1_TO_EV
        wLO = wLO_cm1 * _CM1_TO_EV
        gTO = gTO_cm1 * _CM1_TO_EV
        gLO = gLO_cm1 * _CM1_TO_EV
        num = (wLO * wLO) - (w * w) - 1j * w * gLO
        den = (wTO * wTO) - (w * w) - 1j * w * gTO
        eps *= num / den
    return eps


# --- hBN (uniaxial, aligned with z) ---
def _hbn_eps_x(omega, eps0x=6.41, epsInfx=4.53, oT_x=168e-3, gamma_x=0.82e-3):
    """hBN in-plane ε∥(ω) Lorentz model (ω in eV)."""
    return _lorentz_eps_from_eps0(omega, eps0x, epsInfx, oT_x, gamma_x)


def _hbn_eps_z(omega, eps0z=3.0, epsInfz=2.5, oT_z=94.2e-3, gamma_z=0.23e-3):
    """hBN out-of-plane ε_z(ω) Lorentz model (ω in eV)."""
    return _lorentz_eps_from_eps0(omega, eps0z, epsInfz, oT_z, gamma_z)


# --- SiO2 (isotropic, mid-IR LO-TO factorized model) ---
# Parameters from Aguilar-Merino et al., Nanomaterials 11, 120 (2021), Table 3:
# ε_inf = 2, and three phonon pairs (ω_TO, ω_LO, γ) in cm^-1. citeturn9view0
_SIO2_EPS_INF = 2.0
_SIO2_POLES_CM1 = (
    # (ω_TO, ω_LO, γ_TO, γ_LO)  [γ_LO not provided explicitly; we set γ_LO = γ_TO for this dataset]
    (450.0, 505.0, 51.0, 51.0),
    (800.0, 830.0, 10.0, 10.0),
    (1045.0, 1240.0, 10.0, 10.0),
)


def _sio2_eps(omega):
    """SiO2 isotropic ε(ω) in factorized LO-TO form (ω in eV)."""
    return _factorized_lo_to_eps(omega, _SIO2_EPS_INF, _SIO2_POLES_CM1)


# --- α-Al2O3 / Sapphire (uniaxial, c-axis aligned with z) ---
# Parameters from Schubert, Tiwald, Herzinger, Phys. Rev. B 61, 8187 (2000):
# Table II (TO/LO frequencies), Table III (γ_TO, γ_LO), Table IV (ε_inf). citeturn13view0turn17view1turn18view1
_AL2O3_EPS_INF_PERP = 3.064   # E ⟂ c  (maps to eps_par in our uniaxial tensor)
_AL2O3_EPS_INF_PAR = 3.038    # E ∥ c  (maps to eps_z)
_AL2O3_POLES_EU_CM1 = (
    # Eu modes (E ⟂ c): (ω_TO, ω_LO, γ_TO, γ_LO) in cm^-1
    (384.99, 387.60, 3.3, 3.1),
    (439.10, 481.68, 3.1, 1.9),
    (569.00, 629.50, 4.7, 5.9),
    (633.63, 906.6, 5.0, 14.7),
)
_AL2O3_POLES_A2U_CM1 = (
    # A2u modes (E ∥ c):
    (397.52, 510.87, 5.3, 1.1),
    (582.41, 881.1, 3.0, 15.4),
)


def _al2o3_eps_perp(omega):
    """α-Al2O3 ordinary permittivity ε_perp(ω) (E ⟂ c), ω in eV."""
    return _factorized_lo_to_eps(omega, _AL2O3_EPS_INF_PERP, _AL2O3_POLES_EU_CM1)


def _al2o3_eps_par(omega):
    """α-Al2O3 extraordinary permittivity ε_parallel_to_c(ω) (E ∥ c), ω in eV."""
    return _factorized_lo_to_eps(omega, _AL2O3_EPS_INF_PAR, _AL2O3_POLES_A2U_CM1)


def _epsilon_preset(name: str, omega: complex):
    """
    Map a preset material name to (eps_par(ω), eps_z(ω)).

    Conventions:
      - eps_par is the in-plane component (x,y) relative to the layered geometry
      - eps_z is the out-of-plane component (z) relative to the layered geometry

    Built-ins:
      - "hBN"   : uniaxial Lorentz (same as your notebook)
      - "SiO2"  : isotropic factorized LO-TO model (mid-IR)
      - "Al2O3" : sapphire α-Al2O3 uniaxial factorized LO-TO model (c-axis || z)
    """
    key = str(name).strip().lower()
    if key in ("hbn", "h-bn", "hexagonal boron nitride"):
        return _hbn_eps_x(omega), _hbn_eps_z(omega)
    if key in ("sio2", "silica", "silicon dioxide", "quartz", "fused silica"):
        eps = _sio2_eps(omega)
        return eps, eps
    if key in ("al2o3", "alumina", "sapphire", "alpha-al2o3", "α-al2o3"):
        return _al2o3_eps_perp(omega), _al2o3_eps_par(omega)
    raise ValueError(f"Unknown dispersive epsilon preset: {name!r}")


def _build_region_epsilon_provider(n_regions: int, epsilon):
    """
    Build an internal epsilon provider from a user-supplied `epsilon` specification.

    Accepted `epsilon` forms:
      1) scalar number -> isotropic constant in all regions
      2) sequence length n_regions of numbers -> isotropic per-region constants
      3) sequence length n_regions containing a mix of:
           - numbers (isotropic constant for that region)
           - strings (built-in dispersive presets, e.g. "hBN", "SiO2", "Al2O3")
         -> creates a dispersive provider (region, omega) -> (eps_par, eps_z)

    Returns (mode, provider):
      mode == "static" and provider is epsilon_static compatible
      mode == "dispersive" and provider is epsilon_dispersive callable
    """
    if np.isscalar(epsilon) and not isinstance(epsilon, (str, bytes)):
        return "static", float(epsilon)

    if not isinstance(epsilon, (list, tuple, np.ndarray)):
        raise ValueError("epsilon must be a scalar or a sequence of length N+1.")

    eps_list = list(epsilon)
    if len(eps_list) != n_regions:
        raise ValueError(f"epsilon must have length N+1={n_regions}, got {len(eps_list)}.")

    has_string = any(isinstance(x, (str, bytes)) for x in eps_list)
    if not has_string:
        return "static", np.array([complex(x) for x in eps_list], dtype=np.complex128)

    def eps_disp(region: int, omega: complex):
        x = eps_list[int(region)]
        if isinstance(x, (str, bytes)):
            return _epsilon_preset(x, omega)
        val = complex(x)
        return val, val

    return "dispersive", eps_disp


def _screened_head_W00(inv_qo: np.ndarray, V_qo: np.ndarray) -> np.ndarray:
    """
    G=G'=0 layer block of W = V·ε⁻¹, computed in the full (layer, G) × (layer', G') space.

    inv_qo : (N, N, n_G, n_G), inv_qo[i, j, G, G'] = (ε⁻¹)_{(i,G),(j,G')}
    V_qo   : (N, N, n_G),      V_qo[i, j, G] = V_ij(q+G); V is diagonal in G.
    Both full matrices use the row/column index i*n_G + G.
    """
    N, n_G = inv_qo.shape[0], inv_qo.shape[2]
    inv_full = inv_qo.transpose(0, 2, 1, 3).reshape(N * n_G, N * n_G)
    V_full = np.zeros((N * n_G, N * n_G), dtype=np.complex128)
    head = np.arange(N) * n_G                     # rows/cols (i, G=0)
    for g in range(n_G):
        idx = head + g
        V_full[np.ix_(idx, idx)] = V_qo[:, :, g]
    W_full = V_full @ inv_full
    return W_full[np.ix_(head, head)]


class layer_helpers_G:
    def __init__(
        self,
        crystal: Any,
        n_layers: int,
        epsilon_static: Optional[Union[float, Sequence[complex]]] = None,
        d: Optional[Sequence[float]] = None,
        density_operators=None,
        n_jobs: int = 1,
        use_cache: bool = True,
        epsilon_uniaxial: Optional[Tuple[Union[float, Sequence[complex]], Union[float, Sequence[complex]]]] = None,
        epsilon_dispersive: Optional[Callable[[int, complex], Tuple[complex, complex]]] = None,
        top_gate_d: Optional[float] = None,
        bottom_gate_d: Optional[float] = None,
    ):
        self.crystal = crystal
        self.n_layers = int(n_layers)
        self.n_jobs = int(n_jobs)
        self.use_cache = bool(use_cache)
        self.density_operators = density_operators
        # v2.7: grounded gate distances (nm); None => open half-space
        self.top_gate_d = None if top_gate_d is None else float(top_gate_d)
        self.bottom_gate_d = None if bottom_gate_d is None else float(bottom_gate_d)
        _gate_kw = dict(top_gate_d=self.top_gate_d, bottom_gate_d=self.bottom_gate_d)

        if d is None:
            d_arr = []
        else:
            d_arr = list(np.asarray(d, dtype=float).ravel())

        # Construct consistent Coulomb core
        # Priority:
        #   1) explicit epsilon_dispersive callable
        #   2) explicit epsilon_uniaxial tuple
        #   3) epsilon_static numeric or list OR mixed preset list like ["hBN", 1, "hBN"]
        if epsilon_dispersive is not None:
            self._lc = LayerCoulomb(n_layers=self.n_layers, d=d_arr, epsilon_dispersive=epsilon_dispersive, **_gate_kw)
            self._epsilon_dispersive = epsilon_dispersive
            self.epsilon_mode = "dispersive"
        elif epsilon_uniaxial is not None:
            self._lc = LayerCoulomb(n_layers=self.n_layers, d=d_arr, epsilon_uniaxial=epsilon_uniaxial, **_gate_kw)
            self._epsilon_dispersive = None
            self.epsilon_mode = "uniaxial"
        else:
            if epsilon_static is None:
                raise ValueError(
                    "Provide one of: epsilon_static, epsilon_uniaxial, epsilon_dispersive. "
                    "epsilon_static may also be a mixed list like ['hBN', 1, 'hBN']."
                )

            mode, provider = _build_region_epsilon_provider(self.n_layers + 1, epsilon_static)
            if mode == "static":
                self._lc = LayerCoulomb(n_layers=self.n_layers, d=d_arr, epsilon_static=provider, **_gate_kw)
                self._epsilon_dispersive = None
                self.epsilon_mode = "static"
            else:
                self._lc = LayerCoulomb(n_layers=self.n_layers, d=d_arr, epsilon_dispersive=provider, **_gate_kw)
                self._epsilon_dispersive = provider
                self.epsilon_mode = "dispersive"
        self.n_regions = self.n_layers + 1
        self.d = np.asarray(d_arr, dtype=float)

        self.densityL = None
        self.chi_nn = None

        if self.crystal is not None and self.density_operators is not None:
            try:
                self.densityL = self.compute_density_layer()
            except Exception as e:
                warnings.warn(f"Could not compute density layer matrices at init: {e}")

    # ---------------------------------------------------------------------
    # Coulomb / A-matrix API (consistent physical V)
    # ---------------------------------------------------------------------

    def build_A_matrix_fast(self, q_mag: float, omega_val: complex = 0.0) -> np.ndarray:
        return self._lc.A_matrix(q=float(q_mag), omega=omega_val)

    def invert_A_cached(self, q_mag: float, omega_val: complex = 0.0) -> np.ndarray:
        return self._lc.A_inverse(q=float(q_mag), omega=omega_val)

    def coulomb_potential_matrix(self, q_mag: float, omega_val: complex = 0.0) -> np.ndarray:
        return self._lc.coulomb_matrix(q=float(q_mag), omega=omega_val)

    def coulomb_potential(self, q_mag: float, l: int, lp: int, omega_val: complex = 0.0):
        V = self.coulomb_potential_matrix(q_mag=q_mag, omega_val=omega_val)
        return V[int(l), int(lp)]

    def coulomb_potential_matrix_vectorized(
        self,
        q_vector: Union[np.ndarray, Sequence[float]],
        reciprocal_vectors_T: Union[np.ndarray, Sequence[Sequence[float]]],
        omega: Optional[Union[float, complex, np.ndarray]] = None,
    ) -> np.ndarray:
        """
        Vectorized V(q+G, ω). Supports q and G as 2D or 3D vectors.
        Returns:
          - omega None/scalar: (N,N,nG)
          - omega array: (N,N,nG,nω)
        """
        qv = np.asarray(q_vector, dtype=float).ravel()
        if qv.size == 2:
            qv = np.array([qv[0], qv[1], 0.0], dtype=float)
        elif qv.size == 3:
            pass
        else:
            raise ValueError(f"q_vector must have length 2 or 3, got {qv.size}.")

        GT = np.asarray(reciprocal_vectors_T, dtype=float)
        if GT.ndim != 2:
            raise ValueError("reciprocal_vectors_T must be 2D with shape (D, n_G).")
        if GT.shape[0] == 2:
            GT = np.vstack([GT, np.zeros((1, GT.shape[1]), dtype=float)])
        elif GT.shape[0] == 3:
            pass
        else:
            raise ValueError(f"reciprocal_vectors_T must have 2 or 3 rows, got {GT.shape[0]}.")

        n_G = GT.shape[1]
        q_plus_G = qv[:, None] + GT
        q_mags = np.linalg.norm(q_plus_G, axis=0)

        if omega is None or np.isscalar(omega):
            w = 0.0 if omega is None else omega
            out = np.zeros((self.n_layers, self.n_layers, n_G), dtype=np.complex128)
            valid = q_mags > 1e-12
            for gi in np.flatnonzero(valid):
                out[:, :, gi] = self.coulomb_potential_matrix(q_mag=float(q_mags[gi]), omega_val=np.real(w))
            return out

        omega_arr = np.asarray(omega, dtype=np.complex128).ravel()
        n_omega = omega_arr.size
        out = np.zeros((self.n_layers, self.n_layers, n_G, n_omega), dtype=np.complex128)
        valid = q_mags > 1e-12

        is_dispersive = self._epsilon_dispersive is not None
        if not is_dispersive:
            for gi in np.flatnonzero(valid):
                Vg = self.coulomb_potential_matrix(q_mag=float(q_mags[gi]), omega_val=0.0)
                out[:, :, gi, :] = Vg[:, :, np.newaxis]
        else:
            # Use branch-tracked κ,Y across ω to avoid π jumps in Reststrahlen band (non-passive V)
            eps_par_omega, eps_z_omega = self._lc.eps_arrays_over_omega(omega_arr)
            for gi in np.flatnonzero(valid):
                q_mag = float(q_mags[gi])
                kappa_om, Y_om = compute_kappa_Y_tracked(q_mag, eps_par_omega, eps_z_omega)
                for oi in range(n_omega):
                    out[:, :, gi, oi] = self._lc.coulomb_matrix_from_kappa_Y(
                        q_mag, kappa_om[:, oi], Y_om[:, oi]
                    )
        return out

    # ---------------------------------------------------------------------
    # Density operator matrix elements (ported)
    # ---------------------------------------------------------------------

    def compute_density_matrix(self, band_structure, density_operator):
        if (band_structure._energies is None) or (band_structure._wavefunctions is None):
            warnings.warn("bandstructure not available, ignoring command")
            return None

        if (band_structure._q_sample is None) or (band_structure._G_sample is None):
            warnings.warn("q_sample or G_sample not available", UserWarning)
            return None

        density_list = []
        for ni in density_operator:
            matrix, table = band_structure.compute_matrix_elements(ni)
            density_list.append(matrix)
            band_structure._k_plus_q_ind = table
        return np.stack(density_list, axis=-1)

    def compute_density_layer(self):
        if self.crystal is None:
            raise ValueError("crystal must be provided to compute density operators.")
        if self.density_operators is None:
            raise ValueError("density_operators must be provided to compute density operators.")

        density_matrix = []
        for i in range(self.n_layers):
            density_matrix.append(
                self.compute_density_matrix(
                    band_structure=self.crystal.band_structure,
                    density_operator=self.density_operators[i],
                )
            )
        return np.array(density_matrix)

    # ---------------------------------------------------------------------
    # Polarization / chi (ported; requires pybandstructure)
    # ---------------------------------------------------------------------

    @staticmethod
    def _polarization_functions():
        try:
            from pybandstructure.crystal.analysis import polarization_functions
            return polarization_functions
        except Exception as e:
            raise ImportError(
                "pybandstructure is required for polarization_* and dielectric_function_layer. "
                f"Import failed: {e}"
            )

    def polarization_layered(
        self, omega, q_index, G_index, G1_index, index1=0, index2=1, use_bands=slice(None)
    ):
        if self.crystal is None:
            raise ValueError("crystal must be provided for polarization computation.")
        if self.densityL is None:
            raise ValueError("densityL is not available; call compute_density_layer() first.")

        if self.crystal.geometry._point_group.cardinality != 1:
            warnings.warn("Computing with non-trivial point group is experimental")

        polf = self._polarization_functions()
        polarization = np.zeros_like(omega, dtype=complex)

        for g in range(self.crystal.geometry._point_group.cardinality):
            polarization += polf.polarization(
                omega=omega,
                energies=self.crystal.band_structure.energies[use_bands, :],
                occupations=self.crystal.band_structure.occupations[use_bands, :],
                densityG=np.nan_to_num(
                    self.densityL[index1][
                        use_bands,
                        use_bands,
                        :,
                        self.crystal._q_symmetry_repr[g, q_index],
                        self.crystal._G_symmetry_repr[g, G_index],
                    ]
                ),
                densityG1=np.nan_to_num(
                    self.densityL[index2][
                        use_bands,
                        use_bands,
                        :,
                        self.crystal._q_symmetry_repr[g, q_index],
                        self.crystal._G_symmetry_repr[g, G1_index],
                    ]
                ),
                k_p_q_ind=self.crystal.band_structure._k_plus_q_ind[
                    self.crystal._q_symmetry_repr[g, q_index], :
                ],
                k_weights=self.crystal.band_structure.k_sample.weights,
                integration_constant=self.crystal._k_int_const,
            )
        return polarization / self.crystal.geometry._point_group.cardinality

    def polarization_tot(self, omega, q_index, G_index, G1_index, use_bands=slice(None)):
        polarization_tot = np.zeros([self.n_layers, self.n_layers, len(omega)], dtype=complex)
        for i in range(self.n_layers):
            for j in range(self.n_layers):
                polarization_tot[i, j, :] = self.polarization_layered(
                    omega, q_index, G_index, G1_index, i, j, use_bands
                )
        return polarization_tot

    def _compute_chi_for_q(self, q_index, omega, reciprocal_vectors, use_bands, idx):
        n_G = len(reciprocal_vectors)
        chi_q = np.zeros([self.n_layers, self.n_layers, len(omega), n_G, n_G], dtype=complex)
        for G_index in range(n_G):
            for G1_index in range(n_G):
                chi_q[:, :, :, G_index, G1_index] = self.polarization_tot(
                    omega, q_index, G_index, G1_index, use_bands
                )
        return idx, chi_q

    # ---------------------------------------------------------------------
    # Dielectric and loss function (ported with eigenbasis consistency fix)
    # ---------------------------------------------------------------------

    def dielectric_function_layer(
        self,
        omega=np.array([0], np.float64),
        use_bands=slice(None, None, None),
        contour=None,
        compute_eigenbasis=False,
        retarded=True,
        legacy_contour_indexing=False,
    ):
        """
        Compute ε(q,ω) = I - V·χ and optionally coulomb_matrix, eigenbasis.

        retarded : bool, default True
            If True, χ is evaluated at ω + iη (retarded); if False, at ω - iη (advanced).
            Caller typically passes omega = ω + 1j*eta; then retarded=True uses that,
            retarded=False uses ω - 1j*eta. Use retarded=True for the standard causal
            response (positive loss in passive systems).
        """
        if self.crystal is None:
            raise ValueError("crystal must be provided.")

        q_vectors = self.crystal.band_structure.q_sample.coords
        if contour is not None:
            q_contour = self.crystal.band_structure.q_sample.contour(contour)
            indices = q_contour["indices"]
            q_vectors = q_vectors[indices]
        else:
            indices = None

        reciprocal_vectors = self.crystal.band_structure.G_sample.coords
        n_q = len(q_vectors)
        n_G = len(reciprocal_vectors)
        omega = np.asarray(omega, dtype=np.complex128).ravel()
        n_omega = len(omega)
        # Retarded: χ(ω+iη). Advanced: χ(ω-iη). Caller passes e.g. omega = ω + 1j*eta.
        # Fixed (v2.7.1 evaluated real(w) + 1j*imag(w), i.e. a no-op): the advanced
        # response is the complex conjugate point w - i eta.
        _eta_abs = np.abs(np.imag(omega))
        omega_for_chi = (np.real(omega) + 1j * _eta_abs) if retarded else (np.real(omega) - 1j * _eta_abs)

        chi = np.zeros([n_q, self.n_layers, self.n_layers, n_omega, n_G, n_G], dtype=complex)
        epsilon_layer = np.zeros([n_q, n_omega, self.n_layers, self.n_layers, n_G, n_G], dtype=np.complex128)
        dispersive = self._epsilon_dispersive is not None

        if dispersive:
            coulomb_matrix = np.zeros([n_q, self.n_layers, self.n_layers, n_G, n_omega], dtype=np.complex128)
        else:
            coulomb_matrix = np.zeros([n_q, self.n_layers, self.n_layers, n_G], dtype=np.float64)

        if compute_eigenbasis:
            coulomb_eigenvalues = np.zeros([n_q, n_G, self.n_layers], dtype=np.complex128)
            coulomb_eigenvectors = np.zeros([n_q, n_G, self.n_layers, self.n_layers], dtype=np.complex128)
            chi_eigenbasis = np.zeros([n_q, self.n_layers, self.n_layers, n_omega, n_G, n_G], dtype=complex)
            epsilon_eigenbasis = np.zeros([n_q, n_omega, self.n_layers, self.n_layers, n_G, n_G], dtype=np.complex128)

        # Compute chi (retarded: ω+iη, advanced: ω-iη)
        if HAS_JOBLIB and self.n_jobs != 1 and n_q > 10 and indices is not None and len(indices) > 1:
            print(f"Computing χ in parallel with {self.n_jobs} jobs...")
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(self._compute_chi_for_q)(q_idx, omega_for_chi, reciprocal_vectors, use_bands, idx=idx)
                for idx, q_idx in tqdm(enumerate(indices), desc="Computing χ")
            )
            for q_idx, chi_q in results:
                chi[q_idx] = chi_q

        elif HAS_JOBLIB and self.n_jobs != 1 and n_q > 10 and indices is None:
            print(f"Computing χ in parallel with {self.n_jobs} jobs...")
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(self._compute_chi_for_q)(q_idx, omega_for_chi, reciprocal_vectors, use_bands, idx=q_idx)
                for q_idx in tqdm(range(n_q), desc="Computing χ")
            )
            for q_idx, chi_q in results:
                chi[q_idx] = chi_q

        else:
            # Fixed: on a contour, chi0 must be evaluated at the *global* q index
            # indices[j] that V is built from (v2.7.1 used the local index j).
            for j in tqdm(range(n_q), desc="Computing χ"):
                q_index = j if (indices is None or legacy_contour_indexing) else int(indices[j])
                for G_index in range(n_G):
                    for G1_index in range(n_G):
                        chi[j, :, :, :, G_index, G1_index] = self.polarization_tot(
                            omega_for_chi, q_index, G_index, G1_index, use_bands
                        )

        self.chi_nn = chi
        eye_layers = np.eye(self.n_layers)

        desc = "Computing ε (layer + eigen basis)" if compute_eigenbasis else "Computing ε"
        for q_idx, q in tqdm(enumerate(q_vectors), desc=desc, total=n_q):
            if dispersive:
                V_q = self.coulomb_potential_matrix_vectorized(q, reciprocal_vectors.T, np.real(omega))
                coulomb_matrix[q_idx] = V_q  # (N,N,nG,nω)

                for o_idx in range(n_omega):
                    e_con = np.einsum('ikG,kjGK->ijGK', V_q[:, :, :, o_idx], chi[q_idx, :, :, o_idx, :, :])
                    epsilon_layer[q_idx, o_idx, :, :, :, :] = -e_con
                    for G_index in range(n_G):
                        epsilon_layer[q_idx, o_idx, :, :, G_index, G_index] += eye_layers

            else:
                V_q = self.coulomb_potential_matrix_vectorized(q, reciprocal_vectors.T, omega=None)
                coulomb_matrix[q_idx] = np.asarray(np.real(V_q), dtype=np.float64)

                e_con = np.einsum('ikG,kjoGK->ijoGK', coulomb_matrix[q_idx], chi[q_idx])
                epsilon_layer[q_idx] = -np.transpose(e_con, (2, 0, 1, 3, 4))
                for G_index in range(n_G):
                    epsilon_layer[q_idx, :, :, :, G_index, G_index] += eye_layers[np.newaxis, :, :]

            if compute_eigenbasis:
                for G_index in range(n_G):
                    q_plus_G = q + reciprocal_vectors[G_index]
                    q_mag = float(np.linalg.norm(np.asarray(q_plus_G, dtype=float)))
                    if q_mag <= 1e-12:
                        continue

                    eigen_omega = 0.0 if not dispersive else omega[0]
                    eigenvals, eigenvecs = self.diagonalize_coulomb_matrix(q_mag, omega_val=eigen_omega, use_bare=False)
                    coulomb_eigenvalues[q_idx, G_index, :] = eigenvals
                    coulomb_eigenvectors[q_idx, G_index, :, :] = eigenvecs

                    chi_G = chi[q_idx, :, :, :, G_index, :]  # (N,N,nω,nG')

                    U = eigenvecs
                    for omega_idx in range(n_omega):
                        for G1_index in range(n_G):
                            chi_ij = chi_G[:, :, omega_idx, G1_index]
                            chi_ab = U.T @ chi_ij @ U
                            chi_eigenbasis[q_idx, :, :, omega_idx, G_index, G1_index] = chi_ab

                    for omega_idx in range(n_omega):
                        for G1_index in range(n_G):
                            chi_ab = chi_eigenbasis[q_idx, :, :, omega_idx, G_index, G1_index]
                            epsilon_ab = eye_layers - np.diag(eigenvals) @ chi_ab
                            epsilon_eigenbasis[q_idx, omega_idx, :, :, G_index, G1_index] = epsilon_ab

        if compute_eigenbasis:
            results_dict = {
                'eigenvalues': coulomb_eigenvalues,
                'eigenvectors': coulomb_eigenvectors,
                'chi_eigenbasis': chi_eigenbasis,
                'epsilon_eigenbasis': epsilon_eigenbasis,
            }
            return coulomb_matrix, epsilon_layer, results_dict

        return coulomb_matrix, epsilon_layer, None

    def loss_function_trace(self, dielectric_tensor, coulomb_matrix=None, debug=False):
        r"""
        Compute the energy-loss function at G=G'=0 (macroscopic head).

        **Definition:**

        - **Dispersive case** (coulomb_matrix provided and complex):
          .. math::
             L(q,\omega) = -\operatorname{Im}\operatorname{Tr}\bigl[ V_{00}(q,\omega)\,
             \varepsilon^{-1}_{00}(q,\omega) \bigr]
          i.e. L = -Im(Tr[W00]) with W00 = V00 @ ε⁻¹_00 (layer block at G=G'=0).
          This is the screened-interaction loss; positive in passive systems.

        - **Static case** (coulomb_matrix is None or real):
          .. math::
             L(q,\omega) = -\operatorname{Tr}\operatorname{Im}\bigl[\varepsilon^{-1}_{00}(q,\omega)\bigr]
          (V real so prefactor only rescales.) This formula is **not** positive definite for
          dispersive ε (e.g. Reststrahlen band); use the screened formula with coulomb_matrix instead.

        **Not** used: elementwise Im/(Re²+Im²) or Tr[Im(ε)/(Re²+Im²)].
        ε is stored as ε[q,ω,i,j,G,G']; inversion is over (i,G)×(j,G').
        Screened loss: W = V·ε⁻¹ is computed in full (i,G)×(j,G') space (row/column index
        i*n_G + G for both V and ε⁻¹), then the G=G'=0 block W00 is taken and L = -Im Tr[W00].

        Parameters
        ----------
        dielectric_tensor : ndarray, shape (n_q, n_ω, N, N, n_G, n_G)
        coulomb_matrix : ndarray, optional
            As returned by ``dielectric_function_layer``:
              - static   : shape (n_q, N, N, n_G),        real
              - dispersive: shape (n_q, N, N, n_G, n_ω),  complex
            If provided and complex, the screened formula L = -Im Tr[V00·ε⁻¹_00] is used.
        debug : bool, optional
            If True, print cross-check at a few (q,ω) points: s1=-Im(Tr(W00)), s2=+Im(Tr(W00)),
            and the value returned by this function, to verify sign convention.

        Returns
        -------
        result : ndarray, shape (n_q, n_ω)
            L(q,ω) = -Im Tr[V00·ε⁻¹_00] (dispersive) or -Tr Im[ε⁻¹_00] (static).
        """
        n_q, n_omega = dielectric_tensor.shape[0], dielectric_tensor.shape[1]

        eps = np.array(dielectric_tensor, copy=True)
        eps[~np.isfinite(eps)] = 0.0
        dielectric_reshape = np.transpose(eps, (0, 1, 2, 4, 3, 5))

        inv = np.empty(
            (n_q, n_omega, dielectric_tensor.shape[2],
             dielectric_tensor.shape[4], dielectric_tensor.shape[3],
             dielectric_tensor.shape[5]),
            dtype=np.complex128,
        )
        for q in range(n_q):
            for o in range(n_omega):
                inv[q, o] = np.linalg.tensorinv(dielectric_reshape[q, o])

        inv = np.transpose(inv, (0, 1, 2, 4, 3, 5))
        # inv[q,o,i,j,G,G'] = (ε⁻¹)_{(i,G),(j,G')}; full matrix index (i,G) = i*n_G + G
        inv_00 = inv[:, :, :, :, 0, 0]  # (n_q, n_ω, N, N) for static path and debug

        use_screened = (
            coulomb_matrix is not None and np.iscomplexobj(coulomb_matrix)
        )
        N_layer = dielectric_tensor.shape[2]
        n_G = dielectric_tensor.shape[4]

        if use_screened:
            if coulomb_matrix.ndim == 5:
                # dispersive: (n_q, N, N, n_G, n_ω)
                V_nd = coulomb_matrix  # [q, i, j, G, o]
            elif coulomb_matrix.ndim == 4:
                # static-complex: (n_q, N, N, n_G), broadcast over ω
                V_nd = coulomb_matrix[:, :, :, :, np.newaxis]  # (n_q, N, N, n_G, 1)
            else:
                raise ValueError(
                    f"coulomb_matrix must be 4-d or 5-d, got {coulomb_matrix.ndim}-d."
                )
            if V_nd.shape[4] not in (1, n_omega):
                raise ValueError(
                    f"coulomb_matrix has {V_nd.shape[4]} frequencies, dielectric_tensor has {n_omega}."
                )
            V_per_omega = V_nd.shape[4] > 1
            # W = V @ ε⁻¹ in full (i,G)×(j,G') space; then take the G=G'=0 block
            result = np.zeros((n_q, n_omega), dtype=np.float64)
            for q_idx in range(n_q):
                for o_idx in range(n_omega):
                    V_qo = V_nd[q_idx, :, :, :, o_idx if V_per_omega else 0]  # (N, N, n_G)
                    W_00 = _screened_head_W00(inv[q_idx, o_idx], V_qo)
                    result[q_idx, o_idx] = -np.imag(np.trace(W_00))
        else:
            result = -np.trace(np.imag(inv_00), axis1=2, axis2=3)

        result = np.asarray(result, dtype=np.float64)
        result[~np.isfinite(result)] = 0.0

        # Naive formula -Tr Im[ε⁻¹] is not positive definite for dispersive environments (e.g. Reststrahlen).
        if not use_screened and np.any(result < -1e-10):
            warnings.warn(
                "loss_function_trace: some values are negative (min={:.4e}). "
                "For dispersive systems (e.g. hBN), pass coulomb_matrix to use the screened loss "
                "L = -Im Tr[V·ε⁻¹], which is positive in passive regimes.".format(float(np.min(result))),
                UserWarning,
                stacklevel=2,
            )

        if debug:
            # Cross-check at a few (q,ω) points: s1 = -Im(Tr(W00)), current_loss_scalar (W00 = (V·ε⁻¹)_{00} block)
            check_q = [0, min(1, n_q - 1), min(10, n_q - 1)]
            check_o = [0, n_omega // 2, n_omega - 1] if n_omega >= 3 else [0, max(0, n_omega - 1)]
            check_q = [q for q in check_q if q < n_q]
            check_o = [o for o in check_o if o < n_omega]
            if use_screened:
                for q_idx in check_q:
                    for o_idx in check_o:
                        V_qo = V_nd[q_idx, :, :, :, o_idx if V_per_omega else 0]
                        W00 = _screened_head_W00(inv[q_idx, o_idx], V_qo)
                        s1 = -np.imag(np.trace(W00))
                        s2 = np.imag(np.trace(W00))
                        current_loss_scalar = float(result[q_idx, o_idx])
                        print(
                            f"[loss_function_trace debug] q_idx={q_idx} omega_idx={o_idx}  "
                            f"s1=-Im(Tr(W00))={s1:.6e}  s2=+Im(Tr(W00))={s2:.6e}  "
                            f"current_loss_scalar={current_loss_scalar:.6e}  "
                            f"match_s1={np.isclose(current_loss_scalar, s1)}  match_s2={np.isclose(current_loss_scalar, s2)}"
                        )
            else:
                for q_idx in check_q:
                    for o_idx in check_o:
                        inv00 = inv_00[q_idx, o_idx, :, :]
                        current_loss_scalar = float(result[q_idx, o_idx])
                        tr_im_inv = np.trace(np.imag(inv00))
                        s1 = -tr_im_inv
                        print(
                            f"[loss_function_trace debug] q_idx={q_idx} omega_idx={o_idx}  "
                            f"(static) -Tr(Im(ε⁻¹))={s1:.6e}  current_loss_scalar={current_loss_scalar:.6e}  "
                            f"match={np.isclose(current_loss_scalar, s1)}"
                        )
        return result

    def diagonalize_coulomb_matrix(
        self, q_mag: float, omega_val: complex = 0.0, use_bare: bool = False, mode: str = "hermitian_part"
    ):
        """
        Diagonalize V(q, omega).

        When *use_bare=True* the uniform-medium analytic formula
        V_ij = (2pi e^2 / eps q) exp(-q|z_i-z_j|) is used regardless of
        omega_val (electrostatic limit for uniform isotropic epsilon).

        NOTE (v2.7): the *use_bare* analytic formula assumes open half-spaces
        and is NOT valid when metallic gates are present.  It raises if the
        instance was created with top_gate_d / bottom_gate_d set.  Use
        use_bare=False (the default) for gated stacks, which diagonalizes the
        physical gate-aware V from the A-matrix.
        """
        if use_bare:
            if self.top_gate_d is not None or self.bottom_gate_d is not None:
                raise NotImplementedError(
                    "use_bare=True uses the open-half-space uniform formula and "
                    "ignores the gates set on this instance. Call with use_bare=False "
                    "to diagonalize the physical gate-aware V(q,ω)."
                )
            eps_par, eps_z = self._lc.eps_arrays(omega=0.0)
            if not (np.allclose(eps_par, eps_par[0]) and np.allclose(eps_par, eps_z)):
                raise NotImplementedError("Bare diagonalization only implemented for uniform isotropic epsilon.")
            eps = float(np.real(eps_par[0]))
            z = np.zeros(self.n_layers, dtype=float)
            for k in range(1, self.n_layers):
                z[k] = z[k-1] + self.d[k-1]
            dist = np.abs(z[:, None] - z[None, :])
            q = float(q_mag)
            V = (TWOPI_E2_EV_NM / (eps * q)) * np.exp(-q * dist)
            V = V.astype(np.complex128)
        else:
            V = self.coulomb_potential_matrix(q_mag=q_mag, omega_val=np.real(omega_val))

        if mode == "hermitian_part":
            VH = 0.5 * (V + V.conj().T)
            w, U = np.linalg.eigh(VH)
            return w, U
        if mode == "real_part":
            w, U = np.linalg.eigh(np.real(V))
            return w, U
        if mode == "full_complex":
            w, U = np.linalg.eig(V)
            return w, U
        raise ValueError("mode must be one of: hermitian_part, real_part, full_complex.")

    def transform_chi_to_eigenbasis(self, chi_matrix: np.ndarray, eigenvectors: np.ndarray) -> np.ndarray:
        U = np.asarray(eigenvectors)
        chi = np.asarray(chi_matrix)
        return U.conj().T @ chi @ U

    def loss_diagnostics(
        self,
        coulomb_matrix: np.ndarray,
        epsilon_layer: np.ndarray,
        q_idx: int,
        omega_idx: int,
        omega_array: np.ndarray,
        q_mag: Optional[float] = None,
        tol: float = 1e-12,
    ):
        """
        Diagnose loss sign at (q_idx, omega_idx, G=0): V00, ε⁻¹_00, W00, VA/WA eigs, κ/Y phases.

        Call after dielectric_function_layer(); runnable from a notebook.
        If VA (anti-Hermitian part of V) has negative eigenvalues, the issue is in V/branching.
        If VA is PSD-ish but WA is not, issue is in ε assembly or scalarization.
        """
        return loss_diagnostics(
            self, coulomb_matrix, epsilon_layer, q_idx, omega_idx, omega_array,
            q_mag=q_mag, tol=tol,
        )


def loss_diagnostics(
    layer_helpers,
    coulomb_matrix: np.ndarray,
    epsilon_layer: np.ndarray,
    q_idx: int,
    omega_idx: int,
    omega_array: np.ndarray,
    q_mag: Optional[float] = None,
    tol: float = 1e-12,
):
    """
    At (q_idx, omega_idx, G=0): print V00, ε⁻¹_00, W00, eigenvalues of VA and WA, κ/Y per region.

    layer_helpers : layer_helpers_G instance (for κ,Y; if crystal is None, pass q_mag).
    """
    omega_arr = np.asarray(omega_array, dtype=np.complex128).ravel()
    omega_val = omega_arr[omega_idx]
    N = coulomb_matrix.shape[1]
    n_G = coulomb_matrix.shape[3]

    # V00: N×N at G=0
    if coulomb_matrix.ndim == 5:
        V00 = coulomb_matrix[q_idx, :, :, 0, omega_idx].copy()
    else:
        V00 = coulomb_matrix[q_idx, :, :, 0].copy()

    # ε⁻¹ block (0,0) in layer space: same reshape as loss_function_trace
    eps = np.array(epsilon_layer[q_idx, omega_idx, :, :, :, :], copy=True)
    eps[~np.isfinite(eps)] = 0.0
    dielectric_reshape = np.transpose(eps, (0, 2, 1, 3))  # (N, n_G, N, n_G)
    inv_block = np.linalg.tensorinv(dielectric_reshape)
    inv00 = inv_block[:, 0, :, 0]  # (N, N)

    W00 = V00 @ inv00

    VA = (V00 - V00.conj().T) / (2.0j)
    WA = (W00 - W00.conj().T) / (2.0j)
    eig_VA = np.linalg.eigvalsh(VA)
    eig_WA = np.linalg.eigvalsh(WA)
    va_psd = np.min(eig_VA) >= -tol
    wa_psd = np.min(eig_WA) >= -tol

    print(f"--- Loss diagnostics q_idx={q_idx} omega_idx={omega_idx} ω={omega_val:.6f} ---")
    print("V00 (G=0):")
    print(V00)
    print("ε⁻¹_00 (layer block G=G'=0):")
    print(inv00)
    print("W00 = V00 @ ε⁻¹_00:")
    print(W00)
    print("VA = (V00 - V00†)/(2i)  min_eig={:.6e} max_eig={:.6e}  PSD-ish: {}".format(
        np.min(eig_VA), np.max(eig_VA), va_psd))
    print("WA = (W00 - W00†)/(2i)  min_eig={:.6e} max_eig={:.6e}  PSD-ish: {}".format(
        np.min(eig_WA), np.max(eig_WA), wa_psd))

    if layer_helpers is not None and q_mag is not None:
        eps_par, eps_z = layer_helpers._lc.eps_arrays(omega=omega_val)
        kappa, Y = compute_kappa_Y(q_mag, eps_par, eps_z)
        print("κ (per region), phase (rad):")
        for r in range(kappa.shape[0]):
            print(f"  region {r}: κ={kappa[r]}, phase={np.angle(kappa[r]):.4f}")
        print("Y (per region), phase (rad):")
        for r in range(Y.shape[0]):
            print(f"  region {r}: Y={Y[r]}, phase={np.angle(Y[r]):.4f}")
    elif layer_helpers is not None and layer_helpers.crystal is not None:
        q_vectors = layer_helpers.crystal.band_structure.q_sample.coords
        q_vec = q_vectors[q_idx]
        q_mag_from_q = float(np.linalg.norm(q_vec))
        eps_par, eps_z = layer_helpers._lc.eps_arrays(omega=omega_val)
        kappa, Y = compute_kappa_Y(q_mag_from_q, eps_par, eps_z)
        print("κ (per region), phase (rad) [q_mag={:.6f}]:".format(q_mag_from_q))
        for r in range(kappa.shape[0]):
            print(f"  region {r}: κ={kappa[r]}, phase={np.angle(kappa[r]):.4f}")
        print("Y (per region), phase (rad):")
        for r in range(Y.shape[0]):
            print(f"  region {r}: Y={Y[r]}, phase={np.angle(Y[r]):.4f}")
    else:
        print("(κ/Y not printed: pass q_mag or use layer_helpers with crystal)")

    return {"V00": V00, "inv00": inv00, "W00": W00, "eig_VA": eig_VA, "eig_WA": eig_WA, "VA_PSD": va_psd, "WA_PSD": wa_psd}


def assert_epsilon_inversion_ordering(epsilon_layer, q_idx=0, omega_idx=0, tol=1e-10):
    """
    Verify that tensorinv(reshape(ε)) matches np.linalg.inv(M) for M[(i,G),(j,G')].
    epsilon_layer shape: (n_q, n_ω, N, N, n_G, n_G).
    """
    eps = np.asarray(epsilon_layer[q_idx, omega_idx, :, :, :, :], dtype=np.complex128)
    # eps is (N, N, n_G, n_G)
    N, n_G = eps.shape[0], eps.shape[2]
    # Same as loss_function_trace: (N, N, n_G, n_G) -> (N, n_G, N, n_G)
    M_reshape = np.transpose(eps, (0, 2, 1, 3))  # (N, n_G, N, n_G)
    M_2d = M_reshape.reshape(N * n_G, N * n_G)
    inv_2d = np.linalg.inv(M_2d)
    inv_tensor = np.linalg.tensorinv(M_reshape)
    inv_tensor_2d = inv_tensor.reshape(N * n_G, N * n_G)
    diff = np.max(np.abs(inv_2d - inv_tensor_2d))
    assert diff < tol, "tensorinv result does not match np.linalg.inv(M): max diff = {}".format(diff)
    return True


# ---------------------------------------------------------------------------
# Fast, memory-bounded chi0 (from tbg_hbn_encapsulated_loss/tbg_hbn_loss.py)
# ---------------------------------------------------------------------------

class LayerHelpers(layer_helpers_G):
    r"""``layer_helpers_G`` with a faster chi0 kernel (same formula, same results).

        chi0^{(l,l')}_{G,G'}(q,w) = C sum_{nmk} w_k (f_{nk} - f_{m,k+q})
                                    rho^{(l)}_{nm}(k; q+G) rho^{(l')}_{nm}(k; q+G')^* / (E_{nk} - E_{m,k+q} + w)

    * chi0 of one q is computed for all (l, G, l', G') together, in a single pass
      over blocks of the (n, m, k) index: the energy denominators are built once
      per q and contracted with all (l', G') as matrix products.  The base class
      rebuilds the (n, m, k, w) denominator for each of the N^2 n_G^2 pairs.
      Transitions with f_{nk} - f_{m,k+q} = 0 are skipped (``weight_cutoff``).
    * ``stream_density=True``: density matrices are computed per q and discarded
      (memory independent of n_q; ``densityL`` is then ``None``, so the
      self-energy, which needs ``densityL``, cannot be computed).  The default
      here is ``False`` because the G0W self-energy needs them.
    * ``layer_coulomb``: a ready-made Coulomb kernel object replacing the one
      built from the epsilon keywords (e.g. :class:`~g0w.layer_coulomb.CappedLayerCoulomb`).

    Falls back to the base class for non-trivial point groups.
    """

    block_bytes: float = 2.5e8      # memory of one (n m k block) x w array
    weight_cutoff: float = 1e-13    # skip transitions with |df w_k| < cutoff * max |df w_k|

    def __init__(self, *args, stream_density: bool = False,
                 layer_coulomb: Optional[LayerCoulomb] = None,
                 share_densityL_from: Optional["layer_helpers_G"] = None, **kwargs):
        self.stream_density = bool(stream_density)
        self._shared_densityL = None if share_densityL_from is None else share_densityL_from.densityL
        super().__init__(*args, **kwargs)
        if layer_coulomb is not None:
            if layer_coulomb.n_layers != self.n_layers:
                raise ValueError("layer_coulomb has a different number of layers.")
            self._lc = layer_coulomb

    # ---- density matrices ---------------------------------------------------
    def compute_density_layer(self):
        if self._shared_densityL is not None:
            return self._shared_densityL
        if self.stream_density:
            return None
        return super().compute_density_layer()

    def _densities_for_q(self, q, use_bands):
        """rho^{(l)}(q+G_g) for one q as (N, n_G, n, m, k), and the k+q index table (n_k,)."""
        bs = self.crystal.band_structure
        if not self.stream_density:
            # take q first (a view), then the band subset: fancy-indexing the bands first
            # would copy the whole (n, m, k, q, G) array for every q
            dens = np.stack([np.moveaxis(np.asarray(self.densityL[l])[:, :, :, q, :][use_bands][:, use_bands], -1, 0)
                             for l in range(self.n_layers)])
            return dens, np.asarray(bs._k_plus_q_ind)[q, :]
        from .compat import import_pybandstructure
        import sys as _sys
        pb = import_pybandstructure()
        bs_module = _sys.modules[type(bs).__module__]
        full = bs._q_sample
        one_q = pb.Sample(basis_vectors=full.basis_vectors, denominator=full.denominator,
                          integer_coords=np.asarray(full.integer_coords)[[q]])
        quiet_tqdm, bs_module.tqdm = bs_module.tqdm, (lambda x, **kw: x)
        dens = None
        table = None
        try:
            bs._q_sample = one_q
            for l in range(self.n_layers):
                for g, op in enumerate(self.density_operators[l]):
                    matrix, table = bs.compute_matrix_elements(op)      # (n, m, k, 1)
                    block = matrix[use_bands][:, use_bands][:, :, :, 0]
                    if dens is None:
                        dens = np.empty((self.n_layers, len(self.density_operators[l])) + block.shape,
                                        dtype=np.complex128)
                    dens[l, g] = block
                    del matrix
        finally:
            bs._q_sample = full
            bs_module.tqdm = quiet_tqdm
        return dens, np.asarray(table[0])

    # ---- chi0 -----------------------------------------------------------------
    def polarization_tot(self, omega, q_index, G_index, G1_index, use_bands=slice(None)):
        if self.crystal.geometry._point_group.cardinality != 1:
            return super().polarization_tot(omega, q_index, G_index, G1_index, use_bands)
        omega = np.asarray(omega)
        key = (int(q_index), omega.shape, hash(omega.tobytes()), repr(use_bands))
        if getattr(self, "_chi_block_key", None) != key:
            self._chi_block_cache = None
            self._chi_block_cache = self.chi0_block(omega, int(q_index), use_bands)
            self._chi_block_key = key
        return self._chi_block_cache[:, :, :, G_index, G1_index]

    def chi0_block(self, omega, q_index, use_bands=slice(None)):
        """chi0[l, l', w, G, G'] for one q (``omega`` already carries + i eta)."""
        cr = self.crystal
        bs = cr.band_structure
        q = cr._q_symmetry_repr[0, q_index]
        G_rep = np.asarray(cr._G_symmetry_repr[0, :])
        dens, kpq = self._densities_for_q(q, use_bands)                    # (N, n_G, n, m, k)
        E = bs.energies[use_bands, :]
        f = bs.occupations[use_bands, :]
        dE = (E[:, None, :] - E[None, :, kpq]).reshape(-1)                 # (nmk,)
        weight = ((f[:, None, :] - f[None, :, kpq]) * bs.k_sample.weights[None, None, :]).reshape(-1)
        omega = np.asarray(omega, dtype=np.complex128).ravel()
        N, nG, n_w = self.n_layers, len(G_rep), len(omega)

        if not np.array_equal(G_rep, np.arange(dens.shape[1])):
            dens = dens[:, G_rep]
        B = np.array(dens.reshape(N * nG, -1))           # rows (l, G); becomes rho^* in place
        del dens
        np.nan_to_num(B, copy=False)
        np.conjugate(B, out=B)
        NG = N * nG
        X = np.zeros((NG * NG, n_w), dtype=np.complex128)
        abs_w = np.abs(weight)
        active = np.flatnonzero(abs_w > self.weight_cutoff * abs_w.max()) if abs_w.size and abs_w.max() > 0 \
            else np.zeros(0, dtype=int)
        step = max(1, int(self.block_bytes // (16 * max(n_w, NG * NG))))
        for s0 in range(0, active.size, step):
            idx = active[s0:s0 + step]
            D = 1.0 / (dE[idx, None] + omega[None, :])                     # (block, w)
            Bb = B[:, idx]                                                 # rho^*(l', G')
            Aw = np.conj(Bb) * weight[None, idx]                           # w_k df rho(l, G)
            P = (Aw[:, None, :] * Bb[None, :, :]).reshape(NG * NG, -1)
            X += P @ D
        X = X.reshape(N, nG, N, nG, n_w).transpose(0, 2, 4, 1, 3)          # [l, l', w, G, G']
        return cr._k_int_const * X


# -----------------------------------------------------------------------
# Self-tests (run via  python Layer_Helpers_plasmons_v2_6.py)
# -----------------------------------------------------------------------

def _self_test():
    from .layer_coulomb import TWOPI_E2_EV_NM, FOURPI_E2_EV_NM

    print("Layer_Helpers_plasmons v2.7 self-test:")

    # 1. _build_region_epsilon_provider — scalar
    mode, prov = _build_region_epsilon_provider(3, 4.0)
    assert mode == "static" and prov == 4.0, f"scalar provider failed: {mode}, {prov}"

    # 2. _build_region_epsilon_provider — numeric list returns ndarray
    mode, prov = _build_region_epsilon_provider(3, [1.0, 2.0, 3.0])
    assert mode == "static" and isinstance(prov, np.ndarray), f"list provider failed: {mode}, {type(prov)}"
    assert prov.dtype == np.complex128

    # 3. _build_region_epsilon_provider — mixed preset list
    mode, prov = _build_region_epsilon_provider(3, [1, "hBN", 1])
    assert mode == "dispersive" and callable(prov), f"mixed provider failed: {mode}"
    ep, ez = prov(1, 0.0)
    assert np.isfinite(ep) and np.isfinite(ez), "hBN preset returned non-finite at ω=0"
    print("  _build_region_epsilon_provider: OK")

    # 4. layer_helpers_G construction (static, no crystal)
    N = 3
    d_val = 0.335
    lh = layer_helpers_G(
        crystal=None, n_layers=N,
        epsilon_static=[1, 4.0, 4.0, 1],
        d=d_val * np.ones(N - 1),
        density_operators=None,
    )
    assert lh.epsilon_mode == "static"
    assert lh.n_regions == N + 1

    # 5. Coulomb potential matrix (static)
    q = 0.5
    V = lh.coulomb_potential_matrix(q_mag=q)
    assert V.shape == (N, N), f"V shape mismatch: {V.shape}"
    assert np.allclose(V, V.T, atol=1e-12), "V not symmetric for symmetric geometry"
    print("  layer_helpers_G static construction + V: OK")

    # 6. coulomb_potential_matrix_vectorized (scalar omega)
    GT = np.array([[0.0, 1.0], [0.0, 0.0]])
    qv = np.array([0.3, 0.0])
    Vv = lh.coulomb_potential_matrix_vectorized(qv, GT, omega=None)
    assert Vv.shape == (N, N, 2), f"vectorized V shape: {Vv.shape}"
    print("  coulomb_potential_matrix_vectorized (scalar ω): OK")

    # 7. coulomb_potential_matrix_vectorized (array omega, non-dispersive shortcut)
    omega_arr = np.array([0.0, 0.1])
    Vvo = lh.coulomb_potential_matrix_vectorized(qv, GT, omega=omega_arr)
    assert Vvo.shape == (N, N, 2, 2), f"vectorized V(ω) shape: {Vvo.shape}"
    assert np.allclose(Vvo[:, :, :, 0], Vvo[:, :, :, 1], atol=1e-12), \
        "static V should be independent of omega"
    print("  coulomb_potential_matrix_vectorized (array ω, static): OK")

    # 8. Dispersive construction (preset)
    lh_d = layer_helpers_G(
        crystal=None, n_layers=2,
        epsilon_static=[1, "hBN", 1],
        d=[d_val],
        density_operators=None,
    )
    assert lh_d.epsilon_mode == "dispersive"
    V0 = lh_d.coulomb_potential_matrix(q_mag=0.3, omega_val=0.0)
    Vw = lh_d.coulomb_potential_matrix(q_mag=0.3, omega_val=0.1)
    assert V0.shape == (2, 2) and Vw.shape == (2, 2)
    assert not np.allclose(V0, Vw), "dispersive V should differ at different ω"
    print("  dispersive (hBN preset): OK")

    # 9. diagonalize_coulomb_matrix — hermitian_part and use_bare
    eps_u = 4.0
    lh_u = layer_helpers_G(
        crystal=None, n_layers=N,
        epsilon_static=eps_u,
        d=d_val * np.ones(N - 1),
    )
    w1, U1 = lh_u.diagonalize_coulomb_matrix(q_mag=0.4, use_bare=False)
    w2, U2 = lh_u.diagonalize_coulomb_matrix(q_mag=0.4, use_bare=True)
    assert np.allclose(np.sort(w1), np.sort(w2), rtol=1e-8), \
        "bare vs screened eigenvalues differ in uniform medium"
    print("  diagonalize_coulomb_matrix: OK")

    # 10. loss_function_trace — smoke test with identity dielectric tensor
    n_q_t, n_omega_t, n_G_t = 2, 3, 1
    eps_tensor = np.zeros((n_q_t, n_omega_t, N, N, n_G_t, n_G_t), dtype=np.complex128)
    for qi in range(n_q_t):
        for oi in range(n_omega_t):
            eps_tensor[qi, oi, :, :, 0, 0] = np.eye(N)
    loss = lh.loss_function_trace(eps_tensor)
    assert loss.shape == (n_q_t, n_omega_t), f"loss shape: {loss.shape}"
    assert np.allclose(loss, 0.0, atol=1e-12), "loss of identity dielectric should be 0"
    print("  loss_function_trace (static, ε=I): OK")

    # 10b. loss_function_trace — screened-interaction formula with complex V
    #      Build a synthetic dispersive scenario: ε = I - V·χ with small χ,
    #      then verify -Im Tr[V·ε⁻¹] ≥ 0 for absorptive V (Im V < 0).
    N2 = 2
    n_q_t2, n_omega_t2, n_G_t2 = 1, 5, 1
    q_test = 0.3
    omegas_test = np.linspace(0.08, 0.20, n_omega_t2)
    lh_hbn = layer_helpers_G(
        crystal=None, n_layers=N2,
        epsilon_static=[1, "hBN", 1],
        d=[d_val],
    )
    cm_disp = np.zeros((n_q_t2, N2, N2, n_G_t2, n_omega_t2), dtype=np.complex128)
    eps_disp = np.zeros((n_q_t2, n_omega_t2, N2, N2, n_G_t2, n_G_t2), dtype=np.complex128)
    for oi, w in enumerate(omegas_test):
        V_w = lh_hbn.coulomb_potential_matrix(q_mag=q_test, omega_val=w)
        cm_disp[0, :, :, 0, oi] = V_w
        chi_fake = -0.01 * np.eye(N2, dtype=np.complex128)
        eps_block = np.eye(N2) - V_w @ chi_fake
        eps_disp[0, oi, :, :, 0, 0] = eps_block
    loss_screened = lh_hbn.loss_function_trace(eps_disp, coulomb_matrix=cm_disp)
    assert loss_screened.shape == (n_q_t2, n_omega_t2)
    assert np.all(loss_screened >= -1e-12), (
        f"screened loss should be >= -1e-12 (regression), got min={loss_screened.min()}"
    )
    loss_naive = lh_hbn.loss_function_trace(eps_disp)
    if np.any(loss_naive < -1e-14):
        print(f"    (confirmed: naive -Tr Im[ε⁻¹] goes negative, min={loss_naive.min():.4e})")
    print("  loss_function_trace (dispersive, V·ε⁻¹): OK")

    # 10c. loss_function_trace sign convention: returned value must be s1 = -Im(Tr(W00))
    loss_screened_no_debug = lh_hbn.loss_function_trace(eps_disp, coulomb_matrix=cm_disp)
    inv_00_test = np.linalg.inv(eps_disp[0, 2, :, :, 0, 0])
    W00_test = cm_disp[0, :, :, 0, 2] @ inv_00_test
    s1_expect = -np.imag(np.trace(W00_test))
    assert np.isclose(loss_screened_no_debug[0, 2], s1_expect, atol=1e-12), (
        "returned loss must equal s1=-Im(Tr(W00))"
    )
    print("  loss_function_trace (sign convention s1=-Im(Tr(W))): OK")

    # 10c'. screened loss with local fields (n_G > 1): head of the FULL inverse, V diagonal in G
    rng = np.random.default_rng(0)
    nG_lf = 3
    eps_lf = np.zeros((1, 2, N2, N2, nG_lf, nG_lf), dtype=np.complex128)
    for i in range(N2):
        for g in range(nG_lf):
            eps_lf[:, :, i, i, g, g] = 1.0
    eps_lf += 0.3 * (rng.normal(size=eps_lf.shape) + 1j * rng.normal(size=eps_lf.shape))
    cm_lf = (rng.normal(size=(1, N2, N2, nG_lf, 2)) + 3.0) * (1.0 - 0.1j)
    loss_lf = lh_hbn.loss_function_trace(eps_lf, coulomb_matrix=cm_lf)
    loss_lf_static = lh_hbn.loss_function_trace(eps_lf, coulomb_matrix=cm_lf[..., 0])  # 4-d, n_ω > 1
    head_idx = np.arange(N2) * nG_lf
    for oi in range(2):
        M = eps_lf[0, oi].transpose(0, 2, 1, 3).reshape(N2 * nG_lf, N2 * nG_lf)
        head = np.linalg.inv(M)[np.ix_(head_idx, head_idx)]
        for loss_val, V0 in ((loss_lf[0, oi], cm_lf[0, :, :, 0, oi]), (loss_lf_static[0, oi], cm_lf[0, :, :, 0, 0])):
            expect = -np.imag(np.trace(V0 @ head))
            assert np.isclose(loss_val, expect, rtol=1e-10, atol=1e-12), (
                f"screened loss with n_G={nG_lf}: {loss_val} vs {expect}")
    print("  loss_function_trace (screened, local fields n_G > 1, 4-d and 5-d V): OK")

    # 10d. ε inversion ordering: tensorinv matches np.linalg.inv(M)
    assert_epsilon_inversion_ordering(eps_tensor, q_idx=0, omega_idx=0, tol=1e-10)
    print("  assert_epsilon_inversion_ordering: OK")

    # 11. Regression: N=2 hBN Reststrahlen — loss non-negative (convention: retarded=True, screened formula) — tracked path runs and V is finite (diagnostics localize VA/WA)
    omega_reststrahlen = np.linspace(0.09, 0.16, 15)
    q_rest = 0.01
    V_tracked = lh_hbn.coulomb_potential_matrix_vectorized(
        np.array([q_rest, 0.0]), np.zeros((3, 1)), omega=omega_reststrahlen
    )
    assert V_tracked.shape == (2, 2, 1, 15)
    assert np.all(np.isfinite(V_tracked)), "Tracked V must be finite in Reststrahlen band"
    # Optional: run loss_diagnostics at one (q,ω) to ensure the pipeline works (no strict VA PSD here)
    eps_fake = np.zeros((1, 15, 2, 2, 1, 1), dtype=np.complex128)
    eps_fake[:, :, :, :, 0, 0] = np.eye(2)
    cm_fake = np.zeros((1, 2, 2, 1, 15), dtype=np.complex128)
    cm_fake[0, :, :, 0, :] = V_tracked[:, :, 0, :]
    _ = loss_diagnostics(lh_hbn, cm_fake, eps_fake, 0, 7, omega_reststrahlen, q_mag=q_rest, tol=1e-12)
    print("  regression N=2 hBN Reststrahlen (tracked V finite, diagnostics run): OK")

    # 11b. Regression: N=2 hBN, ω grid inside Reststrahlen, loss >= -1e-12 (screened formula)
    omega_rest = np.linspace(0.09, 0.16, 8)
    cm_rest = np.zeros((1, 2, 2, 1, len(omega_rest)), dtype=np.complex128)
    eps_rest = np.zeros((1, len(omega_rest), 2, 2, 1, 1), dtype=np.complex128)
    for oi, w in enumerate(omega_rest):
        V_w = lh_hbn.coulomb_potential_matrix(q_mag=0.02, omega_val=w)
        cm_rest[0, :, :, 0, oi] = V_w
        eps_rest[0, oi, :, :, 0, 0] = np.eye(2) - V_w @ (-0.005 * np.eye(2, dtype=np.complex128))
    loss_rest = lh_hbn.loss_function_trace(eps_rest, coulomb_matrix=cm_rest)
    assert np.all(loss_rest >= -1e-12), (
        f"Regression: loss in Reststrahlen must be >= -1e-12, got min={loss_rest.min()}"
    )
    print("  regression N=2 hBN Reststrahlen loss >= -1e-12: OK")

    # 12. transform_chi_to_eigenbasis
    chi_test = np.random.randn(N, N) + 1j * np.random.randn(N, N)
    U_test = np.linalg.qr(np.random.randn(N, N))[0]
    chi_eig = lh.transform_chi_to_eigenbasis(chi_test, U_test)
    chi_back = U_test @ chi_eig @ U_test.conj().T
    assert np.allclose(chi_back, chi_test, atol=1e-12), "eigenbasis round-trip failed"
    print("  transform_chi_to_eigenbasis: OK")

    # 13. v2.7 gates forwarded through layer_helpers_G
    lh_gate = layer_helpers_G(
        crystal=None, n_layers=1,
        epsilon_static=[3.0, 3.0],
        d=[],
        top_gate_d=4.0, bottom_gate_d=4.0,
    )
    for q in [0.05, 0.5, 2.0]:
        V = lh_gate.coulomb_potential_matrix(q_mag=q)[0, 0]
        V_ref = (TWOPI_E2_EV_NM / (3.0 * q)) * np.tanh(q * 4.0)
        assert abs((V - V_ref) / V_ref) < 1e-10, f"gated helper mismatch q={q}"
    assert lh_gate.top_gate_d == 4.0 and lh_gate.bottom_gate_d == 4.0
    print("  layer_helpers_G dual-gate forwarding (tanh(qd)): OK")

    print("All Layer_Helpers_plasmons v2.7 tests passed.")


if __name__ == "__main__":
    _self_test()
