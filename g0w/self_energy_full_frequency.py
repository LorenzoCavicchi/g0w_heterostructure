#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Full-frequency (non-PPA) retarded self-energy from the spectral density of W.

PPA replaces the dissipative part of W by pole δ-functions:

    [-Im W^R(q,ω)]/π  →  Σ_n R_n(q) [δ(ω-ω_n) − δ(ω+ω_n)].

This module integrates the reference positive-frequency form (passive,
time-reversal-symmetric electronic systems):

    Σ^R_{ij}(k,ε) = − Σ_{q,λ} w_q F^λ_{ij}(k,k−q)
        ∫_0^{ω_max} dω/π [−Im W^R_{ij}(q,ω)]
        { A_+(ω,ξ) / [ω + ξ − ε − iη_self]
        + A_−(ω,ξ) / [−ω + ξ − ε − iη_self] },

with A_+(ω,ξ) = n_B(ω) + n_F(−ξ),  A_−(ω,ξ) = n_B(ω) + n_F(+ξ).

Same denominators and population factors as ``compute_self_energy_layer`` (PPA).
For Im W^R(−ω) = −Im W^R(+ω), this matches integrating Im W^R dν/π from −ω_max to
+ω_max with the same kernel.

Slower than PPA — for validation and benchmarks on small grids only.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .compat import trapezoid as _trapezoid

from .self_energy_ppa import (
    FormFactorFn,
    _infer_radial_q_magnitudes,
    _k_minus_q_samples,
    _n_bose,
    _n_fermi,
    _normalize_form_factor_band,
    _reshape_eps_slice,
    coulomb_slice_to_V_full,
    make_angular_quadrature,
    make_radial_q_weights,
    screened_interaction_matrix,
)

SPECTRAL_DENSITY_CONVENTION = "S(q,omega) = -Im W^R(q,omega) / pi"


def build_W_grid_from_eps_coulomb(
    epsilon_layer: np.ndarray,
    coulomb_matrix: np.ndarray,
    omega_grid: np.ndarray,
) -> np.ndarray:
    r"""
    Build W^R(q,ω) = ε⁻¹ V.

    Returns W_grid : (n_q, n_ω, dim, dim) complex.
    """
    epsilon_layer = np.asarray(epsilon_layer, dtype=np.complex128)
    coulomb_matrix = np.asarray(coulomb_matrix, dtype=np.complex128)
    omega_grid = np.asarray(omega_grid, dtype=float).ravel()
    n_q = epsilon_layer.shape[0]
    n_omega = omega_grid.size
    _, N, n_G, dim = _reshape_eps_slice(epsilon_layer[0])
    W_grid = np.zeros((n_q, n_omega, dim, dim), dtype=np.complex128)
    for qi in range(n_q):
        eps_full, _, _, _ = _reshape_eps_slice(epsilon_layer[qi])
        cq = coulomb_matrix[qi] if coulomb_matrix.ndim >= 4 else coulomb_matrix
        V_full = coulomb_slice_to_V_full(cq, n_layers=N, n_G=n_G, n_omega=n_omega)
        for oi in range(n_omega):
            W_grid[qi, oi] = screened_interaction_matrix(eps_full[oi], V_full[oi])
    return W_grid


def spectral_density_minus_im_W(W: np.ndarray) -> np.ndarray:
    """S = [-Im W]/π, real array same shape as W."""
    W = np.asarray(W, dtype=np.complex128)
    return (-np.imag(W) / np.pi).astype(float)


def lorentzian_spectral_density(
    omega_grid: np.ndarray,
    omega_pole: float,
    residue_matrix: np.ndarray,
    gamma: float,
) -> np.ndarray:
    r"""
    S(ω) ≈ R · γ/[π((ω−ω_pole)²+γ²)]  with ∫_{−∞}^{∞} γ/[π((ω−ω_pole)²+γ²)] dω = 1.

    Returns shape (n_ω, *residue_matrix.shape).
    """
    omega_grid = np.asarray(omega_grid, dtype=float).ravel()
    R = np.asarray(residue_matrix, dtype=np.float64)
    L = (gamma / np.pi) / ((omega_grid - omega_pole) ** 2 + gamma ** 2)
    return L.reshape(-1, *([1] * R.ndim)) * R


def compute_self_energy_full_frequency(
    k_vec: np.ndarray,
    eps_values: np.ndarray,
    form_factor_fn: FormFactorFn,
    omega_grid: np.ndarray,
    W_grid: np.ndarray,
    q_vectors: np.ndarray,
    *,
    n_layers: int,
    n_G: int = 1,
    q_weights: Optional[np.ndarray] = None,
    T: float = 0.0,
    eta_self: float = 1e-4,
    head_only: bool = True,
    n_theta: int = 64,
) -> np.ndarray:
    r"""
    Full-frequency retarded self-energy (positive-ω trapezoidal integration).

    Parameters
    ----------
    W_grid : (n_q, n_ω, dim, dim) with dim = n_layers * n_G
    n_layers, n_G : layer / reciprocal-lattice head dimensions
    head_only : if True, Σ shape (N, N, n_ε); else (N, n_G, N, n_G, n_ε) [G=0 only for now]
    n_theta : angular samples for ⟨·⟩_θ on radial |q| grids; must match the
        value used for the PPA run this is being compared against.  See
        ``self_energy_plasmon_pole.compute_self_energy_layer``.

    Returns
    -------
    Sigma : complex ndarray [eV]
    """
    k_vec = np.asarray(k_vec, dtype=float).ravel()
    eps_values = np.asarray(eps_values, dtype=float).ravel()
    omega_grid = np.asarray(omega_grid, dtype=float).ravel()
    W_grid = np.asarray(W_grid, dtype=np.complex128)
    q_vectors = np.asarray(q_vectors, dtype=float)
    N, n_G = int(n_layers), int(n_G)
    dim = N * n_G
    if W_grid.shape[2:] != (dim, dim):
        raise ValueError(f"W_grid matrix dimension {W_grid.shape[2:]} != ({dim},{dim})")
    n_q, n_omega = W_grid.shape[0], W_grid.shape[1]
    if omega_grid.size != n_omega:
        raise ValueError("omega_grid length must match W_grid.shape[1].")
    if q_vectors.shape[0] != n_q:
        raise ValueError("q_vectors first dim must match W_grid.shape[0].")

    head_idx = np.arange(N) * n_G
    n_eps = eps_values.size
    if head_only:
        Sigma = np.zeros((N, N, n_eps), dtype=np.complex128)
    else:
        Sigma = np.zeros((N, n_G, N, n_G, n_eps), dtype=np.complex128)

    if q_weights is None:
        q_radial = _infer_radial_q_magnitudes(q_vectors)
        q_weights = (
            make_radial_q_weights(q_radial) if q_radial is not None else np.full(n_q, 1.0 / n_q)
        )
    else:
        q_weights = np.asarray(q_weights, dtype=float)

    q_is_radial = _infer_radial_q_magnitudes(q_vectors) is not None
    thetas, theta_w = make_angular_quadrature(n_theta if q_is_radial else 1)

    for qi in range(n_q):
        w_q = float(q_weights[qi])
        if w_q == 0:
            continue

        # S(q,ω) is isotropic in the direction of q, so it is built once per |q|
        # and reused for every angular sample.
        S_omega = np.zeros((n_omega, N, N), dtype=float)
        for oi in range(n_omega):
            S_full = spectral_density_minus_im_W(W_grid[qi, oi])
            block = S_full[np.ix_(head_idx, head_idx)] if head_only else S_full
            S_omega[oi] = block
        nB = _n_bose(omega_grid, T)

        kmq_samples = _k_minus_q_samples(
            k_vec, q_vectors[qi], radial=q_is_radial, thetas=thetas,
        )
        for ti in range(kmq_samples.shape[0]):
            w_qt = w_q * theta_w[ti]
            energies_band, F_band = form_factor_fn(k_vec, kmq_samples[ti])
            n_bands = energies_band.size
            F_band = _normalize_form_factor_band(
                F_band, n_bands=n_bands, N=N, n_G=n_G, head_only=head_only,
            )

            for la in range(n_bands):
                xi = float(energies_band[la])
                A_p_om = nB + float(_n_fermi(-xi, T))
                A_m_om = nB + float(_n_fermi(+xi, T))
                for ei, eps in enumerate(eps_values):
                    kern = A_p_om / (omega_grid + xi - eps - 1j * eta_self) + A_m_om / (
                        -omega_grid + xi - eps - 1j * eta_self
                    )
                    for i in range(N):
                        for j in range(N):
                            integrand = S_omega[:, i, j] * kern
                            val = -w_qt * F_band[la, i, j] * _trapezoid(
                                integrand, omega_grid,
                            )
                            if head_only:
                                Sigma[i, j, ei] += val
                            else:
                                Sigma[i, 0, j, 0, ei] += val

    return Sigma


def build_W_grid_from_spectral(
    omega_grid: np.ndarray,
    S_grid: np.ndarray,
) -> np.ndarray:
    r"""
    Synthetic retarded W with Re W = 0 and S = [-Im W]/π.

    Parameters
    ----------
    S_grid : (n_q, n_ω, dim, dim) real nonnegative spectral density.
    """
    S_grid = np.asarray(S_grid, dtype=float)
    return (-1j * np.pi * S_grid).astype(np.complex128)


def compute_self_energy_full_frequency_bilateral(
    k_vec: np.ndarray,
    eps_values: np.ndarray,
    form_factor_fn: FormFactorFn,
    omega_grid: np.ndarray,
    W_grid: np.ndarray,
    q_vectors: np.ndarray,
    *,
    n_layers: int,
    n_G: int = 1,
    q_weights: Optional[np.ndarray] = None,
    T: float = 0.0,
    eta_self: float = 1e-4,
    head_only: bool = True,
) -> np.ndarray:
    r"""
    Brute-force reference: ∫_{−ωmax}^{+ωmax} Im W dν/π · kernel(ν).

    ``omega_grid`` must span negative and positive frequencies.  For odd Im W,
    agrees with ``compute_self_energy_full_frequency`` on ω ≥ 0 only.
    """
    k_vec = np.asarray(k_vec, dtype=float).ravel()
    eps_values = np.asarray(eps_values, dtype=float).ravel()
    omega_grid = np.asarray(omega_grid, dtype=float).ravel()
    W_grid = np.asarray(W_grid, dtype=np.complex128)
    N, n_G = int(n_layers), int(n_G)
    dim = N * n_G
    head_idx = np.arange(N) * n_G
    n_q, n_omega = W_grid.shape[0], W_grid.shape[1]
    n_eps = eps_values.size
    if head_only:
        Sigma = np.zeros((N, N, n_eps), dtype=np.complex128)
    else:
        Sigma = np.zeros((N, n_G, N, n_G, n_eps), dtype=np.complex128)

    if q_weights is None:
        q_radial = _infer_radial_q_magnitudes(q_vectors)
        q_weights = (
            make_radial_q_weights(q_radial) if q_radial is not None else np.full(n_q, 1.0 / n_q)
        )
    else:
        q_weights = np.asarray(q_weights, dtype=float)

    for qi in range(n_q):
        w_q = float(q_weights[qi])
        if w_q == 0:
            continue
        q_vec = q_vectors[qi].ravel()
        Ddim = max(k_vec.size, q_vec.size)
        kk = np.zeros(Ddim)
        kk[: k_vec.size] = k_vec
        qq = np.zeros(Ddim)
        qq[: q_vec.size] = q_vec
        energies_band, F_band = form_factor_fn(k_vec, kk - qq)
        F_band = _normalize_form_factor_band(
            F_band, n_bands=energies_band.size, N=N, n_G=n_G, head_only=head_only,
        )
        S_omega = np.zeros((n_omega, N, N), dtype=float)
        for oi in range(n_omega):
            S_full = spectral_density_minus_im_W(W_grid[qi, oi])
            S_omega[oi] = S_full[np.ix_(head_idx, head_idx)] if head_only else S_full
        nB = _n_bose(omega_grid, T)
        for la in range(energies_band.size):
            xi = float(energies_band[la])
            A_p_om = nB + float(_n_fermi(-xi, T))
            A_m_om = nB + float(_n_fermi(+xi, T))
            for ei, eps in enumerate(eps_values):
                kern = A_p_om / (omega_grid + xi - eps - 1j * eta_self) + A_m_om / (
                    -omega_grid + xi - eps - 1j * eta_self
                )
                for i in range(N):
                    for j in range(N):
                        val = -w_q * F_band[la, i, j] * _trapezoid(
                            S_omega[:, i, j] * kern, omega_grid,
                        )
                        if head_only:
                            Sigma[i, j, ei] += val
                        else:
                            Sigma[i, 0, j, 0, ei] += val
    return Sigma


def compute_self_energy_ppa_from_poles(
    k_vec: np.ndarray,
    eps_values: np.ndarray,
    form_factor_fn: FormFactorFn,
    q_vectors: np.ndarray,
    pole_frequencies: Sequence[Sequence[float]],
    residues: Sequence[Sequence[np.ndarray]],
    *,
    q_weights: Optional[np.ndarray] = None,
    n_layers: int,
    n_G: int = 1,
    T: float = 0.0,
    eta_self: float = 1e-4,
    head_only: bool = True,
) -> np.ndarray:
    """Manual Σ-PPA from explicit (ω_n, R_n) lists (validation helper)."""
    from .self_energy_ppa import population_factors

    k_vec = np.asarray(k_vec, dtype=float).ravel()
    eps_values = np.asarray(eps_values, dtype=float).ravel()
    N, n_G = int(n_layers), int(n_G)
    n_eps = eps_values.size
    head_idx = np.arange(N) * n_G
    if head_only:
        Sigma = np.zeros((N, N, n_eps), dtype=np.complex128)
    else:
        Sigma = np.zeros((N, n_G, N, n_G, n_eps), dtype=np.complex128)
    n_q = len(pole_frequencies)
    if q_weights is None:
        q_weights = np.full(n_q, 1.0 / max(n_q, 1))
    for qi in range(n_q):
        w_q = float(q_weights[qi])
        q_vec = np.asarray(q_vectors[qi], dtype=float).ravel()
        Ddim = max(k_vec.size, q_vec.size)
        kk = np.zeros(Ddim)
        kk[: k_vec.size] = k_vec
        qq = np.zeros(Ddim)
        qq[: q_vec.size] = q_vec
        energies_band, F_band = form_factor_fn(k_vec, kk - qq)
        F_band = _normalize_form_factor_band(
            F_band, n_bands=energies_band.size, N=N, n_G=n_G, head_only=head_only,
        )
        for w_n, R_n in zip(pole_frequencies[qi], residues[qi]):
            R_head = np.asarray(R_n, dtype=np.complex128)[np.ix_(head_idx, head_idx)]
            A_plus, A_minus = population_factors(float(w_n), energies_band, T)
            xi = energies_band
            denom_p = (float(w_n) + xi)[:, None] - eps_values[None, :] - 1j * eta_self
            denom_m = (-float(w_n) + xi)[:, None] - eps_values[None, :] - 1j * eta_self
            contrib = A_plus[:, None] / denom_p + A_minus[:, None] / denom_m
            if head_only:
                Sigma += -w_q * R_head[:, :, None] * np.einsum("lij,le->ije", F_band, contrib)
            else:
                raise NotImplementedError("head_only=False not implemented for manual PPA.")
    return Sigma


def project_W_optical_acoustic(
    W: np.ndarray,
    u_plus: np.ndarray,
    u_minus: np.ndarray,
) -> Tuple[complex, complex]:
    """Scalars W_++ = u_+† W u_+,  W_-- = u_-† W u_-."""
    u_plus = np.asarray(u_plus, dtype=np.complex128).ravel()
    u_minus = np.asarray(u_minus, dtype=np.complex128).ravel()
    W = np.asarray(W, dtype=np.complex128)
    return np.conj(u_plus) @ W @ u_plus, np.conj(u_minus) @ W @ u_minus


def spectral_matrix_optical_acoustic(
    S_layer: np.ndarray,
    u_plus: np.ndarray,
    u_minus: np.ndarray,
) -> Tuple[float, float, np.ndarray]:
    """Return (S_pp, S_mm, off-diagonal u_+† S u_-)."""
    u_plus = np.asarray(u_plus, dtype=np.complex128).ravel()
    u_minus = np.asarray(u_minus, dtype=np.complex128).ravel()
    S = np.asarray(S_layer, dtype=float)
    S_pp = float(np.real(np.conj(u_plus) @ S @ u_plus))
    S_mm = float(np.real(np.conj(u_minus) @ S @ u_minus))
    off = np.conj(u_plus) @ S @ u_minus
    return S_pp, S_mm, off


def compare_sigma_ppa_full(
    Sigma_ppa: np.ndarray,
    Sigma_full: np.ndarray,
    *,
    label: str = "",
) -> Dict[str, float]:
    """Error metrics between PPA and full-frequency Σ (same shape)."""
    diff = np.asarray(Sigma_ppa) - np.asarray(Sigma_full)
    denom = np.maximum(np.abs(Sigma_full), 1e-12)
    return {
        "label": label,
        "max_abs_diff": float(np.max(np.abs(diff))),
        "max_rel_diff": float(np.max(np.abs(diff) / denom)),
        "max_abs_diff_Re": float(np.max(np.abs(np.real(diff)))),
        "max_abs_diff_Im": float(np.max(np.abs(np.imag(diff)))),
        "max_rel_diff_Re": float(np.max(np.abs(np.real(diff)) / denom)),
        "max_rel_diff_Im": float(np.max(np.abs(np.imag(diff)) / denom)),
    }


def compute_self_energy_full_frequency_channels(
    k_vec: np.ndarray,
    eps_values: np.ndarray,
    form_factor_fn: FormFactorFn,
    omega_grid: np.ndarray,
    W_grid: np.ndarray,
    q_vectors: np.ndarray,
    u_plus: np.ndarray,
    u_minus: np.ndarray,
    *,
    n_layers: int = 2,
    n_G: int = 1,
    q_weights: Optional[np.ndarray] = None,
    T: float = 0.0,
    eta_self: float = 1e-4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Channel-resolved full-frequency Σ (optical / acoustic scalars).

    Integrates projected spectral densities S_++ and S_-- separately, then
    maps Σ_layer = U diag(I_++, I_--) U†.

    Returns
    -------
    Sigma_layer, Sigma_optical, Sigma_acoustic : each (N, N, n_eps).
    """
    u_plus = np.asarray(u_plus, dtype=np.complex128).ravel()
    u_minus = np.asarray(u_minus, dtype=np.complex128).ravel()
    U = np.column_stack([u_plus, u_minus])
    N = n_layers
    eps_values = np.asarray(eps_values, dtype=float).ravel()
    n_eps = eps_values.size
    I_pp = np.zeros(n_eps, dtype=np.complex128)
    I_mm = np.zeros(n_eps, dtype=np.complex128)

    omega_grid = np.asarray(omega_grid, dtype=float).ravel()
    W_grid = np.asarray(W_grid, dtype=np.complex128)
    head_idx = np.arange(N) * n_G

    if q_weights is None:
        q_radial = _infer_radial_q_magnitudes(q_vectors)
        q_weights = (
            make_radial_q_weights(q_radial)
            if q_radial is not None
            else np.full(W_grid.shape[0], 1.0 / W_grid.shape[0])
        )

    for qi in range(W_grid.shape[0]):
        w_q = float(q_weights[qi])
        q_vec = q_vectors[qi].ravel()
        Ddim = max(k_vec.size, q_vec.size)
        kk = np.zeros(Ddim)
        kk[: k_vec.size] = k_vec
        qq = np.zeros(Ddim)
        qq[: q_vec.size] = q_vec
        energies_band, F_band = form_factor_fn(k_vec, kk - qq)
        F_band = _normalize_form_factor_band(
            F_band, n_bands=energies_band.size, N=N, n_G=n_G, head_only=True,
        )
        n_omega = omega_grid.size
        S_pp_om = np.zeros(n_omega)
        S_mm_om = np.zeros(n_omega)
        for oi in range(n_omega):
            S_head = spectral_density_minus_im_W(W_grid[qi, oi])[np.ix_(head_idx, head_idx)]
            S_pp_om[oi], S_mm_om[oi], _ = spectral_matrix_optical_acoustic(
                S_head, u_plus, u_minus,
            )
        nB = _n_bose(omega_grid, T)
        for la in range(energies_band.size):
            xi = float(energies_band[la])
            A_p = nB + float(_n_fermi(-xi, T))
            A_m = nB + float(_n_fermi(+xi, T))
            for ei, eps in enumerate(eps_values):
                kern = A_p / (omega_grid + xi - eps - 1j * eta_self) + A_m / (
                    -omega_grid + xi - eps - 1j * eta_self
                )
                I_pp[ei] += -w_q * F_band[la, 0, 0] * _trapezoid(S_pp_om * kern, omega_grid)
                I_mm[ei] += -w_q * F_band[la, 1, 1] * _trapezoid(S_mm_om * kern, omega_grid)

    Sigma_layer = np.zeros((N, N, n_eps), dtype=np.complex128)
    Sigma_opt = np.zeros((N, N, n_eps), dtype=np.complex128)
    Sigma_ac = np.zeros((N, N, n_eps), dtype=np.complex128)
    for ei in range(n_eps):
        Sig_ch = np.diag([I_pp[ei], I_mm[ei]])
        Sigma_layer[:, :, ei] = U @ Sig_ch @ U.conj().T
        Sigma_opt[:, :, ei] = U @ np.diag([I_pp[ei], 0.0]) @ U.conj().T
        Sigma_ac[:, :, ei] = U @ np.diag([0.0, I_mm[ei]]) @ U.conj().T
    return Sigma_layer, Sigma_opt, Sigma_ac


# ---------------------------------------------------------------------------
# Full-frequency self-energy from pybandstructure density-matrix elements
# ---------------------------------------------------------------------------

def compute_self_energy_full_frequency_pybandstructure(
    layer_helper,
    k_index: int,
    band_index: int,
    eps_values: np.ndarray,
    omega_grid: np.ndarray,
    W_grid: np.ndarray,
    *,
    q_weights: Optional[np.ndarray] = None,
    band_index_right: Optional[int] = None,
    bands_use: Optional[Sequence[int]] = None,
    T: float = 0.0,
    eta_self: float = 1e-4,
    head_only: bool = True,
    eps_chunk: int = 32,
) -> np.ndarray:
    r"""
    Full-frequency Σ^R from ``layer_helper.densityL`` — the pybandstructure
    counterpart of :func:`compute_self_energy_full_frequency`.

    Why this exists
    ---------------
    ``compute_self_energy_full_frequency`` takes ``form_factor_fn(k, k−q)``, a
    callable of momentum *vectors*.  pybandstructure form factors are stored as
    ``densityL[l, n, m, ki, qi, G]`` and are addressed by the *index pair*
    ``(ki, qi)``; there is no way to recover ``qi`` from a ``k−q`` vector inside
    that callable.  So the full-frequency route and the pybandstructure route
    could not previously be combined: ``compute_self_energy_pybandstructure``
    is PPA-only, and the full-frequency driver is analytic-form-factor-only.
    This closes that gap.

    Physics — identical to ``compute_self_energy_pybandstructure`` except that
    the pole sum Σ_n R_n(q) δ(ω−ω_n) is replaced by the full spectral density:

        Σ_{ij}(k,ε) = − Σ_q w_q Σ_m F^{n,m}_{ij}(k,q)
                        ∫_0^{ω_max} dω  S_{ij}(q,ω)  D_m(ω,ε)

        S_{ij}      = −Im W_{ij}(q,ω)/π
        F^{n,m}_{ij} = densityL[i,n,m,ki,qi,0] · densityL[j,m',m,ki,qi,0]*
        D_m         = A_+/(ω + ξ_m − ε − iη) + A_−/(−ω + ξ_m − ε − iη)
        A_+ = n_B(ω) + n_F(−ξ_m),  A_− = n_B(ω) + n_F(+ξ_m)
        ξ_m = E_m(k+q) − μ,  index from ``band_structure.k_plus_q_ind[qi, ki]``

    No angular average is applied: a pybandstructure ``q_sample`` is an explicit
    2D mesh, so the q-sum already covers the directions (unlike the radial-grid
    case, where ``n_theta`` is mandatory).

    Returns Σ in the **layer** basis, shape ``(N, N, n_ε)``, matching
    ``compute_self_energy_pybandstructure`` so the two are drop-in
    interchangeable.  The band-diagonal self-energy is the sum over ``(i, j)``.

    Parameters
    ----------
    W_grid : (n_q, n_ω, dim, dim) complex
        Screened interaction on the same q-ordering as ``q_sample``.
    band_index_right : int, optional
        Right band index for an off-diagonal Σ_{n n'}; defaults to ``band_index``.
    bands_use : sequence of int, optional
        Restrict the intermediate-state sum (default: all computed bands).
    """
    if getattr(layer_helper, "densityL", None) is None:
        raise ValueError(
            "layer_helper.densityL is not available; call compute_density_layer().")
    if not head_only:
        raise NotImplementedError(
            "Only the G = G' = 0 head block is implemented (head_only=True).")

    bs = layer_helper.crystal.band_structure
    densityL = np.asarray(layer_helper.densityL)
    eps_values = np.asarray(eps_values, dtype=float).ravel()
    omega_grid = np.asarray(omega_grid, dtype=float).ravel()
    W_grid = np.asarray(W_grid, dtype=np.complex128)

    n_q, n_omega = W_grid.shape[0], W_grid.shape[1]
    if omega_grid.size != n_omega:
        raise ValueError("omega_grid length must match W_grid.shape[1].")
    N = int(densityL.shape[0])
    if W_grid.shape[2] < N or W_grid.shape[3] < N:
        raise ValueError(
            f"W_grid matrix block {W_grid.shape[2:]} smaller than n_layers={N}.")

    n_r = int(band_index) if band_index_right is None else int(band_index_right)
    n_bands = densityL.shape[1]
    m_list = (np.arange(n_bands) if bands_use is None
              else np.asarray(bands_use, dtype=int))

    if q_weights is None:
        q_weights = np.full(n_q, 1.0 / n_q)
    q_weights = np.asarray(q_weights, dtype=float).ravel()

    # trapezoidal weights on the (possibly non-uniform) omega grid
    tw = np.zeros_like(omega_grid)
    if omega_grid.size > 1:
        d = np.diff(omega_grid)
        tw[:-1] += 0.5 * d
        tw[1:] += 0.5 * d
    else:
        tw[:] = 1.0
    nB = _n_bose(omega_grid, T)

    energies = bs.energies
    k_plus_q = bs.k_plus_q_ind
    mu = float(bs.chemical_potential)

    rho_L = densityL[:, int(band_index), :, int(k_index), :, 0]   # (N, n_bands, n_q)
    rho_R = densityL[:, n_r, :, int(k_index), :, 0]

    n_eps = eps_values.size
    Sigma = np.zeros((N, N, n_eps), dtype=np.complex128)
    S_all = -np.imag(W_grid[:, :, :N, :N]) / np.pi            # (n_q, n_ω, N, N)

    for qi in range(n_q):
        w_q = float(q_weights[qi])
        if w_q == 0.0:
            continue
        rl = rho_L[:, m_list, qi]                              # (N, n_m)
        rr = rho_R[:, m_list, qi]
        F = np.einsum("am,bm->mab", rl, np.conj(rr))           # (n_m, N, N)
        # only (a, b) pairs with a nonzero form factor cost anything; with an
        # inert sheet (zero density operator) whole rows/columns drop out.
        active = np.argwhere(np.any(np.abs(F) > 0.0, axis=0))
        if active.size == 0:
            continue

        k_pq = int(k_plus_q[qi, int(k_index)])
        xi = energies[m_list, k_pq] - mu                        # (n_m,)
        A_p = nB[None, :] + _n_fermi(-xi, T)[:, None]           # (n_m, n_ω)
        A_m = nB[None, :] + _n_fermi(+xi, T)[:, None]
        S_q = S_all[qi] * tw[:, None, None]                     # (n_ω, N, N)

        for e0 in range(0, n_eps, int(eps_chunk)):
            e1 = min(e0 + int(eps_chunk), n_eps)
            u = xi[:, None, None] - eps_values[None, None, e0:e1]
            om = omega_grid[None, :, None]
            D = (A_p[:, :, None] / (om + u - 1j * eta_self)
                 + A_m[:, :, None] / (-om + u - 1j * eta_self))  # (n_m, n_ω, n_e)
            for a, b in active:
                Sigma[a, b, e0:e1] += -w_q * np.einsum(
                    "m,w,mwe->e", F[:, a, b], S_q[:, a, b], D)

    return Sigma


# ---------------------------------------------------------------------------
# Hermitian spectral density and band-matrix self-energy (new in g0w)
# ---------------------------------------------------------------------------

def spectral_density_matrix(W: np.ndarray, *, hermitian: bool = True) -> np.ndarray:
    r"""
    Spectral density of the screened interaction on the last two (layer) axes.

    ``hermitian=True`` (default):  S = i (W - W^dagger) / (2 pi),  i.e. minus the
    anti-Hermitian part of W divided by pi.  It is Hermitian and positive
    semidefinite at w > 0 for a passive (causal, dissipative) medium, which is
    what guarantees  Im Sigma^R <= 0  for every band matrix element below.

    ``hermitian=False``:  S = -Im W / pi element by element (the convention of
    the original scripts).  The two coincide when W is symmetric,
    W_ab = W_ba, which holds for reciprocal systems; they differ at O(eta)
    when chi0 is computed for one valley only and multiplied by the
    degeneracy (time reversal between valleys is then not exact).
    """
    W = np.asarray(W, dtype=np.complex128)
    if not hermitian:
        return (-np.imag(W) / np.pi).astype(float)
    Wd = np.conj(np.swapaxes(W, -1, -2))
    return (1j * (W - Wd) / (2.0 * np.pi))


def compute_self_energy_band_matrix(
    densityL: np.ndarray,
    energies: np.ndarray,
    k_plus_q_ind: np.ndarray,
    mu: float,
    S_grid: np.ndarray,
    omega_grid: np.ndarray,
    q_weights: np.ndarray,
    k_index: int,
    bands: Sequence[int],
    eps_values: np.ndarray,
    *,
    bands_use: Optional[Sequence[int]] = None,
    T: float = 0.0,
    eta_self: float = 1e-3,
    eps_chunk: int = 256,
) -> np.ndarray:
    r"""
    Full-frequency G0W correlation self-energy as a **matrix in band space**.

        Sigma_{n n'}(k, eps) = - sum_q w_q sum_m sum_{a b}
              rho^a_{n m}(k, q) [rho^b_{n' m}(k, q)]^*
              int_0^{w_max} dw  S_{a b}(q, w)  D_m(w, eps)

        rho^a_{n m}(k, q) = densityL[a, n, m, k, q, 0] = <k n| rho_a(q) |k+q m>
        D_m = A_+/(w + xi_m - eps - i eta) + A_-/(-w + xi_m - eps - i eta)
        A_+ = n_B(w) + n_F(-xi_m),   A_- = n_B(w) + n_F(xi_m),   xi_m = E_m(k+q) - mu

    This is Wei et al. Eq. (37)+(39) with the sum over the moire G vectors
    restricted to the head (local fields off) and the layer index playing the
    role of the reciprocal-lattice index; with S Hermitian PSD (see
    :func:`spectral_density_matrix`) the anti-Hermitian part of the returned
    matrix is negative semidefinite at every eps.  The diagonal n = n'
    reproduces ``compute_self_energy_full_frequency_pybandstructure`` summed
    over the layer indices.

    Parameters
    ----------
    densityL : (N, n_bands, n_bands, n_k, n_q, n_G) density matrix elements;
        the N sheets must be in the same order as the last two axes of ``S_grid``.
    S_grid : (n_q, n_w, N, N) spectral density of W on the same q mesh.
    q_weights : (n_q,) integration weights for int d^2q/(2 pi)^2 (no degeneracy).
    bands : external band indices (n, n').
    bands_use : intermediate bands m (default: all bands in densityL).

    Returns
    -------
    Sigma : (n_b, n_b, n_eps) complex [eV]
    """
    densityL = np.asarray(densityL)
    energies = np.asarray(energies, dtype=float)
    k_plus_q_ind = np.asarray(k_plus_q_ind, dtype=int)
    S_grid = np.asarray(S_grid)
    omega_grid = np.asarray(omega_grid, dtype=float).ravel()
    eps_values = np.asarray(eps_values, dtype=float).ravel()
    q_weights = np.asarray(q_weights, dtype=float).ravel()
    bands = [int(b) for b in bands]
    m_list = (np.arange(densityL.shape[1]) if bands_use is None
              else np.asarray(bands_use, dtype=int))
    N = densityL.shape[0]
    n_q, n_w = S_grid.shape[0], S_grid.shape[1]
    if S_grid.shape[2:] != (N, N):
        raise ValueError(f"S_grid layer block {S_grid.shape[2:]} != ({N},{N}).")
    if omega_grid.size != n_w:
        raise ValueError("omega_grid length must match S_grid.shape[1].")

    from .compat import trapezoid_weights
    tw = trapezoid_weights(omega_grid)
    nB = _n_bose(omega_grid, T)
    nb, nm, ne = len(bands), len(m_list), eps_values.size
    Sigma = np.zeros((nb, nb, ne), dtype=np.complex128)
    ki = int(k_index)
    # rho[a, n, m, q] for this k only (never fancy-index the full (k, q) array)
    rho_k = np.nan_to_num(np.asarray(densityL[:, :, :, ki, :, 0])[:, bands][:, :, m_list])

    for qi in range(n_q):
        w_q = float(q_weights[qi])
        if w_q == 0.0:
            continue
        R = rho_k[:, :, :, qi]                                              # (N, nb, nm)
        if not np.any(R):
            continue
        Sq = S_grid[qi] * tw[:, None, None]                                  # (n_w, N, N)
        # C[n, n', m, w] = sum_ab R[a,n,m] S[w,a,b] R[b,n',m]^*
        C = np.einsum("anm,wab,bpm->npmw", R, Sq, np.conj(R), optimize=True)
        C = C.reshape(nb * nb, nm * n_w)
        k_pq = int(k_plus_q_ind[qi, ki])
        xi = energies[m_list, k_pq] - float(mu)                              # (nm,)
        A_p = nB[None, :] + _n_fermi(-xi, T)[:, None]                        # (nm, n_w)
        A_m = nB[None, :] + _n_fermi(+xi, T)[:, None]
        for e0 in range(0, ne, int(eps_chunk)):
            e1 = min(e0 + int(eps_chunk), ne)
            u = xi[:, None, None] - eps_values[None, None, e0:e1]            # (nm, 1, ne_c)
            om = omega_grid[None, :, None]
            D = (A_p[:, :, None] / (om + u - 1j * eta_self)
                 + A_m[:, :, None] / (-om + u - 1j * eta_self))              # (nm, n_w, ne_c)
            Sigma[:, :, e0:e1] += (-w_q * (C @ D.reshape(nm * n_w, e1 - e0))).reshape(nb, nb, e1 - e0)
    return Sigma


# ---------------------------------------------------------------------------
# Fast route: tabulated frequency integrals ("resolvent kernels")
# ---------------------------------------------------------------------------

class ResolventKernel:
    r"""
    Tabulated frequency integrals of a spectral density, reused for every k and eps.

    The full-frequency self-energy only needs, for each q and each pair of
    sheet indices (a, b), the four functions of u = xi - eps

        P(u)  = int_0^{w_max} dw  S(q, w) / ( w + u - i eta)
        M(u)  = int_0^{w_max} dw  S(q, w) / (-w + u - i eta)
        Bp(u) = int_0^{w_max} dw  S(q, w) n_B(w) / ( w + u - i eta)
        Bm(u) = int_0^{w_max} dw  S(q, w) n_B(w) / (-w + u - i eta)

    because the population factors split as A_+ = n_B(w) + n_F(-xi),
    A_- = n_B(w) + n_F(xi), and xi enters the denominators only through u:

        int dw S D(w, eps) = n_F(-xi) P(u) + Bp(u) + n_F(xi) M(u) + Bm(u).

    The integrals are evaluated once on a uniform u grid (spacing ``du``, which
    should be <= eta/8 for 0.2 % interpolation accuracy) with the same
    trapezoidal rule as the direct route, then linearly interpolated.  At T = 0
    the Bose terms vanish (the w grid starts above 0).  This is an exact
    reformulation up to the interpolation error; it is validated against the
    direct contraction in the tests.

    Parameters
    ----------
    omega_grid : (n_w,) real, positive frequencies
    S : (n_q, n_w, *block) spectral density (complex or real)
    u_min, u_max : range of u = xi - eps that will be requested
    eta : broadening of the self-energy denominators
    du : u spacing (default eta/8)
    """

    def __init__(self, omega_grid, S, u_min: float, u_max: float, eta: float,
                 T: float = 0.0, du: Optional[float] = None, chunk: int = 2048):
        from .compat import trapezoid_weights
        omega_grid = np.asarray(omega_grid, dtype=float).ravel()
        S = np.asarray(S)
        self.block_shape = S.shape[2:]
        self.n_q = S.shape[0]
        self.eta = float(eta)
        self.T = float(T)
        self.du = float(du) if du is not None else self.eta / 8.0
        self.u0 = float(u_min) - 2 * self.du
        n_u = int(np.ceil((float(u_max) + 2 * self.du - self.u0) / self.du)) + 1
        self.u = self.u0 + self.du * np.arange(n_u)
        tw = trapezoid_weights(omega_grid)
        Sf = (S * tw.reshape((1, -1) + (1,) * len(self.block_shape))).reshape(self.n_q, omega_grid.size, -1)
        nB = _n_bose(omega_grid, self.T)
        with_bose = self.T > 0
        nblk = Sf.shape[2]
        self.P = np.zeros((self.n_q, n_u, nblk), dtype=np.complex128)
        self.M = np.zeros_like(self.P)
        self.Bp = np.zeros_like(self.P) if with_bose else None
        self.Bm = np.zeros_like(self.P) if with_bose else None
        for u0 in range(0, n_u, int(chunk)):
            u1 = min(u0 + int(chunk), n_u)
            uu = self.u[u0:u1, None]
            Kp = 1.0 / (omega_grid[None, :] + uu - 1j * self.eta)          # (n_u_c, n_w)
            Km = 1.0 / (-omega_grid[None, :] + uu - 1j * self.eta)
            self.P[:, u0:u1] = np.einsum("uw,qwb->qub", Kp, Sf, optimize=True)
            self.M[:, u0:u1] = np.einsum("uw,qwb->qub", Km, Sf, optimize=True)
            if with_bose:
                self.Bp[:, u0:u1] = np.einsum("uw,qwb->qub", Kp * nB[None, :], Sf, optimize=True)
                self.Bm[:, u0:u1] = np.einsum("uw,qwb->qub", Km * nB[None, :], Sf, optimize=True)

    def _interp(self, table, qi, u):
        x = (np.asarray(u, float) - self.u0) / self.du
        if np.any(x < 0) or np.any(x > self.u.size - 1):
            raise ValueError(f"u outside the tabulated range [{self.u[0]:.4f}, {self.u[-1]:.4f}] eV")
        i = np.minimum(np.floor(x).astype(int), self.u.size - 2)
        t = (x - i)[..., None]
        T = table[qi]
        return (1.0 - t) * T[i] + t * T[i + 1]                              # (..., nblk)

    def integral(self, qi: int, xi: np.ndarray, eps: np.ndarray) -> np.ndarray:
        r"""
        int dw S(q_i, w) D(w; xi, eps) for xi (n_x,) and eps (n_e,).

        Returns (n_x, n_e, *block).
        """
        xi = np.asarray(xi, float).ravel()
        eps = np.asarray(eps, float).ravel()
        u = xi[:, None] - eps[None, :]
        nFm = _n_fermi(-xi, self.T)[:, None, None]
        nFp = _n_fermi(+xi, self.T)[:, None, None]
        out = nFm * self._interp(self.P, qi, u) + nFp * self._interp(self.M, qi, u)
        if self.Bp is not None:
            out = out + self._interp(self.Bp, qi, u) + self._interp(self.Bm, qi, u)
        return out.reshape(xi.size, eps.size, *self.block_shape)


def compute_self_energy_band_matrix_kernel(
    densityL: np.ndarray,
    energies: np.ndarray,
    k_plus_q_ind: np.ndarray,
    mu: float,
    kernel: ResolventKernel,
    q_weights: np.ndarray,
    k_index: int,
    bands: Sequence[int],
    eps_values: np.ndarray,
    *,
    bands_use: Optional[Sequence[int]] = None,
) -> np.ndarray:
    r"""
    Same quantity as :func:`compute_self_energy_band_matrix`, with the frequency
    integral taken from a precomputed :class:`ResolventKernel` built on the
    (n_q, n_w, N, N) spectral density.  The kernel depends only on q, so it is
    built once and reused for every external k: this is the route to use for
    twist maps.  Returns (n_b, n_b, n_eps).
    """
    densityL = np.asarray(densityL)
    energies = np.asarray(energies, float)
    bands = [int(b) for b in bands]
    m_list = (np.arange(densityL.shape[1]) if bands_use is None
              else np.asarray(bands_use, dtype=int))
    eps_values = np.asarray(eps_values, float).ravel()
    q_weights = np.asarray(q_weights, float).ravel()
    N = densityL.shape[0]
    if kernel.block_shape != (N, N):
        raise ValueError(f"kernel block {kernel.block_shape} != ({N},{N})")
    nb, ne = len(bands), eps_values.size
    acc = np.zeros((ne, nb, nb), dtype=np.complex128)
    ki = int(k_index)
    rho_k = np.nan_to_num(np.asarray(densityL[:, :, :, ki, :, 0])[:, bands][:, :, m_list])  # (N, nb, nm, n_q)
    for qi in range(kernel.n_q):
        w_q = float(q_weights[qi])
        if w_q == 0.0:
            continue
        Rm = np.transpose(rho_k[:, :, :, qi], (2, 1, 0))                     # (nm, nb, N)
        if not np.any(Rm):
            continue
        xi = energies[m_list, int(k_plus_q_ind[qi, ki])] - float(mu)
        I = kernel.integral(qi, xi, eps_values)                               # (nm, ne, N, N)
        # sum_{m a b} R[a,n,m] I[m,e,a,b] R[b,n',m]^*  as two batched matmuls
        Y = np.matmul(Rm[:, None, :, :], I)                                   # (nm, ne, nb, N)
        Z = np.matmul(Y, np.conj(np.transpose(Rm, (0, 2, 1)))[:, None, :, :])  # (nm, ne, nb, nb)
        acc += -w_q * Z.sum(axis=0)
    return np.transpose(acc, (1, 2, 0))
