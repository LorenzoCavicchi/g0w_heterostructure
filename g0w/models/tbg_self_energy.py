#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
G0W self-energy and spectral function of the TBG sample in TBG|hBN|SLG.

Physics
-------
The sample band self-energy, in the band-diagonal approximation, is

    Sigma_nn(k, eps) = - sum_q w_q sum_m sum_{l1 l2 in sample}
                         F^{n,m}_{l1 l2}(k, q)
                         int_0^{w_max} dw  S_{l1 l2}(q, w)  D_m(w, eps)

    F^{n,m}_{l1 l2} = <k,n| rho_{l1}(q) |k+q,m> <k,n| rho_{l2}(q) |k+q,m>*
                    = densityL[l1,n,m,ki,qi,0] * conj(densityL[l2,n,m,ki,qi,0])

    S_{l1 l2}(q,w) = -Im W_{l1 l2}(q,w) / pi          (sample block of the 3x3 W)

    D_m = A_+/(w + xi_m - eps - i eta) + A_-/(-w + xi_m - eps - i eta)
    A_+ = n_B(w) + n_F(-xi_m) -> Theta(xi_m),   A_- -> Theta(-xi_m)   at T=0

    xi_m = E_m(k+q) - mu,   k+q index from band_structure.k_plus_q_ind[qi, ki].

No angular average is applied: ``q_sample`` is a full 2D BZ mesh, so the q-sum
already covers the directions.  (That is the opposite of the radial-grid case in
``../test_SLG_hBN_SLG``, where ``n_theta`` was essential.)

What is approximated
--------------------
* **Band-diagonal only.** Wei et al. Eq. (34) keeps Sigma_{lambda rho} off
  diagonal in band space, which matters where two bands are close.  Here A^S is
  built as a sum of independent band contributions.  ``sigma_band_pair`` below
  computes an off-diagonal element if needed, but the driver does not use it.
* **Correlation only.** As throughout this codebase the integrand is -Im W and
  Im v = 0, so the Fock term Sigma_x is absent.  See
  ``../test_SLG_hBN_SLG/README.md`` caveat 5: this shifts quasiparticle
  energies but cannot produce satellites.
* **Tip is not an intermediate state.** The tip is not tunnel coupled to the
  sample in the weak-tunneling limit, so no density operator connects them and
  l1, l2 run over the two TBG layers only.  The tip enters solely through W.

Units: eV, nm.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from .tbg_stack import SAMPLE_SHEETS
from ..self_energy_ppa import _n_bose, _n_fermi


def _trapz_weights(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float).ravel()
    w = np.zeros_like(x)
    if x.size > 1:
        d = np.diff(x)
        w[:-1] += 0.5 * d
        w[1:] += 0.5 * d
    else:
        w[:] = 1.0
    return w


def sample_spectral_density(W: np.ndarray, *, hermitian: bool = True) -> np.ndarray:
    r"""
    Spectral density of the 2x2 *sample* block of the stack W, (n_q, n_omega, 2, 2).

    ``hermitian=True`` (default): S = i (W - W^dagger)/(2 pi) (complex Hermitian,
    PSD at w > 0).  ``hermitian=False`` (original scripts): S = -Im W / pi element
    by element (real); identical for symmetric W.  See
    :func:`g0w.self_energy_full_frequency.spectral_density_matrix`.
    """
    from ..self_energy_full_frequency import spectral_density_matrix
    W = np.asarray(W, dtype=np.complex128)
    B = W[:, :, SAMPLE_SHEETS, :][:, :, :, SAMPLE_SHEETS]
    return spectral_density_matrix(B, hermitian=hermitian)


def sigma_full_frequency_tbg(
    densityL: np.ndarray,
    energies: np.ndarray,
    k_plus_q_ind: np.ndarray,
    mu: float,
    S_q: np.ndarray,
    omega_grid: np.ndarray,
    q_weights: np.ndarray,
    k_index: int,
    band_index: int,
    eps_values: np.ndarray,
    *,
    band_index_right: Optional[int] = None,
    T: float = 0.0,
    eta_self: float = 1e-3,
    eps_chunk: int = 32,
    bands_use: Optional[Sequence[int]] = None,
) -> np.ndarray:
    r"""
    Sigma_{n n'}(k, eps) for the TBG sample.  Returns (n_eps,) complex.

    ``band_index_right`` defaults to ``band_index`` (the diagonal element).
    ``bands_use`` restricts the intermediate-state sum (default: all bands).
    """
    n_r = band_index if band_index_right is None else int(band_index_right)
    eps_values = np.asarray(eps_values, dtype=float).ravel()
    omega_grid = np.asarray(omega_grid, dtype=float).ravel()
    n_eps, n_om = eps_values.size, omega_grid.size
    n_q = S_q.shape[0]

    tw = _trapz_weights(omega_grid)
    nB = _n_bose(omega_grid, T)
    n_bands = densityL.shape[1]
    m_list = np.arange(n_bands) if bands_use is None else np.asarray(bands_use, dtype=int)

    # rho[l, m, qi] for the fixed external k and bands (n, n')
    rho_L = densityL[:, band_index, :, k_index, :, 0]        # (2, n_bands, n_q)
    rho_R = densityL[:, n_r, :, k_index, :, 0]               # (2, n_bands, n_q)

    out = np.zeros(n_eps, dtype=np.complex128)

    for qi in range(n_q):
        w_q = float(q_weights[qi])
        if w_q == 0.0:
            continue
        # coupling per intermediate band: sum_{l1 l2} F_{l1 l2} S_{l1 l2}(w)
        rl = rho_L[:, m_list, qi]                            # (2, n_m)
        rr = rho_R[:, m_list, qi]                            # (2, n_m)
        F = np.einsum("am,bm->mab", rl, np.conj(rr))         # (n_m, 2, 2)
        coup = np.einsum("mab,wab->mw", F, S_q[qi])          # (n_m, n_om)
        if not np.any(coup):
            continue

        k_pq = int(k_plus_q_ind[qi, k_index])
        xi = energies[m_list, k_pq] - mu                      # (n_m,)
        A_p = nB[None, :] + _n_fermi(-xi, T)[:, None]
        A_m = nB[None, :] + _n_fermi(+xi, T)[:, None]
        cw = coup * tw[None, :]                               # fold in trapz weight

        for e0 in range(0, n_eps, int(eps_chunk)):
            e1 = min(e0 + int(eps_chunk), n_eps)
            u = xi[:, None, None] - eps_values[None, None, e0:e1]
            om = omega_grid[None, :, None]
            kern = (A_p[:, :, None] / (om + u - 1j * eta_self)
                    + A_m[:, :, None] / (-om + u - 1j * eta_self))
            out[e0:e1] += -w_q * np.einsum("mw,mwe->e", cw, kern)

    return out


def sigma_bands_on_k(
    tbg: Dict[str, Any],
    lh,
    S_q: np.ndarray,
    omega_grid: np.ndarray,
    q_weights: np.ndarray,
    k_index: int,
    bands: Sequence[int],
    eps_values: np.ndarray,
    **kwargs,
) -> np.ndarray:
    r"""Band-diagonal Sigma for several external bands; returns (n_bands_sel, n_eps)."""
    bs = tbg["crystal"].band_structure
    densityL = np.asarray(lh.densityL)
    out = np.zeros((len(bands), np.size(eps_values)), dtype=np.complex128)
    for i, n in enumerate(bands):
        out[i] = sigma_full_frequency_tbg(
            densityL, bs.energies, bs.k_plus_q_ind, float(bs.chemical_potential),
            S_q, omega_grid, q_weights, k_index, int(n), eps_values, **kwargs,
        )
    return out


def spectral_function_bands(
    tbg: Dict[str, Any],
    k_index: int,
    bands: Sequence[int],
    eps_values: np.ndarray,
    Sigma: np.ndarray,
    *,
    eta_qp: float = 1e-3,
) -> np.ndarray:
    r"""
    A_n(k, eps) = -(1/pi) Im [ eps - xi_n(k) - Sigma_nn(k,eps) + i eta_qp ]^-1

    Returns (n_bands_sel, n_eps).  Energies are measured from the chemical
    potential, so ``eps = 0`` is the sample Fermi level.
    """
    bs = tbg["crystal"].band_structure
    mu = float(bs.chemical_potential)
    eps_values = np.asarray(eps_values, dtype=float).ravel()
    A = np.zeros((len(bands), eps_values.size), dtype=float)
    for i, n in enumerate(bands):
        xi = float(bs.energies[int(n), int(k_index)] - mu)
        G = 1.0 / (eps_values - xi - Sigma[i] + 1j * float(eta_qp))
        A[i] = -np.imag(G) / np.pi
    return A


def band_energies_at_k(tbg: Dict[str, Any], k_index: int,
                       bands: Sequence[int]) -> np.ndarray:
    """xi_n(k) = E_n(k) - mu  [eV] for the selected bands."""
    bs = tbg["crystal"].band_structure
    mu = float(bs.chemical_potential)
    return np.array([float(bs.energies[int(n), int(k_index)] - mu) for n in bands])


# ---------------------------------------------------------------------------
# Band-matrix self-energy (new in g0w)
# ---------------------------------------------------------------------------

def q_weights_tbg(tbg: Dict[str, Any]) -> np.ndarray:
    r"""int d^2q/(2 pi)^2 weights on ``q_sample`` (no degeneracy), with q = 0 removed."""
    from ..self_energy_ppa import q_weights_from_band_structure
    bs = tbg["crystal"].band_structure
    w = q_weights_from_band_structure(bs)
    q = np.asarray(bs.q_sample.coords, float)
    w[np.linalg.norm(q, axis=1) < 1e-12] = 0.0
    return w


def build_kernel_tbg(tbg: Dict[str, Any], S_q: np.ndarray, omega_grid: np.ndarray,
                     eps_values: np.ndarray, *, eta_self: float = 1e-3, T: float = 0.0,
                     du: Optional[float] = None):
    r"""
    :class:`~g0w.self_energy_full_frequency.ResolventKernel` of the sample
    spectral density, covering every intermediate energy of the computed bands.
    """
    from ..self_energy_full_frequency import ResolventKernel
    bs = tbg["crystal"].band_structure
    E = np.asarray(bs.energies) - float(bs.chemical_potential)
    eps_values = np.asarray(eps_values, float)
    return ResolventKernel(omega_grid, S_q, u_min=E.min() - eps_values.max(),
                           u_max=E.max() - eps_values.min(), eta=eta_self, T=T, du=du)


def sigma_band_matrix(tbg: Dict[str, Any], lh, k_index: int, bands: Sequence[int],
                      eps_values: np.ndarray, *, kernel=None, S_q: Optional[np.ndarray] = None,
                      omega_grid: Optional[np.ndarray] = None, q_weights: Optional[np.ndarray] = None,
                      T: float = 0.0, eta_self: float = 1e-3,
                      bands_use: Optional[Sequence[int]] = None) -> np.ndarray:
    r"""
    Sigma_{n n'}(k, eps) of the TBG sample for the external bands ``bands``.

    Uses the precomputed ``kernel`` if given (fast), otherwise the direct
    frequency contraction with ``S_q``/``omega_grid``.  Returns (n_b, n_b, n_eps).
    """
    from ..self_energy_full_frequency import (compute_self_energy_band_matrix,
                                              compute_self_energy_band_matrix_kernel)
    bs = tbg["crystal"].band_structure
    qw = q_weights_tbg(tbg) if q_weights is None else q_weights
    if kernel is not None:
        return compute_self_energy_band_matrix_kernel(
            np.asarray(lh.densityL), bs.energies, bs.k_plus_q_ind, bs.chemical_potential,
            kernel, qw, k_index, bands, eps_values, bands_use=bands_use)
    return compute_self_energy_band_matrix(
        np.asarray(lh.densityL), bs.energies, bs.k_plus_q_ind, bs.chemical_potential,
        S_q, omega_grid, qw, k_index, bands, eps_values, bands_use=bands_use, T=T,
        eta_self=eta_self)
