#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Plasmon-pole-approximation (PPA) self-energy in *layer basis* for the
multilayer Coulomb / RPA framework of :mod:`g0w.layer_coulomb` and
:mod:`g0w.layer_helpers`.

(Source: ``layer_resolved/test_v2_7/self_energy_plasmon_pole.py``, including the
2026-09-01 fix of the (l,G) flattening in ``coulomb_slice_to_V_full``.  The only
changes here are ``np.trapz`` -> ``g0w.compat.trapezoid`` for numpy >= 2.)

Version 2.1 — W-residue PPA (physical default):
  * plasmon residues R_n(q) from ∫ dω [−Im W], not from ε⁻¹ alone,
  * full residue matrices in the self-energy (rank-one Φ optional),
  * radial q weights ∫ d²q/(2π)² by default on 1D |q| grids.

Theoretical background
======================
With ε(q, ω) = I − V(q, ω) χ⁰(q, ω) (``dielectric_function_layer``),

    W(q, ω) = ε⁻¹(q, ω) · V(q, ω)                         (W convention)

**ε⁻¹ does not enter the self-energy.**  The spectral residue at plasmon
frequency ω_n(q) is the matrix

    R_n(q) = (1/π) ∫_{window} dω [−Im W(q, ω)]              (R-PPA)

(Hermitian PSD).  Optional rank-one vectors Φ_n come from the dominant
eigenpair of R_n; Σ uses R_n by default.

    Σ_{ij}(k, ε) = − Σ_{q,n,λ} w_q R_{n,ij}(q) F^λ_{ij}(k,k−q) D_{n,λ}   (Σ-PPA)

    D_{n,λ} = A_+/(ω_n+ξ−ε−iη) + A_−/(−ω_n+ξ−ε−iη),

with w_q ≈ d²q/(2π)² (``make_radial_q_weights`` for radial meshes) and

    A_+ = n_B(ω_n) + n_F(−ξ),   A_− = n_B(ω_n) + n_F(+ξ),

which reduce to Θ(ξ) and Θ(−ξ) at T=0.  D is the finite-temperature result
of the ν integral over plasmon poles, not a standalone bosonic propagator.

The form factor F^λ_{(l_1 G_1),(l_2 G_2)}(k, k−q) is supplied by the user.
For a generic multilayer band structure with layer-projected Bloch states

    F^λ_{(l_1 G_1),(l_2 G_2)}(k, k−q) = ⟨u_k|P_{l_1}|u_{k−q−G_1}^λ⟩
                                        ⟨u_{k−q−G_2}^λ|P_{l_2}|u_k⟩,

which is exactly what ``layer_helper.densityL`` stores.  For decoupled
multi-2DEGs the helper ``make_decoupled_parabolic_form_factor`` produces a
form factor that is layer-diagonal and head-diagonal in G.

Conventions
-----------
* Energies and frequencies in eV.  Coulomb constant e² = 1.4399764 eV·nm.
* q is in 1/nm.
* The ω-grid for plasmon detection lies on the *real* axis.  The
  retarded-side broadening η is read off ``PlasmonPoleData.eta`` and used
  in (Σ-PPA).

Numerical reliability
---------------------
Passing analytic bilayer checks does not guarantee convergence for production
runs.  Use ``collect_diagnostics=True`` in ``build_plasmon_pole_data`` and
``convergence_plasmon_pole.py`` to scan η, ω resolution, pole windows, and q.
Full-residue mode is safer than rank-one ``ΦΦ†``; low rank-one quality only
invalidates vector interpretation, not the matrix residue Σ.

Full-frequency reference
------------------------
PPA replaces the dissipative spectral density ``[-Im W]/π`` by pole δ-functions.
For validation on small grids, ``self_energy_full_frequency`` integrates the full
ω dependence (reference calculation; slower).  Agreement with PPA requires sharp,
well-separated poles and high spectral capture; large η_W, overlapping poles, or
missing off-pole weight (see ``compare_ppa_full_frequency.py``) indicate that PPA
is not quantitatively reliable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import warnings

import numpy as np

from .compat import trapezoid as _trapezoid

# Rest electron energy m_e c² [eV] for effective-mass conversion helpers.
M_ELECTRON_ENERGY_EV = 510998.950

# Dielectric / screened-interaction convention (matches Layer_Helpers v2.6):
#   ε = I − V χ⁰   ⇒   W = ε⁻¹ V
SCREENED_INTERACTION_CONVENTION = "W = eps_inv @ V  (epsilon = I - V @ chi0)"


# ---------------------------------------------------------------------------
# q-integration weights
# ---------------------------------------------------------------------------

def make_radial_q_weights(q_grid: np.ndarray) -> np.ndarray:
    r"""
    Trapezoidal weights for an isotropic 2D radial |q| grid.

    Returns w_i ≈ q_i Δq_i / (2π), so that

        Σ_i w_i  ≈  ∫_0^{q_max} q dq / (2π)  =  q_max² / (4π).

    These weights implement the continuum measure d²q/(2π)² for a rotationally
    symmetric q-mesh.  Pass explicit ``q_weights`` to
    ``build_plasmon_pole_data`` if your mesh is not a 1D radial grid.

    Parameters
    ----------
    q_grid : ndarray
        Non-negative radial magnitudes |q| (1D).

    Returns
    -------
    weights : ndarray, same shape as ``q_grid``
    """
    q = np.asarray(q_grid, dtype=float).ravel()
    if q.size == 0:
        return np.array([], dtype=float)
    if np.any(q < 0):
        raise ValueError("Radial q weights require non-negative |q|.")
    if q.size == 1:
        dq = float(q[0]) if q[0] > 0 else 1.0
        return np.array([q[0] * dq / (2.0 * np.pi)], dtype=float)
    dq = np.empty_like(q)
    dq[0] = 0.5 * (q[1] - q[0])
    dq[-1] = 0.5 * (q[-1] - q[-2])
    dq[1:-1] = 0.5 * (q[2:] - q[:-2])
    return q * dq / (2.0 * np.pi)


def make_angular_quadrature(n_theta: int) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Uniform quadrature for the angular average ⟨·⟩_θ = (1/2π) ∫₀^{2π} dθ.

    Returns (thetas, weights) with Σ weights = 1.  The uniform (rectangle) rule
    on a periodic integrand converges spectrally, so modest ``n_theta`` is
    usually enough — but the resonant denominators 1/(ω_n+ξ_θ−ε−iη) sharpen as
    η → 0, so always check convergence in ``n_theta`` together with η.
    """
    n_theta = int(n_theta)
    if n_theta < 1:
        raise ValueError("n_theta must be >= 1.")
    thetas = 2.0 * np.pi * np.arange(n_theta) / n_theta
    return thetas, np.full(n_theta, 1.0 / n_theta)


def _k_minus_q_samples(
    k_vec: np.ndarray,
    q_entry: np.ndarray,
    *,
    radial: bool,
    thetas: np.ndarray,
) -> np.ndarray:
    r"""
    Sample k − q over the directions of q required by the integration measure.

    Radial meshes
    -------------
    ``make_radial_q_weights`` returns w_q = q Δq /(2π), which is the 2D measure
    **after** the angular integral has been carried out:

        ∫ d²q/(2π)² f(k−q)  =  ∫ q dq/(2π) ⟨ f(k − q(θ)) ⟩_θ .

    The integrand f depends on the direction of q through ξ_{k−q} and the form
    factor, so it must be averaged over θ here.  Evaluating only the collinear
    θ = 0 point (the pre-fix behaviour, recovered with ``n_theta=1``) pairs a 2D
    radial weight with a single angular sample and badly misplaces ξ_{k−q}.

    Explicit 2D meshes
    ------------------
    When ``q_vectors`` already carries full 2D vectors the angular integral is
    part of the q-sum itself; no extra averaging is applied (``radial=False``).

    Returns
    -------
    kmq : (n_theta, dim) array of k − q(θ) vectors.
    """
    k_vec = np.asarray(k_vec, dtype=float).ravel()
    q_entry = np.asarray(q_entry, dtype=float).ravel()
    if not radial:
        dim = max(k_vec.size, q_entry.size)
        kk = np.zeros(dim)
        kk[: k_vec.size] = k_vec
        qq = np.zeros(dim)
        qq[: q_entry.size] = q_entry
        return (kk - qq)[np.newaxis, :]

    q_mag = float(np.abs(q_entry[0])) if q_entry.size == 1 else float(
        np.linalg.norm(q_entry),
    )
    # Promote a radial |k| to a 2D vector along x; the result only depends on
    # the angle between k and q, so this choice is fully general.
    k2 = np.zeros(2)
    k2[: min(2, k_vec.size)] = k_vec[: min(2, k_vec.size)]
    q2 = q_mag * np.stack([np.cos(thetas), np.sin(thetas)], axis=1)  # (n_theta, 2)
    return k2[np.newaxis, :] - q2


def _infer_radial_q_magnitudes(q_vectors: np.ndarray) -> Optional[np.ndarray]:
    """Return 1D |q| if ``q_vectors`` looks like a radial grid, else None."""
    qv = np.asarray(q_vectors, dtype=float)
    if qv.ndim == 1:
        return np.abs(qv)
    if qv.ndim == 2 and qv.shape[1] == 1:
        return np.abs(qv[:, 0])
    if qv.ndim == 2 and qv.shape[1] >= 2:
        transverse = qv[:, 1:]
        if np.allclose(transverse, 0.0, atol=0.0, rtol=0.0):
            return np.abs(qv[:, 0])
    return None


def q_weights_from_band_structure(
    band_structure,
    *,
    include_degeneracy: bool = False,
) -> np.ndarray:
    r"""
    Compute d²q/(2π)² BZ-integration weights from a pybandstructure Band_Structure.

    Returns ``w_q`` of shape ``(n_q,)`` such that

        Σ_q  w_q  f(q)  ≈  ∫ d²q/(2π)²  f(q).

    Concretely, ``w_q[qi] = q_sample.volume × weights[qi] / (2π)^D``, mirroring
    ``Band_Structure.integrate_bz`` **without** its degeneracy factor.

    Degeneracy
    ----------
    ``integrate_bz`` multiplies by ``band_structure.degeneracy`` because it is
    built for sums over *occupied states* — χ⁰ genuinely carries that factor,
    and ``polarization_layered`` applies it.  The self-energy momentum-transfer
    sum does **not**: in Σ = i G W the intermediate propagator carries the same
    spin/valley as the external electron, so there is no free degeneracy sum
    over q.  Including it here would multiply Σ by g (2 or 4 for graphene) on
    top of the g already inside χ⁰ → ε → R_n.

    ``include_degeneracy=True`` restores the old behaviour for reproducing
    earlier runs; it is not the physical weight for Σ.

    Pass the result as ``q_weights`` to ``build_plasmon_pole_data`` so that the
    PPD and ``compute_self_energy_pybandstructure`` share the same integration
    convention.

    Parameters
    ----------
    band_structure : Band_Structure
        Must have a ``q_sample`` with ``weights`` and ``volume`` attributes
        (i.e. a ``Space_Sample`` or ``Unit_Cell_Sample`` instance).

    Returns
    -------
    w_q : ndarray, shape (n_q,)
    """
    qs = band_structure.q_sample
    dim = int(band_structure.k_sample.lattice_dimension)
    vol = float(qs.volume)
    weights = np.asarray(qs.weights, dtype=float)
    factor = float(band_structure.degeneracy) if include_degeneracy else 1.0
    return vol * factor / (2.0 * np.pi) ** dim * weights


def k_weights_from_band_structure(band_structure) -> np.ndarray:
    r"""
    ∫ d²k/(2π)² weights on the pybandstructure ``k_sample`` (no degeneracy).

    Companion of ``q_weights_from_band_structure`` for the external-momentum
    sum in the QTM current, where the spin/valley factor is applied once at the
    end via ``degeneracy`` rather than folded into the weights.
    """
    ks = band_structure.k_sample
    dim = int(ks.lattice_dimension)
    return float(ks.volume) * np.asarray(ks.weights, dtype=float) / (2.0 * np.pi) ** dim


# ---------------------------------------------------------------------------
# Mass / band helpers
# ---------------------------------------------------------------------------

def effective_mass_ratio_to_energy(m_eff_over_me: Union[float, np.ndarray]) -> np.ndarray:
    r"""
    Convert dimensionless effective mass m*/m_e to rest energy m* c² in eV.

        mass_energy_eV = (m*/m_e) × (m_e c²),   m_e c² = 510998.950 eV.
    """
    return np.asarray(m_eff_over_me, dtype=float) * M_ELECTRON_ENERGY_EV


# ---------------------------------------------------------------------------
# Screened interaction W and residue utilities
# ---------------------------------------------------------------------------

def _reshape_eps_slice(eps_q: np.ndarray) -> Tuple[np.ndarray, int, int, int]:
    """(n_ω, N, N, n_G, n_G) → (n_ω, dim, dim) with row = l*n_G + G."""
    eps_q = np.asarray(eps_q, dtype=np.complex128)
    n_omega = eps_q.shape[0]
    N, _, n_G, _ = eps_q.shape[1:]
    dim = N * n_G
    eps_full = np.transpose(eps_q, (0, 1, 3, 2, 4)).reshape(n_omega, dim, dim)
    return eps_full, N, n_G, dim


def coulomb_slice_to_V_full(
    coulomb_q: np.ndarray,
    *,
    n_layers: int,
    n_G: int,
    n_omega: int,
    omega_index: Optional[int] = None,
) -> np.ndarray:
    r"""
    Build block-diagonal V in (layer, G) space: V_{(i,G),(j,G')} = V_{ij}(G) δ_{GG'}.

    Rows/cols are flattened as ``row = l*n_G + G`` — the *same* ordering as
    ``_reshape_eps_slice`` and ``layer_helpers_G`` — so that W = ε⁻¹ V is a
    product of two matrices in one basis.  (Before v2.8 this routine used the
    transposed ordering ``row = G*N + l``, which silently mixed (l,G) blocks
    whenever N > 1 *and* n_G > 1.)

    ``coulomb_q`` may be (N, N, n_G) or (N, N, n_G, n_ω).
    """
    N, n_G = n_layers, n_G
    dim = N * n_G
    cq = np.asarray(coulomb_q, dtype=np.complex128)
    if cq.ndim == 3:
        if cq.shape != (N, N, n_G):
            raise ValueError(f"coulomb_q shape {cq.shape} != ({N}, {N}, {n_G}).")
        V_full = np.zeros((dim, dim), dtype=np.complex128)
        for g in range(n_G):
            idx = np.arange(N) * n_G + g          # row = l*n_G + G
            V_full[np.ix_(idx, idx)] = cq[:, :, g]
        return np.broadcast_to(V_full, (n_omega, dim, dim)).copy()
    if cq.ndim == 4:
        if cq.shape[:3] != (N, N, n_G) or cq.shape[3] != n_omega:
            raise ValueError(
                f"coulomb_q shape {cq.shape} incompatible with ({N},{N},{n_G},{n_omega})."
            )
        out = np.zeros((n_omega, dim, dim), dtype=np.complex128)
        for g in range(n_G):
            idx = np.arange(N) * n_G + g          # row = l*n_G + G
            out[np.ix_(np.arange(n_omega), idx, idx)] = np.transpose(
                cq[:, :, g, :], (2, 0, 1),
            )
        return out
    raise ValueError("coulomb_q must have 3 or 4 dimensions.")


def screened_interaction_matrix(
    eps_matrix: np.ndarray,
    V_matrix: np.ndarray,
) -> np.ndarray:
    r"""
    RPA screened interaction W = ε⁻¹ V (see ``SCREENED_INTERACTION_CONVENTION``).

    Parameters
    ----------
    eps_matrix, V_matrix : (dim, dim) complex
    """
    eps_matrix = np.asarray(eps_matrix, dtype=np.complex128)
    V_matrix = np.asarray(V_matrix, dtype=np.complex128)
    if eps_matrix.shape != V_matrix.shape:
        raise ValueError(
            f"eps_matrix shape {eps_matrix.shape} != V_matrix shape {V_matrix.shape}"
        )
    return np.linalg.inv(eps_matrix) @ V_matrix


def _hermitize_and_clip_residue(R: np.ndarray, eig_floor: float = 1e-14) -> np.ndarray:
    R = 0.5 * (R + R.conj().T)
    w, U = np.linalg.eigh(R)
    w = np.maximum(w, 0.0)
    w[w < eig_floor] = 0.0
    return (U * w) @ U.conj().T


def rank1_quality_from_residue(R: np.ndarray) -> float:
    r"""s_max / Σ_a s_a for positive eigenvalues s_a of Hermitian R."""
    R = 0.5 * (R + R.conj().T)
    s = np.linalg.eigvalsh(R)
    s_pos = s[s > 0]
    if s_pos.size == 0:
        return 0.0
    return float(np.max(s_pos) / np.sum(s_pos))


def phi_from_residue(R: np.ndarray) -> Tuple[np.ndarray, float]:
    r"""
    Dominant-eigenpair rank-one factorization: Φ = √s_max u_max, plus rank1_quality.
    """
    R = _hermitize_and_clip_residue(R)
    w, U = np.linalg.eigh(R)
    i_max = int(np.argmax(w))
    s_max = max(float(w[i_max]), 0.0)
    Phi = np.sqrt(s_max) * U[:, i_max]
    return Phi, rank1_quality_from_residue(R)


def _minus_im_W_on_window(
    eps_window: np.ndarray,
    V_window: np.ndarray,
) -> np.ndarray:
    """[−Im W] for each ω in a window, W = ε⁻¹ V."""
    n_pts = eps_window.shape[0]
    dim = eps_window.shape[1]
    out = np.zeros((n_pts, dim, dim), dtype=float)
    for oi in range(n_pts):
        out[oi] = -np.imag(screened_interaction_matrix(eps_window[oi], V_window[oi]))
    return out


def _integrate_minus_im_W_residue(
    omega_window: np.ndarray,
    eps_window: np.ndarray,
    V_window: np.ndarray,
) -> np.ndarray:
    r"""
    R = (1/π) ∫ dω [−Im W],  W = ε⁻¹ V, trapezoidal in ω.
    """
    omega_window = np.asarray(omega_window, dtype=float).ravel()
    n_pts = omega_window.size
    if n_pts == 0:
        raise ValueError("Empty omega window for residue integration.")
    minus_im_W = _minus_im_W_on_window(eps_window, V_window)
    if n_pts == 1:
        d_omega = 1.0
        R = minus_im_W[0] * d_omega
    else:
        R = _trapezoid(minus_im_W, omega_window, axis=0)
    return _hermitize_and_clip_residue(R / np.pi)


def _minima_with_prominence(
    y: np.ndarray,
    min_prominence: float,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Local minima of ``y`` (>0) that are genuine dips, with their prominence.

    Prominence = min(flanking local maxima) / y_min.  A true collective mode
    drives |λ_min(ε)| towards zero, so its dip is deep; the shallow ripples that
    a Lindhard continuum or a phonon Reststrahlen band produce in |λ_min| are
    not poles and must not be handed to the PPA as such (each spurious entry
    both adds fictitious weight and steals window space from a real pole).
    """
    idx = _local_minima(y)
    if idx.size == 0:
        return idx, np.empty(0, dtype=float)
    keep: List[int] = []
    proms: List[float] = []
    for i in idx:
        j = int(i)
        while j > 0 and y[j - 1] >= y[j]:
            j -= 1
        left = float(y[j])
        j = int(i)
        while j < y.size - 1 and y[j + 1] >= y[j]:
            j += 1
        right = float(y[j])
        p = min(left, right) / max(float(y[i]), 1e-300)
        if p >= min_prominence:
            keep.append(int(i))
            proms.append(p)
    return np.asarray(keep, dtype=int), np.asarray(proms, dtype=float)


def _pole_windows(
    omegas: np.ndarray,
    omega_grid: np.ndarray,
    halfwin: float,
) -> np.ndarray:
    r"""
    Non-overlapping [lo, hi] integration window for each pole.

    Windows are ±``halfwin`` around each ω_n, then truncated at the midpoint
    between neighbouring poles.  Overlapping windows would integrate the same
    [−Im W] weight into several residues; with a fixed ±5η half-width and poles
    closer than that, Σ_n R_n can exceed the true (1/π)∫dω[−Im W] and *grow*
    under ω-grid refinement.

    Returns
    -------
    windows : (n_poles, 2)
    """
    omegas = np.asarray(omegas, dtype=float).ravel()
    lo = omegas - halfwin
    hi = omegas + halfwin
    order = np.argsort(omegas)
    for a, b in zip(order[:-1], order[1:]):
        mid = 0.5 * (omegas[a] + omegas[b])
        hi[a] = min(hi[a], mid)
        lo[b] = max(lo[b], mid)
    lo = np.maximum(lo, float(omega_grid[0]))
    hi = np.minimum(hi, float(omega_grid[-1]))
    return np.stack([lo, hi], axis=1)


def _lambda_min_abs(eps_full: np.ndarray) -> np.ndarray:
    """|λ_min(ε)| for each ω slice of a (n_ω, dim, dim) stack."""
    out = np.empty(eps_full.shape[0], dtype=float)
    for oi in range(eps_full.shape[0]):
        w, _ = np.linalg.eig(eps_full[oi])
        out[oi] = float(np.min(np.abs(w)))
    return out


def _parabolic_minimum(omega: np.ndarray, y: np.ndarray, i: int) -> float:
    """Sub-grid minimum of log(y) from a parabola through points i-1, i, i+1."""
    if i <= 0 or i >= omega.size - 1:
        return float(omega[i])
    y0, y1, y2 = (np.log(y[i + d] + 1e-300) for d in (-1, 0, 1))
    denom = y0 - 2.0 * y1 + y2
    delta = 0.0 if abs(denom) < 1e-14 else 0.5 * (y0 - y2) / denom
    delta = float(np.clip(delta, -1.0, 1.0))
    return float(omega[i]) + delta * float(omega[i] - omega[i - 1])


@dataclass
class PoleDiagnostics:
    """Per-pole reliability record for one q (see ``find_plasmon_poles_at_q``)."""
    omega_n: float
    window: Tuple[float, float]
    prominence: float
    lambda_min: float
    n_points_in_window: int
    n_points_in_fwhm: int
    residue_trace: float
    refined: bool
    # (1/π)∫ Tr[−Im W] over the same window on the *global* grid.  Compared with
    # ``residue_trace`` it measures how much the coarse grid was missing.
    coarse_window_trace: float = 0.0

    def resolved(self, min_points_per_pole: int = 5) -> bool:
        """True if the residue is trustworthy.

        Either the global grid resolves the peak, or the window was integrated
        on its own refined grid (in which case the global spacing is irrelevant).
        """
        return self.refined or self.n_points_in_fwhm >= int(min_points_per_pole)

    def summary(self) -> str:
        flag = "" if self.resolved() else "  << UNDER-RESOLVED"
        return (
            f"pole ω={self.omega_n:.6f} eV  window=[{self.window[0]:.6f},"
            f"{self.window[1]:.6f}]  prom={self.prominence:.2g}"
            f"  |λ|min={self.lambda_min:.2g}  pts(win/FWHM)="
            f"{self.n_points_in_window}/{self.n_points_in_fwhm}"
            f"  TrR={self.residue_trace:.4g}"
            f"{'  [refined]' if self.refined else ''}{flag}"
        )


@dataclass
class SpectralWeightDiagnostics:
    """Sum rule at one q: Σ_n Tr R_n vs the full (1/π)∫dω Tr[−Im W]."""
    total_weight: float
    captured_weight: float

    @property
    def captured_fraction(self) -> float:
        return self.captured_weight / max(self.total_weight, 1e-300)

    def summary(self) -> str:
        return (
            f"spectral weight: Σ_n Tr R_n = {self.captured_weight:.5g}, "
            f"(1/π)∫dω Tr[−Im W] = {self.total_weight:.5g}, "
            f"captured = {100.0 * self.captured_fraction:.1f}%"
        )


def _residue_weight_scalar(R: np.ndarray, head_idx: np.ndarray) -> float:
    """Integrated spectral weight proxy: Tr R_head (real, ≥ 0)."""
    R_head = R[np.ix_(head_idx, head_idx)]
    return max(float(np.real(np.trace(R_head))), 0.0)


# ---------------------------------------------------------------------------
# Population factors
# ---------------------------------------------------------------------------

def _n_bose(omega: np.ndarray, T: float) -> np.ndarray:
    r"""Bose function n_B(ω) = 1/(exp(ω/T)−1), with overflow-safe evaluation.

    For ω > 0, T → 0  ⇒  n_B → 0.
    For ω < 0, T → 0  ⇒  n_B → −1.
    Uses the identity  n_B(x) = exp(−x)/(1−exp(−x))  for x > 0  and the
    direct formula for x ≤ 0 (no overflow on either side).
    """
    omega = np.asarray(omega, dtype=float)
    if T <= 0:
        return np.where(omega > 0, 0.0, -1.0)
    x = omega / T
    out = np.empty_like(x)
    pos = x > 0
    # x > 0:  exp(-x) / (1 - exp(-x))   — bounded, well-conditioned
    if np.any(pos):
        e = np.exp(-x[pos])
        out[pos] = e / (1.0 - e)
    # x ≤ 0:  1 / expm1(x)   — safe (expm1 in [-1, 0])
    if np.any(~pos):
        out[~pos] = 1.0 / np.expm1(x[~pos])
    return out


def _n_fermi(xi: np.ndarray, T: float) -> np.ndarray:
    r"""Fermi function n_F(ξ) = 1/(exp(ξ/T)+1).

    Numerically safe (no overflow) via the tanh representation:
        n_F(ξ) = (1 − tanh(ξ/(2T))) / 2.
    """
    xi = np.asarray(xi, dtype=float)
    if T <= 0:
        return (xi < 0).astype(float)
    return 0.5 * (1.0 - np.tanh(0.5 * xi / T))


def population_factors(omega_n: float, energies: np.ndarray, T: float) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    Return (A_plus, A_minus) for the (Σ-PPA) formula above:

        A_+ = n_B(ω_n) + n_F(−ξ),
        A_− = n_B(ω_n) + n_F(+ξ).

    At T → 0 these reduce to Θ(ξ) and Θ(−ξ).
    """
    nB = float(_n_bose(np.asarray([omega_n], dtype=float), T)[0])
    nF_pos = _n_fermi(+energies, T)   # n_F( ξ)
    nF_neg = _n_fermi(-energies, T)   # n_F(−ξ)
    return nB + nF_neg, nB + nF_pos


# ---------------------------------------------------------------------------
# Plasmon-pole detection
# ---------------------------------------------------------------------------

def _local_minima(y: np.ndarray, threshold: float = np.inf) -> np.ndarray:
    """Indices of local minima in 1D array y, requiring y[i] < threshold."""
    if y.size < 3:
        return np.empty(0, dtype=int)
    inner = np.arange(1, y.size - 1)
    is_min = (y[inner] < y[inner - 1]) & (y[inner] < y[inner + 1]) & (y[inner] < threshold)
    return inner[is_min]


def find_plasmon_poles_at_q(
    eps_q: np.ndarray,
    omega_grid: np.ndarray,
    *,
    coulomb_q: np.ndarray,
    eta: float,
    n_modes: Optional[int] = None,
    min_weight: float = 0.0,
    weight_window_factor: float = 5.0,
    drop_repeated_within: Optional[float] = None,
    min_pole_prominence: float = 2.0,
    min_points_per_pole: int = 5,
    eps_builder: Optional[Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]] = None,
    n_refine: int = 129,
    return_diagnostics: bool = False,
):
    r"""
    Locate plasmon poles ω_n(q) and W-residue matrices R_n(q).

    Pole condition: a *prominent* minimum of |λ_min(ε(q,ω))| on the real ω axis.
    For each pole, the spectral residue of the screened interaction is

        R_n(q) = (1/π) ∫_{window} dω [−Im W(q, ω)],   W = ε⁻¹ V,

    integrated trapezoidally (see ``SCREENED_INTERACTION_CONVENTION``).
    ``coulomb_q`` supplies V(q, ω) in layer space (same layout as
    ``dielectric_function_layer`` Coulomb output).

    Resolving the residue
    ---------------------
    This integral only means anything if the ω-grid resolves the peak.  The peak
    width is set by the broadening carried by ε, so **χ⁰ must be evaluated at
    ω + iη with the same η as V**.  If χ⁰ is built on the real axis while V is
    broadened, an undamped collective mode stays a δ-function: the trapezoidal
    result is then decided by whether a grid point happens to land on the spike,
    and it neither converges nor is reproducible under refinement.

    Three guards are applied:

    * ``min_pole_prominence`` rejects shallow ripples in |λ_min| (continuum or
      phonon-band structure) that are not collective modes;
    * windows are made **non-overlapping** (``_pole_windows``) so no spectral
      weight is counted twice;
    * ``min_points_per_pole`` reports how many grid points fall inside each
      peak's FWHM, so under-resolution is visible instead of silent.

    Set ``eps_builder`` to remove the resolution constraint entirely: it is
    called as ``eps_builder(omega_array) -> (eps_q, coulomb_q)`` with the same
    layouts as this function's own arguments, and lets each window be integrated
    on its own ``n_refine``-point local grid.  This is the recommended
    production route — it needs no analytic model, only the ability to
    re-evaluate ε and V at requested frequencies.

    Returns
    -------
    omega_n : (n_poles,)
    residues : (n_poles, dim, dim) complex Hermitian PSD
    Phi : (n_poles, dim) — dominant eigenvector × √s_max (diagnostic)
    rank1_quality : (n_poles,) — s_max / Σ_a s_a of R_n
    diagnostics : (only when ``return_diagnostics``) tuple
        ``(List[PoleDiagnostics], SpectralWeightDiagnostics)``
    """
    eps_q = np.asarray(eps_q, dtype=np.complex128)
    omega_grid = np.asarray(omega_grid, dtype=float).ravel()
    if eps_q.shape[0] != omega_grid.size:
        raise ValueError("eps_q.shape[0] must match omega_grid.size")
    n_omega = eps_q.shape[0]
    eps_full, N, n_G, dim = _reshape_eps_slice(eps_q)
    V_full = coulomb_slice_to_V_full(
        coulomb_q, n_layers=N, n_G=n_G, n_omega=n_omega,
    )
    head_idx = np.arange(N) * n_G

    # Total spectral weight on the supplied grid — the sum-rule reference.
    minus_im_W_grid = _minus_im_W_on_window(eps_full, V_full)
    total_weight = float(
        _trapezoid(np.trace(minus_im_W_grid, axis1=1, axis2=2), omega_grid) / np.pi,
    ) if n_omega > 1 else 0.0

    def _empty(with_diag: bool):
        z = np.empty(0, dtype=float)
        zc = np.empty((0, dim), dtype=np.complex128)
        out = (z, np.empty((0, dim, dim), dtype=np.complex128), zc, z)
        if with_diag:
            return out + (([], SpectralWeightDiagnostics(total_weight, 0.0)),)
        return out

    lam_min = np.empty(n_omega, dtype=np.complex128)
    for oi in range(n_omega):
        w, _ = np.linalg.eig(eps_full[oi])
        lam_min[oi] = w[int(np.argmin(np.abs(w)))]

    abs_lam = np.abs(lam_min)
    pole_idx, prominences = _minima_with_prominence(abs_lam, min_pole_prominence)
    if pole_idx.size == 0:
        return _empty(return_diagnostics)

    # Sub-grid pole position from a parabola through log|λ_min|.
    omegas = np.asarray(
        [_parabolic_minimum(omega_grid, abs_lam, int(i)) for i in pole_idx],
        dtype=float,
    )

    if omega_grid.size > 1:
        grid_step = float(np.median(np.diff(omega_grid)))
        if not np.allclose(np.diff(omega_grid), grid_step, rtol=1e-4, atol=0.0):
            grid_step = float(np.min(np.diff(omega_grid)))
    else:
        grid_step = 0.0
    # The window must be set by the physical linewidth η, not by the grid: a
    # grid-dependent half-width makes R_n itself grid-dependent (a coarser grid
    # sweeps in more continuum background).  Widening to the grid step is only
    # a fallback for the unrefined path, which needs ≥2 points in the window.
    halfwin = weight_window_factor * (
        eta if eps_builder is not None and n_refine > 2 else max(eta, grid_step)
    )

    # When the global grid does not resolve the pole width (Δω ≳ η) the
    # parabola above places ω_n only to O(Δω).  That error then shifts the
    # residue window and leaks back into R_n, so re-locate each pole on its own
    # fine local grid whenever ε can be re-evaluated.
    if eps_builder is not None and n_refine > 2:
        refined: List[float] = []
        for w_n in omegas:
            lo = max(float(w_n) - halfwin, float(omega_grid[0]))
            hi = min(float(w_n) + halfwin, float(omega_grid[-1]))
            if hi <= lo:
                refined.append(float(w_n))
                continue
            om_loc = np.linspace(lo, hi, int(n_refine))
            eps_loc, _ = eps_builder(om_loc)
            eps_loc_full, _, _, _ = _reshape_eps_slice(
                np.asarray(eps_loc, dtype=np.complex128),
            )
            lam_loc = _lambda_min_abs(eps_loc_full)
            refined.append(_parabolic_minimum(om_loc, lam_loc, int(np.argmin(lam_loc))))
        omegas = np.asarray(refined, dtype=float)

    def _residues_for(omega_set: np.ndarray):
        """Integrate non-overlapping windows around ``omega_set``."""
        windows = _pole_windows(omega_set, omega_grid, halfwin)
        R_list: List[np.ndarray] = []
        w_list: List[float] = []
        diag_list: List[PoleDiagnostics] = []
        for j, w_n in enumerate(omega_set):
            lo, hi = float(windows[j, 0]), float(windows[j, 1])
            mask = (omega_grid >= lo) & (omega_grid <= hi)
            n_in_window = int(np.count_nonzero(mask))
            refined = False
            if eps_builder is not None and n_refine > 1 and hi > lo:
                om_sub = np.linspace(lo, hi, int(n_refine))
                eps_sub_raw, V_sub_raw = eps_builder(om_sub)
                eps_sub, _, _, _ = _reshape_eps_slice(
                    np.asarray(eps_sub_raw, dtype=np.complex128),
                )
                V_sub = coulomb_slice_to_V_full(
                    V_sub_raw, n_layers=N, n_G=n_G, n_omega=om_sub.size,
                )
                refined = True
            elif n_in_window >= 2:
                om_sub = omega_grid[mask]
                eps_sub = eps_full[mask]
                V_sub = V_full[mask]
            else:
                R_list.append(np.zeros((dim, dim), dtype=np.complex128))
                w_list.append(0.0)
                diag_list.append(PoleDiagnostics(
                    omega_n=float(w_n), window=(lo, hi),
                    prominence=float(prominences[j]) if j < prominences.size else 0.0,
                    lambda_min=float(np.min(abs_lam[mask])) if n_in_window else float("nan"),
                    n_points_in_window=n_in_window, n_points_in_fwhm=0,
                    residue_trace=0.0, refined=False,
                ))
                continue

            R_n = _integrate_minus_im_W_residue(om_sub, eps_sub, V_sub)
            R_list.append(R_n)
            w_list.append(_residue_weight_scalar(R_n, head_idx))

            # Same window, global grid — the reference the total weight uses.
            if n_in_window >= 2:
                coarse_trace = float(_trapezoid(
                    np.trace(
                        _minus_im_W_on_window(eps_full[mask], V_full[mask]),
                        axis1=1, axis2=2,
                    ),
                    omega_grid[mask],
                ) / np.pi)
            else:
                coarse_trace = 0.0

            # How many *original* grid points sit inside the peak's FWHM?
            peak = np.trace(_minus_im_W_on_window(eps_sub, V_sub), axis1=1, axis2=2)
            if peak.size and peak.max() > 0:
                above = peak >= 0.5 * peak.max()
                fwhm = float(om_sub[above].max() - om_sub[above].min())
                n_fwhm = int(np.count_nonzero(
                    (omega_grid >= om_sub[above].min()) & (omega_grid <= om_sub[above].max()),
                ))
            else:
                fwhm, n_fwhm = 0.0, 0
            diag_list.append(PoleDiagnostics(
                omega_n=float(w_n), window=(lo, hi),
                prominence=float(prominences[j]) if j < prominences.size else 0.0,
                lambda_min=float(np.min(abs_lam[mask])) if n_in_window else float("nan"),
                n_points_in_window=n_in_window, n_points_in_fwhm=n_fwhm,
                residue_trace=float(np.real(np.trace(R_n))), refined=refined,
                coarse_window_trace=coarse_trace,
            ))
        return R_list, np.asarray(w_list, dtype=float), diag_list

    # Pass 1: provisional residues used only to rank / threshold the candidates.
    _, weights, _ = _residues_for(omegas)

    keep = weights > min_weight
    omegas = omegas[keep]
    prominences = prominences[keep] if prominences.size == keep.size else prominences
    weights = weights[keep]
    if omegas.size == 0:
        return _empty(return_diagnostics)

    if drop_repeated_within is not None and omegas.size > 1:
        order = np.argsort(omegas)
        merged_o: List[float] = []
        merged_w: List[float] = []
        merged_p: List[float] = []
        cluster = [order[0]]
        for k in order[1:]:
            if abs(omegas[k] - omegas[cluster[-1]]) < drop_repeated_within:
                cluster.append(k)
            else:
                best = cluster[int(np.argmax(weights[cluster]))]
                merged_o.append(float(omegas[best]))
                merged_w.append(float(np.sum(weights[cluster])))
                merged_p.append(float(prominences[best]) if prominences.size else 0.0)
                cluster = [k]
        best = cluster[int(np.argmax(weights[cluster]))]
        merged_o.append(float(omegas[best]))
        merged_w.append(float(np.sum(weights[cluster])))
        merged_p.append(float(prominences[best]) if prominences.size else 0.0)
        omegas = np.asarray(merged_o)
        weights = np.asarray(merged_w)
        prominences = np.asarray(merged_p)

    if n_modes is not None and omegas.size > n_modes:
        sel = np.sort(np.argsort(-weights)[:n_modes])
        omegas = omegas[sel]
        weights = weights[sel]
        prominences = prominences[sel] if prominences.size else prominences

    # Pass 2: final residues, re-windowed so the surviving poles share the axis
    # between them (weight from a discarded candidate is reabsorbed, not lost).
    R_list, weights, diag_list = _residues_for(omegas)
    residues = np.stack(R_list, axis=0) if R_list else np.empty(
        (0, dim, dim), dtype=np.complex128,
    )

    Phis = np.empty((omegas.size, dim), dtype=np.complex128)
    rank1_q = np.zeros(omegas.size, dtype=float)
    for j in range(omegas.size):
        Phis[j], rank1_q[j] = phi_from_residue(residues[j])

    if return_diagnostics:
        captured = float(sum(d.residue_trace for d in diag_list))
        # Put numerator and denominator on the same footing: inside each window
        # the coarse-grid estimate is replaced by the refined one, so a
        # under-resolved global grid cannot fake a >100% capture.
        total = (
            total_weight
            - float(sum(d.coarse_window_trace for d in diag_list))
            + captured
        )
        return omegas, residues, Phis, rank1_q, (
            diag_list, SpectralWeightDiagnostics(max(total, captured), captured),
        )
    return omegas, residues, Phis, rank1_q


def residue_from_eigenvalue_derivative(
    eps_q: np.ndarray,
    omega_grid: np.ndarray,
    coulomb_q: np.ndarray,
    omega_n: float,
    *,
    n_layers: int,
    n_G: int = 1,
) -> np.ndarray:
    r"""
    Pole residue of W from the ε-eigenvalue derivative — **reference only**.

    Near a zero of the eigenvalue λ(ω) of ε(q,ω), with right/left eigenvectors
    v_R, v_L normalised to v_L^† v_R = 1,

        ε⁻¹(ω) ≈ v_R v_L^† / [λ'(ω_n)(ω − ω_n)],

    so that, with ω → ω + i0⁺,

        R_n = (1/π) ∫ dω [−Im W]  =  Re[ v_R (v_L^† V) / λ'(ω_n) ] .

    This is *not* used in production: it presumes an isolated simple pole, which
    a Landau-damped or strongly hybridised mode is not, and it needs λ(ω)
    tracked through the mode crossing.  Its purpose is to give the grid-based
    ``find_plasmon_poles_at_q`` an independent target in the tests, where the
    two must agree for a sharp isolated pole.

    λ'(ω_n) is obtained by finite differences on ``omega_grid`` — no closed form
    for χ⁰ is required.
    """
    eps_q = np.asarray(eps_q, dtype=np.complex128)
    omega_grid = np.asarray(omega_grid, dtype=float).ravel()
    eps_full, N, nG, dim = _reshape_eps_slice(eps_q)
    V_full = coulomb_slice_to_V_full(
        coulomb_q, n_layers=n_layers, n_G=n_G, n_omega=omega_grid.size,
    )
    i0 = int(np.argmin(np.abs(omega_grid - float(omega_n))))
    lo = max(i0 - 1, 0)
    hi = min(i0 + 1, omega_grid.size - 1)
    if hi == lo:
        raise ValueError("Need at least two omega points to differentiate lambda.")

    def _lambda_and_vectors(idx):
        w, vr = np.linalg.eig(eps_full[idx])
        k = int(np.argmin(np.abs(w)))
        return w[k], vr[:, k]

    lam_lo, _ = _lambda_and_vectors(lo)
    lam_hi, _ = _lambda_and_vectors(hi)
    lam_prime = (lam_hi - lam_lo) / (omega_grid[hi] - omega_grid[lo])

    _, v_R = _lambda_and_vectors(i0)
    # Left eigenvector of eps for the same eigenvalue: right eigenvector of eps^H.
    wH, vH = np.linalg.eig(eps_full[i0].conj().T)
    kH = int(np.argmin(np.abs(wH)))
    v_L = vH[:, kH]
    norm = np.vdot(v_L, v_R)
    if abs(norm) < 1e-30:
        raise ValueError("Left/right eigenvectors are orthogonal; pole is defective.")
    v_L = v_L / np.conj(norm)

    M = np.outer(v_R, np.conj(v_L)) @ V_full[i0]
    return _hermitize_and_clip_residue(np.real(M / lam_prime).astype(np.complex128))


# ---------------------------------------------------------------------------
# Plasmon-pole data container + builder
# ---------------------------------------------------------------------------

@dataclass
class PlasmonPoleData:
    r"""
    Cached per-q plasmon poles in full (layer, G) basis.

    Primary object for the self-energy: ``residues[qi][n]`` = R_n(q) with
    R_n = (1/π) ∫ dω [−Im W].  ``Phi`` is a rank-one diagnostic from R_n.
    """
    q_vectors: np.ndarray
    q_weights: np.ndarray
    pole_frequencies: List[np.ndarray]   # alias storage: ω_n(q)
    residues: List[np.ndarray]           # per q: (n_poles, dim, dim)
    Phi: List[np.ndarray]                # per q: (n_poles, dim) diagnostic
    rank1_quality: List[np.ndarray]      # per q: (n_poles,)
    n_layers: int
    n_G: int
    eta: float
    # "radial"    : q_vectors holds |q| magnitudes; the self-energy must average
    #               the integrand over the direction of q (see n_theta).
    # "cartesian" : q_vectors holds explicit q vectors (e.g. a pybandstructure
    #               q_sample); the q-sum already covers the angles.
    # None        : infer from the shape of q_vectors.
    q_mesh: Optional[str] = None
    pole_labels: Optional[List[List[str]]] = None
    dispersive_substrate: bool = False
    screened_interaction_convention: str = SCREENED_INTERACTION_CONVENTION
    notes: str = ""
    # Optional numerical diagnostics (see ``ppa_diagnostics``)
    pole_extraction_diagnostics: Optional[List[Any]] = None
    spectral_weight_diagnostics: Optional[List[Any]] = None
    passivity_diagnostics: Optional[List[Any]] = None
    omega_coverage_diagnostics: Optional[List[Any]] = None

    @property
    def omega_n(self) -> List[np.ndarray]:
        """Backward-compatible alias for ``pole_frequencies``."""
        return self.pole_frequencies

    def is_radial(self) -> bool:
        """Whether the self-energy must supply its own angular average over q."""
        if self.q_mesh is not None:
            if self.q_mesh not in ("radial", "cartesian"):
                raise ValueError(f"q_mesh must be 'radial' or 'cartesian', got {self.q_mesh!r}")
            return self.q_mesh == "radial"
        return _infer_radial_q_magnitudes(self.q_vectors) is not None

    def diagnostics_summary(self, q_index: Optional[int] = None) -> str:
        """Printable summary of stored PPA reliability diagnostics."""
        lines: List[str] = []
        n_q = self.q_vectors.shape[0]
        q_range = [q_index] if q_index is not None else range(n_q)
        for qi in q_range:
            qv = np.asarray(self.q_vectors[qi]).ravel()
            lines.append(f"q[{qi}] = {qv}")
            if self.pole_extraction_diagnostics and qi < len(self.pole_extraction_diagnostics):
                for d in self.pole_extraction_diagnostics[qi]:
                    lines.append("  " + d.summary())
            for name, block in (
                ("spectral", self.spectral_weight_diagnostics),
                ("passivity", self.passivity_diagnostics),
                ("omega_cov", self.omega_coverage_diagnostics),
            ):
                if block and qi < len(block) and block[qi] is not None:
                    lines.append(f"  [{name}] " + block[qi].summary())
            if self.pole_labels and qi < len(self.pole_labels):
                lines.append(f"  branches: {self.pole_labels[qi]}")
        return "\n".join(lines)


def build_plasmon_pole_data(
    layer_helper,
    epsilon_layer: np.ndarray,
    omega_grid: np.ndarray,
    *,
    coulomb_matrix: np.ndarray,
    q_vectors: np.ndarray,
    q_weights: Optional[np.ndarray] = None,
    q_mesh: Optional[str] = None,
    eta: float = 1e-3,
    n_modes: Optional[int] = None,
    min_weight: float = 0.0,
    drop_repeated_within: Optional[float] = None,
    weight_window_factor: float = 5.0,
    min_pole_prominence: float = 2.0,
    min_points_per_pole: int = 5,
    eps_builder: Optional[Callable[[int, np.ndarray], Tuple[np.ndarray, np.ndarray]]] = None,
    n_refine: int = 129,
    collect_diagnostics: bool = False,
    track_branches: bool = False,
    emit_warnings: bool = False,
) -> PlasmonPoleData:
    r"""
    Build per-q plasmon poles and W-residue matrices from ε(q, ω) and V(q, ω).

    Parameters
    ----------
    coulomb_matrix : ndarray
        As returned by ``dielectric_function_layer``: static
        ``(n_q, N, N, n_G)`` or dispersive ``(n_q, N, N, n_G, n_ω)``.
        Required for W = ε⁻¹ V residue extraction.
    q_weights : ndarray, optional
        Weights for Σ_q (must represent d²q/(2π)² or your discrete normalisation).
        If omitted, uses ``make_radial_q_weights`` when ``q_vectors`` is a 1D
        radial |q| grid; otherwise uniform 1/n_q with a warning.
    weight_window_factor : float
        Half-width of each pole residue window in units of max(η, grid step).
        Windows are then truncated at the midpoints between neighbouring poles.
    min_pole_prominence : float
        Minimum dip depth of |λ_min(ε)| for a minimum to count as a collective
        mode.  Guards against continuum ripples and phonon bands being fed to
        the PPA as poles.  See ``_minima_with_prominence``.
    min_points_per_pole : int
        A pole whose peak FWHM contains fewer ω-grid points than this is flagged
        as under-resolved in the diagnostics (and warned about when
        ``emit_warnings``).  Its residue is not trustworthy.
    eps_builder : callable, optional
        ``eps_builder(q_index, omega_array) -> (eps_q, coulomb_q)``.  When given,
        each residue window is integrated on its own ``n_refine``-point local
        grid instead of the global ω-grid, which removes the requirement that
        the global grid resolve every pole.  Recommended whenever ε can be
        re-evaluated on demand.
    n_refine : int
        Points per refined window (used only with ``eps_builder``).
    collect_diagnostics : bool
        If True, attach per-q ``PoleDiagnostics`` / ``SpectralWeightDiagnostics``
        to the returned ``PlasmonPoleData``.
    track_branches : bool
        Not implemented in this module (raises).  Poles are returned sorted in ω.
    emit_warnings : bool
        If True with ``collect_diagnostics``, emit ``UserWarning`` for
        under-resolved poles and sum-rule violations.

    Note
    ----
    The residue integral is only meaningful if χ⁰ carries the same +iη as V, so
    that poles have a finite, resolvable width; see ``find_plasmon_poles_at_q``.
    """
    epsilon_layer = np.asarray(epsilon_layer, dtype=np.complex128)
    coulomb_matrix = np.asarray(coulomb_matrix, dtype=np.complex128)
    if epsilon_layer.ndim != 6:
        raise ValueError("epsilon_layer must have shape (n_q, n_ω, N, N, n_G, n_G).")
    n_q, n_omega, N, _, n_G, _ = epsilon_layer.shape
    q_vectors = np.asarray(q_vectors)
    if q_vectors.shape[0] != n_q:
        raise ValueError("q_vectors first dim must equal epsilon_layer.shape[0].")
    if q_weights is None:
        q_radial = _infer_radial_q_magnitudes(q_vectors)
        if q_radial is not None:
            q_weights = make_radial_q_weights(q_radial)
        else:
            warnings.warn(
                "q_weights not provided and q_vectors is not a 1D radial grid; "
                "using uniform 1/n_q (not a physical d²q/(2π)² measure).",
                UserWarning,
                stacklevel=2,
            )
            q_weights = np.full(n_q, 1.0 / n_q)
    else:
        q_weights = np.asarray(q_weights, dtype=float)
    omega_grid = np.asarray(omega_grid, dtype=float).ravel()
    if omega_grid.size != n_omega:
        raise ValueError("omega_grid length must match epsilon_layer's ω-axis.")

    dispersive = False
    if layer_helper is not None:
        dispersive = getattr(layer_helper, "epsilon_mode", "static") == "dispersive"

    omegas_per_q: List[np.ndarray] = []
    residues_per_q: List[np.ndarray] = []
    phis_per_q: List[np.ndarray] = []
    rank1_per_q: List[np.ndarray] = []
    pole_diag_per_q: Optional[List[List[Any]]] = [] if collect_diagnostics else None
    spectral_per_q: Optional[List[Any]] = [] if collect_diagnostics else None
    # Passivity / ω-coverage checks lived in the absent ppa_diagnostics module;
    # the sum-rule and per-pole resolution records below supersede them.
    passivity_per_q: Optional[List[Any]] = None
    omega_cov_per_q: Optional[List[Any]] = None

    if track_branches:
        raise NotImplementedError(
            "track_branches is not available (the ppa_diagnostics module it "
            "relied on does not exist). Poles are returned sorted in ω; label "
            "branches downstream from the residue eigenvectors if needed."
        )

    for qi in range(n_q):
        eps_q = epsilon_layer[qi]
        if coulomb_matrix.ndim in (4, 5):
            cq = coulomb_matrix[qi]
        else:
            raise ValueError(
                f"coulomb_matrix must be 4-d or 5-d, got {coulomb_matrix.ndim}-d."
            )
        builder = None
        if eps_builder is not None:
            builder = (lambda om, _qi=qi: eps_builder(_qi, om))
        result = find_plasmon_poles_at_q(
            eps_q, omega_grid, coulomb_q=cq, eta=eta,
            n_modes=n_modes, min_weight=min_weight,
            drop_repeated_within=drop_repeated_within,
            weight_window_factor=weight_window_factor,
            min_pole_prominence=min_pole_prominence,
            min_points_per_pole=min_points_per_pole,
            eps_builder=builder, n_refine=n_refine,
            return_diagnostics=collect_diagnostics,
        )
        if collect_diagnostics:
            w_n, R_n, Phi_n, rq, (pdiag, sdiag) = result
            assert pole_diag_per_q is not None and spectral_per_q is not None
            pole_diag_per_q.append(pdiag)
            spectral_per_q.append(sdiag)
            if emit_warnings:
                for d in pdiag:
                    if not d.resolved(min_points_per_pole):
                        warnings.warn(
                            f"q index {qi}: {d.summary()} — residue is not "
                            "converged; refine the ω grid, increase η, or pass "
                            "eps_builder.",
                            UserWarning, stacklevel=2,
                        )
                if sdiag.captured_fraction > 1.05:
                    warnings.warn(
                        f"q index {qi}: {sdiag.summary()} — Σ_n R_n exceeds the "
                        "total spectral weight (double counting).",
                        UserWarning, stacklevel=2,
                    )
        else:
            w_n, R_n, Phi_n, rq = result
        omegas_per_q.append(w_n)
        residues_per_q.append(R_n)
        phis_per_q.append(Phi_n)
        rank1_per_q.append(rq)

    branch_labels: Optional[List[List[str]]] = None

    return PlasmonPoleData(
        q_vectors=q_vectors,
        q_weights=q_weights,
        q_mesh=q_mesh,
        pole_frequencies=omegas_per_q,
        residues=residues_per_q,
        Phi=phis_per_q,
        rank1_quality=rank1_per_q,
        n_layers=N,
        n_G=n_G,
        eta=eta,
        pole_labels=branch_labels,
        dispersive_substrate=dispersive,
        notes=("dispersive substrate detected (κ branch-tracking required)"
               if dispersive else ""),
        pole_extraction_diagnostics=pole_diag_per_q,
        spectral_weight_diagnostics=spectral_per_q,
        passivity_diagnostics=passivity_per_q,
        omega_coverage_diagnostics=omega_cov_per_q,
    )


# ---------------------------------------------------------------------------
# Form-factor helpers
# ---------------------------------------------------------------------------

FormFactorFn = Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]
r"""
Signature: ``form_factor_fn(k, k_minus_q) -> (energies, F)``

  * ``energies`` : ndarray (n_bands,) of band energies ξ^λ_{k-q} [eV].
  * ``F`` : ndarray, shape (n_bands, N, N) for head-only mode, or
            shape (n_bands, N, n_G, N, n_G) for the full off-diagonal-G mode.

For decoupled multi-2DEG (one band per layer, no G coupling), F is
layer-diagonal: F^λ_{l1,l2} = δ_{l1,l2} δ_{l1,λ}, with G-diagonal δ_{G1,G2}
in the full mode.
"""


def form_factor_from_densityL(
    densityL: np.ndarray,
    ki: int,
    qi: int,
    n_ext: int,
    *,
    head_only: bool = True,
) -> np.ndarray:
    r"""
    Build the self-energy form factor F_band from pybandstructure density-matrix
    elements stored in ``layer_helper.densityL``.

    ``densityL[l, n, m, ki, qi, G]`` =
    ``⟨k_{ki}, n | ρ_l(q_{qi}+G) | k_{ki+qi}, m⟩``,
    shape ``(n_layers, n_bands, n_bands, n_k, n_q, n_G)``.

    The form factor for external band ``n_ext`` at transferred momentum ``qi`` is

        F^m_{(l₁,G₁),(l₂,G₂)}(k_{ki}, k_{ki+qi})
            = densityL[l₁, n_ext, m, ki, qi, G₁]
              × conj(densityL[l₂, n_ext, m, ki, qi, G₂]).

    This encodes the physical statement that the density operator projects the
    initial state ``|k_{ki}, n_ext⟩`` onto layer l at G and the intermediate
    state ``|k_{ki+qi}, m⟩``.

    Parameters
    ----------
    densityL : ndarray, shape (n_layers, n_bands, n_bands, n_k, n_q, n_G)
    ki : int
        External k-point index in k_sample.
    qi : int
        Transferred-momentum index (same ordering as PPD q_vectors / q_sample).
    n_ext : int
        External band index (left band = initial state at k).
    head_only : bool
        If True return shape ``(n_bands, n_layers, n_layers)`` (G=G'=0 block).
        If False return shape ``(n_bands, n_layers, n_G, n_layers, n_G)`` (full G).

    Returns
    -------
    F_band : ndarray
        ``F_band[m, l₁, (G₁,) l₂, (G₂,)]`` for use with the existing
        einsum in ``compute_self_energy_layer`` / ``compute_self_energy_pybandstructure``.
    """
    if head_only:
        # rho[l, m] = densityL[l, n_ext, m, ki, qi, 0]
        rho = densityL[:, n_ext, :, ki, qi, 0]            # (n_layers, n_bands)
        return np.einsum("im,jm->mij", rho, np.conj(rho)) # (n_bands, n_layers, n_layers)
    else:
        # rho[l, m, G] = densityL[l, n_ext, m, ki, qi, G]
        rho = densityL[:, n_ext, :, ki, qi, :]               # (n_layers, n_bands, n_G)
        return np.einsum("img,hmj->mighj", rho, np.conj(rho)) # (n_bands, n_layers, n_G, n_layers, n_G)


def make_decoupled_parabolic_form_factor(
    mass_energy_eV: Sequence[float],
    chemical_potential_eV: float,
    *,
    masses_eV: Optional[Sequence[float]] = None,
    n_G: int = 1,
    full: bool = False,
    hbarc_eV_nm: float = 6.582119569e-16 * 3e17,
) -> FormFactorFn:
    r"""
    Form factor for N decoupled 2DEGs: one parabolic band per layer.

    ``mass_energy_eV`` must be the rest energy m* c² in eV (not kg, not a
    dimensionless effective mass).  Use ``effective_mass_ratio_to_energy`` to
    convert m*/m_e.  Deprecated alias: ``masses_eV`` (same meaning).

    Dispersion: ξ = (ℏc)² |k−q|² / (2 m* c²) − μ.
    """
    if masses_eV is not None:
        warnings.warn(
            "Argument masses_eV is deprecated; use mass_energy_eV (m* c² in eV).",
            DeprecationWarning,
            stacklevel=2,
        )
        mass_energy_eV = masses_eV
    masses = np.asarray(mass_energy_eV, dtype=float)
    N = masses.size

    def fn(k_vec: np.ndarray, k_minus_q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        k2 = float(np.dot(k_minus_q, k_minus_q))
        energies = hbarc_eV_nm ** 2 * k2 / (2.0 * masses) - chemical_potential_eV
        if full:
            F = np.zeros((N, N, n_G, N, n_G), dtype=np.complex128)
            for la in range(N):
                for g in range(n_G):
                    F[la, la, g, la, g] = 1.0
        else:
            F = np.zeros((N, N, N), dtype=np.complex128)
            for la in range(N):
                F[la, la, la] = 1.0
        return energies, F

    return fn


# ---------------------------------------------------------------------------
# Self-energy
# ---------------------------------------------------------------------------

def _normalize_form_factor_band(
    F_band: np.ndarray,
    *,
    n_bands: int,
    N: int,
    n_G: int,
    head_only: bool,
) -> np.ndarray:
    F_band = np.asarray(F_band, dtype=np.complex128)
    if head_only:
        if F_band.shape != (n_bands, N, N):
            if F_band.shape == (n_bands, N, n_G, N, n_G):
                return F_band[:, :, 0, :, 0]
            raise ValueError(
                f"F shape {F_band.shape} not compatible with head_only=True."
            )
        return F_band
    if F_band.shape != (n_bands, N, n_G, N, n_G):
        if F_band.shape == (n_bands, N, N):
            F_full = np.zeros((n_bands, N, n_G, N, n_G), dtype=np.complex128)
            F_full[:, :, 0, :, 0] = F_band
            return F_full
        raise ValueError(
            f"F shape {F_band.shape} not compatible with head_only=False."
        )
    return F_band


def _coupling_from_residue_or_phi(
    R_n: np.ndarray,
    phi_full: np.ndarray,
    *,
    N: int,
    n_G: int,
    head_only: bool,
    head_idx: np.ndarray,
    use_full_residue: bool,
) -> np.ndarray:
    if use_full_residue:
        if head_only:
            return R_n[np.ix_(head_idx, head_idx)]
        return R_n.reshape(N, n_G, N, n_G)
    phi_head = phi_full[head_idx]
    if head_only:
        return np.outer(phi_head, phi_head.conj())
    phi_grid = phi_full.reshape(N, n_G)
    return np.einsum("ig,jh->igjh", phi_grid, phi_grid.conj())


def compute_self_energy_layer(
    ppd: PlasmonPoleData,
    k_vec: np.ndarray,
    eps_values: np.ndarray,
    form_factor_fn: FormFactorFn,
    *,
    T: float = 0.0,
    eta: Optional[float] = None,
    head_only: bool = True,
    use_full_residue: bool = True,
    n_theta: int = 64,
    n_jobs: int = 1,
) -> np.ndarray:
    r"""
    Plasmon-pole self-energy (Σ-PPA) using W-residue matrices R_n(q) by default:

        Σ_ij = − Σ_{q,n,λ} w_q ⟨ R_{n,ij}(q) F^λ_{ij}(k,k−q) D_{n,λ} ⟩_θ,

    with D_{n,λ} = A_+/(ω_n+ξ−ε−iη) + A_−/(−ω_n+ξ−ε−iη).

    Parameters
    ----------
    use_full_residue : bool, default True
        If False, use rank-one Φ Φ† (approximation; emits a warning).
    n_theta : int, default 64
        Number of angular samples for ⟨·⟩_θ when ``ppd.q_vectors`` is a radial
        |q| grid (see ``_k_minus_q_samples``).  ω_n(q) and R_n(q) are isotropic,
        so only ξ_{k−q} and the form factor are averaged.  Ignored (forced to 1)
        for explicit 2D q meshes, where the q-sum already covers the angles.
        ``n_theta=1`` reproduces the pre-fix collinear-only behaviour and is
        wrong for any k ≠ 0 — keep it only for regression comparisons.
    """
    if not use_full_residue:
        warnings.warn(
            "use_full_residue=False uses rank-one Φ Φ† PPA; the physical default "
            "is the full W-residue matrix R_n(q).",
            UserWarning,
            stacklevel=2,
        )
    if eta is None:
        eta = ppd.eta
    k_vec = np.asarray(k_vec, dtype=float).ravel()
    eps_values = np.asarray(eps_values, dtype=float).ravel()
    N = ppd.n_layers
    n_G = ppd.n_G
    n_eps = eps_values.size

    if head_only:
        Sigma = np.zeros((N, N, n_eps), dtype=np.complex128)
    else:
        Sigma = np.zeros((N, n_G, N, n_G, n_eps), dtype=np.complex128)

    head_idx = np.arange(N) * n_G

    q_is_radial = ppd.is_radial()
    thetas, theta_w = make_angular_quadrature(n_theta if q_is_radial else 1)

    for qi in range(ppd.q_vectors.shape[0]):
        w_q = ppd.q_weights[qi]
        if w_q == 0:
            continue
        w_n_list = ppd.pole_frequencies[qi]
        if w_n_list.size == 0:
            continue
        residues_q = ppd.residues[qi]
        Phi_list = ppd.Phi[qi]

        kmq_samples = _k_minus_q_samples(
            k_vec, ppd.q_vectors[qi], radial=q_is_radial, thetas=thetas,
        )

        for ti in range(kmq_samples.shape[0]):
            w_qt = w_q * theta_w[ti]
            energies_band, F_band = form_factor_fn(k_vec, kmq_samples[ti])
            n_bands = energies_band.size
            F_band = _normalize_form_factor_band(
                F_band, n_bands=n_bands, N=N, n_G=n_G, head_only=head_only,
            )

            for n_idx in range(w_n_list.size):
                w_n = float(w_n_list[n_idx])
                R_coupling = _coupling_from_residue_or_phi(
                    residues_q[n_idx], Phi_list[n_idx],
                    N=N, n_G=n_G, head_only=head_only, head_idx=head_idx,
                    use_full_residue=use_full_residue,
                )
                A_plus, A_minus = population_factors(w_n, energies_band, T)
                xi = energies_band
                denom_p = (w_n + xi)[:, None] - eps_values[None, :] - 1j * eta
                denom_m = (-w_n + xi)[:, None] - eps_values[None, :] - 1j * eta
                contrib = A_plus[:, None] / denom_p + A_minus[:, None] / denom_m

                if head_only:
                    F_x_contrib = np.einsum("lij,le->ije", F_band, contrib)
                    Sigma += -w_qt * R_coupling[:, :, None] * F_x_contrib
                else:
                    F_x_contrib = np.einsum("lighj,le->ighje", F_band, contrib)
                    Sigma += -w_qt * R_coupling[:, :, :, :, None] * F_x_contrib

    return Sigma


def compute_self_energy_layer_per_pole(
    ppd: PlasmonPoleData,
    k_vec: np.ndarray,
    eps_values: np.ndarray,
    form_factor_fn: FormFactorFn,
    *,
    pole_selector: Callable[..., bool],
    T: float = 0.0,
    eta: Optional[float] = None,
    head_only: bool = True,
    use_full_residue: bool = True,
    n_theta: int = 64,
) -> np.ndarray:
    r"""
    Same as ``compute_self_energy_layer`` (including the angular average over
    the direction of q, see ``n_theta`` there) but only sums poles accepted by
    ``pole_selector``.

    Diagnostic branch selectors (acoustic / optical) project the *residue*
    matrix R_n to the layer head (G=0); they are not rigorous for asymmetric
    or local-field-heavy systems — use as diagnostics only.

    ``pole_selector(q_idx, pole_idx, omegas_at_q, Phis_at_q, residues_at_q=None)``
    """
    if not use_full_residue:
        warnings.warn(
            "use_full_residue=False uses rank-one Φ Φ† PPA; the physical default "
            "is the full W-residue matrix R_n(q).",
            UserWarning,
            stacklevel=2,
        )
    if eta is None:
        eta = ppd.eta
    k_vec = np.asarray(k_vec, dtype=float).ravel()
    eps_values = np.asarray(eps_values, dtype=float).ravel()
    N = ppd.n_layers
    n_G = ppd.n_G
    n_eps = eps_values.size
    if head_only:
        Sigma = np.zeros((N, N, n_eps), dtype=np.complex128)
    else:
        Sigma = np.zeros((N, n_G, N, n_G, n_eps), dtype=np.complex128)
    head_idx = np.arange(N) * n_G
    q_is_radial = ppd.is_radial()
    thetas, theta_w = make_angular_quadrature(n_theta if q_is_radial else 1)
    for qi in range(ppd.q_vectors.shape[0]):
        w_q = ppd.q_weights[qi]
        if w_q == 0:
            continue
        w_n_list = ppd.pole_frequencies[qi]
        Phi_list = ppd.Phi[qi]
        residues_q = ppd.residues[qi]
        if w_n_list.size == 0:
            continue
        kmq_samples = _k_minus_q_samples(
            k_vec, ppd.q_vectors[qi], radial=q_is_radial, thetas=thetas,
        )
        for ti in range(kmq_samples.shape[0]):
            w_qt = w_q * theta_w[ti]
            energies_band, F_band = form_factor_fn(k_vec, kmq_samples[ti])
            n_bands = energies_band.size
            F_band = _normalize_form_factor_band(
                F_band, n_bands=n_bands, N=N, n_G=n_G, head_only=head_only,
            )
            for n_idx in range(w_n_list.size):
                try:
                    keep = pole_selector(qi, n_idx, w_n_list, Phi_list, residues_q)
                except TypeError:
                    keep = pole_selector(qi, n_idx, w_n_list, Phi_list)
                if not keep:
                    continue
                w_n = float(w_n_list[n_idx])
                R_coupling = _coupling_from_residue_or_phi(
                    residues_q[n_idx], Phi_list[n_idx],
                    N=N, n_G=n_G, head_only=head_only, head_idx=head_idx,
                    use_full_residue=use_full_residue,
                )
                A_plus, A_minus = population_factors(w_n, energies_band, T)
                xi = energies_band
                denom_p = (w_n + xi)[:, None] - eps_values[None, :] - 1j * eta
                denom_m = (-w_n + xi)[:, None] - eps_values[None, :] - 1j * eta
                contrib = A_plus[:, None] / denom_p + A_minus[:, None] / denom_m
                if head_only:
                    F_x_contrib = np.einsum("lij,le->ije", F_band, contrib)
                    Sigma += -w_qt * R_coupling[:, :, None] * F_x_contrib
                else:
                    F_x_contrib = np.einsum("lighj,le->ighje", F_band, contrib)
                    Sigma += -w_qt * R_coupling[:, :, :, :, None] * F_x_contrib
    return Sigma


def compute_self_energy_pybandstructure(
    ppd: "PlasmonPoleData",
    layer_helper,
    k_index: int,
    band_index: int,
    eps_values: np.ndarray,
    *,
    T: float = 0.0,
    eta: Optional[float] = None,
    head_only: bool = True,
    use_full_residue: bool = True,
) -> np.ndarray:
    r"""
    Plasmon-pole self-energy using pybandstructure density-matrix elements.

    This is the pybandstructure-consistent counterpart of
    ``compute_self_energy_layer``.  The q-grid, form factors, and
    intermediate-state energies are all taken directly from
    ``layer_helper.crystal.band_structure`` and ``layer_helper.densityL``
    instead of being evaluated analytically via a ``FormFactorFn``.

    Physics (same as ``compute_self_energy_layer``):

        Σ_{ij}(k, ε) = − Σ_{q,n,m}  w_q  R_{n,ij}(q)  F^m_{ij}(k, k+q)  D_{n,m}

    The substitution q → −q has been applied so that the sum runs over the
    pybandstructure q_sample as-is and ``layer_helper.densityL`` at the
    *positive* q directly provides the form factors.  The equality
    ``W(−q) = W(q)`` (valid for systems with spatial inversion or
    time-reversal symmetry) is assumed; for non-centrosymmetric systems
    without TRS the sign-convention inconsistency between the ±q
    contributions would require explicit symmetrisation.

    The form factor at transferred momentum q from the initial state
    ``|k_{ki}, n_ext⟩`` to the intermediate state ``|k_{ki+qi}, m⟩`` is

        F^m_{(l₁,G₁),(l₂,G₂)} = ⟨k,n|ρ_{l₁}(q+G₁)|k+q,m⟩
                                × ⟨k,n|ρ_{l₂}(q+G₂)|k+q,m⟩*

    which is ``form_factor_from_densityL(densityL, ki, qi, n_ext)``.
    Intermediate-state energies are
    ``ξ^m_{k+q} = energies[m, k_plus_q_ind[qi, ki]] − μ``.

    Consistency requirement
    ----------------------
    ``ppd.q_vectors[qi]`` must equal ``band_structure.q_sample.coords[qi]``
    for every qi (i.e. the PPD must have been built from the same q_sample as
    the band structure).  Use ``q_weights_from_band_structure`` to obtain the
    correct ``q_weights`` for ``build_plasmon_pole_data``.

    Parameters
    ----------
    ppd : PlasmonPoleData
        Pre-computed plasmon-pole data built from the **same** q_sample as the
        band structure (same index ordering).
    layer_helper : layer_helpers_G
        Must have ``layer_helper.crystal`` set and ``layer_helper.densityL``
        pre-computed (shape ``(n_layers, n_bands, n_bands, n_k, n_q, n_G)``).
    k_index : int
        Index of the external k-point in ``band_structure.k_sample``.
    band_index : int
        External band index n (the initial state at k).
    eps_values : ndarray, shape (n_ε,)
        Energy arguments ε [eV] at which Σ is evaluated.
    T : float, optional
        Temperature in eV (default 0).
    eta : float or None, optional
        Quasiparticle broadening η; defaults to ``ppd.eta``.
    head_only : bool, default True
        If True, return shape (N, N, n_ε) — only the G=G'=0 block.
        If False, return shape (N, n_G, N, n_G, n_ε) — full local-field matrix.
    use_full_residue : bool, default True
        Use full W-residue matrix R_n(q) (physical default).
        If False, use rank-one Φ Φ† approximation (emits a warning).

    Returns
    -------
    Sigma : ndarray
        Self-energy in layer basis.
        Shape ``(N, N, n_ε)`` (head_only=True) or ``(N, n_G, N, n_G, n_ε)``.
    """
    if layer_helper.crystal is None:
        raise ValueError(
            "layer_helper.crystal must be set to use pybandstructure form factors."
        )
    if layer_helper.densityL is None:
        raise ValueError(
            "layer_helper.densityL is not available; call compute_density_layer() first."
        )
    if not use_full_residue:
        warnings.warn(
            "use_full_residue=False uses rank-one Φ Φ† PPA; the physical default "
            "is the full W-residue matrix R_n(q).",
            UserWarning,
            stacklevel=2,
        )
    if eta is None:
        eta = ppd.eta

    bs = layer_helper.crystal.band_structure
    densityL = layer_helper.densityL   # (n_layers, n_bands, n_bands, n_k, n_q, n_G)

    eps_values = np.asarray(eps_values, dtype=float).ravel()
    N = ppd.n_layers
    n_G = ppd.n_G
    n_eps = eps_values.size
    n_q = ppd.q_vectors.shape[0]

    if head_only:
        Sigma = np.zeros((N, N, n_eps), dtype=np.complex128)
    else:
        Sigma = np.zeros((N, n_G, N, n_G, n_eps), dtype=np.complex128)

    head_idx = np.arange(N) * n_G

    k_plus_q_ind = bs.k_plus_q_ind      # (n_q, n_k)
    energies_all = bs.energies           # (n_bands, n_k)
    mu = float(bs.chemical_potential)

    # Pre-extract the density-matrix slice for (k_index, band_index) across all
    # qi and G to avoid repeated array lookups inside the q-loop.
    # rho_ki[l, m, qi, G] = densityL[l, n_ext, m, k_index, qi, G]
    rho_ki = densityL[:, band_index, :, k_index, :, :]  # (N, n_bands, n_q, n_G)

    for qi in range(n_q):
        w_q = float(ppd.q_weights[qi])
        if w_q == 0.0:
            continue

        w_n_list = ppd.pole_frequencies[qi]
        if w_n_list.size == 0:
            continue
        residues_q = ppd.residues[qi]
        Phi_list = ppd.Phi[qi]

        # Index of k_{k_index} + q_{qi} in k_sample
        k_pq_idx = int(k_plus_q_ind[qi, k_index])

        # Intermediate-state energies ξ^m_{k+q} = E_m(k+q) − μ
        energies_band = energies_all[:, k_pq_idx] - mu   # (n_bands,)

        # Form factor F_band from densityL slice at this qi
        if head_only:
            rho_qi = rho_ki[:, :, qi, 0]                          # (N, n_bands)
            F_band = np.einsum("im,jm->mij", rho_qi, np.conj(rho_qi))
            # shape (n_bands, N, N)
        else:
            rho_qi = rho_ki[:, :, qi, :]                                  # (N, n_bands, n_G)
            F_band = np.einsum("img,hmj->mighj", rho_qi, np.conj(rho_qi))
            # shape (n_bands, N, n_G, N, n_G)

        for n_idx in range(w_n_list.size):
            w_n = float(w_n_list[n_idx])
            R_coupling = _coupling_from_residue_or_phi(
                residues_q[n_idx], Phi_list[n_idx],
                N=N, n_G=n_G, head_only=head_only, head_idx=head_idx,
                use_full_residue=use_full_residue,
            )
            A_plus, A_minus = population_factors(w_n, energies_band, T)
            xi = energies_band
            denom_p = (w_n + xi)[:, None] - eps_values[None, :] - 1j * eta
            denom_m = (-w_n + xi)[:, None] - eps_values[None, :] - 1j * eta
            contrib = A_plus[:, None] / denom_p + A_minus[:, None] / denom_m

            if head_only:
                F_x_contrib = np.einsum("lij,le->ije", F_band, contrib)
                Sigma += -w_q * R_coupling[:, :, None] * F_x_contrib
            else:
                F_x_contrib = np.einsum("lighj,le->ighje", F_band, contrib)
                Sigma += -w_q * R_coupling[:, :, :, :, None] * F_x_contrib

    return Sigma


# ---------------------------------------------------------------------------
# Common pole selectors
# ---------------------------------------------------------------------------

def _head_vector_from_mode(
    mode_vec: np.ndarray,
    N: Optional[int],
    n_G: int = 1,
) -> np.ndarray:
    mode_vec = np.asarray(mode_vec, dtype=np.complex128).ravel()
    if N is None:
        return mode_vec
    return mode_vec[np.arange(N) * n_G]


def _branch_projection_fraction(head: np.ndarray, branch_vec: np.ndarray) -> float:
    branch_vec = np.asarray(branch_vec, dtype=np.complex128)
    branch_vec = branch_vec / (np.linalg.norm(branch_vec) + 1e-300)
    proj = np.abs(np.vdot(branch_vec, head)) ** 2
    norm = np.linalg.norm(head) ** 2 + 1e-300
    return float(proj / norm)


def select_acoustic_pole(
    q_idx,
    pole_idx,
    omegas_at_q,
    Phis_at_q,
    residues_at_q=None,
    *,
    N=None,
    n_G: int = 1,
    layer_signs=None,
    threshold=0.6,
):
    r"""
    Diagnostic pole selector: acoustic-like (out-of-phase) head-layer pattern.

    For a symmetric bilayer, u_ac = (1, −1)/√2.  Uses the dominant head eigenvector
    of the W-residue matrix R when provided; otherwise falls back to Φ.

    Not rigorous for asymmetric, local-field, or dispersive-environment systems.
    """
    if residues_at_q is not None and N is not None:
        R = residues_at_q[pole_idx]
        head_idx = np.arange(N) * n_G
        R_head = R[np.ix_(head_idx, head_idx)]
        w, U = np.linalg.eigh(0.5 * (R_head + R_head.conj().T))
        head = U[:, int(np.argmax(w))]
    else:
        head = _head_vector_from_mode(Phis_at_q[pole_idx], N, n_G)
        if N is None:
            N = head.size
    if layer_signs is None:
        if N == 2:
            layer_signs = np.array([1.0, -1.0], dtype=np.complex128) / np.sqrt(2.0)
        else:
            layer_signs = np.array([(-1) ** i for i in range(N)], dtype=float)
    return _branch_projection_fraction(head, layer_signs) >= threshold


def select_optical_pole(
    q_idx,
    pole_idx,
    omegas_at_q,
    Phis_at_q,
    residues_at_q=None,
    *,
    N=None,
    n_G: int = 1,
    threshold=0.6,
):
    r"""
    Diagnostic pole selector: optical-like (in-phase) head-layer pattern.

    For a symmetric bilayer, u_op = (1, 1)/√2.  See ``select_acoustic_pole`` caveats.
    """
    if residues_at_q is not None and N is not None:
        R = residues_at_q[pole_idx]
        head_idx = np.arange(N) * n_G
        R_head = R[np.ix_(head_idx, head_idx)]
        w, U = np.linalg.eigh(0.5 * (R_head + R_head.conj().T))
        head = U[:, int(np.argmax(w))]
    else:
        head = _head_vector_from_mode(Phis_at_q[pole_idx], N, n_G)
        if N is None:
            N = head.size
    uniform = np.ones(N, dtype=np.complex128) / np.sqrt(N)
    return _branch_projection_fraction(head, uniform) >= threshold


# ---------------------------------------------------------------------------
# Caroli / Meir-Wingreen tunneling current
# ---------------------------------------------------------------------------

def caroli_tunneling_current(
    eps_grid: np.ndarray,
    Sigma_kε: np.ndarray,
    H0: np.ndarray,
    Gamma1: np.ndarray,
    Gamma2: np.ndarray,
    mu1: float,
    mu2: float,
    T: float = 0.0,
    *,
    k_weights: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Evaluate the Caroli tunneling current
        J_{1→2}(V) = (2e/ℏ) ∫ dε/2π  Tr[Γ_1 G^R(ε) Γ_2 G^A(ε)]
                     × [n_F(ε−μ_1) − n_F(ε−μ_2)]
    with
        G^R(ε) = [ε I − H_0 − Σ^R(ε) + i(Γ_1 + Γ_2)/2]^{−1}

    (lead self-energy Σ^R_lead = −iΓ/2).

    The trace is over the *layer* (or general central-region) index.

    Parameters
    ----------
    eps_grid : ndarray (n_ε,)
        ε integration grid (the bias window must lie inside).
    Sigma_kε : ndarray, shape (n_k, N, N, n_ε)  or  (N, N, n_ε)
        Self-energy.  If 4-d, k-resolved and we sum over k with k_weights.
        If 3-d, a single k is assumed.
    H0 : ndarray, shape (n_k, N, N) or (N, N)
        Non-interacting Hamiltonian at each k.
    Gamma1, Gamma2 : ndarray, shape (N, N)
        Lead-coupling matrices (positive semi-definite, energy-independent).
    mu1, mu2 : float
        Lead chemical potentials in eV (bias V = (μ1 − μ2)/e).
    T : float
        Temperature in eV.
    k_weights : ndarray (n_k,), optional
        BZ weights for the k-sum.  Default uniform 1/n_k.

    Returns
    -------
    J_prefactor : float
        ∫ dε/(2π) T_tot(ε) [f(ε−μ₁) − f(ε−μ₂)] in units of **dimensionless × eV**.
        Physical current: ``J [A] = g_deg × E_OVER_HBAR_A_PER_EV × J_prefactor``
        with ``E_OVER_HBAR = 0.5 × TWOE_OVER_HBAR``.  Do not multiply by
        ``TWOE_OVER_HBAR`` and ``g_deg=2`` together — that double-counts spin.
    transmission : ndarray, shape (n_ε,)
        T(ε) = Σ_k w_k Tr[Γ_1 G^R Γ_2 G^A]  (real, ≥ 0).
    fermi_diff : ndarray, shape (n_ε,)
        n_F(ε−μ_1) − n_F(ε−μ_2)  (for diagnostics).
    """
    eps_grid = np.asarray(eps_grid, dtype=float).ravel()
    Sigma_kε = np.asarray(Sigma_kε, dtype=np.complex128)
    H0 = np.asarray(H0, dtype=np.complex128)
    if Sigma_kε.ndim == 3:
        Sigma_kε = Sigma_kε[np.newaxis]   # (1, N, N, n_ε)
    if H0.ndim == 2:
        H0 = H0[np.newaxis]               # (1, N, N)
    n_k = Sigma_kε.shape[0]
    N = Sigma_kε.shape[1]
    n_eps = Sigma_kε.shape[3]
    assert H0.shape[0] == n_k, "Sigma and H0 must share k-axis"
    Gamma1 = np.asarray(Gamma1, dtype=np.complex128)
    Gamma2 = np.asarray(Gamma2, dtype=np.complex128)
    if Gamma1.shape != (N, N) or Gamma2.shape != (N, N):
        raise ValueError("Gamma matrices must have shape (N, N).")
    if k_weights is None:
        k_weights = np.full(n_k, 1.0 / n_k)
    Gamma_tot = Gamma1 + Gamma2

    transmission = np.zeros(n_eps, dtype=float)
    for ki in range(n_k):
        for ei, eps in enumerate(eps_grid):
            M = eps * np.eye(N) - H0[ki] - Sigma_kε[ki, :, :, ei] + 1j * Gamma_tot / 2.0
            GR = np.linalg.inv(M)
            GA = GR.conj().T
            T_eps = np.real(np.trace(Gamma1 @ GR @ Gamma2 @ GA))
            transmission[ei] += k_weights[ki] * T_eps

    # Fermi factor difference
    def _nF(x, T_):
        if T_ <= 0:
            return (x < 0).astype(float)
        return 0.5 * (1.0 - np.tanh(0.5 * x / T_))
    fermi_diff = _nF(eps_grid - mu1, T) - _nF(eps_grid - mu2, T)

    if eps_grid.size > 1:
        J = _trapezoid(transmission * fermi_diff, eps_grid) / (2.0 * np.pi)
    else:
        J = float(transmission[0] * fermi_diff[0]) / (2.0 * np.pi)
    return float(J), transmission, fermi_diff


# ---------------------------------------------------------------------------
# Self-tests / demo
# ---------------------------------------------------------------------------

def _self_test():
    """Quick smoke test on a 1-pole synthetic ε = I − V χ."""
    print("Self-energy PPA self-test (v2.1 W-residue)")
    N, n_G = 1, 1
    n_q = 5
    n_omega = 401
    eta = 1e-3
    eps_layer = np.zeros((n_q, n_omega, N, N, n_G, n_G), dtype=np.complex128)
    coulomb_matrix = np.zeros((n_q, N, N, n_G), dtype=np.complex128)
    omega_grid = np.linspace(1e-3, 0.05, n_omega)
    alpha = 0.1
    qs = np.linspace(0.01, 0.05, n_q)
    V_test = 1.0
    for qi, q in enumerate(qs):
        wp = alpha * np.sqrt(q)
        coulomb_matrix[qi, 0, 0, 0] = V_test
        for oi, w in enumerate(omega_grid):
            chi = wp ** 2 / (w + 1j * eta) ** 2
            eps_layer[qi, oi, 0, 0, 0, 0] = 1.0 - V_test * chi

    ppd = build_plasmon_pole_data(
        layer_helper=None,
        epsilon_layer=eps_layer,
        coulomb_matrix=coulomb_matrix,
        omega_grid=omega_grid,
        q_vectors=qs.reshape(-1, 1),
        eta=eta,
        n_modes=1,
    )
    m_star = effective_mass_ratio_to_energy(0.067)
    ff = make_decoupled_parabolic_form_factor([float(m_star)], 0.0)
    # k_vec chosen so ξ_{k-q} ≠ 0 for every q in qs (avoid Θ vs Fermi(0)=0.5 ambiguity)
    k_vec = np.array([0.0])
    eps_vals = np.linspace(-0.05, 0.05, 11)
    Sigma_head = compute_self_energy_layer(ppd, k_vec, eps_vals, ff, T=0.0, head_only=True)
    assert Sigma_head.shape == (1, 1, 11)
    Sigma_full = compute_self_energy_layer(ppd, k_vec, eps_vals, ff, T=0.0, head_only=False)
    assert Sigma_full.shape == (1, 1, 1, 1, 11)
    assert np.allclose(Sigma_head, Sigma_full[:, 0, :, 0, :]), \
        "head-only and full-G outputs disagree on G=G'=0 block"
    Sigma_Tlow = compute_self_energy_layer(ppd, k_vec, eps_vals, ff, T=1e-8, head_only=True)
    assert np.allclose(Sigma_head, Sigma_Tlow, atol=1e-8), \
        f"T=1e-8 should match T=0 (max diff = {np.max(np.abs(Sigma_head - Sigma_Tlow))})"
    Sigma_Thigh = compute_self_energy_layer(ppd, k_vec, eps_vals, ff, T=0.02, head_only=True)
    assert not np.allclose(Sigma_head, Sigma_Thigh), \
        "T=0.02 should differ from T=0"
    assert ppd.residues[0].shape[1] == N * n_G
    print("  PPA v2.1 self-test OK: W residues + head-only ↔ full-G + finite-T.")


if __name__ == "__main__":
    _self_test()
