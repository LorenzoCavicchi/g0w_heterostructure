#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
G0W self-energy and spectral function of the graphene sample in SLG|hBN|SLG.

Dirac form factor
-----------------
``make_decoupled_parabolic_form_factor`` is *wrong* for graphene: it sets
F = 1 for the single band of each layer, which drops the sublattice
(chirality) overlap.  For a Dirac cone the intermediate-state overlap is

    |<u^{s'}_{k} | u^{s}_{k-q}>|^2 = (1 + s s' cos(angle(k, k-q))) / 2 ,

with s, s' = +/-1 the conduction/valence labels and both momenta measured from
the *Dirac point*.  This suppresses backscattering (theta = pi, s = s' gives 0)
and controls the relative weight of the intraband and interband channels.  It
is the graphene analogue of Eq. (5) of Wei et al., <u_k|u_{k-q}>.

Layer structure
---------------
The two sheets are decoupled (no interlayer tunneling), so an electron in the
sample stays in the sample: the form factor is layer-diagonal *and* projects
onto the sample layer only,

    F^lambda_{l1 l2} = delta_{l1,S} delta_{l2,S} * chirality ,

hence Sigma is layer-diagonal and only ``Sigma[S,S]`` is populated.  The tip
never contributes an intermediate state; it enters solely through the screened
interaction W (its polarizability reshapes R_n(q) and the pole positions).

This is the situation ``qtm_layer_resolved.py`` documents under "Layer-off-
diagonal Sigma needs layer-hybridised states": with decoupled sheets the
layer-coherence channel is absent by construction, and the acoustic mode
reaches the sample through ``R_{SS}``, not through ``Sigma_{ST}``.  Getting
Sigma_{ST} != 0 would require a hybridised (tunnel-coupled) basis, which is a
different physical system from the QTM junction in the weak-tunneling limit.

Units: energies eV, momenta nm^-1 measured from the Dirac point.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np

from . import dirac_chi as cd
from .slg_stack import SAMPLE_LAYER, TIP_LAYER, SLGStackConfig, stack_eps_and_V
from ..self_energy_full_frequency import (
    build_W_grid_from_eps_coulomb,
    compute_self_energy_full_frequency,
)
from ..self_energy_ppa import PlasmonPoleData, compute_self_energy_layer


# ---------------------------------------------------------------------------
# Form factor
# ---------------------------------------------------------------------------

def chirality_overlap(k_vec: np.ndarray, kmq_vec: np.ndarray,
                      s_ext: int, s: int) -> float:
    r"""|<u^{s_ext}_k | u^{s}_{k-q}>|^2 = (1 + s_ext*s*cos(angle))/2."""
    k = np.asarray(k_vec, dtype=float).ravel()[:2]
    p = np.asarray(kmq_vec, dtype=float).ravel()[:2]
    nk, npv = np.linalg.norm(k), np.linalg.norm(p)
    if nk < 1e-14 or npv < 1e-14:
        # One state sits at the Dirac point, where the two bands are degenerate
        # and the overlap is 1/2 for either sign.
        return 0.5
    c = float(np.clip(np.dot(k, p) / (nk * npv), -1.0, 1.0))
    return 0.5 * (1.0 + float(s_ext) * float(s) * c)


def make_dirac_form_factor(
    cfg: SLGStackConfig,
    *,
    band_ext: int = +1,
    n_layers: int = 2,
    sample_layer: int = SAMPLE_LAYER,
) -> Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    r"""
    ``form_factor_fn(k, k-q) -> (energies, F)`` for the graphene sample.

    ``energies`` : (2,) intermediate-band energies  xi^s_{k-q} = s hv |k-q| - E_F
    ``F``        : (2, n_layers, n_layers), nonzero only in the [S, S] entry.

    Parameters
    ----------
    band_ext : +1 (conduction) or -1 (valence)
        The external band whose self-energy is computed.  dI/dV at positive bias
        with an electron-doped sample probes the conduction band.
    """
    if band_ext not in (+1, -1):
        raise ValueError("band_ext must be +1 (conduction) or -1 (valence).")
    hv = float(cfg.hv)
    mu = float(cfg.E_F)
    bands = (+1, -1)

    def fn(k_vec: np.ndarray, kmq_vec: np.ndarray):
        p = np.asarray(kmq_vec, dtype=float).ravel()[:2]
        p_norm = float(np.linalg.norm(p))
        energies = np.array([s * hv * p_norm - mu for s in bands], dtype=float)
        F = np.zeros((len(bands), n_layers, n_layers), dtype=np.complex128)
        for i, s in enumerate(bands):
            F[i, sample_layer, sample_layer] = chirality_overlap(
                k_vec, p, band_ext, s,
            )
        return energies, F

    return fn


# ---------------------------------------------------------------------------
# Self-energy drivers
# ---------------------------------------------------------------------------

def sigma_ppa(
    ppd: PlasmonPoleData,
    cfg: SLGStackConfig,
    k_vec: np.ndarray,
    eps_values: np.ndarray,
    *,
    band_ext: int = +1,
    T: float = 0.0,
    eta: Optional[float] = None,
    n_theta: int = 64,
) -> np.ndarray:
    r"""
    Plasmon-pole Sigma^R(k, eps) of the sample band.  Returns shape (2, 2, n_eps);
    only the [S, S] entry is nonzero (see module docstring).
    """
    ff = make_dirac_form_factor(cfg, band_ext=band_ext)
    return compute_self_energy_layer(
        ppd, np.asarray(k_vec, dtype=float).ravel(), eps_values, ff,
        T=T, eta=eta, head_only=True, use_full_residue=True, n_theta=n_theta,
    )


def build_W_grid(cfg: SLGStackConfig, q_grid: np.ndarray,
                 omega_grid: np.ndarray) -> np.ndarray:
    r"""
    W(q, w) = eps^-1 V on the full (q, w) grid, shape (n_q, n_w, 2, 2), for the
    full-frequency route.
    """
    n_q, n_w = len(q_grid), len(omega_grid)
    eps_layer = np.zeros((n_q, n_w, 2, 2, 1, 1), dtype=np.complex128)
    coulomb = np.zeros((n_q, 2, 2, 1, n_w), dtype=np.complex128)
    for qi, q in enumerate(q_grid):
        e_q, c_q = stack_eps_and_V(float(q), omega_grid, cfg)
        eps_layer[qi] = e_q
        coulomb[qi] = c_q
    return build_W_grid_from_eps_coulomb(eps_layer, coulomb, omega_grid)


def sigma_full_frequency(
    cfg: SLGStackConfig,
    k_vec: np.ndarray,
    eps_values: np.ndarray,
    q_grid: np.ndarray,
    omega_grid: np.ndarray,
    W_grid: np.ndarray,
    q_weights: np.ndarray,
    *,
    band_ext: int = +1,
    T: float = 0.0,
    eta_self: float = 1e-3,
    n_theta: int = 64,
) -> np.ndarray:
    r"""
    Full-frequency Sigma^R(k, eps): integrates the whole -Im W spectrum rather
    than a set of poles.

    This is the *production* route for graphene.  The PPA captures only the
    undamped collective modes, and in doped graphene the plasmon merges into the
    interband particle-hole continuum around q ~ k_F, above which most of the
    spectral weight is not in any pole at all (the ``captured`` fraction
    reported by ``build_slg_ppd(collect_diagnostics=True)`` falls below ~30%).
    """
    ff = make_dirac_form_factor(cfg, band_ext=band_ext)
    return compute_self_energy_full_frequency(
        np.asarray(k_vec, dtype=float).ravel(), eps_values, ff,
        omega_grid, W_grid, q_grid.reshape(-1, 1),
        n_layers=2, n_G=1, q_weights=q_weights,
        T=T, eta_self=eta_self, head_only=True, n_theta=n_theta,
    )


# ---------------------------------------------------------------------------
# Spectral function
# ---------------------------------------------------------------------------

def spectral_function_sample(
    cfg: SLGStackConfig,
    k_vec: np.ndarray,
    eps_values: np.ndarray,
    Sigma: np.ndarray,
    *,
    band_ext: int = +1,
    eta_qp: float = 1e-3,
    sample_layer: int = SAMPLE_LAYER,
) -> np.ndarray:
    r"""
    A^S(k, eps) = -(1/pi) Im [ eps - xi_k - Sigma^R(k, eps) + i*eta_qp ]^-1 .

    ``xi_k = band_ext * hv |k| - E_F`` is the bare band energy of the external
    state, with ``k`` measured from the sample Dirac point.  ``eta_qp`` is a
    small extrinsic width (disorder / finite junction) and should be kept well
    below the plasmon energy so the satellites stay resolvable.

    Returns shape (n_eps,), normalised so that int A deps = 1 for Sigma = 0.
    """
    eps_values = np.asarray(eps_values, dtype=float).ravel()
    k = np.asarray(k_vec, dtype=float).ravel()[:2]
    xi = float(band_ext) * cfg.hv * float(np.linalg.norm(k)) - cfg.E_F
    S = np.asarray(Sigma, dtype=np.complex128)
    sig = S[sample_layer, sample_layer, :] if S.ndim == 3 else S
    G = 1.0 / (eps_values - xi - sig + 1j * float(eta_qp))
    return -np.imag(G) / np.pi


def quasiparticle_energy(cfg: SLGStackConfig, k_vec: np.ndarray,
                         band_ext: int = +1) -> float:
    r"""Bare band energy xi_k = band_ext * hv |k| - E_F  [eV]."""
    k = np.asarray(k_vec, dtype=float).ravel()[:2]
    return float(band_ext) * cfg.hv * float(np.linalg.norm(k)) - cfg.E_F


# ---------------------------------------------------------------------------
# Vectorised full-frequency self-energy
# ---------------------------------------------------------------------------

def sigma_full_frequency_fast(
    cfg: SLGStackConfig,
    k_vec: np.ndarray,
    eps_values: np.ndarray,
    q_grid: np.ndarray,
    omega_grid: np.ndarray,
    W_grid: np.ndarray,
    q_weights: np.ndarray,
    *,
    band_ext: int = +1,
    T: float = 0.0,
    eta_self: float = 1e-3,
    n_theta: int = 64,
    sample_layer: int = SAMPLE_LAYER,
    eps_chunk: int = 48,
) -> np.ndarray:
    r"""
    Same quantity as :func:`sigma_full_frequency`, evaluated with the omega
    integral written as a contraction instead of a ``np.trapz`` per
    (q, theta, band, eps, i, j).

    Specialised to this benchmark: the form factor is nonzero only in the
    [sample, sample] layer block, so the N x N loop collapses to one element,
    and the eps loop becomes a matrix product over the omega axis.  Validated
    against ``compute_self_energy_full_frequency`` in ``test_slg_hbn_slg.py``.

    Returns shape (2, 2, n_eps) with only [S, S] populated, matching
    :func:`sigma_full_frequency`.
    """
    from ..self_energy_ppa import _n_bose, _n_fermi, make_angular_quadrature

    k_vec = np.asarray(k_vec, dtype=float).ravel()[:2]
    eps_values = np.asarray(eps_values, dtype=float).ravel()
    omega_grid = np.asarray(omega_grid, dtype=float).ravel()
    W_grid = np.asarray(W_grid, dtype=np.complex128)
    q_grid = np.asarray(q_grid, dtype=float).ravel()
    q_weights = np.asarray(q_weights, dtype=float).ravel()

    n_eps = eps_values.size
    hv, mu = float(cfg.hv), float(cfg.E_F)
    eta = float(eta_self)

    thetas, theta_w = make_angular_quadrature(int(n_theta))
    cos_t, sin_t = np.cos(thetas), np.sin(thetas)

    # trapezoidal weights on the omega grid
    tw = np.zeros_like(omega_grid)
    if omega_grid.size > 1:
        d = np.diff(omega_grid)
        tw[:-1] += 0.5 * d
        tw[1:] += 0.5 * d
    else:
        tw[:] = 1.0
    nB = _n_bose(omega_grid, T)

    out = np.zeros(n_eps, dtype=np.complex128)

    for qi, q in enumerate(q_grid):
        w_q = float(q_weights[qi])
        if w_q == 0.0:
            continue
        # spectral density of W in the sample-layer block, times trapz weight
        S_om = (-np.imag(W_grid[qi, :, sample_layer, sample_layer]) / np.pi) * tw

        # k - q(theta) for every angular sample
        px = k_vec[0] - q * cos_t
        py = k_vec[1] - q * sin_t
        p_norm = np.hypot(px, py)

        # chirality overlap with the external band
        nk = float(np.linalg.norm(k_vec))
        if nk < 1e-14 or np.any(p_norm < 1e-14):
            cos_ang = np.zeros_like(p_norm)
            good = (p_norm > 1e-14) & (nk > 1e-14)
            cos_ang[good] = np.clip(
                (k_vec[0] * px[good] + k_vec[1] * py[good]) / (nk * p_norm[good]),
                -1.0, 1.0,
            )
        else:
            cos_ang = np.clip((k_vec[0] * px + k_vec[1] * py) / (nk * p_norm), -1.0, 1.0)

        # flatten (theta, band) into one axis
        xi_list, wF_list = [], []
        for s in (+1, -1):
            xi_list.append(s * hv * p_norm - mu)                        # (n_theta,)
            wF_list.append(theta_w * 0.5 * (1.0 + band_ext * s * cos_ang))
        xi = np.concatenate(xi_list)                                    # (n_b,)
        wF = np.concatenate(wF_list)                                    # (n_b,)
        keep = wF > 1e-15
        xi, wF = xi[keep], wF[keep]
        if xi.size == 0:
            continue

        A_p = nB[None, :] + _n_fermi(-xi, T)[:, None]                   # (n_b, n_om)
        A_m = nB[None, :] + _n_fermi(+xi, T)[:, None]

        for e0 in range(0, n_eps, int(eps_chunk)):
            e1 = min(e0 + int(eps_chunk), n_eps)
            u = xi[:, None, None] - eps_values[None, None, e0:e1]       # (n_b,1,n_e)
            om = omega_grid[None, :, None]
            kern = (A_p[:, :, None] / (om + u - 1j * eta)
                    + A_m[:, :, None] / (-om + u - 1j * eta))           # (n_b,n_om,n_e)
            out[e0:e1] += -w_q * np.einsum("b,o,boe->e", wF, S_om, kern)

    Sigma = np.zeros((2, 2, n_eps), dtype=np.complex128)
    Sigma[sample_layer, sample_layer, :] = out
    return Sigma


# ---------------------------------------------------------------------------
# Optical / acoustic channel decomposition
# ---------------------------------------------------------------------------

def sigma_full_frequency_channels(
    cfg: SLGStackConfig,
    k_vec: np.ndarray,
    eps_values: np.ndarray,
    q_grid: np.ndarray,
    omega_grid: np.ndarray,
    W_grid: np.ndarray,
    q_weights: np.ndarray,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Split Sigma_SS into in-phase (optical) and out-of-phase (acoustic) channels.

    For two *identical* sheets the layer-symmetry eigenvectors u_pm = (1, +-1)/sqrt2
    diagonalise the screened interaction exactly, and

        S_++ = S_SS + S_ST ,   S_-- = S_SS - S_ST
        =>   S_SS = (1/2) S_++  +  (1/2) S_--

    so the sample-layer spectral density splits into one half carried by the
    in-phase channel and one half by the out-of-phase channel, and the same
    split passes linearly through the Sigma integral.  Unlike selecting poles by
    the sign of R_ST, this is a *symmetry* projection: it stays exact where the
    modes are Landau damped and not poles at all, which is most of the q range
    here (see the PPA capture numbers).

    Only meaningful when both sheets respond, and the projection itself tells you
    whether that is the case:

    * ``chi_on_tip=True``  -- S_++ and S_-- peak at *different* omega (0.177 vs
      0.154 eV at q = 0.23 nm^-1), i.e. two distinct collective modes.
    * ``chi_on_tip=False`` -- W_ST is still nonzero (V_ST /= 0 even with an inert
      tip), so the split is not trivial, but both projections peak at the *same*
      omega and their heights are in the static ratio
      (1 - V_ST/V_SS)/(1 + V_ST/V_SS) = 0.165.  That is one mode re-weighted by
      geometry, not a second mode.  Calling the minus channel "acoustic" there
      would be an artifact of the projection.

    So compare the peak positions of the two channels before reading the split
    as optical/acoustic.

    Returns ``(Sigma_total, Sigma_optical, Sigma_acoustic)``, each (2, 2, n_eps),
    with ``Sigma_optical + Sigma_acoustic == Sigma_total`` to round-off.
    """
    W_grid = np.asarray(W_grid, dtype=np.complex128)
    S, T = SAMPLE_LAYER, TIP_LAYER

    W_op = W_grid.copy()
    W_ac = W_grid.copy()
    # (1/2) S_++ = (1/2)(S_SS + S_ST)   and   (1/2) S_-- = (1/2)(S_SS - S_ST)
    W_op[:, :, S, S] = 0.5 * (W_grid[:, :, S, S] + W_grid[:, :, S, T])
    W_ac[:, :, S, S] = 0.5 * (W_grid[:, :, S, S] - W_grid[:, :, S, T])

    args = (cfg, k_vec, eps_values, q_grid, omega_grid)
    tail = (q_weights,)
    Sig_tot = sigma_full_frequency_fast(*args, W_grid, *tail, **kwargs)
    Sig_op = sigma_full_frequency_fast(*args, W_op, *tail, **kwargs)
    Sig_ac = sigma_full_frequency_fast(*args, W_ac, *tail, **kwargs)
    return Sig_tot, Sig_op, Sig_ac


# ---------------------------------------------------------------------------
# Fast route: tabulated frequency integrals (new in g0w)
# ---------------------------------------------------------------------------

def build_sample_kernel(cfg: SLGStackConfig, omega_grid: np.ndarray, W_grid: np.ndarray,
                        eps_values: np.ndarray, q_grid: np.ndarray, *, eta_self: float = 1e-3,
                        T: float = 0.0, sample_layer: int = SAMPLE_LAYER, du: Optional[float] = None,
                        k_max: Optional[float] = None):
    r"""
    :class:`~g0w.self_energy_full_frequency.ResolventKernel` of the sample-layer
    spectral density S_SS(q, w) = -Im W_SS/pi on the radial q grid.

    The tabulated u = xi - eps range covers |xi| <= hv (k_max + q_max) + E_F for
    external momenta |k| <= ``k_max`` (default 3 k_F).
    """
    from ..self_energy_full_frequency import ResolventKernel
    eps_values = np.asarray(eps_values, float)
    k_max = 3.0 * cfg.k_F if k_max is None else float(k_max)
    xi_max = cfg.hv * (k_max + float(np.max(q_grid))) + cfg.E_F
    S = -np.imag(np.asarray(W_grid)[:, :, sample_layer, sample_layer]) / np.pi
    return ResolventKernel(omega_grid, S, u_min=-xi_max - eps_values.max(),
                           u_max=xi_max - eps_values.min(), eta=eta_self, T=T, du=du)


def sigma_full_frequency_kernel(
    cfg: SLGStackConfig,
    k_vec: np.ndarray,
    eps_values: np.ndarray,
    q_grid: np.ndarray,
    kernel,
    q_weights: np.ndarray,
    *,
    band_ext: int = +1,
    n_theta: int = 64,
    sample_layer: int = SAMPLE_LAYER,
) -> np.ndarray:
    r"""
    Full-frequency Sigma_SS(k, eps) from a precomputed :func:`build_sample_kernel`.

    Identical physics to :func:`sigma_full_frequency_fast` (radial q measure,
    angular average over the direction of q, Dirac chirality form factor); the
    frequency integral is read from the kernel, so the cost per k is ~100x lower.
    Returns (2, 2, n_eps) with only [S, S] populated.
    """
    from ..self_energy_ppa import make_angular_quadrature
    k_vec = np.asarray(k_vec, float).ravel()[:2]
    eps_values = np.asarray(eps_values, float).ravel()
    q_grid = np.asarray(q_grid, float).ravel()
    q_weights = np.asarray(q_weights, float).ravel()
    hv, mu = float(cfg.hv), float(cfg.E_F)
    thetas, theta_w = make_angular_quadrature(int(n_theta))
    ct, st = np.cos(thetas), np.sin(thetas)
    nk = float(np.linalg.norm(k_vec))
    out = np.zeros(eps_values.size, dtype=np.complex128)
    for qi, q in enumerate(q_grid):
        w_q = float(q_weights[qi])
        if w_q == 0.0:
            continue
        px, py = k_vec[0] - q * ct, k_vec[1] - q * st
        pn = np.hypot(px, py)
        cos_ang = np.zeros_like(pn)
        good = (pn > 1e-14) & (nk > 1e-14)
        cos_ang[good] = np.clip((k_vec[0] * px[good] + k_vec[1] * py[good]) / (nk * pn[good]), -1, 1)
        xi = np.concatenate([s * hv * pn - mu for s in (+1, -1)])
        wF = np.concatenate([theta_w * 0.5 * (1.0 + band_ext * s * cos_ang) for s in (+1, -1)])
        keep = wF > 1e-15
        I = kernel.integral(qi, xi[keep], eps_values)                       # (n_b, n_e)
        out += -w_q * (wF[keep] @ I)
    Sigma = np.zeros((2, 2, eps_values.size), dtype=np.complex128)
    Sigma[sample_layer, sample_layer] = out
    return Sigma


def sigma_full_frequency_channels_kernel(
    cfg: SLGStackConfig,
    k_vec: np.ndarray,
    eps_values: np.ndarray,
    q_grid: np.ndarray,
    omega_grid: np.ndarray,
    W_grid: np.ndarray,
    q_weights: np.ndarray,
    *,
    band_ext: int = +1,
    eta_self: float = 1e-3,
    T: float = 0.0,
    n_theta: int = 64,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Kernel-route version of :func:`sigma_full_frequency_channels`:
    ``(Sigma_total, Sigma_optical, Sigma_acoustic)`` with
    S_SS = (1/2) S_++ + (1/2) S_-- split before the frequency integral.
    """
    W_grid = np.asarray(W_grid, dtype=np.complex128)
    S, Tl = SAMPLE_LAYER, TIP_LAYER
    W_op = W_grid.copy()
    W_ac = W_grid.copy()
    W_op[:, :, S, S] = 0.5 * (W_grid[:, :, S, S] + W_grid[:, :, S, Tl])
    W_ac[:, :, S, S] = 0.5 * (W_grid[:, :, S, S] - W_grid[:, :, S, Tl])
    out = []
    for Wx in (W_grid, W_op, W_ac):
        ker = build_sample_kernel(cfg, omega_grid, Wx, eps_values, q_grid, eta_self=eta_self, T=T)
        out.append(sigma_full_frequency_kernel(cfg, k_vec, eps_values, q_grid, ker, q_weights,
                                               band_ext=band_ext, n_theta=n_theta))
    return tuple(out)  # type: ignore[return-value]
