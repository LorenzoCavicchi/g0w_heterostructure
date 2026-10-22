# -*- coding: utf-8 -*-
r"""
Interpretation of plasmon satellites (notes, Sec. "Interpretation of the results").

The satellite of the G0W spectral function is the one-plasmon emission edge of
-Im Sigma.  Its energy is a stationary value of the final-state energy

    E_k(q) = xi_{k-q} + omega_pl(q)

over the momentum transfer q, not omega_pl at any single q.  This module
provides

* :func:`plasmon_branch`      omega_pl(q) as the maximum of the spectral density S(q, w)
                              above the intraband edge hbar v q;
* :func:`dispersive_edge`     Delta = max_q [omega_pl(q) - hbar v q]: satellite at
                              eps ~ xi_k + Delta for a dispersive plasmon (graphene);
* :func:`sqrt_plasmon_edge`   the closed form for omega_pl = sqrt(g e^2 E_F q / 2 eps):
                              Delta = (g/8) alpha_g E_F, q* = (g/8) alpha_g k_F;
* :func:`slg_energy_loss`,
  :func:`tbg_energy_loss`     decomposition of -Im Sigma(k, eps) (eta -> 0, T = 0) into
                              contributions (energy lost w, final energy xi', q, weight),
                              i.e. which plasmons and which final states build the satellite;
* :func:`weighted_summary`    mean, median and inter-quartile range of a weighted sample.

Units: eV, nm^-1.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Plasmon branch and kinematic estimates
# ---------------------------------------------------------------------------

def plasmon_branch(q: np.ndarray, omega: np.ndarray, S: np.ndarray, hv: float,
                   edge_factor: float = 1.02) -> np.ndarray:
    r"""
    omega_pl(q): frequency of the maximum of ``S[q, w]`` above ``edge_factor * hv * q``.

    ``S`` is (n_q, n_w), the (sheet-resolved or traced) spectral density of W.
    Restricting to w > hv q excludes the intraband particle-hole continuum, so the
    maximum is the plasmon wherever it is a well-defined peak.
    """
    q = np.asarray(q, float); omega = np.asarray(omega, float); S = np.asarray(S, float)
    out = np.zeros(q.size)
    for i, qq in enumerate(q):
        row = np.where(omega > edge_factor * hv * qq, S[i], -np.inf)
        out[i] = omega[int(np.argmax(row))]
    return out


def dispersive_edge(q: np.ndarray, omega_pl: np.ndarray, hv: float, *,
                    k: Optional[float] = None, k_F: Optional[float] = None) -> Dict[str, float]:
    r"""
    Delta = max_q [omega_pl(q) - hv q] and the q* where it is reached.

    For an electron of momentum k on a cone of velocity v, emission collinear with
    the velocity gives the lowest final energy at fixed q,
    E(q) = xi_k - hv q + omega_pl(q).  Its maximum over q is a caustic where a range
    of q contributes at the same energy: the edge (peak) of -Im Sigma sits at
    eps_sat ~ xi_k + Delta.  It is reached where d omega_pl / dq = v (plasmon group
    velocity = electron velocity).  If ``k`` and ``k_F`` are given, q is restricted
    to q <= k - k_F (final state above E_F); the edge is then cut by Pauli blocking
    when q* > k - k_F.

    Returns ``Delta``, ``q_star``, ``omega_star`` = omega_pl(q*), ``slope`` =
    (d omega_pl/dq)/hv at q*, ``pauli_limited`` (bool).
    """
    q = np.asarray(q, float); w = np.asarray(omega_pl, float)
    sel = np.ones(q.size, bool)
    if k is not None and k_F is not None:
        sel = q <= float(k) - float(k_F)
    if not np.any(sel):
        return dict(Delta=np.nan, q_star=np.nan, omega_star=np.nan, slope=np.nan, pauli_limited=True)
    D = w[sel] - hv * q[sel]
    j = int(np.argmax(D))
    # slope of omega_pl at q* from a local quadratic fit (omega_pl taken from a frequency grid is a staircase)
    jj = int(np.flatnonzero(sel)[j])
    dq = float(np.median(np.diff(q)))
    win = np.abs(q - q[jj]) <= max(0.15 * q[jj], 4 * dq)
    cf = np.polyfit(q[win] - q[jj], w[win], 2)
    slope = cf[1] / hv
    jall = int(np.argmax(w - hv * q))
    return dict(Delta=float(D[j]), q_star=float(q[sel][j]), omega_star=float(w[sel][j]),
                slope=float(slope), pauli_limited=bool(np.flatnonzero(sel)[j] != jall))


def sqrt_plasmon_edge(E_F: float, k_F: float, alpha_g: float, g: float = 4.0) -> Dict[str, float]:
    r"""
    Closed form for the long-wavelength 2D plasmon (hbar omega)^2 = (g/2)(e^2/eps) E_F q:

        maximise a sqrt(q) - hv q   ->   q* = a^2 / (4 hv^2) = (g/8) alpha_g k_F,
                                         Delta = a^2 / (4 hv) = (g/8) alpha_g E_F,

    with alpha_g = e^2 / (eps hbar v_F).  For g = 4: Delta = alpha_g E_F / 2.
    """
    return dict(Delta=g / 8.0 * alpha_g * E_F, q_star=g / 8.0 * alpha_g * k_F)


# ---------------------------------------------------------------------------
# Energy-loss decomposition of -Im Sigma
# ---------------------------------------------------------------------------

def weighted_summary(x: np.ndarray, w: np.ndarray) -> Dict[str, float]:
    r"""Weighted mean, median and quartiles of ``x`` with non-negative weights ``w``."""
    x = np.asarray(x, float).ravel(); w = np.asarray(w, float).ravel()
    tot = float(w.sum())
    if tot <= 0:
        return dict(mean=np.nan, median=np.nan, q25=np.nan, q75=np.nan, total=0.0)
    o = np.argsort(x); cw = np.cumsum(w[o]) / tot
    q25, med, q75 = x[o][np.minimum(np.searchsorted(cw, [0.25, 0.5, 0.75]), x.size - 1)]
    return dict(mean=float(np.sum(x * w) / tot), median=float(med), q25=float(q25), q75=float(q75), total=tot)


def slg_energy_loss(cfg, k_vec: np.ndarray, eps: float, q: np.ndarray, omega: np.ndarray,
                    S: np.ndarray, q_weights: np.ndarray, *, band_ext: int = +1,
                    n_phi: int = 512) -> Dict[str, np.ndarray]:
    r"""
    Contributions to -Im Sigma_s(k, eps) of the graphene sample at T = 0, eta -> 0.

    From the anti-Hermitian part of the full-frequency self-energy (notes, causality
    section), with the radial q measure and the angular average:

        -Im Sigma = pi sum_q w_q < sum_s' F^{s' s}(k, k-q) S(q, w) > ,
        w = eps - xi'  (eps > 0, xi' > 0)   or   w = xi' - eps  (eps < 0, xi' < 0),

    where xi' = s' hv |k - q| - E_F and S = -Im W_SS / pi on the (q, omega) grid.
    Returns ``omega_lost``, ``xi_final``, ``q``, ``weight`` (eV; weights sum to -Im Sigma).
    """
    k = np.asarray(k_vec, float).ravel()[:2]
    nk = float(np.linalg.norm(k))
    phis = np.linspace(0.0, 2 * np.pi, int(n_phi), endpoint=False)
    out = {"omega_lost": [], "xi_final": [], "q": [], "weight": []}
    for i, qq in enumerate(np.asarray(q, float)):
        if q_weights[i] == 0:
            continue
        px, py = k[0] - qq * np.cos(phis), k[1] - qq * np.sin(phis)
        pn = np.hypot(px, py)
        cosang = np.where((pn > 1e-14) & (nk > 1e-14), (k[0] * px + k[1] * py) / np.maximum(nk * pn, 1e-300), 1.0)
        for s in (+1, -1):
            xf = s * cfg.hv * pn - cfg.E_F
            F = 0.5 * (1.0 + band_ext * s * cosang)
            if eps > 0:
                w = eps - xf; ok = xf > 0
            else:
                w = xf - eps; ok = xf < 0
            ok &= (w > omega[0]) & (w < omega[-1])
            if not np.any(ok):
                continue
            Sv = np.interp(w[ok], omega, S[i])
            out["omega_lost"].append(w[ok]); out["xi_final"].append(xf[ok])
            out["q"].append(np.full(int(ok.sum()), qq))
            out["weight"].append(np.pi * q_weights[i] / phis.size * F[ok] * Sv)
    return {key: (np.concatenate(v) if v else np.zeros(0)) for key, v in out.items()}


def tbg_energy_loss(tbg: Dict[str, Any], lh, S: np.ndarray, omega: np.ndarray,
                    q_weights: np.ndarray, k_index: int, band: int, eps: float) -> Dict[str, np.ndarray]:
    r"""
    Contributions to -Im Sigma_nn(k, eps) of TBG (band ``band`` at mesh point ``k_index``),
    T = 0, eta -> 0:

        -Im Sigma_nn = pi sum_q w_q sum_m [rho_m S(q, w) rho_m^dagger]_nn ,
        w = eps - xi_m(k+q) for eps > 0 (empty m),  w = xi_m(k+q) - eps for eps < 0 (filled m),

    with rho_m^a = densityL[a, n, m, k, q, 0] and S the 2x2 sample spectral density.
    Returns ``omega_lost``, ``xi_final``, ``final_band``, ``q_index``, ``weight``.
    """
    bs = tbg["crystal"].band_structure
    E = np.asarray(bs.energies) - float(bs.chemical_potential)
    kpq = np.asarray(bs.k_plus_q_ind)
    dL = np.asarray(lh.densityL)
    S = np.asarray(S)
    rows = []
    for qi in range(S.shape[0]):
        if q_weights[qi] == 0:
            continue
        kq = int(kpq[qi, int(k_index)])
        for m in range(E.shape[0]):
            xm = float(E[m, kq])
            if eps > 0 and xm <= 0 or eps < 0 and xm >= 0:
                continue
            w = eps - xm if eps > 0 else xm - eps
            if not (omega[0] < w < omega[-1]):
                continue
            r = dL[:, int(band), m, int(k_index), qi, 0]
            Sab = np.empty((2, 2), complex)
            for a in range(2):
                for b in range(2):
                    Sab[a, b] = np.interp(w, omega, S[qi, :, a, b].real) + 1j * np.interp(w, omega, S[qi, :, a, b].imag)
            rows.append((w, xm, m, qi, np.pi * q_weights[qi] * float(np.real(r @ Sab @ np.conj(r)))))
    a = np.array(rows) if rows else np.zeros((0, 5))
    return dict(omega_lost=a[:, 0], xi_final=a[:, 1], final_band=a[:, 2].astype(int),
                q_index=a[:, 3].astype(int), weight=a[:, 4])


def minus_im_sigma_from_loss(decomp: Dict[str, np.ndarray]) -> float:
    r"""-Im Sigma (eta -> 0) recovered from a decomposition: the sum of its weights."""
    return float(np.sum(decomp["weight"]))
