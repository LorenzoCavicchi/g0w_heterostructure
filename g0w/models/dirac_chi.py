#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Density-density response of doped monolayer graphene (T = 0, RPA bubble).

This is the Dirac counterpart of ``chi_2D`` (parabolic 2DEG) and follows the
same contract, so the two can be swapped inside the same stack builder.

Complex frequency
-----------------
``omega`` may be **real** or **complex**.  A real ``omega`` is interpreted as
the retarded boundary value ``omega + i0+``; a complex ``omega`` with positive
imaginary part is evaluated on the analytic continuation into the upper half
plane, so that

    chi0(q, omega + i*eta)

carries the *same* broadening eta as the Coulomb matrix V(q, omega + i*eta).
This matters for the plasmon-pole route: a graphene plasmon outside the
particle-hole continuum is undamped, so if V is broadened but chi0 is not, the
RPA pole stays a delta-function in -Im W and the residue integral
(1/pi) int dw [-Im W] is decided by whether a grid point happens to land on the
spike.  See ``GEOMETRY_AND_SELF_ENERGY.md`` Sec. 8.1.

Branch
------
Everything is built from

    G(z) = z*sqrt(z-1)*sqrt(z+1) - log(z + sqrt(z-1)*sqrt(z+1))
         = z*sqrt(z^2-1) - arccosh(z),

written with the *split* inner root sqrt(z-1)*sqrt(z+1) rather than
sqrt(z^2-1): the split form is analytic on C minus the segment [-1, 1], which
is the cut structure the retarded response actually has, whereas sqrt(z^2-1)
carries a spurious cut along the imaginary axis.

``massless_dirac_fermions.polarization`` selects between ``G(-D_minus)`` and
``G(D_minus) + i*pi`` with ``np.where(Re D_minus < -1, ...)``.  That switch is
unnecessary: the two expressions are the *same analytic function* on the whole
upper half plane (verified to 3e-14 over 4000 random UHP points, and provable
because they agree along the line Re z = -1).  Dropping it makes this module
manifestly analytic in the UHP and removes a real-axis criterion applied to a
complex argument.  ``chi_dirac_branch_switched_reference`` keeps the original
form for regression.

Validation
----------
``chi_dirac_phys`` is checked against a brute-force Lindhard k-sum
(``chi_dirac_lindhard_reference``) to better than 0.05% in
``test_slg_hbn_slg.py``.  The brute-force route integrates only the *doped
minus undoped* difference, which is UV-convergent; the undoped piece is added
back in closed form via ``chi_dirac_intrinsic``.

Units: omega [eV], q [nm^-1], n [nm^-2], hv = hbar*v_F [eV*nm].
Returns chi0 in [nm^-2 eV^-1], with chi0(q->0, 0) = -DOS(E_F) < 0, which is the
sign convention required by  epsilon = I - V chi0.
"""

from __future__ import annotations

import numpy as np

# Physical constants (shared with chi_2D / self_energy_plasmon_pole).
HBAR_EV_S = 6.582119569e-16       # eV*s
C_NM_PER_S = 2.99792458e17        # nm/s
HBARC_EV_NM = HBAR_EV_S * C_NM_PER_S

# hbar * v_F for graphene.  v_F = 1e6 m/s  ->  0.6582 eV*nm.  The value below is
# the usual continuum-model number (v_F = 1.0e6 m/s); pass ``hv`` explicitly to
# use a renormalised velocity.
GRAPHENE_HV_EV_NM = 0.658

# Spin x valley degeneracy of monolayer graphene.
GRAPHENE_DEGENERACY = 4.0


# ---------------------------------------------------------------------------
# Band-structure helpers
# ---------------------------------------------------------------------------

def fermi_momentum(n: float, g: float = GRAPHENE_DEGENERACY) -> float:
    r"""k_F = sqrt(4 pi n / g)  [nm^-1]  for carrier density n [nm^-2]."""
    return float(np.sqrt(4.0 * np.pi * float(n) / float(g)))


def fermi_energy(n: float, g: float = GRAPHENE_DEGENERACY,
                 hv: float = GRAPHENE_HV_EV_NM) -> float:
    r"""E_F = hbar v_F k_F  [eV]."""
    return float(hv) * fermi_momentum(n, g)


def density_from_fermi_energy(E_F: float, g: float = GRAPHENE_DEGENERACY,
                              hv: float = GRAPHENE_HV_EV_NM) -> float:
    r"""Inverse of :func:`fermi_energy`:  n = g E_F^2 / (4 pi (hbar v_F)^2)."""
    return float(g) * float(E_F) ** 2 / (4.0 * np.pi * float(hv) ** 2)


def dos_at_fermi(n: float, g: float = GRAPHENE_DEGENERACY,
                 hv: float = GRAPHENE_HV_EV_NM) -> float:
    r"""D(E_F) = g k_F / (2 pi hbar v_F)  [nm^-2 eV^-1]."""
    return float(g) * fermi_momentum(n, g) / (2.0 * np.pi * float(hv))


# ---------------------------------------------------------------------------
# The G function
# ---------------------------------------------------------------------------

def _G(z: np.ndarray) -> np.ndarray:
    r"""
    G(z) = z sqrt(z-1) sqrt(z+1) - log(z + sqrt(z-1) sqrt(z+1)).

    Analytic on C \ [-1, 1].  Satisfies G(-z) = G(z) + i*pi for Im z > 0.
    """
    z = np.asarray(z, dtype=np.complex128)
    r = np.sqrt(z - 1.0) * np.sqrt(z + 1.0)
    return z * r - np.log(z + r)


# ---------------------------------------------------------------------------
# Polarization
# ---------------------------------------------------------------------------

def _as_grid(omega, q):
    """Broadcast to (n_omega, 1) and (1, n_q); returns (om, qq, squeeze_flags)."""
    om = np.atleast_1d(np.asarray(omega, dtype=np.complex128))
    qq = np.atleast_1d(np.asarray(q, dtype=np.float64))
    return om[:, None], qq[None, :]


def chi_dirac_intrinsic(omega, q, g: float = GRAPHENE_DEGENERACY,
                        hv: float = GRAPHENE_HV_EV_NM) -> np.ndarray:
    r"""
    Undoped (E_F = 0) graphene polarization, exact:

        chi0(q, w) = -(g/16) q^2 / sqrt((hbar v_F q)^2 - w^2)

    with the principal square root, which is the retarded branch for Im w > 0.

    Returns shape (n_omega, n_q) [nm^-2 eV^-1].
    """
    om, qq = _as_grid(omega, q)
    return -(float(g) / 16.0) * qq ** 2 / np.sqrt(
        (float(hv) * qq) ** 2 - om ** 2 + 0.0j,
    )


def chi_dirac_phys(omega, q, n: float, g: float = GRAPHENE_DEGENERACY,
                   hv: float = GRAPHENE_HV_EV_NM) -> np.ndarray:
    r"""
    Retarded T=0 polarization of doped graphene  [nm^-2 eV^-1].

        chi0(q, w) = -(g / 16 pi) * [ 8 k_F / (hbar v_F)
                                      + q^2 (G(D_-) + i pi - G(D_+)) / s ]

        D_pm = (w / (hbar v_F) +/- 2 k_F) / q,
        s    = sqrt(w^2 - (hbar v_F q)^2)          (principal root)

    Parameters
    ----------
    omega : array_like, real or complex
        Frequencies [eV].  Real values are taken as the w + i0+ boundary value.
        For residue extraction pass ``omega + 1j*eta`` (see module docstring).
    q : array_like
        Momenta [nm^-1], measured from the Dirac point.  q = 0 returns 0.
    n : float
        Carrier density [nm^-2] (electron doping; the T=0 result is even in the
        doping sign).
    g : float
        Spin x valley degeneracy (4 for monolayer graphene).
    hv : float
        hbar * v_F [eV*nm].

    Returns
    -------
    chi0 : ndarray, shape (n_omega, n_q), complex
    """
    om, qq = _as_grid(omega, q)
    hv = float(hv)
    k_F = fermi_momentum(n, g)

    safe_q = np.where(qq == 0.0, 1.0, qq)
    Dp = (om / hv + 2.0 * k_F) / safe_q
    Dm = (om / hv - 2.0 * k_F) / safe_q
    s = np.sqrt(om ** 2 - (hv * safe_q) ** 2 + 0.0j)

    intra = 8.0 * k_F / (hv * safe_q ** 2)
    inter = (_G(Dm) + 1j * np.pi - _G(Dp)) / s

    out = -(float(g) / (16.0 * np.pi)) * (intra + inter) * safe_q ** 2
    return np.where(qq == 0.0, 0.0, out)


def chi_dirac_branch_switched_reference(omega, q, n: float,
                                        g: float = GRAPHENE_DEGENERACY,
                                        hv: float = GRAPHENE_HV_EV_NM) -> np.ndarray:
    r"""
    ``massless_dirac_fermions.polarization`` (T=0) transcribed verbatim, with its
    ``np.where(Re D_- < -1, G(-D_-), G(D_-) + i pi)`` branch switch.

    Regression reference only -- it must agree with :func:`chi_dirac_phys`
    everywhere in the upper half plane.  Kept so that a future change to the
    branch handling is caught against the original implementation, and so this
    module can be compared to the existing repo code without needing ``mpmath``
    (which ``massless_dirac_fermions`` imports at module scope for the
    finite-temperature Maldague average only).
    """
    om, qq = _as_grid(omega, q)
    hv = float(hv)
    k_F = fermi_momentum(n, g)
    safe_q = np.where(qq == 0.0, 1.0, qq)
    Dp = (om / hv + 2.0 * k_F) / safe_q
    Dm = (om / hv - 2.0 * k_F) / safe_q
    intra = 8.0 * k_F / (hv * safe_q ** 2)
    inter = (
        np.where(np.real(Dm) < -1.0, _G(-Dm), _G(Dm) + 1j * np.pi) - _G(Dp)
    ) / np.sqrt(om ** 2 - hv ** 2 * safe_q ** 2 + 0.0j)
    out = -(float(g) / (16.0 * np.pi)) * (intra + inter) * safe_q ** 2
    return np.where(qq == 0.0, 0.0, out)


# ---------------------------------------------------------------------------
# Brute-force Lindhard reference (test-only)
# ---------------------------------------------------------------------------

def chi_dirac_lindhard_reference(
    omega: float,
    q,
    n: float,
    g: float = GRAPHENE_DEGENERACY,
    hv: float = GRAPHENE_HV_EV_NM,
    *,
    eta: float = 1e-3,
    n_k: int = 900,
    n_theta: int = 900,
) -> np.ndarray:
    r"""
    Brute-force Lindhard sum for chi0, as an independent check on the closed form.

    **Test-only** -- slow, and never on the production path.

    The bare interband sum is UV divergent, so this routine integrates only the
    *doped-minus-undoped* difference,

        delta-chi = chi0(mu) - chi0(mu=0),

    whose occupation-factor numerator vanishes unless |k| < k_F or |k+q| < k_F,
    and adds the undoped piece back from :func:`chi_dirac_intrinsic`.  The polar
    grid therefore runs out to k_F + q_max; truncating it at k_F instead silently
    drops the interband term and corrupts Re chi0 by tens of percent while
    leaving Im chi0 apparently fine.

        chi0 = g int d2k/(2pi)^2 sum_{l,l'} F_{l l'}(k, k+q)
                 [f(E_l(k)) - f(E_l'(k+q))] / (w + i eta + E_l(k) - E_l'(k+q))

        F_{l l'} = (1 + l l' cos(angle between k and k+q)) / 2      (chirality)
        E_l(k)   = l * hbar v_F |k|

    Returns shape (n_q,) complex.
    """
    q_arr = np.atleast_1d(np.asarray(q, dtype=float))
    hv = float(hv)
    k_F = fermi_momentum(n, g)
    mu = hv * k_F

    k_max = k_F + float(np.max(q_arr)) + 1e-9
    kr = (np.arange(n_k) + 0.5) * k_max / n_k
    th = (np.arange(n_theta) + 0.5) * 2.0 * np.pi / n_theta
    dk = k_max / n_k
    dth = 2.0 * np.pi / n_theta
    KR, TH = np.meshgrid(kr, th, indexing="ij")
    kx, ky = KR * np.cos(TH), KR * np.sin(TH)
    w = complex(omega) + 1j * float(eta)

    out = np.empty(q_arr.size, dtype=np.complex128)
    for iq, qv in enumerate(q_arr):
        px, py = kx + float(qv), ky
        P = np.hypot(px, py)
        cos_ang = np.clip(
            np.divide(kx * px + ky * py, np.maximum(KR * P, 1e-300)), -1.0, 1.0,
        )
        acc = 0.0 + 0.0j
        for lam in (+1, -1):
            for lam_p in (+1, -1):
                E_k = lam * hv * KR
                E_p = lam_p * hv * P
                num = ((E_k < mu).astype(float) - (E_p < mu).astype(float)) \
                    - ((E_k < 0.0).astype(float) - (E_p < 0.0).astype(float))
                if not np.any(num):
                    continue
                F = 0.5 * (1.0 + lam * lam_p * cos_ang)
                acc += np.sum(F * num / (w + E_k - E_p) * KR) * dk * dth
        out[iq] = float(g) * acc / (2.0 * np.pi) ** 2

    return out + chi_dirac_intrinsic(np.array([w]), q_arr, g, hv)[0]


# ---------------------------------------------------------------------------
# Particle-hole continuum boundaries (for plotting / pole sanity checks)
# ---------------------------------------------------------------------------

def ph_continuum_bounds(q, n: float, g: float = GRAPHENE_DEGENERACY,
                        hv: float = GRAPHENE_HV_EV_NM):
    r"""
    Boundaries of the T=0 particle-hole continuum of doped graphene [eV].

    Returns ``(w_intra_upper, w_inter_lower)``:

    * ``w_intra_upper = hbar v_F q``  -- above the *intraband* continuum, a
      plasmon here is Landau-undamped by intraband transitions.
    * ``w_inter_lower = |hbar v_F q - 2 E_F|`` for q < 2k_F, else
      ``hbar v_F q - 2 E_F`` -- below the *interband* continuum.

    A plasmon branch is undamped where ``w_intra_upper < w < w_inter_lower``.
    """
    q_arr = np.atleast_1d(np.asarray(q, dtype=float))
    hv = float(hv)
    E_F = fermi_energy(n, g, hv)
    return hv * q_arr, np.abs(hv * q_arr - 2.0 * E_F)
