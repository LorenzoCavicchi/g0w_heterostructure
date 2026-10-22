#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
SLG | hBN | SLG stack: geometry, RPA dielectric matrix, and plasmon poles.

Geometry
--------
Two graphene sheets separated by an hBN spacer of thickness ``d_hbn``, embedded
in a uniaxial hBN environment (optionally gated)::

    ------------------------------  (optional top gate, top_gate_d)
      region 0   hBN
    ============================== sheet 0
      region 1   hBN   d_hbn
    ============================== sheet 1
      region 2   hBN
    ------------------------------  (optional bottom gate)

``SAMPLE_LAYER`` / ``TIP_LAYER`` name the two sheets.  With a symmetric
environment (the default: the same permittivity in all three regions and equal
gate distances) the Coulomb matrix is invariant under swapping the two sheets,
so the labelling is a bookkeeping choice, not a physical one.

The tip's polarizability
------------------------
The two cases the benchmark compares are set by ``chi_on_tip``:

* ``False`` -- only the sample responds.  ``chi0 = diag(chi_S, 0)``.  A sheet
  with no charge response does not screen, so this reproduces the *isolated*
  sample plasmon exactly, while still being evaluated in the 2x2 layer basis.
  That makes it a genuine internal consistency check of the matrix machinery
  rather than a separate code path.
* ``True`` -- both sheets respond.  ``chi0 = diag(chi_S, chi_T)``.  With both
  doped this splits the single sqrt(q) mode into an optical (in-phase, sqrt(q))
  and an acoustic (out-of-phase, linear-in-q) branch; the acoustic residue has
  ``R_01 < 0``.  With the tip near charge neutrality ``chi_T`` is the interband
  response only -- no Drude weight, hence no second branch, but still extra
  screening of the sample mode.  That is the tip-backaction question raised in
  Wei et al. Sec. VI, and it is why this is worth running both ways.

Conventions
-----------
``epsilon = I - V chi0`` and ``W = eps^-1 V`` (matching
``SCREENED_INTERACTION_CONVENTION`` in ``self_energy_plasmon_pole``).  Both V
and chi0 are evaluated at ``omega + i*eta`` so the residue integral
``(1/pi) int dw [-Im W]`` is resolvable -- see ``chi_dirac`` module docstring.

Units: energies eV, momenta nm^-1, lengths nm, densities nm^-2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from . import dirac_chi as cd
from ..layer_coulomb import LayerCoulomb
from ..self_energy_ppa import (
    PlasmonPoleData,
    build_plasmon_pole_data,
    make_radial_q_weights,
)

# Static head permittivities of hBN (the values used in GEOMETRY_AND_SELF_ENERGY.md).
HBN_EPS_PAR_STATIC = 6.41
HBN_EPS_Z_STATIC = 3.0

# e^2 in eV*nm (cgs-ESU, matching layer_coulomb_v2_7).
E2_EV_NM = 1.4399764

SAMPLE_LAYER = 0
TIP_LAYER = 1


@dataclass
class SLGStackConfig:
    r"""
    Parameters of the SLG|hBN|SLG junction.

    Attributes
    ----------
    n_sample, n_tip : float
        Carrier densities [nm^-2].  1e12 cm^-2 = 0.01 nm^-2.
    chi_on_tip : bool
        Whether the tip sheet carries a polarization (see module docstring).
    d_hbn : float
        hBN spacer thickness [nm].  0.334 nm per layer; 1.0 nm ~ 3 layers.
    eps_par, eps_z : float
        Uniaxial hBN permittivity, applied to every region.
    top_gate_d, bottom_gate_d : float or None
        Grounded-gate distances [nm]; ``None`` = open half-space.  Ungated is
        the default: V(q->0) ~ 1/q diverges but the 2D measure q dq cancels it,
        so the self-energy integrand still vanishes at small q.
    hv : float
        hbar v_F [eV nm].
    g : float
        Spin x valley degeneracy per sheet.
    eta : float
        Broadening [eV] carried by *both* V and chi0.
    """
    n_sample: float = 0.01
    n_tip: float = 0.01
    chi_on_tip: bool = True
    d_hbn: float = 1.0
    eps_par: float = HBN_EPS_PAR_STATIC
    eps_z: float = HBN_EPS_Z_STATIC
    top_gate_d: Optional[float] = None
    bottom_gate_d: Optional[float] = None
    hv: float = cd.GRAPHENE_HV_EV_NM
    g: float = cd.GRAPHENE_DEGENERACY
    eta: float = 1e-3

    # ---- derived quantities -------------------------------------------------
    @property
    def k_F(self) -> float:
        return cd.fermi_momentum(self.n_sample, self.g)

    @property
    def E_F(self) -> float:
        return cd.fermi_energy(self.n_sample, self.g, self.hv)

    @property
    def eps_bar(self) -> float:
        r"""Effective isotropic background constant sqrt(eps_par * eps_z)."""
        return float(np.sqrt(self.eps_par * self.eps_z))

    def label(self) -> str:
        tip = (f"tip n={self.n_tip:g}" if self.chi_on_tip else "tip inert")
        return (f"n_S={self.n_sample:g} nm^-2, {tip}, d={self.d_hbn:g} nm, "
                f"eps=({self.eps_par:g},{self.eps_z:g})")


# ---------------------------------------------------------------------------
# Electrostatics
# ---------------------------------------------------------------------------

def stack_coulomb(cfg: SLGStackConfig) -> LayerCoulomb:
    """Two-sheet ``LayerCoulomb`` for the configuration (3 dielectric regions)."""
    return LayerCoulomb(
        n_layers=2,
        d=[float(cfg.d_hbn)],
        epsilon_uniaxial=([float(cfg.eps_par)] * 3, [float(cfg.eps_z)] * 3),
        top_gate_d=cfg.top_gate_d,
        bottom_gate_d=cfg.bottom_gate_d,
    )


def chi0_matrix(omega, q: float, cfg: SLGStackConfig) -> np.ndarray:
    r"""
    Layer-diagonal chi0(q, omega) for the stack, shape (n_omega, 2, 2).

    The two sheets are decoupled (no interlayer tunneling), so chi0 is diagonal
    in the layer index; all interlayer physics enters through V.
    """
    omega = np.atleast_1d(np.asarray(omega, dtype=np.complex128))
    chi_S = cd.chi_dirac_phys(omega, np.array([float(q)]),
                              cfg.n_sample, cfg.g, cfg.hv)[:, 0]
    if cfg.chi_on_tip:
        chi_T = cd.chi_dirac_phys(omega, np.array([float(q)]),
                                  cfg.n_tip, cfg.g, cfg.hv)[:, 0]
    else:
        chi_T = np.zeros_like(chi_S)
    out = np.zeros((omega.size, 2, 2), dtype=np.complex128)
    out[:, SAMPLE_LAYER, SAMPLE_LAYER] = chi_S
    out[:, TIP_LAYER, TIP_LAYER] = chi_T
    return out


def stack_eps_and_V(
    q: float,
    omega: np.ndarray,
    cfg: SLGStackConfig,
    lc: Optional[LayerCoulomb] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""
    eps(q, w+i.eta) and V(q, w+i.eta) at one q for an array of w.

    Returns
    -------
    eps_q : (n_omega, 2, 2, 1, 1)     -- layout expected by build_plasmon_pole_data
    coulomb_q : (2, 2, 1, n_omega)    -- dispersive Coulomb layout
    """
    lc = lc if lc is not None else stack_coulomb(cfg)
    omega = np.asarray(omega, dtype=float).ravel()
    n_om = omega.size
    eta = float(cfg.eta)

    chi = chi0_matrix(omega + 1j * eta, q, cfg)          # (n_om, 2, 2)
    eps_q = np.zeros((n_om, 2, 2, 1, 1), dtype=np.complex128)
    cq = np.zeros((2, 2, 1, n_om), dtype=np.complex128)
    I2 = np.eye(2, dtype=np.complex128)
    for oi, om in enumerate(omega):
        V = np.asarray(lc.coulomb_matrix(float(q), om + 1j * eta),
                       dtype=np.complex128)
        eps_q[oi, :, :, 0, 0] = I2 - V @ chi[oi]
        cq[:, :, 0, oi] = V
    return eps_q, cq


# ---------------------------------------------------------------------------
# Plasmon poles
# ---------------------------------------------------------------------------

def build_slg_ppd(
    cfg: SLGStackConfig,
    *,
    q_min: Optional[float] = None,
    q_max: Optional[float] = None,
    n_q: int = 40,
    omega_max: Optional[float] = None,
    n_omega: int = 400,
    n_refine: int = 129,
    refine_residues: bool = True,
    collect_diagnostics: bool = False,
    min_pole_prominence: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray, PlasmonPoleData]:
    r"""
    Build eps(q,w), V(q,w) and the ``PlasmonPoleData`` for the stack.

    Defaults: ``q`` from 0.02 k_F to 3 k_F, ``omega`` up to 1.4 * the interband
    threshold 2 E_F, which comfortably contains the sqrt(q) optical branch over
    the whole q range while staying below the hBN phonons (~94 meV for eps_z)
    for typical dopings.  ``sigma_q_weight_profile`` in the validation module
    checks that the large-q end is not truncating real spectral weight.

    Returns (q_grid, omega_grid, ppd).
    """
    k_F = cfg.k_F
    q_min = 0.02 * k_F if q_min is None else float(q_min)
    q_max = 3.0 * k_F if q_max is None else float(q_max)
    omega_max = 1.4 * 2.0 * cfg.E_F if omega_max is None else float(omega_max)

    q_grid = np.linspace(q_min, q_max, int(n_q))
    q_weights = make_radial_q_weights(q_grid)
    omega_grid = np.linspace(1e-4, omega_max, int(n_omega))

    lc = stack_coulomb(cfg)
    N, n_G = 2, 1
    epsilon_layer = np.zeros((n_q, n_omega, N, N, n_G, n_G), dtype=np.complex128)
    coulomb_matrix = np.zeros((n_q, N, N, n_G, n_omega), dtype=np.complex128)
    for qi, q in enumerate(q_grid):
        e_q, c_q = stack_eps_and_V(float(q), omega_grid, cfg, lc)
        epsilon_layer[qi] = e_q
        coulomb_matrix[qi] = c_q

    def eps_builder(q_index: int, om: np.ndarray):
        return stack_eps_and_V(float(q_grid[q_index]), om, cfg, lc)

    ppd = build_plasmon_pole_data(
        layer_helper=None,
        epsilon_layer=epsilon_layer,
        coulomb_matrix=coulomb_matrix,
        omega_grid=omega_grid,
        q_vectors=q_grid.reshape(-1, 1),
        q_weights=q_weights,
        eta=cfg.eta,
        eps_builder=eps_builder if refine_residues else None,
        n_refine=int(n_refine),
        min_pole_prominence=float(min_pole_prominence),
        collect_diagnostics=collect_diagnostics,
    )
    return q_grid, omega_grid, ppd


# ---------------------------------------------------------------------------
# Analytic references
# ---------------------------------------------------------------------------

def plasmon_optical_long_wavelength(q, cfg: SLGStackConfig) -> np.ndarray:
    r"""
    **Single-sheet** graphene plasmon in the q -> 0 limit:

        w_pl(q) = sqrt( g e^2 E_F q / (2 eps_bar) )

    which for g = 4 is the familiar sqrt(2 e^2 E_F q / eps_bar).  Valid while
    w_pl >> hbar v_F q, i.e. q << (g e^2 k_F)/(2 eps_bar hbar v_F).

    This is the right reference for ``chi_on_tip=False`` only.  Two *doped*
    sheets at q d << 1 share the same restoring field, so the in-phase mode
    carries twice the Drude weight and sits a factor sqrt(2) higher; compare
    against this curve only in the ``sample_only`` case.
    """
    q = np.atleast_1d(np.asarray(q, dtype=float))
    return np.sqrt(cfg.g * E2_EV_NM * cfg.E_F * q / (2.0 * cfg.eps_bar))


def acoustic_velocity_bound(cfg: SLGStackConfig) -> float:
    r"""
    Rigorous lower bound on the acoustic sound velocity: ``v_ac > v_F``.

    The out-of-phase mode only exists as a pole where it is outside the
    intraband particle-hole continuum, i.e. above ``w = hbar v_F q``.  That is
    the only statement about ``v_ac`` this module makes analytically.

    A closed-form long-wavelength expression was tried here and removed: the
    usual two-sheet expansion  v_ac/v_F = sqrt(1 + 2x/(1+x)),
    x = alpha_g g k_F d / 2, gives 1.140 at the default parameters, while the
    computed branch gives 1.028 -- an 11% overshoot.  The expansion assumes
    q d << 1 *and* a mode well clear of the continuum edge, and here the
    acoustic branch rides only 3-7% above it over the whole q range, so neither
    holds.  Use :func:`acoustic_velocity_from_poles` on the computed poles
    instead; that is what the benchmark reports.

    Returns hbar * v_F [eV nm], i.e. the bound on the slope of w_ac(q).
    """
    return float(cfg.hv)


def acoustic_velocity_from_poles(
    ppd: PlasmonPoleData,
    q_grid: np.ndarray,
    cfg: SLGStackConfig,
    *,
    q_max_fit: Optional[float] = None,
) -> Tuple[float, np.ndarray, np.ndarray]:
    r"""
    Sound velocity of the acoustic branch, fitted from the computed poles.

    Uses only poles labelled ``acoustic`` *and* flagged ``reliable`` by
    :func:`classify_branches`, fits w_ac = v_ac q through the origin, and
    returns ``(v_ac [eV nm], q_used, w_used)``.

    The dispersion is not exactly linear -- w/(hbar v_F q) drifts from ~1.07 at
    q = 0.25 k_F to ~1.04 at k_F as the mode is pulled toward the continuum
    edge -- so the fitted slope is a summary, and ``q_used``/``w_used`` are
    returned so the drift can be inspected rather than hidden.
    """
    q_max_fit = float(q_max_fit) if q_max_fit is not None else float(np.max(q_grid))
    branches = classify_branches(ppd, q_grid, cfg)
    xs, ys = [], []
    for qi, rec in enumerate(branches):
        if q_grid[qi] > q_max_fit:
            continue
        for j in range(rec["omega"].size):
            if rec["label"][j] == "acoustic" and rec["reliable"][j]:
                xs.append(float(q_grid[qi])); ys.append(float(rec["omega"][j]))
    if not xs:
        return float("nan"), np.array([]), np.array([])
    x = np.asarray(xs); y = np.asarray(ys)
    v_ac = float(np.dot(x, y) / np.dot(x, x))     # least squares through origin
    return v_ac, x, y


def loss_function_head(eps_q: np.ndarray, coulomb_q: np.ndarray,
                       layer: int = SAMPLE_LAYER) -> np.ndarray:
    r"""
    L(q, w) = -Im [ (eps^-1 V)_{ll} ]  for one layer -- the quantity whose peaks
    the pole finder locates.  ``eps_q`` (n_w,2,2,1,1), ``coulomb_q`` (2,2,1,n_w).
    """
    n_om = eps_q.shape[0]
    out = np.zeros(n_om, dtype=float)
    for oi in range(n_om):
        E = eps_q[oi, :, :, 0, 0]
        V = coulomb_q[:, :, 0, oi]
        W = np.linalg.inv(E) @ V
        out[oi] = -float(np.imag(W[layer, layer]))
    return out


# ---------------------------------------------------------------------------
# Branch classification and PPA reliability
# ---------------------------------------------------------------------------

def classify_branches(
    ppd: PlasmonPoleData,
    q_grid: np.ndarray,
    cfg: SLGStackConfig,
    *,
    edge_margin_eta: float = 2.0,
    weight_window_factor: float = 5.0,
):
    r"""
    Label each pole ``acoustic`` / ``optical`` by the sign of R_{ST}, and flag
    those too close to the intraband particle-hole edge to be trusted.

    Why the flag is needed
    ----------------------
    The residue window is +/- ``weight_window_factor * eta`` wide.  When a mode
    sits closer than that to the intraband edge ``w = hbar v_F q``, the window
    straddles the continuum and the integral picks up single-particle weight of
    the *opposite* layer symmetry; the measured R_{ST} shrinks and can change
    sign.

    The test has to be an *absolute* distance in units of eta, not a ratio:
    the acoustic mode of two ungated Dirac sheets rides only 4-8% above the edge
    at every q (v_ac ~ 1.04-1.08 v_F), so any ratio threshold either accepts
    everything or rejects everything.  At n = 1e12 cm^-2, d = 1 nm, eta = 1 meV
    the edge distance grows from 1.7 meV at q = 0.17 k_F to 4.4 meV at k_F, and
    it is exactly the first point -- below 2 eta -- whose R_{ST} comes out
    positive.  Those points are not a broken residue extraction; they are a
    regime where the acoustic mode is not a well-defined pole, and they must be
    excluded rather than fitted.

    Returns
    -------
    list of per-q dicts with keys ``omega``, ``R_ST``, ``label``, ``edge_ratio``
    (= w / hbar v_F q), ``edge_distance_eta`` (= (w - hbar v_F q)/eta) and
    ``reliable`` (edge_distance_eta >= edge_margin_eta, and the mode is above
    the edge at all).
    """
    out = []
    hv = float(cfg.hv)
    for qi in range(len(q_grid)):
        w = np.asarray(ppd.pole_frequencies[qi], dtype=float)
        R = ppd.residues[qi]
        edge = hv * float(q_grid[qi])
        rec = {"omega": w,
               "R_ST": np.array([R[j][SAMPLE_LAYER, TIP_LAYER].real
                                 for j in range(w.size)]),
               "edge_ratio": (w / edge if edge > 0 else np.full(w.size, np.inf))}
        rec["label"] = np.array(
            ["acoustic" if r < 0 else "optical" for r in rec["R_ST"]], dtype=object,
        )
        rec["edge_distance_eta"] = (w - edge) / float(cfg.eta)
        rec["window_half_width"] = float(weight_window_factor) * float(cfg.eta)
        rec["reliable"] = rec["edge_distance_eta"] >= float(edge_margin_eta)
        out.append(rec)
    return out


def undamped_window_closes_at(cfg: SLGStackConfig) -> float:
    r"""
    q at which graphene's undamped window ``hbar v_F q < w < 2 E_F - hbar v_F q``
    closes, i.e. q = E_F / (hbar v_F) = k_F.

    Above this the plasmon has entered the interband continuum, it stops being a
    pole of ``eps``, and the plasmon-pole approximation loses most of the
    spectral weight.  Returned in nm^-1.
    """
    return cfg.E_F / cfg.hv
