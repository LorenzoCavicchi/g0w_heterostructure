#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
TBG | hBN | SLG stack: three sheets, RPA dielectric matrix, screened interaction.

Geometry (LayerCoulomb sheet 0 is the top one)::

    ------------------------------  optional top gate
      region 0   hBN
    ============================== sheet 0  SLG tip
      region 1   hBN     d_tip
    ============================== sheet 1  TBG top layer
      region 2   vacuum (eps_interlayer, default 1.0)   d_tbg = 0.335 nm
    ============================== sheet 2  TBG bottom layer
      region 3   hBN
    ------------------------------  optional bottom gate

The polarizability is block diagonal across the two subsystems -- the tip and the
sample are not tunnel coupled, so no density operator connects them::

    chi0 = [ chi_SLG    0        0     ]
           [   0     chi_TBG_00  chi_TBG_01 ]
           [   0     chi_TBG_10  chi_TBG_11 ]

with ``chi_SLG`` the analytic Dirac bubble from ``../test_SLG_hBN_SLG/chi_dirac``
and the 2x2 TBG block from ``tbg_sample.chi0_tbg``.  All interlayer coupling is
in V, exactly as in the SLG|hBN|SLG case.

The tip is treated as unstructured on the moire scale: it carries no moire
periodicity of its own, so its polarizability is diagonal in G and evaluated at
|q + G|.  With head-only local fields (n_G = 1, the default) this is just
``chi_Dirac(|q|)``.

Conventions: ``eps = I - V chi0``, ``W = eps^-1 V``, both evaluated at
``omega + i eta`` with the *same* eta.  Units: eV, nm.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

from . import dirac_chi as cd
from . import tbg_model as ts
from ..layer_coulomb import LayerCoulomb

HBN_EPS_PAR_STATIC = 6.41
HBN_EPS_Z_STATIC = 3.0
E2_EV_NM = 1.4399764

# Sheet indices in the three-sheet stack
TIP, SAM_TOP, SAM_BOT = 0, 1, 2
SAMPLE_SHEETS = (SAM_TOP, SAM_BOT)


@dataclass
class TBGSLGConfig:
    r"""
    Parameters of the TBG|hBN|SLG junction.

    ``n_tip`` is the tip carrier density [nm^-2].  The paper keeps the tip near
    charge neutrality so it perturbs the sample plasmons as little as possible;
    ``n_tip = 0`` reproduces that (the tip still screens through its *interband*
    response, which is nonzero at charge neutrality, but carries no Drude
    weight).  ``chi_on_tip=False`` removes the tip's response entirely and is the
    reference for measuring back-action.
    """
    tbg: ts.TBGParams = field(default_factory=ts.TBGParams)
    d_tip: float = 1.0                  # hBN between tip and TBG top layer
    d_tbg: float = 0.335                # TBG interlayer spacing
    n_tip: float = 0.0                  # tip density [nm^-2]; 0 = charge neutral
    chi_on_tip: bool = True
    eps_par: float = HBN_EPS_PAR_STATIC
    eps_z: float = HBN_EPS_Z_STATIC
    eps_interlayer: Optional[float] = 1.0
    """Dielectric constant of region 2, *between the two TBG sheets*.

    Defaults to **1.0 (vacuum)**, which is the physical situation: the two
    graphene layers of a TBG are in van der Waals contact 0.335 nm apart and
    there is no hBN between them.  This matches Cavicchi et al.,
    arXiv:2401.16384 (their eps_2 = 1 with eps_1 = eps_3 = 4.9).  Isotropic: the
    same value is used for the in-plane and out-of-plane components of region 2.

    ``None`` restores the historical behaviour of filling that gap with hBN like
    every other region.  **That was a bug**, and it matters a great deal for the
    acoustic channel, whose drive is ``V_- = V_11 - V_12``: it roughly halves
    ``1 - Re eps_--`` and cuts the acoustic share of Sigma by ~3.3x at 5 deg.
    Keep ``None`` only to reproduce results published before 2026-09-07.
    """
    top_gate_d: Optional[float] = None
    bottom_gate_d: Optional[float] = None
    hv_tip: float = cd.GRAPHENE_HV_EV_NM
    g_tip: float = cd.GRAPHENE_DEGENERACY

    @property
    def eta(self) -> float:
        return self.tbg.eta

    @property
    def eps_bar(self) -> float:
        return float(np.sqrt(self.eps_par * self.eps_z))

    def label(self) -> str:
        gate = "ungated" if self.top_gate_d is None else f"d_gate={self.top_gate_d:g} nm"
        tip = (f"tip n={self.n_tip:g} nm^-2" if self.chi_on_tip else "tip inert")
        e2 = "hBN" if self.eps_interlayer is None else f"{self.eps_interlayer:g}"
        return (f"d_tip={self.d_tip:g} nm, {tip}, {gate}, "
                f"eps=({self.eps_par:g},{self.eps_z:g}), eps_gap={e2}")


# ---------------------------------------------------------------------------
# Electrostatics
# ---------------------------------------------------------------------------

def region_epsilon(cfg: TBGSLGConfig) -> Tuple[list, list]:
    r"""
    Per-region ``(eps_parallel, eps_z)`` for the four regions of the stack.

    Region 2 -- the gap between the two TBG sheets -- takes
    ``cfg.eps_interlayer`` (default 1.0, vacuum); ``None`` restores the old
    hBN-filled gap.  See :attr:`TBGSLGConfig.eps_interlayer`.
    """
    par = [float(cfg.eps_par)] * 4
    z = [float(cfg.eps_z)] * 4
    if cfg.eps_interlayer is not None:
        par[2] = z[2] = float(cfg.eps_interlayer)
    return par, z


def stack_coulomb(cfg: TBGSLGConfig) -> LayerCoulomb:
    """Three-sheet ``LayerCoulomb`` (4 dielectric regions)."""
    par, z = region_epsilon(cfg)
    return LayerCoulomb(
        n_layers=3,
        d=[float(cfg.d_tip), float(cfg.d_tbg)],
        epsilon_uniaxial=(par, z),
        top_gate_d=cfg.top_gate_d,
        bottom_gate_d=cfg.bottom_gate_d,
    )


def chi_tip(omega: np.ndarray, q: float, cfg: TBGSLGConfig) -> np.ndarray:
    r"""Analytic Dirac bubble of the SLG tip at |q|, shape (n_omega,)."""
    if not cfg.chi_on_tip:
        return np.zeros(np.size(omega), dtype=np.complex128)
    n = max(float(cfg.n_tip), 0.0)
    if n <= 0.0:
        # charge neutral: interband response only (no Drude weight)
        return cd.chi_dirac_intrinsic(omega, np.array([float(q)]),
                                      cfg.g_tip, cfg.hv_tip)[:, 0]
    return cd.chi_dirac_phys(omega, np.array([float(q)]), n,
                             cfg.g_tip, cfg.hv_tip)[:, 0]


def build_stack_eps_W(
    cfg: TBGSLGConfig,
    q_coords: np.ndarray,
    omega_grid: np.ndarray,
    chi_tbg: np.ndarray,
    *,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Assemble eps, V and W for the three-sheet stack on the (q, omega) grid.

    Parameters
    ----------
    q_coords : (n_q, 2) moire q vectors [nm^-1]
    omega_grid : (n_omega,) real frequencies [eV]; +i eta is added here
    chi_tbg : (n_q, 2, 2, n_omega) sample polarizability from ``tbg_sample.chi0_tbg``

    Returns
    -------
    eps_layer : (n_q, n_omega, 3, 3, 1, 1)   — layout for build_plasmon_pole_data
    coulomb   : (n_q, 3, 3, 1, n_omega)
    W         : (n_q, n_omega, 3, 3)         — eps^-1 V

    The q = 0 point is skipped (V diverges); its row is left at eps = I, W = 0
    and it should be given zero integration weight.
    """
    q_coords = np.asarray(q_coords, dtype=float)
    omega_grid = np.asarray(omega_grid, dtype=float).ravel()
    n_q, n_om = q_coords.shape[0], omega_grid.size
    eta = float(cfg.eta)
    lc = stack_coulomb(cfg)

    eps_layer = np.zeros((n_q, n_om, 3, 3, 1, 1), dtype=np.complex128)
    coulomb = np.zeros((n_q, 3, 3, 1, n_om), dtype=np.complex128)
    W = np.zeros((n_q, n_om, 3, 3), dtype=np.complex128)
    I3 = np.eye(3, dtype=np.complex128)
    om_c = omega_grid + 1j * eta

    for qi in range(n_q):
        q_mag = float(np.linalg.norm(q_coords[qi]))
        if q_mag < 1e-12:
            eps_layer[qi, :, :, :, 0, 0] = I3
            continue
        chi_t = chi_tip(om_c, q_mag, cfg)
        for oi in range(n_om):
            V = np.asarray(lc.coulomb_matrix(q_mag, complex(om_c[oi])),
                           dtype=np.complex128)
            chi = np.zeros((3, 3), dtype=np.complex128)
            chi[TIP, TIP] = chi_t[oi]
            chi[np.ix_(SAMPLE_SHEETS, SAMPLE_SHEETS)] = chi_tbg[qi, :, :, oi]
            E = I3 - V @ chi
            eps_layer[qi, oi, :, :, 0, 0] = E
            coulomb[qi, :, :, 0, oi] = V
            W[qi, oi] = np.linalg.solve(E, V)
        if verbose and (qi + 1) % max(n_q // 8, 1) == 0:
            print(f"    eps/W: {qi+1}/{n_q} q-points", flush=True)

    return eps_layer, coulomb, W


# ---------------------------------------------------------------------------
# Layer-symmetry channels of the sample block
# ---------------------------------------------------------------------------

def split_sample_block_channels(W: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Split the 2x2 *sample* block of W into layer-symmetric / antisymmetric parts.

    With ``u_pm = (1, +-1)/sqrt2`` and ``P_pm = |u_pm><u_pm|`` acting on the two
    TBG layers,

        W_block = P_+ W P_+  +  P_- W P_-  +  (P_+ W P_- + P_- W P_+)

    which is exact and additive by construction.  The three returned arrays are
    full (n_q, n_omega, 3, 3) copies of ``W`` with the sample block replaced by
    each term (tip entries zeroed in the split copies, since only the sample
    block enters the sample self-energy).

    Unlike the symmetric SLG|hBN|SLG bilayer, the tip sits above one TBG layer
    and breaks the up/down symmetry, so ``u_pm`` are no longer exact eigenvectors
    and the **cross term is not zero**.  Its size is a direct measure of how much
    the tip's presence mixes the optical and acoustic channels; it is returned
    rather than discarded.
    """
    W = np.asarray(W, dtype=np.complex128)
    s = np.ix_(SAMPLE_SHEETS, SAMPLE_SHEETS)
    B = W[:, :, SAMPLE_SHEETS, :][:, :, :, SAMPLE_SHEETS]      # (n_q,n_om,2,2)

    u_p = np.array([1.0, 1.0]) / np.sqrt(2.0)
    u_m = np.array([1.0, -1.0]) / np.sqrt(2.0)
    P_p = np.outer(u_p, u_p)
    P_m = np.outer(u_m, u_m)

    B_op = np.einsum("ab,qwbc,cd->qwad", P_p, B, P_p)
    B_ac = np.einsum("ab,qwbc,cd->qwad", P_m, B, P_m)
    B_x = (np.einsum("ab,qwbc,cd->qwad", P_p, B, P_m)
           + np.einsum("ab,qwbc,cd->qwad", P_m, B, P_p))

    out = []
    for B_part in (B_op, B_ac, B_x):
        Wp = np.zeros_like(W)
        Wp[np.ix_(np.arange(W.shape[0]), np.arange(W.shape[1]),
                  SAMPLE_SHEETS, SAMPLE_SHEETS)] = B_part
        out.append(Wp)
    return tuple(out)  # type: ignore[return-value]


def loss_function_sample(W: np.ndarray, layer: int = SAM_TOP) -> np.ndarray:
    r"""L(q, w) = -Im W_{ll}  for one sheet; peaks locate the collective modes."""
    return -np.imag(np.asarray(W)[:, :, layer, layer])
