# -*- coding: utf-8 -*-
r"""
Quantum twisting microscope (QTM): twist kinematics and differential conductance.

(Sources: ``layer_resolved/test_SLG_hBN_SLG/slg_qtm.py`` and
``layer_resolved/test_TBG_hBN_SLG/tbg_qtm.py``; the TBG trajectory, tunneling
amplitude and band-matrix conductance are corrected here, the old behaviour is
kept behind ``legacy=True`` switches.  See ``docs/notes.pdf`` Sec. 7.)

Differential conductance (Wei, Guinea, von Oppen, Glazman, PRB 112, 155157,
Eq. (33)): for a monolayer-graphene tip with a small Fermi momentum, whose
chemical potential follows the bias while the sample's does not move,

    dI/dV(V, theta) = e^2 Gamma sum_{lambda rho} Lambda_lambda(K_theta)
                      A^S_{lambda rho}(K_theta, eV) Lambda_rho(K_theta)^*        (T = 0)

* ``K_theta = O(theta) K_top`` is the tip Dirac point, ``O`` the clockwise
  rotation of Wei et al. Eq. (16) and theta the tip--top-layer twist;
* the sample is probed at the (extended-zone) momentum ``K_theta``;
* ``Lambda_lambda = u^lambda_{K_theta, top, A} + u^lambda_{K_theta, top, B}``
  (Eq. 32) is the top-layer sublattice sum of the plane-wave component of the
  Bloch state at the *unfolded* momentum ``K_theta``;
* ``A^S`` is the (band-matrix) G0W spectral function at the folded momentum;
* ``eV = eps``: the bias axis is the sample energy axis.

Dropped constant prefactors: Gamma = 2 pi w^2 Omega nu_T N_b N_f / hbar (tip
DOS, junction area, Bragg-channel multiplicity N_b = 3, flavours N_f = 4).
Output is dI/dV / (e^2 Gamma) in eV^-1.

Units: energies eV, momenta nm^-1, angles in degrees unless stated.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from .constants import GRAPHENE_K_NM_INV


# ---------------------------------------------------------------------------
# Common kinematics
# ---------------------------------------------------------------------------

def rotation_matrix(theta_rad: float) -> np.ndarray:
    """Clockwise rotation O(theta) = [[cos, sin], [-sin, cos]] (Wei et al. Eq. 16)."""
    c, s = np.cos(theta_rad), np.sin(theta_rad)
    return np.array([[c, s], [-s, c]], dtype=float)


def twist_to_momentum(theta_deg, K: float = GRAPHENE_K_NM_INV) -> np.ndarray:
    r"""|K_theta - K| = 2 K |sin(theta/2)|  [nm^-1]  (0.297 nm^-1 per degree)."""
    th = np.deg2rad(np.asarray(theta_deg, dtype=float))
    return 2.0 * float(K) * np.abs(np.sin(0.5 * th))


def momentum_to_twist(q, K: float = GRAPHENE_K_NM_INV) -> np.ndarray:
    r"""Inverse of :func:`twist_to_momentum`, in degrees."""
    q = np.asarray(q, dtype=float)
    return np.rad2deg(2.0 * np.arcsin(np.clip(q / (2.0 * float(K)), -1.0, 1.0)))


def bias_derivative(didv: np.ndarray, bias: np.ndarray, axis: int = -1) -> np.ndarray:
    r"""d^2I/dV^2 = d(dI/dV)/dV by centred finite differences along ``axis``."""
    return np.gradient(np.asarray(didv, dtype=float), np.asarray(bias, dtype=float), axis=axis)


def smooth_bias(didv: np.ndarray, bias: np.ndarray, width: float, axis: int = -1) -> np.ndarray:
    r"""Gaussian smoothing along the bias axis (``width`` = sigma in eV), e.g. thermal / lock-in."""
    if width <= 0:
        return np.asarray(didv, dtype=float)
    bias = np.asarray(bias, float)
    dv = float(np.mean(np.diff(bias)))
    n = int(np.ceil(4 * width / dv))
    x = np.arange(-n, n + 1) * dv
    ker = np.exp(-0.5 * (x / width) ** 2)
    ker /= ker.sum()
    return np.apply_along_axis(lambda y: np.convolve(y, ker, mode="same"), axis, np.asarray(didv, float))


# ---------------------------------------------------------------------------
# SLG sample (Dirac cone at K, isotropic)
# ---------------------------------------------------------------------------

def slg_probe_momentum(theta_deg: float, K: float = GRAPHENE_K_NM_INV) -> np.ndarray:
    r"""
    p_theta = K_theta - K = (O(theta) - 1) (K, 0): sample momentum measured from its
    Dirac point.  |p_theta| = 2 K sin(theta/2), direction ~ perpendicular to K.
    """
    K_vec = np.array([float(K), 0.0])
    return rotation_matrix(np.deg2rad(float(theta_deg))) @ K_vec - K_vec


def slg_tunneling_weight(p_vec: np.ndarray, band: int, model: str = "chiral") -> float:
    r"""
    |Lambda_s(p)|^2 for a Dirac sample (p from the Dirac point, band s = +-1).

    ``chiral``: Lambda_s = (1 + s e^{i phi_p})/sqrt2 -> |Lambda|^2 = 1 + s cos(phi_p)
    (Eq. 32 with the sublattice spinor (1, s e^{i phi})/sqrt2);
    ``constant``: 1 (reads the spectral function directly).
    In the QTM geometry p_theta is ~ perpendicular to K, cos(phi_p) ~ 0 and the
    two models differ by < 2 %.
    """
    if model == "constant":
        return 1.0
    if model != "chiral":
        raise ValueError("model must be 'chiral' or 'constant'.")
    p = np.asarray(p_vec, dtype=float).ravel()[:2]
    if np.linalg.norm(p) < 1e-14:
        return 1.0
    return float(1.0 + float(band) * np.cos(np.arctan2(p[1], p[0])))


def slg_spectral_map(cfg, theta_deg: np.ndarray, eps_values: np.ndarray, *,
                     q_grid: np.ndarray, omega_grid: np.ndarray, W_grid: np.ndarray,
                     q_weights: np.ndarray, bands: Tuple[int, ...] = (+1, -1),
                     n_theta_ang: int = 64, eta_qp: float = 2e-3, eta_self: float = 1e-3,
                     T: float = 0.0, verbose: bool = False) -> Dict[str, np.ndarray]:
    r"""
    A^S_s(p_theta, eps) and Sigma_s(p_theta, eps) on a (theta, eps) grid for the
    SLG sample of an SLG|hBN|SLG junction (full-frequency G0W, both Dirac bands).

    ``W_grid`` from :func:`g0w.models.slg_self_energy.build_W_grid`.
    Returns ``theta_deg, q_theta, xi (n_b, n_th), eps_values, A, Sigma (n_b, n_th, n_eps), bands``.
    """
    from .models import slg_self_energy as se
    theta_deg = np.atleast_1d(np.asarray(theta_deg, dtype=float))
    eps_values = np.asarray(eps_values, dtype=float).ravel()
    nb, nth, ne = len(bands), theta_deg.size, eps_values.size
    A = np.zeros((nb, nth, ne))
    Sig = np.zeros((nb, nth, ne), dtype=np.complex128)
    xi = np.zeros((nb, nth))
    for ib, band in enumerate(bands):
        for it, th in enumerate(theta_deg):
            k_vec = slg_probe_momentum(th)
            S = se.sigma_full_frequency_fast(cfg, k_vec, eps_values, q_grid, omega_grid, W_grid,
                                             q_weights, band_ext=band, T=T, eta_self=eta_self,
                                             n_theta=n_theta_ang)
            Sig[ib, it] = S[0, 0]
            A[ib, it] = se.spectral_function_sample(cfg, k_vec, eps_values, S, band_ext=band,
                                                    eta_qp=eta_qp)
            xi[ib, it] = se.quasiparticle_energy(cfg, k_vec, band)
            if verbose:
                print(f"  band {band:+d} theta = {th:6.3f} deg  xi = {xi[ib, it]*1e3:+7.1f} meV", flush=True)
    return dict(theta_deg=theta_deg, q_theta=twist_to_momentum(theta_deg), xi=xi,
                eps_values=eps_values, A=A, Sigma=Sig, bands=np.asarray(bands))


def slg_didv(smap: Dict[str, np.ndarray], tunneling_model: str = "chiral") -> np.ndarray:
    r"""dI/dV(V, theta)/(e^2 Gamma) = sum_s |Lambda_s(p_theta)|^2 A^S_s(p_theta, eV); (n_theta, n_eps)."""
    A, bands, th = smap["A"], smap["bands"], smap["theta_deg"]
    out = np.zeros(A.shape[1:])
    for ib, b in enumerate(bands):
        for it, t in enumerate(th):
            out[it] += slg_tunneling_weight(slg_probe_momentum(t), int(b), tunneling_model) * A[ib, it]
    return out


# ---------------------------------------------------------------------------
# TBG sample
# ---------------------------------------------------------------------------

def tbg_tip_momentum(tbg: Dict[str, Any], theta_deg: float, *, legacy: bool = False,
                     theta_offset_deg: float = 0.0) -> np.ndarray:
    r"""
    Model-frame (extended-zone) sample momentum probed by the tip Dirac point.

    Physical (default):  k_theta = O(theta) K_top^lab - Gamma^lab, so that
        theta = 0         -> K_1 = top-layer Dirac point (kappa_top),
        theta = theta_TBG -> K_2 = bottom-layer Dirac point (kappa_bot),
        theta = -theta_TBG-> ~ Gamma of a neighbouring moire BZ, 2 K_1 - K_2 (to O(theta^2)),
    i.e. an arc of radius |K| that is, to O(theta^2), a straight line through kappa_top
    in the direction K_2 - K_1.

    ``legacy=True`` reproduces ``tbg_qtm.probe_momentum`` of the original scripts,
    p = (O(theta) - 1) K_top, which puts theta = 0 on the moire gamma point.
    That trajectory is shifted by -K_1 from the physical one.
    """
    from .models import tbg_model as tm
    fr = tm.model_frame(tbg)
    th = np.deg2rad(float(theta_deg) + float(theta_offset_deg))
    O = rotation_matrix(th)
    if legacy:
        return (O - np.eye(2)) @ fr["K_top_lab"]
    return O @ fr["K_top_lab"] - fr["gamma_lab"]


def tbg_trajectory(tbg: Dict[str, Any], theta_deg: Sequence[float], *, legacy: bool = False,
                   theta_offset_deg: float = 0.0) -> Dict[str, np.ndarray]:
    r"""
    Map a twist sweep onto the k mesh.

    Returns ``theta_deg``, ``k_ext`` (n,2) extended-zone momentum, ``k_index`` (n,),
    ``G_int`` (n,2) moire G with k_ext ~ k_mesh + G, ``snap_error`` (n,) and
    ``mesh_spacing``.  The snapping error is at most 1/sqrt(3) ~ 0.58 mesh spacings (nearest mesh point).
    """
    from .models import tbg_model as tm
    theta_deg = np.atleast_1d(np.asarray(theta_deg, dtype=float))
    k_ext = np.array([tbg_tip_momentum(tbg, t, legacy=legacy, theta_offset_deg=theta_offset_deg)
                      for t in theta_deg])
    k_idx = np.zeros(theta_deg.size, dtype=int)
    G_int = np.zeros((theta_deg.size, 2), dtype=int)
    err = np.zeros(theta_deg.size)
    for i, k in enumerate(k_ext):
        f = tm.fold_to_mesh(tbg, k)
        k_idx[i], G_int[i], err[i] = f["k_index"], f["G_int"], f["snap_error"]
    p = tbg["params"]
    return dict(theta_deg=theta_deg, k_ext=k_ext, k_index=k_idx, G_int=G_int, snap_error=err,
                mesh_spacing=p.G_moire / p.n_sites, legacy=legacy)


def tbg_tunneling_amplitudes(tbg: Dict[str, Any], k_index: int, bands: Sequence[int], *,
                             G_int: Tuple[int, int] = (0, 0), model: str = "chiral",
                             legacy: bool = False) -> np.ndarray:
    r"""
    Complex tunneling amplitudes Lambda_lambda for the selected bands at a mesh k.

    ``chiral``     : u^lambda_{k+G, top, A} + u^lambda_{k+G, top, B}  (Wei Eq. 32), with
                     G = ``G_int`` the moire vector that unfolds k to the tip momentum;
                     0 if k+G lies outside the plane-wave cutoff.
    ``constant``   : 1 for every band (spectral function, no matrix element).
    ``top_weight`` : sqrt of the top-layer probability (foil: not what the tip couples to).

    The eigenvectors are those stored in the band structure (``bs.wavefunctions``),
    i.e. the *same gauge* as the density matrices that build Sigma_{lambda rho}:
    the band-matrix conductance Lambda^T A Lambda^* is then gauge invariant.
    ``legacy=True``: G = 0 always and a fresh ``numpy.linalg.eigh`` of H(k) (the
    original ``tbg_qtm.tunneling_amplitudes``; returns |Lambda| only, fine for the
    band-diagonal conductance).
    """
    from .models import tbg_model as tm
    bands = [int(b) for b in bands]
    if model == "constant":
        return np.ones(len(bands), dtype=np.complex128)
    if legacy:
        H = np.asarray(tbg["model"]["hamiltonian"](np.asarray(tbg["k_sample"].coords[int(k_index)], float)))
        _, vecs = np.linalg.eigh(H)
        V = vecs[:, [tm.band_offset(tbg) + b for b in bands]]
        g = 0
    else:
        V = np.asarray(tbg["crystal"].band_structure.wavefunctions)[:, bands, int(k_index)]
        pw = tm.plane_waves(tbg)
        key = tuple(int(x) for x in G_int)
        if key not in pw["index"]:
            return np.zeros(len(bands), dtype=np.complex128)
        g = pw["index"][key]
    comp = V[4 * g: 4 * g + 4, :]                       # (top A, top B, bot A, bot B) x bands
    if model == "chiral":
        lam = comp[0] + comp[1]
        return np.abs(lam).astype(np.complex128) if legacy else lam
    if model == "top_weight":
        return np.sqrt(np.sum(np.abs(V.reshape(-1, 4, V.shape[1])[:, 0:2, :]) ** 2, axis=(0, 1))).astype(complex)
    raise ValueError("model must be 'chiral', 'constant' or 'top_weight'.")


def tbg_spectral_along_trajectory(tbg: Dict[str, Any], lh, traj: Dict[str, np.ndarray],
                                  S_grid: Optional[np.ndarray], omega_grid: Optional[np.ndarray],
                                  q_weights: np.ndarray, eps_values: np.ndarray, *,
                                  kernel=None, bands: Optional[Sequence[int]] = None,
                                  band_diagonal: bool = False, eta_self: float = 1e-3,
                                  eta_qp: float = 1e-3, T: float = 0.0,
                                  verbose: bool = False) -> Dict[str, np.ndarray]:
    r"""
    G0W self-energy and spectral function at every distinct k of a trajectory.

    ``band_diagonal=False`` (default): Sigma_{lambda rho}(k, eps) is the full band
    matrix (Wei Eq. 34/50) and A the matrix spectral function.
    ``band_diagonal=True``: only Sigma_{lambda lambda} (the original scripts).
    ``kernel``: a :class:`~g0w.self_energy_full_frequency.ResolventKernel` of
    ``S_grid`` (fast route); otherwise the frequency integral is done directly.

    Returns ``Sigma`` (n_k_unique, n_b, n_b, n_eps), ``A`` (n_k_unique, n_eps, n_b, n_b),
    ``xi`` (n_k_unique, n_b), ``k_unique``, ``inverse`` (trajectory -> unique), ``bands``.
    """
    from .self_energy_full_frequency import (compute_self_energy_band_matrix,
                                             compute_self_energy_band_matrix_kernel)
    from .spectral import spectral_function_matrix
    bs = tbg["crystal"].band_structure
    nb_tot = np.asarray(bs.energies).shape[0]
    bands = list(range(nb_tot)) if bands is None else [int(b) for b in bands]
    eps_values = np.asarray(eps_values, float).ravel()
    k_unique, inverse = np.unique(np.asarray(traj["k_index"], int), return_inverse=True)
    mu = float(bs.chemical_potential)
    E = np.asarray(bs.energies)
    nb, ne = len(bands), eps_values.size
    Sig = np.zeros((k_unique.size, nb, nb, ne), dtype=np.complex128)
    A = np.zeros((k_unique.size, ne, nb, nb), dtype=np.complex128)
    xi = np.zeros((k_unique.size, nb))
    for i, ki in enumerate(k_unique):
        if kernel is not None:
            S = compute_self_energy_band_matrix_kernel(np.asarray(lh.densityL), E, bs.k_plus_q_ind, mu,
                                                       kernel, q_weights, int(ki), bands, eps_values)
        else:
            S = compute_self_energy_band_matrix(np.asarray(lh.densityL), E, bs.k_plus_q_ind, mu, S_grid,
                                                omega_grid, q_weights, int(ki), bands, eps_values,
                                                T=T, eta_self=eta_self)
        if band_diagonal:
            S = S * np.eye(nb)[:, :, None]
        Sig[i] = S
        xi[i] = E[bands, int(ki)] - mu
        A[i] = spectral_function_matrix(eps_values, xi[i], S, eta_qp=eta_qp)
        if verbose:
            print(f"  k {i+1}/{k_unique.size} (index {ki})", flush=True)
    return dict(Sigma=Sig, A=A, xi=xi, k_unique=k_unique, inverse=inverse, bands=np.asarray(bands),
                eps_values=eps_values, band_diagonal=band_diagonal)


def tbg_didv(tbg: Dict[str, Any], traj: Dict[str, np.ndarray], spec: Dict[str, np.ndarray], *,
             model: str = "chiral", legacy_amplitude: bool = False) -> Dict[str, np.ndarray]:
    r"""
    dI/dV(V, theta)/(e^2 Gamma) = sum_{lambda rho} Lambda_lambda A_{lambda rho}(eV) Lambda_rho^*
    along a trajectory.  Returns ``didv`` (n_theta, n_eps) and ``Lambda`` (n_theta, n_b).
    """
    bands = spec["bands"]
    n_th = traj["theta_deg"].size
    ne = spec["eps_values"].size
    out = np.zeros((n_th, ne))
    Lam = np.zeros((n_th, len(bands)), dtype=np.complex128)
    for it in range(n_th):
        iu = int(spec["inverse"][it])
        L = tbg_tunneling_amplitudes(tbg, int(traj["k_index"][it]), bands, G_int=tuple(traj["G_int"][it]),
                                     model=model, legacy=legacy_amplitude)
        Lam[it] = L
        A = spec["A"][iu]                                   # (n_eps, n_b, n_b)
        out[it] = np.real(np.einsum("l,elr,r->e", L, A, np.conj(L)))
    return dict(didv=out, Lambda=Lam)


def tbg_bare_didv(tbg: Dict[str, Any], traj: Dict[str, np.ndarray], eps_values: np.ndarray, *,
                  bands: Optional[Sequence[int]] = None, eta_qp: float = 1e-3, model: str = "chiral",
                  legacy: bool = False) -> np.ndarray:
    r"""Non-interacting dI/dV = sum_lambda |Lambda_lambda|^2 delta_eta(eV - xi_lambda) along a trajectory."""
    bs = tbg["crystal"].band_structure
    E = np.asarray(bs.energies) - float(bs.chemical_potential)
    bands = list(range(E.shape[0])) if bands is None else list(bands)
    eps_values = np.asarray(eps_values, float).ravel()
    out = np.zeros((traj["theta_deg"].size, eps_values.size))
    for it in range(traj["theta_deg"].size):
        ki = int(traj["k_index"][it])
        L = tbg_tunneling_amplitudes(tbg, ki, bands, G_int=tuple(traj["G_int"][it]), model=model, legacy=legacy)
        lor = (eta_qp / np.pi) / ((eps_values[None, :] - E[bands, ki][:, None]) ** 2 + eta_qp ** 2)
        out[it] = np.einsum("l,le->e", np.abs(L) ** 2, lor)
    return out
