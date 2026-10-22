# -*- coding: utf-8 -*-
r"""
Dyson equation, spectral functions and quasiparticle analysis.

Conventions
-----------
Energies are measured from the chemical potential of the sample (eps = 0 is
the Fermi level).  The retarded Green's function of the external state(s) is

    scalar :  G(k, eps) = [eps - xi_k - Sigma(k, eps) + i eta_qp]^{-1}
    matrix :  G(k, eps) = [(eps + i eta_qp) 1 - diag(xi_k) - Sigma(k, eps)]^{-1}

(Wei et al. Eqs. (27)-(28) and (50)), with ``eta_qp`` a small extrinsic width
(finite junction, disorder) that keeps the quasiparticle peak finite where
Im Sigma -> 0.  The spectral function is

    scalar :  A = -(1/pi) Im G               (int A d eps = 1)
    matrix :  A = i (G - G^dagger) / (2 pi)  (Hermitian, PSD; Tr int A = n_bands)

which is Wei et al. Eq. (26), A = -G''/pi with G'' = i (G^dagger - G)/2.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

from .compat import trapezoid


# ---------------------------------------------------------------------------
# Scalar (band-diagonal)
# ---------------------------------------------------------------------------

def green_function_scalar(eps: np.ndarray, xi: float, Sigma: np.ndarray,
                          eta_qp: float = 1e-3) -> np.ndarray:
    """G(eps) = 1 / (eps - xi - Sigma(eps) + i eta_qp)."""
    eps = np.asarray(eps, dtype=float)
    return 1.0 / (eps - float(xi) - np.asarray(Sigma, dtype=np.complex128) + 1j * float(eta_qp))


def spectral_function_scalar(eps: np.ndarray, xi: float, Sigma: np.ndarray,
                             eta_qp: float = 1e-3) -> np.ndarray:
    """A(eps) = -(1/pi) Im G(eps)  [eV^-1]."""
    return -np.imag(green_function_scalar(eps, xi, Sigma, eta_qp)) / np.pi


# ---------------------------------------------------------------------------
# Matrix (band-off-diagonal self-energy)
# ---------------------------------------------------------------------------

def green_function_matrix(eps: np.ndarray, xi: Sequence[float], Sigma: Optional[np.ndarray],
                          eta_qp: float = 1e-3) -> np.ndarray:
    r"""
    G_{n n'}(eps) = [(eps + i eta_qp) - diag(xi) - Sigma(eps)]^{-1}.

    ``Sigma`` has shape (n_b, n_b, n_eps) (or None for the bare propagator).
    Returns (n_eps, n_b, n_b).
    """
    eps = np.asarray(eps, dtype=float).ravel()
    xi = np.asarray(xi, dtype=float).ravel()
    nb = xi.size
    M = np.zeros((eps.size, nb, nb), dtype=np.complex128)
    M[:, np.arange(nb), np.arange(nb)] = (eps[:, None] + 1j * float(eta_qp)) - xi[None, :]
    if Sigma is not None:
        M -= np.moveaxis(np.asarray(Sigma, dtype=np.complex128), -1, 0)
    return np.linalg.inv(M)


def spectral_function_matrix(eps: np.ndarray, xi: Sequence[float], Sigma: Optional[np.ndarray],
                             eta_qp: float = 1e-3) -> np.ndarray:
    r"""A(eps) = i (G - G^dagger)/(2 pi), shape (n_eps, n_b, n_b), Hermitian PSD."""
    G = green_function_matrix(eps, xi, Sigma, eta_qp)
    return 1j * (G - np.conj(np.swapaxes(G, -1, -2))) / (2.0 * np.pi)


def is_passive(Sigma: np.ndarray, atol: float = 1e-10) -> bool:
    r"""
    True if the anti-Hermitian part of Sigma^R is negative semidefinite at every eps
    (scalar: Im Sigma <= 0).  Required for A >= 0.
    """
    S = np.asarray(Sigma, dtype=np.complex128)
    if S.ndim == 1:
        return bool(np.all(np.imag(S) <= atol))
    St = np.moveaxis(S, -1, 0)
    A = (St - np.conj(np.swapaxes(St, -1, -2))) / 2j
    return bool(np.all(np.linalg.eigvalsh(A) <= atol))


# ---------------------------------------------------------------------------
# Quasiparticle analysis
# ---------------------------------------------------------------------------

def quasiparticle_solution(eps: np.ndarray, xi: float, Sigma: np.ndarray,
                           window: Optional[float] = None) -> Dict[str, float]:
    r"""
    Solve the quasiparticle equation  E - xi - Re Sigma(E) = 0  on the eps grid.

    Returns the root ``E_qp`` closest to ``xi`` (within ``window`` if given), the
    renormalisation factor ``Z = [1 - d Re Sigma/d eps]^{-1}`` at the root and
    ``Gamma = -2 Z Im Sigma(E_qp)`` (full width of the Lorentzian peak).  The
    linearised solution  E_lin = xi + Z Re Sigma(xi)  is also returned.
    """
    eps = np.asarray(eps, dtype=float).ravel()
    S = np.asarray(Sigma, dtype=np.complex128).ravel()
    f = eps - float(xi) - np.real(S)
    roots = []
    for i in np.flatnonzero(np.sign(f[:-1]) * np.sign(f[1:]) <= 0):
        if f[i] == f[i + 1]:
            continue
        t = f[i] / (f[i] - f[i + 1])
        roots.append((eps[i] + t * (eps[i + 1] - eps[i]), i, t))
    dRe = np.gradient(np.real(S), eps)
    if not roots:
        return dict(E_qp=np.nan, Z=np.nan, Gamma=np.nan, E_lin=np.nan)
    # physical root: increasing f (d f/d eps > 0 <=> 0 < Z < inf), closest to xi
    good = [r for r in roots if (1.0 - (dRe[r[1]] + r[2] * (dRe[min(r[1] + 1, eps.size - 1)] - dRe[r[1]]))) > 0]
    cand = good if good else roots
    if window is not None:
        cand = [r for r in cand if abs(r[0] - xi) <= window] or cand
    E, i, t = min(cand, key=lambda r: abs(r[0] - xi))
    dR = dRe[i] + t * (dRe[min(i + 1, eps.size - 1)] - dRe[i])
    Z = 1.0 / (1.0 - dR)
    ImS = np.imag(S[i]) + t * (np.imag(S[min(i + 1, eps.size - 1)]) - np.imag(S[i]))
    j = int(np.argmin(np.abs(eps - xi)))
    Zx = 1.0 / (1.0 - dRe[j])
    return dict(E_qp=float(E), Z=float(Z), Gamma=float(-2.0 * Z * ImS),
                E_lin=float(xi + Zx * np.real(S[j])))


def spectral_weight(eps: np.ndarray, A: np.ndarray) -> float:
    """int A(eps) d eps on the grid (should be ~1 per band if the window is wide)."""
    return float(trapezoid(np.asarray(A, float), np.asarray(eps, float)))
