# -*- coding: utf-8 -*-
r"""
RPA screening of an arbitrary stack of 2D sheets (head only, n_G = 1).

The stack is defined by a :class:`~g0w.layer_coulomb.LayerCoulomb` (N sheets,
N+1 dielectric regions, optional gates) and by the independent-particle
polarizability of each *electronic subsystem*.  Subsystems that are not tunnel
coupled (e.g. a QTM tip and a sample) have no density operator connecting them,
so chi0 is block diagonal::

    chi0(q, w) = blockdiag( chi0_A(q, w), chi0_B(q, w), ... )      (N x N)

with each block covering the sheets of one subsystem (a 1x1 analytic Dirac
bubble, the 2x2 layer-resolved chi0 of TBG from pybandstructure, a zero block
for an inert sheet...).  All inter-subsystem physics enters through V:

    eps(q, w) = 1 - V(q, w) chi0(q, w),     W(q, w) = eps^{-1}(q, w) V(q, w).

Both V and chi0 are evaluated at w + i eta with the *same* eta (an undamped
collective mode then has a Lorentzian of width ~eta in -Im W instead of a
delta function, which makes frequency integrals of W resolvable).
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from .layer_coulomb import LayerCoulomb


def assemble_chi0(n_sheets: int, blocks: Sequence[Tuple[Sequence[int], np.ndarray]]) -> np.ndarray:
    r"""
    Block-diagonal stack polarizability.

    ``blocks``: list of ``(sheet_indices, chi)`` with ``chi`` of shape
    (n_q, m, m, n_w) and ``m = len(sheet_indices)``.  Sheets not listed are inert
    (chi0 = 0).  Returns (n_q, N, N, n_w) complex.
    """
    n_q = n_w = None
    for idx, chi in blocks:
        chi = np.asarray(chi)
        n_q, n_w = chi.shape[0], chi.shape[-1]
    out = np.zeros((n_q, n_sheets, n_sheets, n_w), dtype=np.complex128)
    seen = set()
    for idx, chi in blocks:
        idx = [int(i) for i in idx]
        if seen.intersection(idx):
            raise ValueError("a sheet appears in two chi0 blocks")
        seen.update(idx)
        out[:, np.asarray(idx)[:, None], np.asarray(idx)[None, :], :] = np.asarray(chi)
    return out


def screened_interaction(lc: LayerCoulomb, q_vectors: np.ndarray, omega_grid: np.ndarray,
                         chi0: np.ndarray, eta: float, *,
                         V_at_complex_omega: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""
    eps, V and W = eps^{-1} V on a (q, w) grid.

    Parameters
    ----------
    q_vectors : (n_q, d) or (n_q,) momenta [nm^-1] (only |q| enters V)
    omega_grid : (n_w,) real frequencies [eV]; ``+ i eta`` is added here
    chi0 : (n_q, N, N, n_w) polarizability already evaluated at w + i eta
    V_at_complex_omega : evaluate a dispersive V at w + i eta (default) or at w

    Returns ``eps, V, W`` each (n_q, n_w, N, N).  q = 0 is skipped (V diverges):
    eps = 1, V = W = 0 there; give it zero integration weight.
    """
    q_vectors = np.asarray(q_vectors, dtype=float)
    qmag = np.abs(q_vectors) if q_vectors.ndim == 1 else np.linalg.norm(q_vectors, axis=1)
    omega_grid = np.asarray(omega_grid, dtype=float).ravel()
    N = lc.n_layers
    n_q, n_w = qmag.size, omega_grid.size
    eps = np.zeros((n_q, n_w, N, N), dtype=np.complex128)
    V = np.zeros_like(eps)
    W = np.zeros_like(eps)
    I = np.eye(N, dtype=np.complex128)
    om_c = omega_grid + 1j * float(eta)
    for qi in range(n_q):
        if qmag[qi] < 1e-12:
            eps[qi] = I
            continue
        for oi in range(n_w):
            w = complex(om_c[oi]) if V_at_complex_omega else float(omega_grid[oi])
            Vq = np.asarray(lc.coulomb_matrix(float(qmag[qi]), w), dtype=np.complex128)
            E = I - Vq @ chi0[qi, :, :, oi]
            eps[qi, oi], V[qi, oi] = E, Vq
            W[qi, oi] = np.linalg.solve(E, Vq)
    return eps, V, W


def loss_function(W: np.ndarray, sheets: Optional[Sequence[int]] = None) -> np.ndarray:
    r"""L(q, w) = -Im Tr_{sheets} W(q, w)  (>= 0 at w > 0 in a passive system)."""
    W = np.asarray(W)
    s = list(range(W.shape[-1])) if sheets is None else list(sheets)
    return -np.imag(np.trace(W[..., s, :][..., :, s], axis1=-2, axis2=-1))


def minimal_eigenvalue_abs(eps: np.ndarray) -> np.ndarray:
    r"""|lambda_min(eps)| over the layer index: zeros mark collective modes."""
    lam = np.linalg.eigvals(np.asarray(eps))
    return np.min(np.abs(lam), axis=-1)


def split_layer_channels(W: np.ndarray, pair: Sequence[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Layer-symmetry decomposition of the 2x2 block of W on the sheets ``pair``.

    With u_pm = (1, +-1)/sqrt2 and P_pm = u_pm u_pm^T,

        W_block = P_+ W P_+  +  P_- W P_-  +  (P_+ W P_- + P_- W P_+)
                  in-phase      out-of-phase   cross

    is an identity (exact and additive).  For two equivalent sheets the cross
    term vanishes and P_+ W P_+ / P_- W P_- are the optical / acoustic
    channels; a third object (a tip above one sheet) makes the cross term
    nonzero.  Returns three arrays shaped like ``W`` with the block replaced by
    each term and every other entry set to zero.
    """
    W = np.asarray(W, dtype=np.complex128)
    a, b = int(pair[0]), int(pair[1])
    B = W[..., [a, b], :][..., :, [a, b]]
    up = np.array([1.0, 1.0]) / np.sqrt(2.0)
    um = np.array([1.0, -1.0]) / np.sqrt(2.0)
    Pp, Pm = np.outer(up, up), np.outer(um, um)
    parts = (Pp @ B @ Pp, Pm @ B @ Pm, Pp @ B @ Pm + Pm @ B @ Pp)
    out = []
    for part in parts:
        Wp = np.zeros_like(W)
        Wp[..., a, a], Wp[..., a, b] = part[..., 0, 0], part[..., 0, 1]
        Wp[..., b, a], Wp[..., b, b] = part[..., 1, 0], part[..., 1, 1]
        out.append(Wp)
    return tuple(out)  # type: ignore[return-value]
