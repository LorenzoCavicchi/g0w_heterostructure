#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Layer-resolved Coulomb interaction of N parallel 2D sheets in a stratified medium.

(Source: ``layer_resolved/test_v2_7/layer_coulomb_v2_7.py``, v2.7 with gates;
``CappedLayerCoulomb`` from ``tbg_hbn_encapsulated_loss/tbg_hbn_loss.py``.)

Electrostatic (quasi-static) Poisson problem in Gaussian units with sheet charges::

    E = -grad phi,   D = eps . E,   div D = 4 pi rho   =>   div(eps grad phi) = -4 pi rho

For a uniaxial medium aligned with z in region i,
``eps^(i) = diag(eps_par^(i)(w), eps_par^(i)(w), eps_z^(i)(w))`` and

    kappa_i(q, w) = q sqrt(eps_par^(i) / eps_z^(i)),
    Y_i(q, w)     = eps_z^(i) kappa_i = q sqrt(eps_par^(i) eps_z^(i)).

N sheets define N+1 regions: region 0 above sheet 0, region j (1 <= j <= N-1)
between sheets j-1 and j, region N below sheet N-1.  ``d[j]`` (length N-1) is the
spacing between sheets j and j+1.  At each (q, w) the sheet potentials obey the
tridiagonal system  A(q,w) Phi = 4 pi e s_m  and the physical kernel is

    V(q, w) = 4 pi e^2 A(q, w)^{-1}         [eV nm^2 for q in nm^-1].

The outer regions are open half-spaces or are terminated by grounded gates
(``top_gate_d``/``bottom_gate_d``); a gate replaces the boundary admittance
Y -> Y coth(kappa d_gate).  ``CappedLayerCoulomb`` instead closes region 0 with a
finite slab backed by a half-space (e.g. vacuum above a finite hBN flake).

Closed-form checks (isotropic eps, single sheet)::

    open           V = 2 pi e^2 / (eps q)
    dual gate d    V = 2 pi e^2 / (eps q) tanh(q d)
    single gate d  V = 2 pi e^2 / (eps q) (1 - exp(-2 q d))

See ``docs/notes.pdf`` Sec. 2 for the derivation.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Sequence, Tuple, Union, Any

import numpy as np

# Coulomb constant in Gaussian-like units expressed in eV·nm
E2_EV_NM: float = 1.4399764  # e^2 in eV·nm
FOURPI_E2_EV_NM: float = 4.0 * np.pi * E2_EV_NM
TWOPI_E2_EV_NM: float = 2.0 * np.pi * E2_EV_NM

# Numeric stability thresholds for hyperbolic functions
Z_CUT: float = np.inf#50.0           # large |Re(z)| asymptotics
Z_SMALL_SERIES: float = 0#1e-6  # small |z| series
KAPPA_BRANCH_TOL: float = 1e-12
_D_MIN: float = 1e-290  # below this |D| use fallback to avoid 0/0 in coth_csch_stable


def fourpi_e2() -> float:
    """Return 4π e^2 in eV·nm (consistent prefactor for V = (4π e^2) A^{-1})."""
    return FOURPI_E2_EV_NM


def _as_complex_array(x: Any, dtype=np.complex128) -> np.ndarray:
    return np.asarray(x, dtype=dtype)


def coth_csch_stable(
    z: Union[complex, np.ndarray],
    z_cut: float = np.inf,
    z_small: float = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Numerically stable coth(z) and csch(z) for complex z (vectorized, scalar-safe).

    Uses an exact algebraic rewrite factoring out exp(-z) to avoid overflow/underflow
    and cancellation:

        u = exp(-2z),   D = 1 - u  (computed as -expm1(-2z) for accuracy when u≈1)
        coth(z) = (1 + u) / D
        csch(z) = 2*exp(-z) / D

    Regimes:
      - |z| < z_small:  series  coth ≈ 1/z + z/3,  csch ≈ 1/z - z/6
      - Re(z) > z_cut:  coth ≈ 1,  csch ≈ 2*exp(-z)  (no overflow)
      - Re(z) < -z_cut: coth(-z)=-coth(z), csch(-z)=-csch(z)  (parity)
      - else:            exp-based formulas above (single exp(-z), then u=exp(-z)^2)

    Supports vectorized numpy arrays and complex dtype. Used everywhere in A-matrix
    assembly for spacer couplings (any N, isotropic/uniaxial, complex κ).
    """
    z = np.atleast_1d(_as_complex_array(z))
    coth = np.empty_like(z)
    csch = np.empty_like(z)

    absz = np.abs(z)
    re_z = np.real(z)
    small = absz < z_small
    large_pos = re_z > z_cut
    large_neg = re_z < -z_cut
    mid = ~(small | large_pos | large_neg)

    # Small |z|: series
    if np.any(small):
        zs = z[small]
        inv = 1.0 / zs
        coth[small] = inv + zs / 3.0
        csch[small] = inv - zs / 6.0

    # Large positive Re(z): coth→1, csch→2*exp(-z)
    if np.any(large_pos):
        ez = np.exp(-z[large_pos])
        coth[large_pos] = 1.0
        csch[large_pos] = 2.0 * ez

    # Large negative Re(z): parity coth(z)=-coth(-z), csch(z)=-csch(-z)
    if np.any(large_neg):
        zn = -z[large_neg]
        ez = np.exp(-zn)
        coth[large_neg] = -1.0
        csch[large_neg] = -2.0 * ez

    # Mid: u = exp(-2z), D = -expm1(-2z), coth = (1+u)/D, csch = 2*exp(-z)/D
    # Compute exp(-z) once, then u = exp(-z)^2 = exp(-2z)
    if np.any(mid):
        zm = z[mid]
        exp_neg_z = np.exp(-zm)
        u = exp_neg_z * exp_neg_z
        D = -np.expm1(-2.0 * zm)
        use_exp = np.abs(D) >= _D_MIN
        coth_exp = (1.0 + u) / np.where(use_exp, D, 1.0)
        csch_exp = (2.0 * exp_neg_z) / np.where(use_exp, D, 1.0)
        coth_fb = np.cosh(zm) / np.sinh(zm)
        csch_fb = 1.0 / np.sinh(zm)
        coth[mid] = np.where(use_exp, coth_exp, coth_fb)
        csch[mid] = np.where(use_exp, csch_exp, csch_fb)
    return coth, csch


def hyperbolic_stable(z) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stable coth(z) and csch(z). Alias for coth_csch_stable with default thresholds.
    Accepts scalar or array input; always returns arrays.
    """
    return coth_csch_stable(z, z_cut=Z_CUT, z_small=Z_SMALL_SERIES)


def _choose_kappa_branch(kappa: np.ndarray) -> np.ndarray:
    """
    Enforce decay branch for κ: Re(κ) > 0, or if ~0 then Im(κ) > 0.
    Operates elementwise and returns a possibly sign-flipped κ.
    """
    kappa = _as_complex_array(kappa)
    kr = np.real(kappa)
    ki = np.imag(kappa)
    flip = (kr < 0.0) | ((np.abs(kr) < KAPPA_BRANCH_TOL) & (ki < 0.0))
    kappa = np.where(flip, -kappa, kappa)
    return kappa


def compute_kappa_Y(
    q: Union[float, np.ndarray],
    eps_par: np.ndarray,
    eps_z: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute κ_i(q,ω) and Y_i(q,ω) for each region i (single ω).

    Parameters
    ----------
    q : float or array
        In-plane wavevector magnitude (1/nm if d is in nm).
    eps_par, eps_z : arrays of shape (n_regions,)
        Uniaxial dielectric components at the desired ω (may be complex).

    Returns
    -------
    kappa : array of shape (n_regions,)
    Y : array of shape (n_regions,)
    """
    q = np.asarray(q, dtype=np.complex128)  # allow complex for consistency
    eps_par = _as_complex_array(eps_par)
    eps_z = _as_complex_array(eps_z)
    if eps_par.shape != eps_z.shape:
        raise ValueError("eps_par and eps_z must have the same shape (n_regions,).")

    # κ_i = q * sqrt(eps_par/eps_z)
    ratio = eps_par / eps_z
    kappa = q * np.sqrt(ratio)
    kappa = _choose_kappa_branch(kappa)

    # Y_i = eps_z * κ_i
    Y = eps_z * kappa
    return kappa, Y


def compute_kappa_Y_tracked(
    q: Union[float, np.ndarray],
    eps_par: np.ndarray,
    eps_z: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute κ_i(q,ω) and Y_i(q,ω) with analytic continuation in ω (no π jumps).

    Use for dispersive runs (n_omega > 1) to avoid branch discontinuities across
    the Reststrahlen band that would make V(q,ω) non-passive (VA indefinite).

    Parameters
    ----------
    q : float
        In-plane wavevector magnitude.
    eps_par, eps_z : arrays of shape (n_regions, n_omega)
        Uniaxial dielectric at each ω (may be complex).

    Returns
    -------
    kappa : array of shape (n_regions, n_omega)
    Y : array of shape (n_regions, n_omega), with Y = eps_z * κ (consistent).
    """
    q = np.asarray(q, dtype=np.complex128)
    if np.isscalar(q):
        q = complex(q)
    eps_par = _as_complex_array(eps_par)
    eps_z = _as_complex_array(eps_z)
    if eps_par.shape != eps_z.shape:
        raise ValueError("eps_par and eps_z must have the same shape (n_regions, n_omega).")
    if eps_par.ndim != 2:
        raise ValueError("eps_par and eps_z must be 2D (n_regions, n_omega).")

    n_regions, n_omega = eps_par.shape
    ratio = eps_par / eps_z  # (n_regions, n_omega)
    # Principal sqrt; we will fix branch by continuity
    raw_kappa = q * np.sqrt(ratio)  # (n_regions, n_omega)

    kappa = np.empty((n_regions, n_omega), dtype=np.complex128)
    # At ω[0]: use decay branch; if that yields non-passive V, try -κ (caller can use LayerCoulomb to test)
    kappa[:, 0] = _choose_kappa_branch(raw_kappa[:, 0])

    # For each subsequent ω, choose sign to minimize |κ[n] - κ[n-1]| (analytic continuation)
    for n in range(1, n_omega):
        prev = kappa[:, n - 1]
        cand = raw_kappa[:, n]
        alt = -cand
        keep = np.abs(cand - prev) <= np.abs(alt - prev)
        kappa[:, n] = np.where(keep, cand, alt)

    # Y = eps_z * κ (no independent branch for Y)
    Y = eps_z * kappa
    return kappa, Y


def _scalar_coth_csch(z):
    """coth(z), csch(z) of a single complex number as Python complex scalars.

    (``coth_csch_stable`` always returns 1-element arrays; numpy >= 2 refuses to
    assign those to array elements.)
    """
    c, s = coth_csch_stable(z)
    return complex(np.asarray(c).ravel()[0]), complex(np.asarray(s).ravel()[0])


def build_tridiagonal_A_diagonals(
    kappa: np.ndarray,
    Y: np.ndarray,
    d: np.ndarray,
    top_gate_d: Optional[float] = None,
    bottom_gate_d: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Build tridiagonal diagonals (lower, diag, upper) for A at fixed (q,ω).

    kappa, Y are region arrays of length n_regions = N+1.
    d is the interior-spacer thickness array of length N-1.

    Boundary conditions (v2.7)
    --------------------------
    The top and bottom boundaries can be either an open half-space (the
    default, identical to v2.6) or a grounded perfect-conductor *gate* at a
    finite distance.

      * ``top_gate_d = None``    : region 0 is a half-space ⇒ boundary term Y[0]
      * ``top_gate_d = d_t > 0`` : grounded gate at distance d_t above sheet 0
                                   ⇒ boundary term Y[0]·coth(κ[0] d_t)
      * ``bottom_gate_d`` analogous for region N below sheet N-1.

    Derivation: with a grounded gate (φ=0) at distance d_t the solution in
    region 0 is  φ(z) = Φ_0 sinh(κ_0 z)/sinh(κ_0 d_t)  (z measured from the
    gate), giving a normal-derivative contribution to the sheet equation of
    Y[0]·coth(κ_0 d_t) instead of the half-space value Y[0].  As d_t→∞,
    coth→1 and the half-space result is recovered.

    Returns
    -------
    lower, diag, upper : arrays of lengths N, N, N
        with lower[0]=0 and upper[-1]=0 by convention.
    """
    kappa = _as_complex_array(kappa)
    Y = _as_complex_array(Y)
    d = np.asarray(d, dtype=float)

    n_regions = kappa.shape[0]
    N = n_regions - 1
    if N < 1:
        raise ValueError("Need at least 1 sheet (n_regions >= 2).")

    if top_gate_d is not None and not (top_gate_d > 0):
        raise ValueError("top_gate_d must be positive (or None for half-space).")
    if bottom_gate_d is not None and not (bottom_gate_d > 0):
        raise ValueError("bottom_gate_d must be positive (or None for half-space).")

    # Effective top/bottom boundary admittance.
    #   half-space:  Y_boundary = Y[region]
    #   gate:        Y_boundary = Y[region] * coth(kappa[region] * d_gate)
    if top_gate_d is None:
        Y_top = Y[0]
    else:
        coth_top, _ = _scalar_coth_csch(kappa[0] * float(top_gate_d))
        Y_top = Y[0] * coth_top
    if bottom_gate_d is None:
        Y_bot = Y[N]
    else:
        coth_bot, _ = _scalar_coth_csch(kappa[N] * float(bottom_gate_d))
        Y_bot = Y[N] * coth_bot
    # Make sure scalars (coth returns 1-element arrays)
    Y_top = np.asarray(Y_top).reshape(-1)[0]
    Y_bot = np.asarray(Y_bot).reshape(-1)[0]

    if N == 1:
        if d.size != 0:
            raise ValueError("For N=1, d must be empty.")
        # Single sheet: A = [Y_top + Y_bot]
        lower = np.zeros(1, dtype=np.complex128)
        upper = np.zeros(1, dtype=np.complex128)
        diag = np.array([Y_top + Y_bot], dtype=np.complex128)
        return lower, diag, upper

    if d.size != N - 1:
        raise ValueError(f"Expected len(d) = N-1 = {N-1}, got {d.size}.")
    if np.any(d <= 0):
        raise ValueError("All spacer thicknesses d must be positive.")

    lower = np.zeros(N, dtype=np.complex128)
    upper = np.zeros(N, dtype=np.complex128)
    diag = np.zeros(N, dtype=np.complex128)

    # Helper: region j (1..N-1) is spacer between sheets j and j+1 in 0-based sheets.
    # For sheet index ℓ (0-based):
    #   above spacer is region ℓ (for ℓ>=1) with thickness d[ℓ-1]
    #   below spacer is region ℓ+1 (for ℓ<=N-2) with thickness d[ℓ]
    # Outer boundaries (regions 0 and N) are either half-spaces (Y[0], Y[N])
    # or gates (Y_top, Y_bot) as computed above.

    # Top sheet ℓ=0 uses Y_top (boundary) and spacer region 1 with thickness d0
    z = kappa[1] * d[0]
    coth, csch = _scalar_coth_csch(z)
    diag[0] = Y_top + Y[1] * coth
    upper[0] = -Y[1] * csch

    # Interior sheets
    for ell in range(1, N - 1):
        z_up = kappa[ell] * d[ell - 1]
        coth_up, csch_up = _scalar_coth_csch(z_up)
        z_dn = kappa[ell + 1] * d[ell]
        coth_dn, csch_dn = _scalar_coth_csch(z_dn)

        diag[ell] = Y[ell] * coth_up + Y[ell + 1] * coth_dn
        lower[ell] = -Y[ell] * csch_up
        upper[ell] = -Y[ell + 1] * csch_dn

    # Bottom sheet ℓ=N-1 uses Y_bot (boundary) and spacer region N-1 with thickness d[N-2]
    z = kappa[N - 1] * d[N - 2]
    coth, csch = _scalar_coth_csch(z)
    diag[N - 1] = Y_bot + Y[N - 1] * coth
    lower[N - 1] = -Y[N - 1] * csch

    return lower, diag, upper


_THOMAS_SINGULAR_TOL: float = 1e-30

def tridiagonal_lu_factor(
    lower: np.ndarray, diag: np.ndarray, upper: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    LU factorization of tridiagonal matrix A with diagonals (lower, diag, upper).
    Returns (c, dtilde) for Thomas algorithm where:
      c[i] is modified upper diag
      dtilde[i] is modified main diag
    Suitable for multiple RHS solves.

    Raises LinAlgError if a pivot is smaller than _THOMAS_SINGULAR_TOL.
    """
    lower = _as_complex_array(lower)
    diag = _as_complex_array(diag)
    upper = _as_complex_array(upper)
    n = diag.size
    c = np.zeros(n, dtype=np.complex128)
    dtilde = np.zeros(n, dtype=np.complex128)

    dtilde[0] = diag[0]
    if np.abs(dtilde[0]) < _THOMAS_SINGULAR_TOL:
        raise np.linalg.LinAlgError("Singular tridiagonal matrix at row 0.")
    c[0] = upper[0] / dtilde[0]

    for i in range(1, n):
        dtilde[i] = diag[i] - lower[i] * c[i - 1]
        if np.abs(dtilde[i]) < _THOMAS_SINGULAR_TOL:
            raise np.linalg.LinAlgError(f"Singular tridiagonal matrix at row {i}.")
        c[i] = upper[i] / dtilde[i] if i < n - 1 else 0.0

    return c, dtilde


def tridiagonal_solve_factorized(
    lower: np.ndarray, c: np.ndarray, dtilde: np.ndarray, B: np.ndarray
) -> np.ndarray:
    """
    Solve A X = B for tridiagonal A using pre-factorized Thomas data (c, dtilde).
    B can be vector (n,) or matrix (n, nrhs).
    """
    lower = _as_complex_array(lower)
    c = _as_complex_array(c)
    dtilde = _as_complex_array(dtilde)
    B = _as_complex_array(B)

    n = dtilde.size
    if B.ndim == 1:
        Bm = B.reshape(n, 1)
    else:
        if B.shape[0] != n:
            raise ValueError("B has incompatible shape.")
        Bm = B

    # Forward substitution for modified RHS
    Y = np.zeros_like(Bm, dtype=np.complex128)
    Y[0] = Bm[0] / dtilde[0]
    for i in range(1, n):
        Y[i] = (Bm[i] - lower[i] * Y[i - 1]) / dtilde[i]

    # Back substitution
    X = np.zeros_like(Y, dtype=np.complex128)
    X[-1] = Y[-1]
    for i in range(n - 2, -1, -1):
        X[i] = Y[i] - c[i] * X[i + 1]

    return X[:, 0] if B.ndim == 1 else X


def tridiagonal_inverse(lower: np.ndarray, diag: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """
    Compute A^{-1} explicitly by solving against the identity using a single LU factorization.
    Complexity O(N^2), suitable for moderate N.
    """
    n = diag.size
    c, dtilde = tridiagonal_lu_factor(lower, diag, upper)
    I = np.eye(n, dtype=np.complex128)
    invA = tridiagonal_solve_factorized(lower, c, dtilde, I)
    return invA


@dataclass
class LayerCoulomb:
    """
    Layer-resolved Coulomb calculator.

    Parameters
    ----------
    n_layers : int
        Number of 2D sheets N.
    d : sequence of float
        Spacer thicknesses (length N-1). For N=1, pass empty list/tuple/array.
    epsilon_static : optional
        Isotropic permittivity. Either:
          - scalar (uniform in all regions), or
          - sequence length N+1 giving per-region ε (regions 0..N).
        Mutually exclusive with epsilon_uniaxial / epsilon_dispersive.
    epsilon_uniaxial : optional
        Uniaxial permittivity specified as a pair (eps_par, eps_z), where each is
        either scalar (uniform in all regions) or sequence length N+1.
        Mutually exclusive with epsilon_static / epsilon_dispersive.
    epsilon_dispersive : optional
        Callable returning (eps_par, eps_z) given (region_index, omega).
        Signature: epsilon_dispersive(region: int, omega: complex|float) -> (complex, complex)
        For isotropic dispersive, return same value for eps_par and eps_z.
        Mutually exclusive with epsilon_static / epsilon_uniaxial.
    top_gate_d : optional float (v2.7)
        Distance (nm) from the top sheet (sheet 0) to a grounded metallic
        gate above the stack.  ``None`` (default) means an open half-space
        above (identical to v2.6).  The dielectric of region 0 still applies
        between the gate and sheet 0.
    bottom_gate_d : optional float (v2.7)
        Distance (nm) from the bottom sheet (sheet N-1) to a grounded
        metallic gate below the stack.  ``None`` (default) means an open
        half-space below.

    Notes (v2.7)
    ------------
    With a grounded gate the boundary admittance Y of the corresponding
    outer region is replaced  Y -> Y·coth(κ·d_gate)  in the A-matrix.
    Dual-gate single sheet reproduces  V(q) = (2π e²/εq) tanh(q d).
    Single gate reproduces V(q) = (2π e²/εq)(1 - e^{-2 q d}).
    """
    n_layers: int
    d: Sequence[float]
    epsilon_static: Optional[Union[float, Sequence[complex]]] = None
    epsilon_uniaxial: Optional[Tuple[Union[float, Sequence[complex]], Union[float, Sequence[complex]]]] = None
    epsilon_dispersive: Optional[Callable[[int, complex], Tuple[complex, complex]]] = None
    top_gate_d: Optional[float] = None
    bottom_gate_d: Optional[float] = None

    _cache: Dict[Tuple, np.ndarray] = field(default_factory=dict, init=False, repr=False)
    _cache_max: int = field(default=256, init=False, repr=False)

    def __post_init__(self):
        self.n_layers = int(self.n_layers)
        if self.n_layers < 1:
            raise ValueError("n_layers must be >= 1.")
        if self.top_gate_d is not None:
            self.top_gate_d = float(self.top_gate_d)
            if self.top_gate_d <= 0:
                raise ValueError("top_gate_d must be positive (or None for half-space).")
        if self.bottom_gate_d is not None:
            self.bottom_gate_d = float(self.bottom_gate_d)
            if self.bottom_gate_d <= 0:
                raise ValueError("bottom_gate_d must be positive (or None for half-space).")
        self.d = np.asarray(self.d, dtype=float)
        if self.n_layers == 1:
            self.d = np.asarray([], dtype=float)
        else:
            if self.d.size != self.n_layers - 1:
                raise ValueError(f"len(d) must be N-1={self.n_layers-1}, got {self.d.size}.")
            if np.any(self.d <= 0):
                raise ValueError("All spacer thicknesses must be positive.")

        # exactly one epsilon specification
        specs = [self.epsilon_static is not None, self.epsilon_uniaxial is not None, self.epsilon_dispersive is not None]
        if sum(specs) != 1:
            raise ValueError("Provide exactly one of: epsilon_static, epsilon_uniaxial, epsilon_dispersive.")

        self.n_regions = self.n_layers + 1

        # normalize epsilon providers to a callable get_eps(region, omega)->(eps_par, eps_z)
        if self.epsilon_static is not None:
            eps = self.epsilon_static
            if np.isscalar(eps):
                eps_arr = np.full(self.n_regions, complex(eps), dtype=np.complex128)
            else:
                eps_arr = _as_complex_array(eps)
                if eps_arr.size != self.n_regions:
                    raise ValueError(f"epsilon_static must have length N+1={self.n_regions}.")
            def _get(region: int, omega: complex) -> Tuple[complex, complex]:
                val = eps_arr[region]
                return val, val
            self._get_eps = _get

        elif self.epsilon_uniaxial is not None:
            eps_par, eps_z = self.epsilon_uniaxial
            if np.isscalar(eps_par):
                eps_par_arr = np.full(self.n_regions, complex(eps_par), dtype=np.complex128)
            else:
                eps_par_arr = _as_complex_array(eps_par)
                if eps_par_arr.size != self.n_regions:
                    raise ValueError(f"epsilon_uniaxial eps_par must have length N+1={self.n_regions}.")
            if np.isscalar(eps_z):
                eps_z_arr = np.full(self.n_regions, complex(eps_z), dtype=np.complex128)
            else:
                eps_z_arr = _as_complex_array(eps_z)
                if eps_z_arr.size != self.n_regions:
                    raise ValueError(f"epsilon_uniaxial eps_z must have length N+1={self.n_regions}.")
            def _get(region: int, omega: complex) -> Tuple[complex, complex]:
                return eps_par_arr[region], eps_z_arr[region]
            self._get_eps = _get

        else:
            if not callable(self.epsilon_dispersive):
                raise ValueError("epsilon_dispersive must be callable.")
            self._get_eps = self.epsilon_dispersive

    def eps_arrays(self, omega: Optional[Union[float, complex]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Return eps_par and eps_z arrays (length N+1) evaluated at omega (default 0)."""
        w = 0.0 if omega is None else omega
        eps_par = np.empty(self.n_regions, dtype=np.complex128)
        eps_z = np.empty(self.n_regions, dtype=np.complex128)
        for r in range(self.n_regions):
            ep, ez = self._get_eps(r, w)
            eps_par[r] = complex(ep)
            eps_z[r] = complex(ez)
        return eps_par, eps_z

    def eps_arrays_over_omega(
        self, omega_array: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return eps_par and eps_z with shape (n_regions, n_omega) for branch tracking."""
        omega_array = np.asarray(omega_array, dtype=np.complex128).ravel()
        n_omega = omega_array.size
        eps_par = np.empty((self.n_regions, n_omega), dtype=np.complex128)
        eps_z = np.empty((self.n_regions, n_omega), dtype=np.complex128)
        for r in range(self.n_regions):
            for o in range(n_omega):
                ep, ez = self._get_eps(r, omega_array[o])
                eps_par[r, o] = complex(ep)
                eps_z[r, o] = complex(ez)
        return eps_par, eps_z

    def A_diagonals(self, q: float, omega: Optional[Union[float, complex]] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return tridiagonal diagonals of A(q,omega)."""
        if q <= 0:
            raise ValueError("q must be positive.")
        eps_par, eps_z = self.eps_arrays(omega=omega)
        kappa, Y = compute_kappa_Y(q=q, eps_par=eps_par, eps_z=eps_z)
        return build_tridiagonal_A_diagonals(
            kappa=kappa, Y=Y, d=self.d,
            top_gate_d=self.top_gate_d, bottom_gate_d=self.bottom_gate_d,
        )

    def A_matrix(self, q: float, omega: Optional[Union[float, complex]] = None) -> np.ndarray:
        """Return the full A(q,omega) as an (N,N) dense matrix (mainly for debugging)."""
        lower, diag, upper = self.A_diagonals(q=q, omega=omega)
        N = self.n_layers
        A = np.zeros((N, N), dtype=np.complex128)
        np.fill_diagonal(A, diag)
        for i in range(1, N):
            A[i, i-1] = lower[i]
        for i in range(0, N-1):
            A[i, i+1] = upper[i]
        return A

    def A_inverse(self, q: float, omega: Optional[Union[float, complex]] = None) -> np.ndarray:
        """Return A(q,omega)^{-1} (cached)."""
        w = 0.0 if omega is None else complex(omega)
        key = (float(q), w)
        cached = self._cache.get(key)
        if cached is not None:
            return cached.copy()
        lower, diag, upper = self.A_diagonals(q=q, omega=omega)
        inv = tridiagonal_inverse(lower, diag, upper)
        if len(self._cache) >= self._cache_max:
            self._cache.pop(next(iter(self._cache)))
        self._cache[key] = inv
        return inv.copy()

    def clear_cache(self):
        """Discard all cached A-inverse results."""
        self._cache.clear()

    def coulomb_matrix(self, q: float, omega: Optional[Union[float, complex]] = None) -> np.ndarray:
        """
        Return the *physical* Coulomb kernel V(q,omega) in eV·nm:
            V(q,omega) = (4π e^2) A(q,omega)^{-1}.
        """
        A_inv = self.A_inverse(q=q, omega=omega)
        return fourpi_e2() * A_inv

    def coulomb_matrix_from_kappa_Y(
        self, q: float, kappa: np.ndarray, Y: np.ndarray
    ) -> np.ndarray:
        """
        Return V(q) = (4π e^2) A^{-1} from given κ and Y (e.g. from branch-tracked values).
        kappa, Y must have shape (n_regions,) with n_regions = N+1.
        """
        lower, diag, upper = build_tridiagonal_A_diagonals(
            kappa=kappa, Y=Y, d=self.d,
            top_gate_d=self.top_gate_d, bottom_gate_d=self.bottom_gate_d,
        )
        A_inv = tridiagonal_inverse(lower, diag, upper)
        return fourpi_e2() * A_inv

    def coulomb_matrix_uniform_formula(self, q: float, epsilon: float) -> np.ndarray:
        """
        Uniform 3D dielectric formula (matches Layer_Helpers.py for uniform epsilon):
            V_ij(q) = (2π e^2)/(ε q) * exp(-q |z_i - z_j|).

        This is provided as a check/benchmark for the OPEN (no-gate) half-space
        geometry only.  It does NOT include gate boundary conditions; compare
        against ``coulomb_matrix`` only when top_gate_d = bottom_gate_d = None.
        """
        if q <= 0:
            raise ValueError("q must be positive.")
        eps = float(epsilon)
        N = self.n_layers
        z = np.zeros(N, dtype=float)
        for k in range(1, N):
            z[k] = z[k-1] + self.d[k-1]
        dist = np.abs(z[:, None] - z[None, :])
        return (TWOPI_E2_EV_NM / (eps * q)) * np.exp(-q * dist)

    def diagonalize(self, q: float, omega: Optional[Union[float, complex]] = None, mode: str = "hermitian_part"):
        """
        Diagonalize the *physical* V(q,omega) unless otherwise specified.

        mode:
          - "hermitian_part" (default): diagonalize 0.5*(V + V†), returns real eigenvalues and orthonormal eigenvectors
          - "full_complex": diagonalize V (eig), returns complex eigenvalues/eigenvectors
          - "real_part": diagonalize Re(V), returns real eigenvalues/vectors
        """
        V = self.coulomb_matrix(q=q, omega=omega)
        if mode == "hermitian_part":
            VH = 0.5 * (V + V.conj().T)
            w, U = np.linalg.eigh(VH)
            return w, U
        if mode == "real_part":
            w, U = np.linalg.eigh(np.real(V))
            return w, U
        if mode == "full_complex":
            w, U = np.linalg.eig(V)
            return w, U
        raise ValueError("mode must be one of: hermitian_part, full_complex, real_part.")



def capped_slab_admittance(q, kappa_s, Y_s, d: float, eps_cap: float):
    r"""Boundary admittance of a slab (kappa_s, Y_s = eps_z kappa_s, thickness d) on a half-space eps_cap.

        Y_eff = Y_s (Y_c + Y_s tanh kappa_s d) / (Y_s + Y_c tanh kappa_s d),   Y_c = eps_cap q

    Limits: d -> inf gives Y_s (half-space of slab material), d -> 0 gives Y_c,
    and Y_c -> inf (metal) gives the gate result Y_s coth(kappa_s d).  Y_eff is
    even in kappa_s, so it does not depend on the kappa branch inside
    Reststrahlen bands.
    """
    Y_c = eps_cap * q
    t = np.tanh(kappa_s * d)
    return Y_s * (Y_c + Y_s * t) / (Y_s + Y_c * t)


@dataclass
class CappedLayerCoulomb(LayerCoulomb):
    """``LayerCoulomb`` whose top region is a finite slab with a half-space above it.

    Region 0 (above sheet 0) has thickness ``top_slab_d`` and is capped by a static
    isotropic half-space ``top_cap_eps`` (1 = vacuum).  Only the top boundary
    admittance of A changes, so gates, dispersive eps and branch-tracked kappa(w)
    work as in :class:`LayerCoulomb`.
    """
    top_slab_d: Optional[float] = None
    top_cap_eps: float = 1.0

    def __post_init__(self):
        super().__post_init__()
        if self.top_slab_d is not None:
            if self.top_gate_d is not None:
                raise ValueError("top_slab_d and top_gate_d are mutually exclusive.")
            self.top_slab_d = float(self.top_slab_d)
            if self.top_slab_d <= 0:
                raise ValueError("top_slab_d must be positive (or None for a half-space).")
        if not self.top_cap_eps > 0:
            raise ValueError("top_cap_eps must be positive.")

    def _diagonals(self, q: float, kappa: np.ndarray, Y: np.ndarray):
        lower, diag, upper = build_tridiagonal_A_diagonals(
            kappa=kappa, Y=Y, d=self.d, top_gate_d=self.top_gate_d,
            bottom_gate_d=self.bottom_gate_d)
        if self.top_slab_d is not None:
            diag[0] += capped_slab_admittance(q, kappa[0], Y[0], self.top_slab_d,
                                              self.top_cap_eps) - Y[0]
        return lower, diag, upper

    def A_diagonals(self, q: float, omega=None):
        if q <= 0:
            raise ValueError("q must be positive.")
        eps_par, eps_z = self.eps_arrays(omega=omega)
        kappa, Y = compute_kappa_Y(q=q, eps_par=eps_par, eps_z=eps_z)
        return self._diagonals(q, kappa, Y)

    def coulomb_matrix_from_kappa_Y(self, q: float, kappa: np.ndarray, Y: np.ndarray) -> np.ndarray:
        lower, diag, upper = self._diagonals(q, np.asarray(kappa, np.complex128),
                                             np.asarray(Y, np.complex128))
        return fourpi_e2() * tridiagonal_inverse(lower, diag, upper)


def _self_test():
    print("LayerCoulomb v2.7 (gated) self-test:")

    # isotropic single-layer in uniform medium
    eps = 5.0
    lc = LayerCoulomb(n_layers=1, d=[], epsilon_static=[eps, eps])
    for q in [0.05, 0.3, 2.0]:
        V = lc.coulomb_matrix(q)[0, 0]
        V_ref = (TWOPI_E2_EV_NM / (eps * q))
        rel = abs((V - V_ref) / V_ref)
        assert rel < 1e-10, f"single-layer isotropic mismatch at q={q}: {rel}"
    print("  isotropic single-layer: OK")

    # ------------------------------------------------------------------
    # v2.7 GATE tests
    # ------------------------------------------------------------------
    # G1) Dual-gate single sheet:  V(q) = (2π e²/εq) tanh(q d)
    eps_g = 3.0
    d_g = 4.0
    lc_dg = LayerCoulomb(n_layers=1, d=[], epsilon_static=[eps_g, eps_g],
                         top_gate_d=d_g, bottom_gate_d=d_g)
    for q in [0.02, 0.2, 1.0, 3.0]:
        V = lc_dg.coulomb_matrix(q)[0, 0]
        V_ref = (TWOPI_E2_EV_NM / (eps_g * q)) * np.tanh(q * d_g)
        rel = abs((V - V_ref) / V_ref)
        assert rel < 1e-10, f"dual-gate single sheet mismatch q={q}: {rel}"
    print("  dual-gate single sheet  V=(2πe²/εq)tanh(qd): OK")

    # G2) Single (top) gate single sheet: V(q) = (2π e²/εq)(1 - e^{-2 q d})
    lc_tg = LayerCoulomb(n_layers=1, d=[], epsilon_static=[eps_g, eps_g],
                         top_gate_d=d_g)
    for q in [0.02, 0.2, 1.0, 3.0]:
        V = lc_tg.coulomb_matrix(q)[0, 0]
        V_ref = (TWOPI_E2_EV_NM / (eps_g * q)) * (1.0 - np.exp(-2.0 * q * d_g))
        rel = abs((V - V_ref) / V_ref)
        assert rel < 1e-10, f"single-gate single sheet mismatch q={q}: {rel}"
    print("  single-gate single sheet  V=(2πe²/εq)(1-e^{-2qd}): OK")

    # G3) Large d_gate limit -> half-space (v2.6) value
    lc_open = LayerCoulomb(n_layers=1, d=[], epsilon_static=[eps_g, eps_g])
    lc_far = LayerCoulomb(n_layers=1, d=[], epsilon_static=[eps_g, eps_g],
                          top_gate_d=200.0, bottom_gate_d=200.0)
    for q in [0.05, 0.5, 2.0]:
        V_open = lc_open.coulomb_matrix(q)[0, 0]
        V_far = lc_far.coulomb_matrix(q)[0, 0]
        assert abs((V_far - V_open) / V_open) < 1e-8, "large-d gate != half-space"
    print("  d_gate -> ∞ recovers half-space: OK")

    # G4) Dual-gate N=2: V is symmetric, real, positive-definite; and the
    #     symmetric-mode eigenvalue is the dual-gate single-sheet value with
    #     an effective coupling.  Just check structure + finiteness here.
    lc_g2 = LayerCoulomb(n_layers=2, d=[2.0], epsilon_static=[eps_g] * 3,
                         top_gate_d=3.0, bottom_gate_d=3.0)
    for q in [0.05, 0.5, 1.5]:
        V = lc_g2.coulomb_matrix(q)
        assert np.allclose(V, V.T, atol=1e-12), "gated N=2 V not symmetric"
        w = np.linalg.eigvalsh(np.real(V))
        assert np.all(w > 0), f"gated N=2 V not PD at q={q}: {w}"
        # gating must reduce the interaction relative to the open stack
        V_open2 = LayerCoulomb(n_layers=2, d=[2.0], epsilon_static=[eps_g]*3).coulomb_matrix(q)
        assert np.max(np.abs(V)) < np.max(np.abs(V_open2)) + 1e-12, \
            "gating should not increase |V|"
    print("  dual-gate N=2: symmetric, PD, screened vs open: OK")

    # uniaxial single-layer uniform
    eps_par, eps_z = 4.0, 9.0
    lc = LayerCoulomb(n_layers=1, d=[], epsilon_uniaxial=([eps_par, eps_par], [eps_z, eps_z]))
    for q in [0.07, 0.8]:
        V = lc.coulomb_matrix(q)[0, 0]
        V_ref = (TWOPI_E2_EV_NM / (q * np.sqrt(eps_par * eps_z)))
        rel = abs((V - V_ref) / V_ref)
        assert rel < 1e-10, f"single-layer uniaxial mismatch at q={q}: {rel}"
    print("  uniaxial single-layer: OK")

    # N=2 uniform medium should match uniform formula exp(-qd)
    eps = 7.0
    d = [3.2]
    lc = LayerCoulomb(n_layers=2, d=d, epsilon_static=[eps, eps, eps])
    for q in [0.03, 0.4, 1.1]:
        V = lc.coulomb_matrix(q)
        V_ref = lc.coulomb_matrix_uniform_formula(q, epsilon=eps)
        rel = np.max(np.abs((V - V_ref) / V_ref))
        assert rel < 1e-10, f"N=2 uniform mismatch at q={q}: {rel}"
    print("  N=2 uniform medium: OK")

    # Hyperbolic stability spot-check (via coth_csch_stable)
    zvals = np.array([1e-9+0j, 1e-3+0.2j, 5+0j, 60+0j, -60+0j], dtype=np.complex128)
    coth, csch = coth_csch_stable(zvals)
    assert np.all(np.isfinite(coth)), "coth produced non-finite"
    assert np.all(np.isfinite(csch)), "csch produced non-finite"
    print("  coth_csch_stable (legacy spot-check): OK")

    # --- coth_csch_stable targeted tests (expm1-based stable evaluation) ---

    # A) Consistency with direct numpy in moderate regime
    np.random.seed(42)
    z_mod = np.random.randn(50) * 2.5 + 1j * np.random.randn(50) * 2.5
    coth_np = np.cosh(z_mod) / np.sinh(z_mod)
    csch_np = 1.0 / np.sinh(z_mod)
    coth_s, csch_s = coth_csch_stable(z_mod)
    rel_coth = np.abs(coth_s - coth_np) / (np.abs(coth_np) + 1e-20)
    rel_csch = np.abs(csch_s - csch_np) / (np.abs(csch_np) + 1e-20)
    assert np.max(rel_coth) < 1e-12 and np.max(rel_csch) < 1e-12, \
        f"moderate regime: max rel_coth={np.max(rel_coth):.2e}, rel_csch={np.max(rel_csch):.2e}"
    print("  coth_csch_stable test A (moderate z vs numpy): OK")

    # B) Large Re(z): no overflow, coth≈1, csch≈2*exp(-z)
    z_large = 100.0 + 0.3j
    coth_b, csch_b = coth_csch_stable(np.array([z_large]))
    assert np.isfinite(coth_b.flat[0]) and np.isfinite(csch_b.flat[0]), "large z: non-finite"
    assert np.abs(coth_b.flat[0] - 1.0) < 1e-14, f"large z: coth≈1 got {coth_b.flat[0]}"
    assert np.abs(csch_b.flat[0] - 2.0 * np.exp(-z_large)) < 1e-14 * np.abs(np.exp(-z_large)), \
        f"large z: csch≈2*exp(-z) mismatch"
    print("  coth_csch_stable test B (large Re(z)): OK")

    # C) Small |z|: finite, dominated by 1/z
    z_small = 1e-9 + 1e-9j
    coth_c, csch_c = coth_csch_stable(np.array([z_small]))
    assert np.isfinite(coth_c.flat[0]) and np.isfinite(csch_c.flat[0]), "small z: non-finite"
    ref_coth = 1.0 / z_small + z_small / 3.0
    ref_csch = 1.0 / z_small - z_small / 6.0
    # relative tolerance: with Z_SMALL_SERIES = 0 the expm1-based branch is used,
    # whose relative error is ~1e-16 / |z| ... still far below 1e-6 here
    assert np.abs(coth_c.flat[0] - ref_coth) < 1e-6 * np.abs(ref_coth), "small z: coth mismatch"
    assert np.abs(csch_c.flat[0] - ref_csch) < 1e-6 * np.abs(ref_csch), "small z: csch mismatch"
    print("  coth_csch_stable test C (small |z|): OK")

    # D) N=2 uniform Coulomb: V12 = (2π e²)/(ε q) exp(-q d), finite for extreme qd
    eps_d = 4.0
    d_d = 2.0
    lc_d = LayerCoulomb(n_layers=2, d=[d_d], epsilon_static=[eps_d, eps_d, eps_d])
    for q_val in [1e-3, 1e-1, 1.0, 10.0, 50.0, 100.0]:
        V = lc_d.coulomb_matrix(q=q_val)
        assert np.all(np.isfinite(V)), f"N=2 q={q_val}: V has non-finite"
        V12_ref = (TWOPI_E2_EV_NM / (eps_d * q_val)) * np.exp(-q_val * d_d)
        rel = np.abs(V[0, 1] - V12_ref) / (np.abs(V12_ref) + 1e-30)
        assert rel < 1e-10, f"N=2 uniform V12 at q={q_val}: rel err {rel}"
    print("  coth_csch_stable test D (N=2 uniform, extreme qd): OK")

    # D2) N=2, epsilon = ['hBN','hBN','hBN'] (uniaxial): verify A-matrix V vs user analytic formula.
    #     User formula (η = sqrt(ε_par/ε_z), κ = η q, Y = εz κ):
    #     V11 = (2π e²)/(η q εz) * exp(-dηq/2)*(exp(-dηq/2)+exp(dηq/2)) = (2π e²)/Y * (1 + exp(-κd))
    #     V12 = V21 = (2π e²)/(η q εz) * exp(-d η q) = (2π e²)/Y * exp(-κd),  V22 = V11.
    #     A-matrix gives V11 = V22 = 2π e²/Y (no (1+exp(-κd)) factor); V12 = V21 match user formula.
    eps_par_hbn, eps_z_hbn = 6.41, 3.0  # hBN static (ω=0) from Lorentz defaults
    d_n2 = 5.0
    lc_hbn = LayerCoulomb(
        n_layers=2,
        d=[d_n2],
        epsilon_uniaxial=(
            [eps_par_hbn] * 3,
            [eps_z_hbn] * 3,
        ),
    )
    eta = np.sqrt(eps_par_hbn / eps_z_hbn)
    for q_val in [0.1, 0.5, 1.0]:
        V = lc_hbn.coulomb_matrix(q=q_val)
        kappa = q_val * eta
        Y = eps_z_hbn * kappa
        # User analytic formulas
        pref = TWOPI_E2_EV_NM / (eta * q_val * eps_z_hbn)
        V11_user = pref * np.exp(-0.5 * d_n2 * eta * q_val) * (
            np.exp(-0.5 * d_n2 * eta * q_val) + np.exp(0.5 * d_n2 * eta * q_val)
        )
        V12_user = pref * np.exp(-d_n2 * eta * q_val)
        assert np.allclose(V[0, 1], V12_user, atol=1e-12, rtol=1e-10), (
            f"N=2 hBN V12: A-matrix {V[0,1]} vs user {V12_user}"
        )
        assert np.allclose(V[1, 0], V12_user, atol=1e-12, rtol=1e-10), "N=2 hBN V21"
        # Diagonals: user formula has extra factor (1+exp(-κd)); A-matrix gives 2π e²/Y
        assert np.allclose(V[0, 0], TWOPI_E2_EV_NM / Y, atol=1e-12, rtol=1e-10), "N=2 hBN V11 = 2πe²/Y"
        assert np.allclose(V[1, 1], TWOPI_E2_EV_NM / Y, atol=1e-12, rtol=1e-10), "N=2 hBN V22 = 2πe²/Y"
        assert np.allclose(
            V11_user, (TWOPI_E2_EV_NM / Y) * (1.0 + np.exp(-kappa * d_n2)), atol=1e-12
        ), "user V11 = (2πe²/Y)*(1+exp(-κd))"
    print("  N=2 all-hBN (uniaxial): V12/V21 = user formula; V11/V22 = 2πe²/Y (user has (1+exp(-κd))): OK")

    # E) compute_kappa_Y_tracked: matches compute_kappa_Y at ω[0]; continuity
    n_r, n_o = 3, 8
    ep_par = np.random.randn(n_r, n_o) + 1.0 + 1j * (np.random.randn(n_r, n_o) * 0.1)
    ep_z = np.random.randn(n_r, n_o) + 1.0 + 1j * (np.random.randn(n_r, n_o) * 0.1)
    k_t, Y_t = compute_kappa_Y_tracked(0.5, ep_par, ep_z)
    k0, Y0 = compute_kappa_Y(0.5, ep_par[:, 0], ep_z[:, 0])
    assert np.allclose(k_t[:, 0], k0), "tracked κ at ω[0] should match compute_kappa_Y"
    assert np.allclose(Y_t[:, 0], Y0), "tracked Y at ω[0] should match compute_kappa_Y"
    for n in range(1, n_o):
        dk = np.abs(k_t[:, n] - k_t[:, n - 1])
        dk_alt = np.abs(k_t[:, n] + k_t[:, n - 1])
        assert np.all(dk <= dk_alt + 1e-14), "tracked κ should be continuous (min jump)"
    print("  compute_kappa_Y_tracked: OK")

    print("All tests passed.")


if __name__ == "__main__":
    _self_test()
