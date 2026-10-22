#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
TBG sample: continuum model, band structure, layer-resolved density matrix, chi0.

(Source: ``layer_resolved/test_TBG_hBN_SLG/tbg_sample.py``; Hartree from
``tbg_hbn_encapsulated_loss/tbg_hbn_loss.py``; new: model-frame geometry,
plane-wave bookkeeping and folding used by the QTM module.)

Model
-----
One valley of the Koshino/Bistritzer-MacDonald continuum model (pybandstructure
``koshino``) in a plane-wave basis ``k + G`` (``G`` in the moire reciprocal
lattice, ``|G| <= G_max |G_moire|``).  The eigenvector of band n at k has
components ``u[4 g + 2 l + s]``: g = plane-wave index, l = 0 (top) / 1 (bottom),
s = 0 (A) / 1 (B).  In the model frame the top/bottom Dirac points of the G = 0
block sit at the moire kappa points

    K_1 = (g_M/sqrt3)(-sqrt3/2, +1/2),    K_2 = (g_M/sqrt3)(-sqrt3/2, -1/2),

and the *lab* (graphene-frame) momentum is ``p = k + Gamma_lab`` with
``Gamma_lab = gamma_point_tblg(theta)`` along +x, so that ``K_top = Gamma_lab + K_1``
and ``K_bot = Gamma_lab + K_2`` have |K| = 4 pi / 3a and sit at angles +-theta/2.

What the stack needs from the sample
------------------------------------
* ``densityL[l, n, m, ki, qi, G] = <k_ki, n| rho_l(q_qi + G) |k_ki + q_qi, m>``:
  the layer-projected form factors of the self-energy, and
* ``chi0_tbg(q, omega)``: the 2x2 layer-resolved polarizability of the sample
  alone, one diagonal block of the stack chi0 (the SLG tip is a different
  Hilbert space and is added in :mod:`g0w.models.tbg_stack`).

Units: energies eV, momenta nm^-1, lengths nm.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from ..compat import import_pybandstructure
from ..layer_helpers import LayerHelpers, layer_helpers_G

class _LazyPybandstructure:
    """Imports pybandstructure (with the compat fixes) on first attribute access."""

    def __getattr__(self, name):
        return getattr(import_pybandstructure(), name)


pb = _LazyPybandstructure()

# Layer index inside the TBG sample (as ordered in densityL / density_operators)
TBG_TOP, TBG_BOT = 0, 1


@dataclass
class TBGParams:
    r"""
    Continuum-model and mesh parameters.

    Defaults follow ``benchmark_v8zx1vhr.ipynb`` (paper Figs 4-7): theta = 1.20
    deg, uAB = 117 meV, uAA/uAB = 0.7, charge neutral.

    ``n_sites`` is the BZ mesh denominator (n_k = n_q = n_sites^2 for a
    Wigner-Seitz sample) and ``n_bands`` the number of bands kept on each side of
    charge neutrality, so 2*n_bands in total.  Memory is the binding constraint:
    densityL is (2, 2 n_bands, 2 n_bands, n_k, n_q, n_G) complex128, i.e.
    0.68 GB at (24, 4) and 3.7 GB at (30, 6).  n_bands = 4 keeps the two flat
    bands plus three remote ones per side, which covers "flat + first remote"
    with room for intermediate states.
    """
    theta_deg: float = 1.20
    a: float = 0.246
    hv: float = 197e6 / 3e8          # eV nm
    uAB: float = 0.117
    uAA_ratio: float = 0.7
    valley: int = 1
    G_max: float = 3.0001
    G_cut_off: float = 0.0001        # head only (n_G = 1)
    n_sites: int = 24
    n_bands: int = 4
    filling: float = 0.0             # charge neutral
    temperature: float = (0.026 / 300) * 5
    eta_override: Optional[float] = None   # [eV]; None = notebook heuristic

    @property
    def uAA(self) -> float:
        return self.uAA_ratio * self.uAB

    @property
    def K(self) -> float:
        return 4.0 * np.pi / (3.0 * self.a)

    @property
    def G_moire(self) -> float:
        r"""|G_moire| = 8 pi sin(theta/2) / (sqrt3 a)  [nm^-1]."""
        return abs(8.0 * np.pi * np.sin(np.deg2rad(self.theta_deg) / 2.0)
                   / (np.sqrt(3.0) * self.a))

    @property
    def alpha(self) -> float:
        r"""BM coupling alpha = uAB / (hv * k_theta); flat bands near alpha ~ 0.6."""
        return self.uAB / (19.81 * np.sin(np.deg2rad(self.theta_deg) / 2.0))

    @property
    def v_renorm(self) -> float:
        r"""Renormalised Dirac velocity hv (1-3a^2)/(1+6a^2) [eV nm] (informational)."""
        a2 = self.alpha ** 2
        return self.hv * (1.0 - 3.0 * a2) / (1.0 + 6.0 * a2)

    @property
    def eta_heuristic(self) -> float:
        r"""The notebook's mesh-tied broadening ``2 v_n |G_moire| / n_sites``.

        This is *only* sensible in the flat-band regime.  It scales as
        ``v_n |G_moire|``, and both factors blow up away from the magic angle:
        at theta = 1.20 deg it gives 0.53 meV (n_sites = 24), but at 5 deg
        ``v_n/v`` is 0.851 instead of 0.016 and ``|G_moire|`` is 4.2x larger, so
        it returns **119.9 meV** -- wider than most of the features it is meant
        to resolve.  Use ``eta_override`` there; see ``eta`` below.
        """
        return 2.0 * self.v_renorm * self.G_moire / self.n_sites

    @property
    def eta_mesh(self) -> float:
        r"""Half the heuristic, ``v_n |G_moire| / n_sites``.

        The energy a Dirac-like band sweeps across **one** mesh spacing
        ``|G_moire|/n_sites``.  Below this the Lindhard sum over a finite k-mesh
        resolves individual mesh points rather than a continuum, so chi0
        develops mesh-period ripples.  It is the natural floor on eta, and the
        heuristic above is just 2x it.
        """
        return self.v_renorm * self.G_moire / self.n_sites

    @property
    def eta(self) -> float:
        r"""Broadening used everywhere (chi0, eps, W and the Sigma denominators).

        Defaults to the notebook heuristic :attr:`eta_heuristic`; set
        ``eta_override`` to pick it explicitly.  The heuristic is a *read-only*
        property, so before ``eta_override`` existed there was no way to change
        eta at all without editing this file -- and at theta = 5 deg the
        heuristic is unusable (see :attr:`eta_heuristic`).
        """
        if self.eta_override is not None:
            return float(self.eta_override)
        return self.eta_heuristic

    def label(self) -> str:
        return (f"TBG theta={self.theta_deg}deg, uAB={self.uAB*1e3:.0f} meV, "
                f"filling={self.filling:g}, n_sites={self.n_sites}, "
                f"{2*self.n_bands} bands")


# ---------------------------------------------------------------------------
# Band structure
# ---------------------------------------------------------------------------

def build_tbg(params: Optional[TBGParams] = None,
              verbose: bool = True) -> Dict[str, Any]:
    r"""
    Build the Koshino model, BZ samples, band structure and Crystal.

    Returns a dict with ``params, model, geometry, k_sample, q_sample, G_sample,
    crystal, mu, n_bands_total``.
    """
    p = params or TBGParams()
    t0 = time.time()
    model = pb.koshino(valley=p.valley, uAA=p.uAA, uAB=p.uAB,
                       uAA_nl=0, uAB_nl=0, theta=p.theta_deg,
                       hv=p.hv, a=p.a, G_max=p.G_max)
    bv = model["reciprocal_lattice_vectors"].basis_vectors

    point_group = pb.Point_Group.trivial_group(space_dimension=2)
    geometry = pb.Crystal_Geometry(reciprocal_lattice_basis=bv,
                                   point_group=point_group)
    n = int(p.n_sites)
    Mxy, Kxy = n // 2, n // 3
    special = {r"$\Gamma$": (0, 0), r"$M$": (Mxy, 0),
               r"$M^\prime$": (0, Mxy), r"$K$": (Kxy, Kxy)}

    k_sample = pb.Unit_Cell_Sample.wigner_seitz_cell(
        basis_vectors=bv, denominator=n, special_points=special)
    q_sample = pb.Unit_Cell_Sample.wigner_seitz_cell(
        basis_vectors=bv, denominator=n, special_points=special)
    G_sample = pb.Sample.lattice_sample(
        basis_vectors=bv, cut_off=p.G_moire * p.G_cut_off)

    density = p.filling * 4.0 / geometry.unit_cell_volume
    band_structure = pb.Band_Structure.from_hamiltonian(
        k_sample=k_sample, q_sample=q_sample, G_sample=G_sample,
        hamiltonian=model["hamiltonian"],
        momentum_operator=model["momentum_operator"],
        degeneracy=4, zero_filling=p.n_bands,
        density=density, temperature=p.temperature)
    crystal = pb.Crystal(geometry=geometry, band_structure=band_structure,
                         eta=p.eta)

    n_half = 4 * len(model["reciprocal_lattice_vectors"]) // 2
    crystal.band_structure.compute_bands(
        eigvals=(n_half - p.n_bands, n_half + p.n_bands - 1))

    mu = float(crystal.band_structure.chemical_potential)
    if verbose:
        print(f"  {p.label()}", flush=True)
        print(f"  n_k = {len(k_sample)}, n_q = {len(q_sample)}, "
              f"n_G = {len(G_sample)}, G_moire = {p.G_moire:.4f} nm^-1", flush=True)
        print(f"  alpha = {p.alpha:.4f}, v_n = {p.v_renorm:.4f} eV nm, "
              f"eta = {p.eta*1e3:.3f} meV", flush=True)
        print(f"  mu = {mu*1e3:+.3f} meV   [{time.time()-t0:.1f} s]", flush=True)

    return dict(params=p, model=model, geometry=geometry, k_sample=k_sample,
                q_sample=q_sample, G_sample=G_sample, crystal=crystal, mu=mu,
                n_bands_total=2 * p.n_bands)


def band_indices(tbg: Dict[str, Any]) -> Dict[str, Any]:
    r"""
    Identify flat and remote bands by energy ordering around charge neutrality.

    With ``zero_filling = n_bands`` the computed bands run 0 .. 2 n_bands - 1 and
    charge neutrality sits between indices ``n_bands - 1`` (flat valence) and
    ``n_bands`` (flat conduction).

    Returns ``{'flat': (iv, ic), 'remote_val': [...], 'remote_cond': [...],
    'flat_plus_first_remote': [...]}``.
    """
    nb = tbg["params"].n_bands
    iv, ic = nb - 1, nb
    remote_val = list(range(nb - 1))[::-1]        # nearest remote first
    remote_cond = list(range(nb + 1, 2 * nb))
    sel = [iv, ic]
    if remote_val:
        sel.insert(0, remote_val[0])
    if remote_cond:
        sel.append(remote_cond[0])
    return dict(flat=(iv, ic), remote_val=remote_val, remote_cond=remote_cond,
                flat_plus_first_remote=sorted(sel))


def band_summary(tbg: Dict[str, Any]) -> str:
    """Bandwidths and gaps, for the report."""
    bs = tbg["crystal"].band_structure
    mu = tbg["mu"]
    idx = band_indices(tbg)
    iv, ic = idx["flat"]
    E = bs.energies
    lines = []
    for name, i in [("flat valence", iv), ("flat conduction", ic)]:
        e = E[i] - mu
        lines.append(f"    {name:16s} band {i}: "
                     f"[{e.min()*1e3:+8.3f}, {e.max()*1e3:+8.3f}] meV, "
                     f"BW = {(e.max()-e.min())*1e3:6.3f} meV")
    for name, lst in [("remote valence", idx["remote_val"]),
                      ("remote conduction", idx["remote_cond"])]:
        if lst:
            i = lst[0]
            e = E[i] - mu
            lines.append(f"    first {name:14s} band {i}: "
                         f"[{e.min()*1e3:+8.3f}, {e.max()*1e3:+8.3f}] meV")
    gap_c = (E[idx["remote_cond"][0]].min() - E[ic].max()) * 1e3 if idx["remote_cond"] else np.nan
    gap_v = (E[iv].min() - E[idx["remote_val"][0]].max()) * 1e3 if idx["remote_val"] else np.nan
    lines.append(f"    gap flat->remote: conduction {gap_c:.3f} meV, "
                 f"valence {gap_v:.3f} meV")
    return "\n".join(lines)


def band_regime(tbg: Dict[str, Any]) -> Dict[str, Any]:
    r"""
    Measure whether the sample is actually in the flat-band regime.

    ``band_indices`` *labels* bands ``n_bands - 1`` and ``n_bands`` "flat" purely
    from their position in the ordering.  That label is a definition, not a
    measurement, and away from the magic angle it is simply false:

    ==========  ==========  ==============  ==================
    theta       alpha       central BW      central->remote gap
    ==========  ==========  ==============  ==================
    1.20 deg    0.564       25 meV          +44 meV
    5.00 deg    0.135       745 meV         **-173 meV**
    ==========  ==========  ==============  ==================

    At 5 deg the two central bands are 30x wider than at 1.2 deg and *overlap*
    the remote bands in energy, so nothing about "a flat manifold separated from
    remote bands" survives.  This routine returns the numbers and a boolean so
    downstream code can branch on the measurement instead of on the label.

    Returns a dict with ``central`` (iv, ic), ``bw_val``, ``bw_cond``,
    ``gap_cond``, ``gap_val`` (negative = the bands overlap in energy),
    ``bandwidth`` (the wider of the two central bands), ``span`` (full energy
    range of every computed band, relative to mu), ``alpha`` and
    ``is_flat_band_regime``.

    The criterion is deliberately crude -- both central bands narrower than
    100 meV **and** a positive gap to the nearest remote band on both sides --
    because it only has to separate two regimes that differ by a factor 30.
    """
    p = tbg["params"]
    bs = tbg["crystal"].band_structure
    E = np.asarray(bs.energies)
    mu = float(tbg["mu"])
    nb = p.n_bands
    iv, ic = nb - 1, nb

    bw_val = float(E[iv].max() - E[iv].min())
    bw_cond = float(E[ic].max() - E[ic].min())
    gap_cond = float(E[ic + 1].min() - E[ic].max()) if ic + 1 < E.shape[0] else np.nan
    gap_val = float(E[iv].min() - E[iv - 1].max()) if iv - 1 >= 0 else np.nan

    flat = (max(bw_val, bw_cond) < 0.100
            and (np.isnan(gap_cond) or gap_cond > 0.0)
            and (np.isnan(gap_val) or gap_val > 0.0))
    return dict(central=(iv, ic), bw_val=bw_val, bw_cond=bw_cond,
                gap_cond=gap_cond, gap_val=gap_val,
                bandwidth=max(bw_val, bw_cond),
                span=(float(E.min() - mu), float(E.max() - mu)),
                alpha=float(p.alpha), is_flat_band_regime=bool(flat))


def select_bands(tbg: Dict[str, Any], n_remote: int = 1) -> Dict[str, Any]:
    r"""
    Band selection that does not assume a flat manifold exists.

    Returns the two bands straddling charge neutrality (``central``) plus
    ``n_remote`` bands on each side, as ``{'central': (iv, ic), 'selected':
    [...], 'is_flat_band_regime': bool, 'regime': <band_regime dict>}``.

    The *selection* is the same list ``band_indices(...)['flat_plus_first_remote']``
    returns at ``n_remote = 1`` -- picking the four bands nearest charge
    neutrality is right in both regimes.  What differs is the claim attached to
    it: this routine calls them "central", reports the measured bandwidth and
    gap, and never asserts they are flat or separated.  Use it (and
    ``regime['is_flat_band_regime']``) wherever the physics depends on
    flatness; ``band_indices`` remains for the 1.2 deg code paths that already
    checked the gap themselves.
    """
    p = tbg["params"]
    nb = p.n_bands
    iv, ic = nb - 1, nb
    lo = [i for i in range(max(iv - n_remote, 0), iv)]
    hi = [i for i in range(ic + 1, min(ic + 1 + n_remote, 2 * nb))]
    reg = band_regime(tbg)
    return dict(central=(iv, ic), selected=sorted(lo + [iv, ic] + hi),
                remote_val=lo[::-1], remote_cond=hi,
                is_flat_band_regime=reg["is_flat_band_regime"], regime=reg)


def regime_summary(tbg: Dict[str, Any]) -> str:
    """One-paragraph regime report, for the run logs."""
    r = band_regime(tbg)
    p = tbg["params"]
    iv, ic = r["central"]
    g = lambda x: "n/a" if np.isnan(x) else f"{x*1e3:+.1f} meV"
    return (f"    theta = {p.theta_deg:g} deg, alpha = {r['alpha']:.4f}: "
            f"{'FLAT-BAND' if r['is_flat_band_regime'] else 'NOT flat-band'} regime\n"
            f"    central bands ({iv}, {ic}): BW = {r['bw_val']*1e3:.1f} / "
            f"{r['bw_cond']*1e3:.1f} meV\n"
            f"    gap to nearest remote: conduction {g(r['gap_cond'])}, "
            f"valence {g(r['gap_val'])}"
            f"{'   (negative = bands overlap)' if (r['gap_cond'] < 0 or r['gap_val'] < 0) else ''}\n"
            f"    all computed bands span [{r['span'][0]*1e3:+.0f}, "
            f"{r['span'][1]*1e3:+.0f}] meV about mu\n"
            f"    eta = {p.eta*1e3:.2f} meV "
            f"(heuristic {p.eta_heuristic*1e3:.2f}, mesh floor {p.eta_mesh*1e3:.2f})")


# ---------------------------------------------------------------------------
# Layer-resolved density matrix
# ---------------------------------------------------------------------------

def build_layer_helper(tbg: Dict[str, Any], verbose: bool = True) -> layer_helpers_G:
    r"""
    ``layer_helpers_G`` over the two TBG layers, used **only** for ``densityL``
    and ``polarization_layered``.

    The ``epsilon_static``/``d`` given here are placeholders: the physical
    Coulomb matrix of the TBG|hBN|SLG stack is built by ``tbg_slg_stack`` over
    three sheets, and nothing downstream reads this helper's own V.
    """
    model = tbg["model"]
    G_sample = tbg["G_sample"]
    mat_top = np.kron(pb.pauli_matrix("u"), np.eye(2))
    mat_bot = np.kron(pb.pauli_matrix("l"), np.eye(2))
    ops = [[pb.build_density_operator(
                matrix=m, G=G_vec,
                reciprocal_lattice_vectors=model["reciprocal_lattice_vectors"],
                difference_vectors=None)
            for G_vec in G_sample]
           for m in (mat_top, mat_bot)]
    t0 = time.time()
    lh = LayerHelpers(crystal=tbg["crystal"], n_layers=2,
                      density_operators=ops,
                      epsilon_static=[1.0, 1.0, 1.0], d=[0.335])
    if verbose:
        print(f"  densityL {np.asarray(lh.densityL).shape}  "
              f"[{time.time()-t0:.1f} s, "
              f"{np.asarray(lh.densityL).nbytes/1e9:.2f} GB]", flush=True)
    return lh


def minus_q_index(tbg: Dict[str, Any]) -> Dict[str, np.ndarray]:
    r"""
    For every point of ``q_sample`` the index of the mesh point equivalent to -q.

    Returns ``index`` (n_q,) and ``on_boundary`` (n_q,) bool: True where -q is
    equivalent to the stored representative only up to a reciprocal-lattice
    vector G0 != 0 (Wigner-Seitz boundary points, O(1/n_sites) of the mesh).
    """
    qs = tbg["q_sample"]
    ints = np.asarray(qs.integer_coords, dtype=int)
    idx = np.zeros(len(ints), dtype=int)
    bnd = np.zeros(len(ints), dtype=bool)
    for i, X in enumerate(ints):
        d = qs(tuple(int(v) for v in -X))
        idx[i] = int(d["index"])
        bnd[i] = any(int(v) != 0 for v in d["integer_part"])
    return dict(index=idx, on_boundary=bnd)


def chi0_tbg(lh: layer_helpers_G, omega: np.ndarray, *,
             n_q: Optional[int] = None, verbose: bool = True,
             q_indices: Optional[Sequence[int]] = None,
             layers: Tuple[int, int] = (TBG_TOP, TBG_BOT),
             valley_symmetrize: bool = True,
             tbg: Optional[Dict[str, Any]] = None) -> np.ndarray:
    r"""
    Layer-resolved sample polarizability chi0_{l1 l2}(q, omega), head only, both valleys.

    ``omega`` should already carry its +i eta.  Returns ``(n_q, 2, 2, n_omega)``
    complex [nm^-2 eV^-1], negative at omega = 0 (the sign eps = 1 - V chi0 needs).

    Valleys
    -------
    The band structure is computed for valley K only and pybandstructure
    multiplies chi0 by the full degeneracy g = 4.  Time reversal maps valley K
    at q onto valley K' at -q with the layer indices exchanged,
    chi^{K'}_{ab}(q) = chi^K_{ba}(-q), so the physical two-valley response is

        chi0(q) = (g/2) [chi^K(q) + chi^K(-q)^T] = (1/2) [X(q) + X(-q)^T],   X = g chi^K.

    ``valley_symmetrize=True`` (default) returns this.  ``False`` returns X(q),
    the convention of the original scripts, which is not time-reversal
    symmetric (W(q) != W(-q)^T) because the single-valley bands are not
    inversion symmetric, E_K(k) != E_K(-k).  For the ~1/n_sites Wigner-Seitz
    boundary points where -q is stored as -q + G0, the head at the stored
    representative is used (same |q|).  Needs ``tbg`` (the dict of
    :func:`build_tbg`) for the -q map; it is taken from ``lh`` otherwise.

    ``q_indices`` restricts the calculation to a subset of the q mesh (rows in
    the order given; the -q partners are computed internally).  ``layers``
    selects the two sample sheets of ``lh``.
    """
    omega = np.asarray(omega, dtype=np.complex128).ravel()
    bs = lh.crystal.band_structure
    if q_indices is None:
        n_q = int(n_q if n_q is not None else len(bs.q_sample))
        q_list = list(range(n_q))
    else:
        q_list = [int(i) for i in q_indices]
    l_a, l_b = int(layers[0]), int(layers[1])
    if valley_symmetrize:
        if tbg is None:
            tbg = dict(q_sample=bs.q_sample)
        mq = minus_q_index(tbg)["index"]
        needed = sorted(set(q_list) | {int(mq[i]) for i in q_list})
    else:
        needed = list(q_list)
    fast = hasattr(lh, "chi0_block") and lh.crystal.geometry._point_group.cardinality == 1
    blocks: Dict[int, np.ndarray] = {}
    t0 = time.time()
    for j, qi in enumerate(needed):
        if fast:
            blk = lh.chi0_block(omega, qi)                     # [l, l', w, G, G']
            blocks[qi] = blk[np.ix_([l_a, l_b], [l_a, l_b])][:, :, :, 0, 0]
        else:
            out_q = np.zeros((2, 2, omega.size), dtype=np.complex128)
            for l1, la in enumerate((l_a, l_b)):
                for l2, lb in enumerate((l_a, l_b)):
                    out_q[l1, l2] = lh.polarization_layered(
                        omega=omega, q_index=qi, G_index=0, G1_index=0,
                        index1=la, index2=lb)
            blocks[qi] = out_q
        if verbose and (j + 1) % max(len(needed) // 8, 1) == 0:
            print(f"    chi0: {j+1}/{len(needed)} q-points [{time.time()-t0:.0f} s]", flush=True)
    out = np.zeros((len(q_list), 2, 2, omega.size), dtype=np.complex128)
    for j, qi in enumerate(q_list):
        if valley_symmetrize:
            out[j] = 0.5 * (blocks[qi] + np.transpose(blocks[int(mq[qi])], (1, 0, 2)))
        else:
            out[j] = blocks[qi]
    return out


# ---------------------------------------------------------------------------
# Model-frame geometry, plane waves and folding (used by g0w.qtm)
# ---------------------------------------------------------------------------

def model_frame(tbg: Dict[str, Any]) -> Dict[str, np.ndarray]:
    r"""
    Positions of the special points in the model frame and in the lab frame.

    Returns ``gamma_lab`` (lab position of the moire gamma point),
    ``K_top``/``K_bot`` (model-frame top/bottom Dirac points K_1, K_2) and
    ``K_top_lab``/``K_bot_lab`` (= gamma_lab + K_1/2, |.| = 4 pi/3a), all nm^-1.
    """
    import pybandstructure.models.twisted_bilayer as _tb
    p = tbg["params"]
    g = p.G_moire
    v = float(p.valley)
    K1 = v * g / np.sqrt(3.0) * np.array([-np.sqrt(3.0) / 2.0, 0.5])
    K2 = v * g / np.sqrt(3.0) * np.array([-np.sqrt(3.0) / 2.0, -0.5])
    gamma_lab = np.asarray(_tb.gamma_point_tblg(p.theta_deg, p.a, p.valley), dtype=float)
    return dict(gamma_lab=gamma_lab, K_top=K1, K_bot=K2,
                K_top_lab=gamma_lab + K1, K_bot_lab=gamma_lab + K2,
                basis=np.asarray(tbg["model"]["reciprocal_lattice_vectors"].basis_vectors, float))


def plane_waves(tbg: Dict[str, Any]) -> Dict[str, Any]:
    r"""Plane-wave basis of the Hamiltonian: integer coords, coordinates, index lookup."""
    rlv = tbg["model"]["reciprocal_lattice_vectors"]
    ints = np.asarray(rlv.integer_coords, dtype=int)
    return dict(sample=rlv, integer_coords=ints,
                coords=np.asarray(rlv.coords, dtype=float),
                index={tuple(int(x) for x in v): i for i, v in enumerate(ints)},
                n_pw=len(rlv))


def band_offset(tbg: Dict[str, Any]) -> int:
    r"""Index, in the full spectrum of H(k), of computed band 0."""
    n_half = 4 * len(tbg["model"]["reciprocal_lattice_vectors"]) // 2
    return n_half - int(tbg["params"].n_bands)


def fold_to_mesh(tbg: Dict[str, Any], k_ext: np.ndarray) -> Dict[str, Any]:
    r"""
    Decompose a model-frame momentum as ``k_ext ~ k_mesh[k_index] + G``.

    ``k_ext`` is snapped to the nearest point of the k mesh lattice
    ``{B X / n_sites}``; returns ``k_index``, ``G_int`` (integer coords of G in
    the moire basis), ``G`` (nm^-1) and ``snap_error`` = |k_ext - k_mesh - G|.
    """
    ks = tbg["k_sample"]
    B = np.asarray(tbg["model"]["reciprocal_lattice_vectors"].basis_vectors, float)
    n = int(tbg["params"].n_sites)
    k2 = np.asarray(k_ext, float).ravel()[:2]
    x = np.linalg.solve(B, k2) * n
    # nearest point of the (oblique) mesh lattice: component-wise rounding is not
    # nearest on a triangular lattice, so test the four corners of the cell
    base = np.floor(x).astype(int)
    cands = [base + np.array(c) for c in ((0, 0), (1, 0), (0, 1), (1, 1))]
    X = tuple(int(v) for v in min(cands, key=lambda c: np.linalg.norm(B @ (c / n) - k2)))
    d = ks(X)
    G_int = tuple(int(v) for v in d["integer_part"])
    G = B @ np.asarray(G_int, float)
    k_mesh = np.asarray(ks.coords[int(d["index"])], float)
    err = float(np.linalg.norm(np.asarray(k_ext, float)[:2] - k_mesh - G))
    return dict(k_index=int(d["index"]), G_int=G_int, G=G, k_mesh=k_mesh, snap_error=err)


def path_indices(tbg: Dict[str, Any], labels: Sequence[str] = ("K", "Gamma", "M", "K")) -> Dict[str, Any]:
    r"""
    k-mesh indices along a high-symmetry path of the moire BZ.

    ``labels`` from {'Gamma', 'M', "M'", 'K'}.  Returns ``indices`` (into
    ``k_sample``), the path coordinate ``x`` [nm^-1], ``vertices`` (positions of
    the labelled points in the index list) and ``labels``.
    """
    names = {"Gamma": r"$\Gamma$", "G": r"$\Gamma$", "M": r"$M$", "M'": r"$M^\prime$", "K": r"$K$"}
    c = tbg["k_sample"].contour([names[l] for l in labels])
    return dict(indices=np.asarray(c["indices"], int), x=np.asarray(c["x"], float),
                vertices=np.asarray(c["vertices"], int), labels=list(labels))


def density_of_states(tbg: Dict[str, Any], eps: np.ndarray, eta: float = 1e-3,
                      bands: Optional[Sequence[int]] = None) -> np.ndarray:
    r"""
    DOS per unit area and per energy, all flavours:  D(e) = g int d2k/(2pi)^2
    sum_n delta_eta(e - xi_nk), Lorentzian width ``eta``.  [eV^-1 nm^-2]
    """
    bs = tbg["crystal"].band_structure
    E = np.asarray(bs.energies) - float(bs.chemical_potential)
    if bands is not None:
        E = E[list(bands)]
    w = np.asarray(bs.k_sample.weights, float)
    vol = float(bs.k_sample.volume)
    eps = np.asarray(eps, float).ravel()
    L = (eta / np.pi) / ((eps[:, None, None] - E[None, :, :]) ** 2 + eta ** 2)
    return float(bs.degeneracy) * vol / (2 * np.pi) ** 2 * np.einsum("enk,k->e", L, w)


# ---------------------------------------------------------------------------
# Optional Hartree self-consistency (doped TBG; needs the pybandstructure
# development version, see g0w.compat)
# ---------------------------------------------------------------------------

def hartree_operators(tbg: Dict[str, Any]):
    r"""First-shell layer-symmetric (U_S) and antisymmetric (U_A) potential operators."""
    rlv = tbg["model"]["reciprocal_lattice_vectors"]
    shell_1 = [(1, 0), (0, -1), (-1, 1)]
    shell_2 = [(-1, 0), (0, 1), (1, -1)]
    comp_S = np.zeros(len(rlv), dtype=complex)
    comp_A = np.zeros(len(rlv), dtype=complex)
    for key in shell_1:
        comp_S[rlv[key]] = 1.0 / 6.0
        comp_A[rlv[key]] = -1.0j / 6.0
    for key in shell_2:
        comp_S[rlv[key]] = 1.0 / 6.0
        comp_A[rlv[key]] = 1.0j / 6.0
    US = pb.build_potential_hamiltonian(U=np.eye(4, dtype=complex), potential_components=comp_S,
                                        reciprocal_lattice_vectors=rlv)
    UA = pb.build_potential_hamiltonian(U=np.eye(4, dtype=complex), potential_components=comp_A,
                                        reciprocal_lattice_vectors=rlv)
    return [US, UA]


def apply_hartree(tbg: Dict[str, Any], eps_hartree: float, *,
                  initial_values=(-0.001, 0.001), tol: float = 1e-8, maxiter: int = 50,
                  verbose: bool = True) -> Dict[str, Any]:
    r"""
    Self-consistent Hartree potential on the first moire shell (doped TBG).

        V_H(r) = sum_{|G| = g_M} (2 pi e^2 / (eps_H |G|)) delta n(G) e^{i G r}

    The two first-shell components (symmetric x_S, antisymmetric x_A) are
    solved with Broyden's method via ``Band_Structure.compute_sc_bands``; the
    Hartree Hamiltonian then replaces the bare one and the bands are
    recomputed.  At charge neutrality the Hartree potential of the
    (particle-hole symmetric) BM model vanishes and this step can be skipped.

    Returns a new ``tbg`` dict (bands with Hartree) with ``hartree`` info.
    """
    from ..compat import has_hartree_support
    from ..constants import E2_EV_NM
    if not has_hartree_support(pb):
        raise ImportError("Hartree needs the pybandstructure development version (g0w.compat.PB_INSTALL_HARTREE).")
    p = tbg["params"]
    alpha = 2.0 * np.pi * E2_EV_NM / (float(eps_hartree) * p.G_moire)      # [eV nm^2]
    operators = hartree_operators(tbg)

    def transfer(x):
        return [np.real(6.0 * alpha * x[0]), np.real(6.0 * alpha * x[1])]

    it = [0]

    def callback(x, f):
        it[0] += 1
        if verbose:
            print(f"  Hartree iter {it[0]:2d}: x = [{x[0]: .6e}, {x[1]: .6e}]  |res| = {np.linalg.norm(f):.2e}",
                  flush=True)

    n_half = 4 * len(tbg["model"]["reciprocal_lattice_vectors"]) // 2
    eigvals = (n_half - p.n_bands, n_half + p.n_bands - 1)
    density = p.filling * 4.0 / tbg["geometry"].unit_cell_volume
    # The SCF needs occupations regularised against the charge-neutral reference
    # (densities measured from neutrality); chi0 must NOT use them.  So the SCF
    # runs on a separate, regularised copy and tbg["crystal"] is left untouched.
    bs0 = pb.Band_Structure.from_hamiltonian(
        k_sample=tbg["k_sample"], q_sample=tbg["q_sample"], G_sample=tbg["G_sample"],
        hamiltonian=tbg["model"]["hamiltonian"], momentum_operator=tbg["model"]["momentum_operator"],
        degeneracy=4, zero_filling=p.n_bands, density=density, temperature=p.temperature,
        regularize_occupations=True)
    bs0.compute_bands(eigvals=eigvals)
    sol = bs0.compute_sc_bands(
        operators=operators, transfer_function=transfer,
        initial_values=list(initial_values), eigvals=eigvals,
        options={"method": "broyden1", "options": {"line_search": None, "fatol": tol, "maxiter": maxiter},
                 "callback": callback})
    couplings = transfer(sol.x)
    ham = tbg["model"]["hamiltonian"]
    for c, op in zip(couplings, operators):
        ham = ham + float(np.real(c)) * op
    ham.collapse_constant_coefficients()
    bs = pb.Band_Structure.from_hamiltonian(
        k_sample=tbg["k_sample"], q_sample=tbg["q_sample"], G_sample=tbg["G_sample"],
        hamiltonian=ham, momentum_operator=tbg["model"]["momentum_operator"],
        degeneracy=4, zero_filling=p.n_bands, density=density, temperature=p.temperature)
    crystal = pb.Crystal(geometry=tbg["geometry"], band_structure=bs, eta=p.eta)
    crystal.band_structure.compute_bands(eigvals=eigvals)
    out = dict(tbg)
    model = dict(tbg["model"]); model["hamiltonian"] = ham
    out.update(model=model, crystal=crystal, mu=float(bs.chemical_potential),
               hartree=dict(eps_H=float(eps_hartree), alpha=alpha, x=np.asarray(sol.x, float),
                            couplings=[float(c) for c in couplings], iterations=it[0],
                            success=bool(getattr(sol, "success", True))))
    return out
