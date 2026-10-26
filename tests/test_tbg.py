# -*- coding: utf-8 -*-
"""TBG sample (needs pybandstructure): model, densityL, chi0, stack, band-matrix Sigma, QTM."""
import functools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from g0w.models import tbg_model as tm, tbg_stack as tst, tbg_self_energy as tse  # noqa: E402
from g0w import qtm, spectral, self_energy_full_frequency as sff  # noqa: E402
from g0w.compat import import_pybandstructure  # noqa: E402

# These tests need pybandstructure; without it they are skipped (pytest and tests/run_tests.py).
try:
    import_pybandstructure()
    SKIP_REASON = None
except ImportError as _exc:
    SKIP_REASON = "pybandstructure not available: " + str(_exc).splitlines()[0]
try:
    import pytest
    pytestmark = pytest.mark.skipif(SKIP_REASON is not None, reason=str(SKIP_REASON))
except ImportError:
    pass


@functools.lru_cache(maxsize=None)
def _tbg(n_sites=12):
    p = tm.TBGParams(n_sites=n_sites, n_bands=4)
    tbg = tm.build_tbg(p, verbose=False)
    lh = tm.build_layer_helper(tbg, verbose=False)
    return tbg, lh


@functools.lru_cache(maxsize=None)
def _stack(n_sites=12):
    tbg, lh = _tbg(n_sites)
    p = tbg["params"]
    om = np.linspace(2 * p.eta, 0.15, 151)
    chi = tm.chi0_tbg(lh, om + 1j * p.eta, verbose=False)
    cfg = tst.TBGSLGConfig(tbg=p)
    eps, V, W = tst.build_stack_eps_W(cfg, np.asarray(tbg["q_sample"].coords), om, chi, verbose=False)
    return om, chi, W


def test_flat_bands_at_1p2_degrees():
    tbg, _ = _tbg()
    r = tm.band_regime(tbg)
    assert r["is_flat_band_regime"] and r["bandwidth"] < 0.03 and r["gap_cond"] > 0.02


def test_model_frame_dirac_points():
    tbg, _ = _tbg()
    p, fr = tbg["params"], tm.model_frame(tbg)
    assert np.isclose(np.linalg.norm(fr["K_top_lab"]), p.K) and np.isclose(np.linalg.norm(fr["K_bot_lab"]), p.K)
    assert np.isclose(np.linalg.norm(fr["K_top"] - fr["K_bot"]), p.G_moire / np.sqrt(3))
    # decoupled layers: zero-energy top-layer state at K_top
    m = tm.pb.koshino(valley=1, uAA=0, uAB=0, uAA_nl=0, uAB_nl=0, theta=p.theta_deg, hv=p.hv, a=p.a, G_max=1.0001)
    w, v = np.linalg.eigh(m["hamiltonian"](fr["K_top"]))
    i = np.argmin(np.abs(w))
    assert abs(w[i]) < 1e-12 and np.sum(np.abs(v[0:2, i]) ** 2) > 0.999


def test_density_matrix_completeness():
    tbg, lh = _tbg()
    dL = np.asarray(lh.densityL)
    q0 = int(np.argmin(np.linalg.norm(np.asarray(tbg["q_sample"].coords), axis=1)))
    comp = dL[0, :, :, :, q0, 0] + dL[1, :, :, :, q0, 0]
    assert np.max(np.abs(comp - np.eye(comp.shape[0])[:, :, None])) < 1e-10


def test_eigenvector_ordering_matches_densityL():
    """|top-layer weight| from bs.wavefunctions (4 g + 2 l + s) equals densityL[0, n, n, k, q=0]."""
    tbg, lh = _tbg()
    dL = np.asarray(lh.densityL)
    q0 = int(np.argmin(np.linalg.norm(np.asarray(tbg["q_sample"].coords), axis=1)))
    V = np.asarray(tbg["crystal"].band_structure.wavefunctions)
    for ki in (0, 7, 40):
        top = np.sum(np.abs(V.reshape(-1, 4, V.shape[1], V.shape[2])[:, 0:2, :, ki]) ** 2, axis=(0, 1))
        assert np.allclose(top, dL[0, np.arange(8), np.arange(8), ki, q0, 0].real, atol=1e-12)


def test_fast_chi0_matches_pybandstructure_polarization():
    tbg, lh = _tbg()
    p = tbg["params"]
    om = np.linspace(0.001, 0.1, 21) + 1j * p.eta
    fast = lh.chi0_block(om, 5)[:, :, :, 0, 0]
    base = super(type(lh), lh)
    for a in range(2):
        for b in range(2):
            slow = base.polarization_layered(om, 5, 0, 0, a, b)
            assert np.max(np.abs(fast[a, b] - slow)) <= 1e-12 * np.max(np.abs(slow)) + 1e-14


def test_chi0_sign_and_passivity_above_eta():
    _, chi, _ = _stack()
    tbg, _ = _tbg()
    assert np.all(np.linalg.eigvalsh(-np.moveaxis(chi[1:], -1, 1)[:, 0].real) > -1e-6)   # static: chi < 0
    tr = np.trace(chi, axis1=1, axis2=2)
    assert np.all(tr.imag[1:, 2:] <= 1e-4 * np.abs(tr.imag).max())


def test_W_symmetric_and_spectral_density_psd():
    om, _, W = _stack()
    assert np.max(np.abs(W - np.swapaxes(W, -1, -2))) < 1e-10 * np.max(np.abs(W))
    S = tse.sample_spectral_density(W, hermitian=True)
    ev = np.linalg.eigvalsh(S[1:])
    assert ev.min() > -1e-8 * np.abs(ev).max()


def test_band_matrix_sigma_diagonal_matches_original_routine():
    tbg, lh = _tbg()
    om, _, W = _stack()
    p = tbg["params"]
    bs = tbg["crystal"].band_structure
    S_el = tse.sample_spectral_density(W, hermitian=False)     # legacy -Im W / pi
    qw = tse.q_weights_tbg(tbg)
    eps = np.linspace(-0.1, 0.1, 81)
    Sm = tse.sigma_band_matrix(tbg, lh, 7, list(range(8)), eps, S_q=S_el, omega_grid=om, eta_self=p.eta)
    for n in (3, 4):
        legacy = tse.sigma_full_frequency_tbg(np.asarray(lh.densityL), bs.energies, bs.k_plus_q_ind,
                                              bs.chemical_potential, S_el, om, qw, 7, n, eps, eta_self=p.eta)
        assert np.max(np.abs(Sm[n, n] - legacy)) < 1e-12 * np.max(np.abs(legacy))
    assert spectral.is_passive(Sm)


def test_band_matrix_sigma_kernel_route_and_gauge_covariance():
    tbg, lh = _tbg()
    om, _, W = _stack()
    p = tbg["params"]
    S = tse.sample_spectral_density(W, hermitian=True)
    eps = np.linspace(-0.12, 0.12, 121)
    ker = tse.build_kernel_tbg(tbg, S, om, eps, eta_self=p.eta)
    a = tse.sigma_band_matrix(tbg, lh, 11, list(range(8)), eps, kernel=ker)
    b = tse.sigma_band_matrix(tbg, lh, 11, list(range(8)), eps, S_q=S, omega_grid=om, eta_self=p.eta)
    assert np.max(np.abs(a - b)) / np.max(np.abs(b)) < 2e-3
    # off-diagonal elements are not negligible inside the flat-band manifold
    off = np.max(np.abs(a[3, 4])) / np.max(np.abs(a[4, 4]))
    assert off > 1e-3


def test_tip_trajectory_hits_the_dirac_points():
    tbg, _ = _tbg()
    p, fr = tbg["params"], tm.model_frame(tbg)
    assert np.allclose(qtm.tbg_tip_momentum(tbg, 0.0), fr["K_top"], atol=1e-12)
    assert np.allclose(qtm.tbg_tip_momentum(tbg, p.theta_deg), fr["K_bot"], atol=1e-12)
    k = qtm.tbg_tip_momentum(tbg, -p.theta_deg)              # gamma of a neighbouring zone
    f = tm.fold_to_mesh(tbg, k)
    assert np.linalg.norm(f["k_mesh"]) < 1e-12 and f["G_int"] != (0, 0)
    legacy = qtm.tbg_tip_momentum(tbg, 0.0, legacy=True)
    assert np.linalg.norm(legacy) < 1e-12                     # the old gamma-centred trajectory


def test_trajectory_snapping_within_mesh():
    tbg, _ = _tbg(12)
    tr = qtm.tbg_trajectory(tbg, np.linspace(-1.5, 2.5, 41))
    assert np.max(tr["snap_error"]) <= tr["mesh_spacing"] / np.sqrt(3) + 1e-9   # nearest point of a triangular mesh


def test_decoupled_layers_follow_the_extended_zone_dirac_cone():
    p = tm.TBGParams(n_sites=12, n_bands=8, uAB=1e-9, uAA_ratio=0.0, eta_override=2e-3)
    tbg = tm.build_tbg(p, verbose=False)
    fr = tm.model_frame(tbg)
    th = np.linspace(-1.4, 2.2, 19)
    ev = np.linspace(-0.4, 0.4, 3201)
    tr = qtm.tbg_trajectory(tbg, th)
    d = qtm.tbg_bare_didv(tbg, tr, ev, eta_qp=2e-3)
    B = fr["basis"]
    for i in range(th.size):
        k_snap = np.asarray(tbg["k_sample"].coords)[tr["k_index"][i]] + B @ tr["G_int"][i]
        e_exp = p.hv * np.linalg.norm(k_snap - fr["K_top"])
        if e_exp > 0.38:
            continue
        assert abs(abs(ev[np.argmax(d[i])]) - e_exp) < 1.5e-3, (th[i], ev[np.argmax(d[i])], e_exp)


def test_band_matrix_didv_is_gauge_invariant_and_positive():
    tbg, lh = _tbg()
    om, _, W = _stack()
    S = tse.sample_spectral_density(W, hermitian=True)
    eps = np.linspace(-0.1, 0.1, 81)
    ker = tse.build_kernel_tbg(tbg, S, om, eps, eta_self=tbg["params"].eta)
    tr = qtm.tbg_trajectory(tbg, np.array([0.0, 0.5, 1.0]))
    spec = qtm.tbg_spectral_along_trajectory(tbg, lh, tr, None, None, tse.q_weights_tbg(tbg), eps, kernel=ker)
    d = qtm.tbg_didv(tbg, tr, spec)["didv"]
    assert np.all(d > -1e-10)
    # random band phases change Sigma_{lr} and Lambda, but not Lambda^T A Lambda^*
    phase = np.exp(1j * np.random.default_rng(0).uniform(0, 2 * np.pi, 8))
    A = spec["A"][spec["inverse"][1]]
    L = qtm.tbg_tunneling_amplitudes(tbg, tr["k_index"][1], range(8), G_int=tuple(tr["G_int"][1]))
    A2 = np.conj(phase)[None, :, None] * A * phase[None, None, :]
    L2 = L * phase                                   # u -> u e^{i phi}: Lambda -> Lambda e^{i phi}
    assert np.allclose(np.einsum("l,elr,r->e", L2, A2, np.conj(L2)), np.einsum("l,elr,r->e", L, A, np.conj(L)))


def test_constant_tunneling_is_trace_and_gauge_invariant():
    """model="constant" must give Tr A, independent of the band phases (not sum_{lr} A_lr)."""
    tbg, lh = _tbg()
    om, _, W = _stack()
    S = tse.sample_spectral_density(W, hermitian=True)
    eps = np.linspace(-0.1, 0.1, 81)
    ker = tse.build_kernel_tbg(tbg, S, om, eps, eta_self=tbg["params"].eta)
    tr = qtm.tbg_trajectory(tbg, np.array([0.0, 0.5, 1.0]))
    spec = qtm.tbg_spectral_along_trajectory(tbg, lh, tr, None, None, tse.q_weights_tbg(tbg), eps, kernel=ker)
    d = qtm.tbg_didv(tbg, tr, spec, model="constant")["didv"]
    trA = np.real(np.trace(spec["A"], axis1=2, axis2=3))[spec["inverse"]]
    assert np.allclose(d, trA)
    phase = np.exp(1j * np.random.default_rng(1).uniform(0, 2 * np.pi, spec["A"].shape[-1]))
    spec2 = dict(spec, A=np.conj(phase)[None, None, :, None] * spec["A"] * phase[None, None, None, :])
    assert np.allclose(qtm.tbg_didv(tbg, tr, spec2, model="constant")["didv"], d)


def test_valley_symmetrized_chi0_is_time_reversal_symmetric():
    """chi(q) = (1/2)[X(q) + X(-q)^T] restores W(q) = W(-q)^T; the single-valley X does not."""
    tbg, lh = _tbg()
    p = tbg["params"]
    om = np.linspace(0.005, 0.08, 16)
    mq = tm.minus_q_index(tbg)
    sel = [i for i in range(1, 40) if not mq["on_boundary"][i]][:12]
    partners = [int(mq["index"][i]) for i in sel]
    sym = tm.chi0_tbg(lh, om + 1j * p.eta, q_indices=sel + partners, verbose=False, tbg=tbg)
    raw = tm.chi0_tbg(lh, om + 1j * p.eta, q_indices=sel + partners, verbose=False, valley_symmetrize=False)
    n = len(sel)
    d_sym = np.max(np.abs(sym[:n] - np.transpose(sym[n:], (0, 2, 1, 3)))) / np.max(np.abs(sym))
    d_raw = np.max(np.abs(raw[:n] - np.transpose(raw[n:], (0, 2, 1, 3)))) / np.max(np.abs(raw))
    assert d_sym < 1e-12 and d_raw > 1e-2
    # the static long-wavelength response is unchanged (it is valley-even)
    assert np.allclose(sym[:, :, :, 0].real.sum(), raw[:, :, :, 0].real.sum(), rtol=1e-2)


def test_stack_W_is_time_reversal_symmetric():
    tbg, _ = _tbg()
    _, _, W = _stack()
    mq = tm.minus_q_index(tbg)
    ok = ~mq["on_boundary"]
    Wm = np.transpose(W[mq["index"]], (0, 1, 3, 2))
    assert np.max(np.abs(W[ok] - Wm[ok])) < 1e-10 * np.max(np.abs(W))
