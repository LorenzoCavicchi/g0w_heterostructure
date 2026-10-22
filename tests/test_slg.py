# -*- coding: utf-8 -*-
"""Graphene sample (SLG | hBN | SLG): chi0, plasmons, self-energy, QTM kinematics."""
import functools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from g0w.models import dirac_chi as cd, slg_stack as ss, slg_self_energy as se  # noqa: E402
from g0w import self_energy_ppa as ppa, qtm, spectral  # noqa: E402


@functools.lru_cache(maxsize=None)
def _grid(chi_on_tip=True, n_q=40, n_w=320):
    cfg = ss.SLGStackConfig(chi_on_tip=chi_on_tip)
    q = np.linspace(0.02 * cfg.k_F, 3 * cfg.k_F, n_q)
    om = np.linspace(1e-4, 1.4 * 2 * cfg.E_F, n_w)
    W = se.build_W_grid(cfg, q, om)
    return cfg, q, om, W, ppa.make_radial_q_weights(q)


def test_chi_static_long_wavelength_is_minus_dos():
    n = 0.01
    chi = cd.chi_dirac_phys(np.array([1e-9]), np.array([1e-4]), n)[0, 0]
    assert np.isclose(chi.real, -cd.dos_at_fermi(n), rtol=1e-3)


def test_chi_matches_brute_force_lindhard():
    n = 0.01
    for w in (0.05, 0.2):
        q = np.array([0.1, 0.25])
        ref = cd.chi_dirac_lindhard_reference(w, q, n, eta=1e-3, n_k=1000, n_theta=1000)
        cf = cd.chi_dirac_phys(np.array([w + 1e-3j]), q, n)[0]
        # the brute-force sum converges slowly at eta = 1 meV (1e-2, 6e-3, 1e-3 for n_k = 250, 500, 1000)
        assert np.max(np.abs(cf - ref)) / cd.dos_at_fermi(n) < 2e-3


def test_chi_retarded_is_passive():
    w = np.linspace(1e-3, 0.5, 200) + 1e-3j
    chi = cd.chi_dirac_phys(w, np.linspace(0.01, 0.6, 60), 0.01)
    assert np.all(chi.imag <= 1e-12)


def test_inert_tip_reproduces_single_sheet_coulomb():
    cfg = ss.SLGStackConfig(chi_on_tip=False)
    q = 0.1
    eps_q, Vq = ss.stack_eps_and_V(q, np.array([0.05]), cfg)
    W = np.linalg.inv(eps_q[0, :, :, 0, 0]) @ Vq[:, :, 0, 0]
    chi = cd.chi_dirac_phys(np.array([0.05 + 1j * cfg.eta]), np.array([q]), cfg.n_sample)[0, 0]
    V = Vq[0, 0, 0, 0]
    assert np.isclose(W[0, 0], V / (1 - V * chi), rtol=1e-10)


def test_optical_plasmon_long_wavelength_law():
    cfg = ss.SLGStackConfig(chi_on_tip=False)
    q, om, ppd = ss.build_slg_ppd(cfg, q_max=0.25 * cfg.k_F, n_q=6, n_omega=400, omega_max=0.15)
    ref = ss.plasmon_optical_long_wavelength(q, cfg)
    w = np.array([p[int(np.argmax([np.trace(r).real for r in ppd.residues[i]]))] if len(p) else np.nan
                  for i, p in enumerate(ppd.pole_frequencies)])
    good = np.isfinite(w)
    assert good.sum() >= 4 and np.nanmax(np.abs(w[good] / ref[good] - 1)) < 0.12


def test_self_energy_is_passive_and_kernel_matches_direct():
    cfg, q, om, W, qw = _grid()
    eps = np.linspace(-0.3, 0.3, 241)
    k = qtm.slg_probe_momentum(1.0)
    direct = se.sigma_full_frequency_fast(cfg, k, eps, q, om, W, qw, band_ext=+1, eta_self=1e-3, n_theta=32)
    ker = se.build_sample_kernel(cfg, om, W, eps, q, eta_self=1e-3)
    fast = se.sigma_full_frequency_kernel(cfg, k, eps, q, ker, qw, band_ext=+1, n_theta=32)
    assert spectral.is_passive(direct[0, 0]) and spectral.is_passive(fast[0, 0])
    assert np.max(np.abs(fast - direct)) / np.max(np.abs(direct)) < 1e-3
    assert np.all(fast[1, :] == 0) and np.all(fast[:, 1] == 0)          # sample block only


def test_chirality_form_factor_sum_rule():
    cfg = ss.SLGStackConfig()
    ff = se.make_dirac_form_factor(cfg, band_ext=+1)
    rng = np.random.default_rng(3)
    for _ in range(20):
        k, kq = rng.normal(size=2) * 0.2, rng.normal(size=2) * 0.2
        _, F = ff(k, kq)
        assert np.isclose(F[:, 0, 0].real.sum(), 1.0)


def test_channel_split_is_additive():
    cfg, q, om, W, qw = _grid()
    eps = np.linspace(-0.3, 0.3, 121)
    tot, op, ac = se.sigma_full_frequency_channels_kernel(cfg, qtm.slg_probe_momentum(1.2), eps, q, om, W, qw,
                                                         n_theta=32)
    assert np.max(np.abs(op + ac - tot)) / np.max(np.abs(tot)) < 1e-6


def test_quasiparticle_peak_obeys_dyson_root():
    cfg, q, om, W, qw = _grid(chi_on_tip=False)
    eps = np.linspace(-0.35, 0.35, 1401)
    ker = se.build_sample_kernel(cfg, om, W, eps, q, eta_self=1e-3)
    k = np.array([0.0, 1.8 * cfg.k_F])
    S = se.sigma_full_frequency_kernel(cfg, k, eps, q, ker, qw, band_ext=+1, n_theta=32)
    A = se.spectral_function_sample(cfg, k, eps, S, band_ext=+1, eta_qp=2e-3)
    sol = spectral.quasiparticle_solution(eps, se.quasiparticle_energy(cfg, k, +1), S[0, 0], window=0.1)
    assert abs(eps[np.argmax(A)] - sol["E_qp"]) < 2e-3
    assert 0.7 < sol["Z"] < 1.0


def test_twist_kinematics():
    for t in (0.1, 0.7, 1.5):
        assert np.isclose(np.linalg.norm(qtm.slg_probe_momentum(t)), qtm.twist_to_momentum(t), rtol=1e-12)
        assert np.isclose(qtm.momentum_to_twist(qtm.twist_to_momentum(t)), t)
    assert np.isclose(qtm.twist_to_momentum(1.0), 0.2972, atol=1e-4)


def test_slg_didv_reduces_to_spectral_function_for_constant_weights():
    smap = dict(A=np.random.default_rng(0).random((2, 3, 5)), bands=np.array([1, -1]),
                theta_deg=np.array([0.2, 0.5, 1.0]))
    assert np.allclose(qtm.slg_didv(smap, "constant"), smap["A"].sum(0))


def test_satellite_edge_of_sqrt_plasmon_matches_closed_form():
    from g0w import satellites as sat
    E_F, k_F, alpha, hv = 0.1166, 0.1772, 0.5, 0.658
    a2 = 0.5 * 4 * alpha * hv * E_F                  # (hbar w)^2 = (g/2)(e^2/eps) E_F q, e^2/eps = alpha hv
    q = np.linspace(1e-4, 1.0, 20001)
    d = sat.dispersive_edge(q, np.sqrt(a2 * q), hv)
    ref = sat.sqrt_plasmon_edge(E_F, k_F, alpha)
    assert abs(d["Delta"] - ref["Delta"]) < 1e-6 and abs(d["q_star"] - ref["q_star"]) < 1e-3
    assert abs(d["slope"] - 1.0) < 1e-2                # plasmon group velocity = electron velocity


def test_energy_loss_decomposition_reproduces_minus_im_sigma():
    from g0w import satellites as sat
    from g0w import self_energy_ppa as ppa
    cfg = ss.SLGStackConfig(n_sample=0.01, n_tip=0.01, chi_on_tip=True, d_hbn=1.0, eta=1e-3)
    q = np.linspace(0.02 * cfg.k_F, 3 * cfg.k_F, 48); om = np.linspace(1e-4, 2.8 * cfg.E_F, 480)
    W = se.build_W_grid(cfg, q, om); qw = ppa.make_radial_q_weights(q)
    S = -np.imag(W[:, :, ss.SAMPLE_LAYER, ss.SAMPLE_LAYER]) / np.pi
    p = np.array([0.0, 2.0 * cfg.k_F]); eps = np.array([0.17])
    d = sat.slg_energy_loss(cfg, p, 0.17, q, om, S, qw, n_phi=256)
    ker = se.build_sample_kernel(cfg, om, W, np.linspace(0.1, 0.25, 151), q, eta_self=2e-4)
    Sk = se.sigma_full_frequency_kernel(cfg, p, eps, q, ker, qw, band_ext=+1, n_theta=256)[0, 0, 0]
    assert abs(sat.minus_im_sigma_from_loss(d) + Sk.imag) < 0.05 * abs(Sk.imag)
