# -*- coding: utf-8 -*-
"""Core: Coulomb kernel, dielectric matrix, self-energy conventions, spectral functions.

Runs with pytest or with ``python tests/run_tests.py``.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from g0w.constants import TWOPI_E2_EV_NM  # noqa: E402
from g0w.layer_coulomb import LayerCoulomb, CappedLayerCoulomb  # noqa: E402
from g0w import self_energy_ppa as ppa, self_energy_full_frequency as sff, spectral, screening  # noqa: E402


# ---------------------------------------------------------------- Coulomb kernel

def test_single_sheet_closed_forms():
    eps, d = 3.0, 4.0
    for q in (0.02, 0.2, 1.0, 3.0):
        V0 = TWOPI_E2_EV_NM / (eps * q)
        assert np.isclose(LayerCoulomb(1, [], epsilon_static=[eps, eps]).coulomb_matrix(q)[0, 0], V0, rtol=1e-10)
        Vg = LayerCoulomb(1, [], epsilon_static=[eps, eps], top_gate_d=d, bottom_gate_d=d).coulomb_matrix(q)[0, 0]
        assert np.isclose(Vg, V0 * np.tanh(q * d), rtol=1e-10)
        V1 = LayerCoulomb(1, [], epsilon_static=[eps, eps], top_gate_d=d).coulomb_matrix(q)[0, 0]
        assert np.isclose(V1, V0 * (1 - np.exp(-2 * q * d)), rtol=1e-10)


def test_uniaxial_bilayer_closed_form():
    ep, ez, d = 6.41, 3.0, 1.0
    lc = LayerCoulomb(2, [d], epsilon_uniaxial=([ep] * 3, [ez] * 3))
    for q in (0.05, 0.3, 1.5):
        V = lc.coulomb_matrix(q)
        V11 = TWOPI_E2_EV_NM / (np.sqrt(ep * ez) * q)
        assert np.allclose(V[0, 0], V11, rtol=1e-10) and np.allclose(V[1, 1], V11, rtol=1e-10)
        assert np.allclose(V[0, 1], V11 * np.exp(-q * np.sqrt(ep / ez) * d), rtol=1e-10)


def test_capped_slab_limits():
    q = 0.4
    open_ = LayerCoulomb(1, [], epsilon_static=[4.0, 4.0]).coulomb_matrix(q)[0, 0]
    thick = CappedLayerCoulomb(1, [], epsilon_static=[4.0, 4.0], top_slab_d=1e4).coulomb_matrix(q)[0, 0]
    thin = CappedLayerCoulomb(1, [], epsilon_static=[4.0, 4.0], top_slab_d=1e-7, top_cap_eps=1.0).coulomb_matrix(q)[0, 0]
    assert np.isclose(thick, open_, rtol=1e-10)
    assert np.isclose(thin, 4 * np.pi * 1.4399764 / (q * (1.0 + 4.0)), rtol=1e-5)


def test_gate_suppresses_in_phase_more_than_out_of_phase():
    q = 0.05
    o = LayerCoulomb(2, [1.0], epsilon_static=[4.0] * 3).coulomb_matrix(q).real
    g = LayerCoulomb(2, [1.0], epsilon_static=[4.0] * 3, top_gate_d=1.0, bottom_gate_d=1.0).coulomb_matrix(q).real
    r_plus = (g[0, 0] + g[0, 1]) / (o[0, 0] + o[0, 1])
    r_minus = (g[0, 0] - g[0, 1]) / (o[0, 0] - o[0, 1])
    assert r_plus < 0.2 < r_minus


# ---------------------------------------------------------------- weights and conventions

def test_radial_q_weights_measure():
    q = np.linspace(0, 2.0, 2001)
    assert np.isclose(np.sum(ppa.make_radial_q_weights(q)), 2.0 ** 2 / (4 * np.pi), rtol=1e-6)


def test_angular_quadrature_normalised():
    th, w = ppa.make_angular_quadrature(64)
    assert np.isclose(w.sum(), 1.0) and np.isclose(np.sum(w * np.cos(th)), 0.0, atol=1e-14)


def test_population_factors_zero_temperature():
    xi = np.array([-0.1, 0.1])
    Ap, Am = ppa.population_factors(0.05, xi, 0.0)
    assert np.allclose(Ap, [0, 1]) and np.allclose(Am, [1, 0])


def test_lG_flattening_consistent_between_eps_and_V():
    """W = eps^-1 V must multiply matrices in one (l, G) basis (row = l*n_G + G)."""
    rng = np.random.default_rng(1)
    N, nG, nw = 2, 3, 2
    V = rng.normal(size=(N, N, nG)); V = V + np.transpose(V, (1, 0, 2))
    Vf = ppa.coulomb_slice_to_V_full(V, n_layers=N, n_G=nG, n_omega=nw)[0]
    for l in range(N):
        for lp in range(N):
            for g in range(nG):
                assert Vf[l * nG + g, lp * nG + g] == V[l, lp, g]


# ---------------------------------------------------------------- self-energy kernels

def _toy_problem():
    rng = np.random.default_rng(0)
    n_q, n_w, N, nb = 5, 120, 2, 3
    om = np.linspace(1e-3, 0.3, n_w)
    lor = lambda w0, g: (g / np.pi) / ((om - w0) ** 2 + g ** 2)
    S = np.zeros((n_q, n_w, N, N), dtype=complex)
    for qi in range(n_q):
        v = rng.normal(size=N) + 1j * rng.normal(size=N)
        S[qi] = lor(0.05 + 0.02 * qi, 0.004)[:, None, None] * np.outer(v, v.conj())[None] + 0.01 * np.eye(N)
    densityL = rng.normal(size=(N, nb, nb, 1, n_q, 1)) + 1j * rng.normal(size=(N, nb, nb, 1, n_q, 1))
    energies = rng.uniform(-0.1, 0.1, size=(nb, n_q))
    kpq = np.arange(n_q)[:, None] * np.ones((1, 1), int)
    return om, S, densityL, energies, kpq, np.full(n_q, 0.1)


def test_band_matrix_sigma_is_passive_and_hermitian_structured():
    om, S, dL, E, kpq, w = _toy_problem()
    eps = np.linspace(-0.4, 0.4, 161)
    Sg = sff.compute_self_energy_band_matrix(dL, E, kpq, 0.0, S, om, w, 0, [0, 1, 2], eps, eta_self=2e-3)
    assert spectral.is_passive(Sg)
    A = spectral.spectral_function_matrix(eps, [0.0, 0.05, -0.05], Sg, eta_qp=1e-3)
    assert np.allclose(A, np.conj(np.swapaxes(A, 1, 2)), atol=1e-10)
    assert np.min(np.linalg.eigvalsh(A)) > -1e-9


def test_kernel_route_matches_direct_contraction():
    om, S, dL, E, kpq, w = _toy_problem()
    eps = np.linspace(-0.4, 0.4, 161)
    for T in (0.0, 0.003):
        direct = sff.compute_self_energy_band_matrix(dL, E, kpq, 0.0, S, om, w, 0, [0, 1, 2], eps, eta_self=2e-3, T=T)
        ker = sff.ResolventKernel(om, S, u_min=E.min() - eps.max(), u_max=E.max() - eps.min(), eta=2e-3, T=T)
        fast = sff.compute_self_energy_band_matrix_kernel(dL, E, kpq, 0.0, ker, w, 0, [0, 1, 2], eps)
        assert np.max(np.abs(fast - direct)) / np.max(np.abs(direct)) < 2e-3


def test_hermitian_and_elementwise_spectral_density_agree_for_symmetric_W():
    rng = np.random.default_rng(2)
    A = rng.normal(size=(4, 3, 3)) + 1j * rng.normal(size=(4, 3, 3))
    W = A + np.swapaxes(A, 1, 2)                      # complex symmetric
    assert np.allclose(sff.spectral_density_matrix(W), sff.spectral_density_matrix(W, hermitian=False))


# ---------------------------------------------------------------- spectral function

def test_spectral_function_normalised_without_interactions():
    eps = np.linspace(-2, 2, 40001)
    A = spectral.spectral_function_scalar(eps, 0.1, np.zeros_like(eps), eta_qp=1e-3)
    assert abs(spectral.spectral_weight(eps, A) - 1) < 1e-3
    Am = spectral.spectral_function_matrix(eps, [0.1, -0.2], None, eta_qp=1e-3)
    assert abs(spectral.spectral_weight(eps, np.real(np.trace(Am, axis1=1, axis2=2))) - 2) < 2e-3


def test_quasiparticle_solution_linear_sigma():
    eps = np.linspace(-0.5, 0.5, 2001)
    Sig = -0.2 * eps - 1j * 0.01                      # Re Sigma = -0.2 eps -> Z = 1/1.2, E = xi/1.2
    sol = spectral.quasiparticle_solution(eps, 0.12, Sig)
    assert np.isclose(sol["E_qp"], 0.1, atol=1e-6) and np.isclose(sol["Z"], 1 / 1.2, rtol=1e-6)
    assert np.isclose(sol["Gamma"], 2 * 0.01 / 1.2, rtol=1e-6)


# ---------------------------------------------------------------- screening utilities

def test_assemble_chi0_and_channel_split_are_exact():
    chiA = np.ones((2, 1, 1, 3)) * (-0.1)
    chiB = np.ones((2, 2, 2, 3)) * (-0.05)
    chi = screening.assemble_chi0(3, [((0,), chiA), ((1, 2), chiB)])
    assert np.all(chi[:, 0, 1:] == 0) and np.all(chi[:, 1:, 0] == 0)
    lc = LayerCoulomb(3, [1.0, 0.335], epsilon_static=[4.0, 4.0, 1.0, 4.0])
    eps, V, W = screening.screened_interaction(lc, np.array([0.1, 0.3]), np.linspace(0.01, 0.1, 3), chi, 1e-3)
    op, ac, x = screening.split_layer_channels(W, (1, 2))
    blk = np.ix_(range(2), range(3), [1, 2], [1, 2])
    assert np.allclose((op + ac + x)[blk], W[blk], atol=1e-14)
