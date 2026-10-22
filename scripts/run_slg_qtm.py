#!/usr/bin/env python3
r"""
dI/dV(V, theta) of an SLG tip | hBN | SLG sample quantum twisting microscope.

Full-frequency G0W self-energy of the sample (both Dirac bands), Dyson spectral
function, Wei et al. Eq. (33) conductance and d^2I/dV^2.  Writes
``<out>.npz`` (theta, eV, didv, d2IdV2, A, xi) and ``<out>.png``.

    python scripts/run_slg_qtm.py --n 1e12 --d-hbn 1.0 --theta-max 1.6 --out slg_qtm

Notes: docs/notes.pdf, Secs. 5-7.  Units: n in cm^-2, d in nm, energies in meV
on the command line; eV / nm internally.
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from g0w import qtm, self_energy_ppa as ppa  # noqa: E402
from g0w.models import slg_stack as ss, slg_self_energy as se  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=float, default=1e12, help="sample and tip density [cm^-2]")
    ap.add_argument("--n-tip", type=float, default=None, help="tip density [cm^-2] (default: = --n)")
    ap.add_argument("--inert-tip", action="store_true", help="tip does not screen (chi_tip = 0)")
    ap.add_argument("--d-hbn", type=float, default=1.0, help="hBN barrier thickness [nm]")
    ap.add_argument("--eta", type=float, default=1.0, help="broadening of chi0, V, W and Sigma [meV]")
    ap.add_argument("--eta-qp", type=float, default=2.0, help="extrinsic width in the Dyson equation [meV]")
    ap.add_argument("--theta-max", type=float, default=1.6, help="largest twist [deg]")
    ap.add_argument("--n-theta", type=int, default=120)
    ap.add_argument("--emax", type=float, default=350.0, help="bias window +-emax [meV]")
    ap.add_argument("--n-eps", type=int, default=701)
    ap.add_argument("--n-q", type=int, default=96, help="radial q points on [0.02, 3] k_F")
    ap.add_argument("--n-omega", type=int, default=480)
    ap.add_argument("--smooth", type=float, default=1.5, help="Gaussian smoothing before d/dV [meV]")
    ap.add_argument("--tunneling", choices=("chiral", "constant"), default="chiral")
    ap.add_argument("--out", default="slg_qtm")
    ap.add_argument("--no-plot", action="store_true")
    a = ap.parse_args(argv)

    n = a.n * 1e-14                                   # cm^-2 -> nm^-2
    n_tip = (a.n if a.n_tip is None else a.n_tip) * 1e-14
    cfg = ss.SLGStackConfig(n_sample=n, n_tip=n_tip, chi_on_tip=not a.inert_tip, d_hbn=a.d_hbn, eta=a.eta * 1e-3)
    eps_v = np.linspace(-a.emax, a.emax, a.n_eps) * 1e-3
    theta = np.linspace(0.02, a.theta_max, a.n_theta)
    print(f"k_F = {cfg.k_F:.4f} nm^-1 (theta = {qtm.momentum_to_twist(cfg.k_F):.3f} deg), E_F = {cfg.E_F*1e3:.1f} meV")

    t0 = time.time()
    q_grid = np.linspace(0.02 * cfg.k_F, 3.0 * cfg.k_F, a.n_q)
    om = np.linspace(1e-4, 1.4 * 2 * cfg.E_F, a.n_omega)
    W = se.build_W_grid(cfg, q_grid, om)
    k_max = max(3.2 * cfg.k_F, 1.05 * qtm.twist_to_momentum(a.theta_max))
    ker = se.build_sample_kernel(cfg, om, W, eps_v, q_grid, eta_self=cfg.eta, k_max=k_max)
    qw = ppa.make_radial_q_weights(q_grid)
    print(f"W and resolvent kernel: {time.time()-t0:.1f} s")

    t0 = time.time()
    A = np.zeros((2, theta.size, eps_v.size))
    xi = np.zeros((2, theta.size))
    for ib, b in enumerate((+1, -1)):
        for it, t in enumerate(theta):
            pv = qtm.slg_probe_momentum(t)
            Sg = se.sigma_full_frequency_kernel(cfg, pv, eps_v, q_grid, ker, qw, band_ext=b)
            A[ib, it] = se.spectral_function_sample(cfg, pv, eps_v, Sg, band_ext=b, eta_qp=a.eta_qp * 1e-3)
            xi[ib, it] = se.quasiparticle_energy(cfg, pv, b)
    smap = dict(theta_deg=theta, A=A, xi=xi, bands=np.array([+1, -1]), eps_values=eps_v)
    didv = qtm.slg_didv(smap, tunneling_model=a.tunneling)
    d2 = qtm.bias_derivative(qtm.smooth_bias(didv, eps_v, a.smooth * 1e-3), eps_v)
    print(f"{2 * theta.size} self-energies: {time.time()-t0:.1f} s")

    np.savez_compressed(a.out + ".npz", theta_deg=theta, eV=eps_v, didv=didv, d2IdV2=d2, A=A, xi=xi,
                        config=str(cfg.label() if hasattr(cfg, "label") else cfg))
    print(f"wrote {a.out}.npz")
    if not a.no_plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
        ax[0].pcolormesh(theta, eps_v * 1e3, np.log10(didv.T + 1e-2), shading="auto", cmap="inferno", vmin=-2, vmax=2.3)
        for ib in range(2):
            ax[0].plot(theta, xi[ib] * 1e3, "c--", lw=0.7)
        ax[0].set_title(r"log$_{10}$ $dI/dV/e^2\Gamma$ (cyan: bare bands)", fontsize=9)
        lim = np.percentile(np.abs(d2), 98)
        ax[1].pcolormesh(theta, eps_v * 1e3, d2.T, shading="auto", cmap="RdBu_r", vmin=-lim, vmax=lim)
        ax[1].set_title(r"$d^2I/dV^2$", fontsize=9)
        for x in ax:
            x.set_xlabel(r"$\theta$ (deg)")
        ax[0].set_ylabel(r"$eV$ (meV)")
        ax[0].set_ylim(-a.emax, a.emax)
        fig.tight_layout()
        fig.savefig(a.out + ".png", dpi=130)
        print(f"wrote {a.out}.png")


if __name__ == "__main__":
    main()
