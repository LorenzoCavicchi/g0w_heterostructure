#!/usr/bin/env python3
r"""
dI/dV(V, theta) of an SLG tip | hBN | TBG sample quantum twisting microscope.

Continuum model of TBG (pybandstructure), layer-resolved RPA screening of the
three-sheet stack (both valleys), band-matrix full-frequency G0W self-energy at
every momentum of the tip trajectory k_theta = O(theta) K_top^lab - Gamma^lab,
matrix Dyson spectral function, Wei et al. Eq. (33) with the unfolded tunneling
amplitude, and d^2I/dV^2.  Writes ``<out>.npz`` and ``<out>.png``.

    python scripts/run_tbg_qtm.py --theta-tbg 1.2 --n-sites 18 --out tbg_qtm

``--band-diagonal`` keeps only Sigma_{lambda lambda} (band-diagonal approximation).
Notes: docs/notes.pdf, Secs. 3-7.  Requires pybandstructure (see README).
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from g0w import qtm  # noqa: E402
from g0w.models import tbg_model as tm, tbg_stack as tst, tbg_self_energy as tse  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--theta-tbg", type=float, default=1.20, help="TBG twist angle [deg]")
    ap.add_argument("--uab", type=float, default=117.0, help="interlayer AB hopping [meV]")
    ap.add_argument("--uaa-ratio", type=float, default=0.7)
    ap.add_argument("--n-sites", type=int, default=18, help="moire mesh N_s x N_s")
    ap.add_argument("--n-bands", type=int, default=4, help="bands per side of neutrality")
    ap.add_argument("--filling", type=float, default=0.0)
    ap.add_argument("--eta", type=float, default=None, help="broadening [meV] (default: mesh heuristic)")
    ap.add_argument("--eta-qp", type=float, default=0.5, help="extrinsic Dyson width [meV]")
    ap.add_argument("--d-tip", type=float, default=1.0, help="tip--top-layer distance (hBN) [nm]")
    ap.add_argument("--n-tip", type=float, default=0.0, help="tip density [cm^-2]")
    ap.add_argument("--omega-max", type=float, default=160.0, help="top of the W frequency grid [meV]")
    ap.add_argument("--n-omega", type=int, default=320)
    ap.add_argument("--theta-min", type=float, default=-1.5)
    ap.add_argument("--theta-max", type=float, default=2.5)
    ap.add_argument("--n-theta", type=int, default=161)
    ap.add_argument("--emax", type=float, default=150.0, help="bias window +-emax [meV]")
    ap.add_argument("--n-eps", type=int, default=601)
    ap.add_argument("--smooth", type=float, default=1.0, help="Gaussian smoothing before d/dV [meV]")
    ap.add_argument("--band-diagonal", action="store_true", help="band-diagonal approximation for Sigma")
    ap.add_argument("--out", default="tbg_qtm")
    ap.add_argument("--no-plot", action="store_true")
    a = ap.parse_args(argv)

    params = tm.TBGParams(theta_deg=a.theta_tbg, uAB=a.uab * 1e-3, uAA_ratio=a.uaa_ratio, n_sites=a.n_sites,
                          n_bands=a.n_bands, filling=a.filling,
                          eta_override=None if a.eta is None else a.eta * 1e-3)
    t0 = time.time()
    tbg = tm.build_tbg(params, verbose=True)
    lh = tm.build_layer_helper(tbg, verbose=False)
    eta = params.eta
    omega = np.linspace(2 * eta, a.omega_max * 1e-3, a.n_omega)
    chi = tm.chi0_tbg(lh, omega + 1j * eta, verbose=False)
    cfg = tst.TBGSLGConfig(tbg=params, d_tip=a.d_tip, n_tip=a.n_tip * 1e-14)
    _, _, W = tst.build_stack_eps_W(cfg, np.asarray(tbg["q_sample"].coords), omega, chi, verbose=False)
    S = tse.sample_spectral_density(W)
    eps_v = np.linspace(-a.emax, a.emax, a.n_eps) * 1e-3
    kernel = tse.build_kernel_tbg(tbg, S, omega, eps_v, eta_self=eta)
    qw = tse.q_weights_tbg(tbg)
    print(f"band structure, form factors, W and kernel: {time.time()-t0:.1f} s  ({cfg.label()})")

    theta = np.linspace(a.theta_min, a.theta_max, a.n_theta)
    traj = qtm.tbg_trajectory(tbg, theta)
    print(f"trajectory: {np.unique(traj['k_index']).size} distinct mesh points, "
          f"max snapping error {traj['snap_error'].max()/traj['mesh_spacing']:.2f} spacings")
    t0 = time.time()
    spec = qtm.tbg_spectral_along_trajectory(tbg, lh, traj, None, None, qw, eps_v, kernel=kernel,
                                             band_diagonal=a.band_diagonal, eta_qp=a.eta_qp * 1e-3)
    res = qtm.tbg_didv(tbg, traj, spec)
    didv = res["didv"]
    d2 = qtm.bias_derivative(qtm.smooth_bias(didv, eps_v, a.smooth * 1e-3), eps_v)
    print(f"self-energies along the trajectory: {time.time()-t0:.1f} s")

    np.savez_compressed(a.out + ".npz", theta_deg=theta, eV=eps_v, didv=didv, d2IdV2=d2, Lambda=res["Lambda"],
                        k_ext=traj["k_ext"], k_index=traj["k_index"], G_int=traj["G_int"],
                        xi=spec["xi"], inverse=spec["inverse"], band_diagonal=a.band_diagonal, config=cfg.label())
    print(f"wrote {a.out}.npz")
    if not a.no_plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
        ax[0].pcolormesh(theta, eps_v * 1e3, np.log10(didv.T + 1e-3), shading="auto", cmap="inferno",
                         vmin=-2.5, vmax=2.5)
        ax[0].set_title(r"log$_{10}$ $dI/dV/e^2\Gamma$" + (" (band-diagonal)" if a.band_diagonal else ""), fontsize=10)
        lim = np.percentile(np.abs(d2), 97)
        ax[1].pcolormesh(theta, eps_v * 1e3, d2.T, shading="auto", cmap="RdBu_r", vmin=-lim, vmax=lim)
        ax[1].set_title(r"$d^2I/dV^2$", fontsize=10)
        for x in ax:
            for t, lab in ((0, r"$\kappa_{top}$"), (a.theta_tbg, r"$\kappa_{bot}$"), (-a.theta_tbg, r"$\gamma'$")):
                x.axvline(t, color="w", lw=0.5, ls=":")
                x.text(t, 0.93 * a.emax, lab, color="w", ha="center", fontsize=8)
        for x in ax:
            x.set_xlabel(r"$\theta$ (deg)")
        ax[0].set_ylabel(r"$eV$ (meV)")
        fig.tight_layout()
        fig.savefig(a.out + ".png", dpi=130)
        print(f"wrote {a.out}.png")


if __name__ == "__main__":
    main()
