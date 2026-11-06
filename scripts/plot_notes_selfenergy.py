r"""
Figures for the self-energy section of docs/notes.tex.

    python scripts/plot_notes_selfenergy.py quadrature   -> docs/figures/fig_quadrature.png
    python scripts/plot_notes_selfenergy.py anatomy      -> docs/figures/fig_sigma_anatomy.png
                                                           (+ docs/figures/fig_sigma_anatomy.npz)

``anatomy`` evaluates Eq. (sigma) of the notes for the conduction band of the doped
SLG sample (SLG | 1 nm hBN | SLG, both sheets at 1e12 cm^-2) at |k| = 1.5 k_F and
keeps every momentum transfer q and every term of D_m separately, so that the
figure shows which (q, omega) region of the spectral density S_SS builds -Im Sigma.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FIG = ROOT / "docs" / "figures"

plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8,
                     "figure.dpi": 150, "savefig.dpi": 200})


# ---------------------------------------------------------------------------
def quadrature():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.5))

    # (a) moire Brillouin-zone mesh, N_s = 6
    Ns = 6
    b1 = np.array([np.sqrt(3) / 2, 0.5])
    b2 = np.array([np.sqrt(3) / 2, -0.5])
    shifts = [m * b1 + n * b2 for m in (-1, 0, 1) for n in (-1, 0, 1)]
    hexv = np.array([[np.cos(j * np.pi / 3), np.sin(j * np.pi / 3)]
                     for j in range(7)]) / np.sqrt(3)

    def fold(p):
        best = min(shifts, key=lambda G: np.linalg.norm(p - G) - 1e-9 * (G[0] + 0.1 * G[1]))
        return p - best, best

    pts = np.array([fold((i / Ns) * b1 + (j / Ns) * b2)[0] for i in range(Ns) for j in range(Ns)])
    for G in shifts:
        if np.linalg.norm(G) > 0:
            ax1.plot(hexv[:, 0] + G[0], hexv[:, 1] + G[1], color="0.8", lw=0.7)
    ax1.fill(hexv[:, 0], hexv[:, 1], color="C0", alpha=0.07)
    ax1.plot(hexv[:, 0], hexv[:, 1], color="C0", lw=1.2)
    ax1.plot(pts[:, 0], pts[:, 1], "o", ms=3.2, color="C0")
    k = fold((1 / Ns) * b1 + (1 / Ns) * b2)[0]
    q = (2 / Ns) * b1 + (1 / Ns) * b2
    kq = k + q
    kq_f, G0 = fold(kq)
    arr = dict(arrowstyle="-|>", lw=1.2, mutation_scale=9)
    ax1.annotate("", kq, k, arrowprops=dict(color="C3", **arr))
    ax1.annotate("", kq_f, kq, arrowprops=dict(color="C2", ls="--", **arr))
    ax1.plot(*k, "o", ms=5, color="k")
    ax1.plot(*kq, "o", ms=5, mfc="w", mec="C3")
    ax1.plot(*kq_f, "s", ms=5, color="C2")
    ax1.text(*(k + [-0.03, 0.05]), r"$\mathbf{k}$", ha="right")
    ax1.text(*((k + kq) / 2 + [0.0, 0.04]), r"$\mathbf{q}$", color="C3", ha="center")
    ax1.text(*(kq + [0.04, 0.0]), r"$\mathbf{k}+\mathbf{q}$", color="C3", va="center")
    ax1.text(*((kq + kq_f) / 2 + [0.06, -0.04]), r"$-\mathbf{G}_0$", color="C2")
    ax1.text(*(kq_f + [-0.05, -0.04]), r"$\mathbf{k}'=\mathbf{k}+\mathbf{q}-\mathbf{G}_0$", color="C2", ha="right", va="top", fontsize=8)
    ax1.text(-0.75, 0.68, r"$w_{\mathbf{q}}=\dfrac{A_{\rm BZ}}{(2\pi)^2N_s^2}$", fontsize=9)
    ax1.set_title("(a) moiré mesh ($N_s=6$ shown)", fontsize=9)
    ax1.set_aspect("equal"); ax1.set_xlim(-0.75, 1.15); ax1.set_ylim(-0.75, 0.8); ax1.axis("off")

    # (b) radial-angular grid around an external momentum k (graphene)
    kF = 1.0
    kvec = np.array([1.5, 0.0])
    th = np.linspace(0, 2 * np.pi, 200)
    ax2.plot(kF * np.cos(th), kF * np.sin(th), color="0.5", lw=0.8, ls=":")
    ax2.text(-0.7, -1.0, r"Fermi line $|\mathbf{p}|=k_F$", color="0.4", fontsize=8, va="top")
    qs = np.array([0.15, 0.35, 0.55, 0.75])
    nth = 16
    thj = 2 * np.pi * np.arange(nth) / nth
    for qi in qs:
        ax2.plot(kvec[0] - qi * np.cos(th), kvec[1] - qi * np.sin(th), color="C0", lw=0.5, alpha=0.6)
        ax2.plot(kvec[0] - qi * np.cos(thj), kvec[1] - qi * np.sin(thj), "o", ms=2.2, color="C0")
    j = 5
    qv = qs[2] * np.array([np.cos(thj[j]), np.sin(thj[j])])
    ax2.annotate("", kvec, [0, 0], arrowprops=dict(color="k", **arr))
    ax2.annotate("", kvec - qv, kvec, arrowprops=dict(color="C3", **arr))
    ax2.annotate("", kvec - qv, [0, 0], arrowprops=dict(color="C2", ls="--", **arr))
    ax2.plot(0, 0, "k+", ms=7)
    ax2.text(-0.05, 0.08, "Dirac\npoint", fontsize=8, ha="right")
    ax2.text(0.75, 0.06, r"$\mathbf{k}$")
    ax2.text(*(kvec - qv / 2 + [0.04, 0.05]), r"$-\mathbf{q}$", color="C3")
    ax2.text(*(kvec - qv + [-0.35, -0.15]), r"$\mathbf{k}-\mathbf{q}$", color="C2")
    ax2.text(1.05, 0.95, r"$w_q=\dfrac{q\,\Delta q}{2\pi}$,  $\langle\cdot\rangle_\vartheta$: $n_\vartheta$ angles",
             fontsize=8.5)
    ax2.set_title(r"(b) radial grid (graphene)", fontsize=9)
    ax2.set_aspect("equal"); ax2.set_xlim(-1.1, 2.5); ax2.set_ylim(-1.2, 1.15); ax2.axis("off")

    fig.tight_layout()
    fig.savefig(FIG / "fig_quadrature.png")
    print("wrote", FIG / "fig_quadrature.png")


# ---------------------------------------------------------------------------
def anatomy():
    from g0w.models import slg_stack as ss, slg_self_energy as se
    from g0w.self_energy_ppa import make_radial_q_weights, make_angular_quadrature

    cfg = ss.SLGStackConfig(n_sample=0.01, n_tip=0.01, chi_on_tip=True, d_hbn=1.0, eta=1e-3)
    kF, EF, hv = cfg.k_F, cfg.E_F, cfg.hv
    n_q, n_w, n_th = 96, 480, 64
    q_grid = np.linspace(0.02, 3.0, n_q) * kF
    om = np.linspace(1e-4, 2.8 * EF, n_w)
    eps = np.linspace(-0.35, 0.35, 701)
    W = se.build_W_grid(cfg, q_grid, om)
    S = -W[:, :, 0, 0].imag / np.pi                                  # S_SS(q, w)  [eV nm^2 / eV]
    kern = se.build_sample_kernel(cfg, om, W, eps, q_grid, eta_self=1e-3)
    wq = make_radial_q_weights(q_grid)
    thetas, tw = make_angular_quadrature(n_th)
    kvec = np.array([1.5 * kF, 0.0])
    nk = np.linalg.norm(kvec)

    # per-q, per-term contributions to Sigma (conduction band, band_ext = +1)
    parts = {key: np.zeros((n_q, eps.size), complex) for key in ("e+", "h+", "h-")}
    for qi, q in enumerate(q_grid):
        px, py = kvec[0] - q * np.cos(thetas), kvec[1] - q * np.sin(thetas)
        pn = np.hypot(px, py)
        cosang = np.clip((kvec[0] * px + kvec[1] * py) / (nk * pn), -1, 1)
        for s in (+1, -1):
            xi = s * hv * pn - EF
            wF = tw * 0.5 * (1.0 + s * cosang)
            u = xi[:, None] - eps[None, :]
            P = kern._interp(kern.P, qi, u)[..., 0]
            M = kern._interp(kern.M, qi, u)[..., 0]
            occ = (xi < 0).astype(float)[:, None]
            e_term = -wq[qi] * (wF[:, None] * (1 - occ) * P).sum(0)
            h_term = -wq[qi] * (wF[:, None] * occ * M).sum(0)
            if s > 0:
                parts["e+"][qi] += e_term
                parts["h+"][qi] += h_term
            else:
                parts["h-"][qi] += h_term + e_term
    Sigma = sum(p.sum(0) for p in parts.values())
    Sig_ref = se.sigma_full_frequency_kernel(cfg, kvec, eps, q_grid, kern, wq, n_theta=n_th)[0, 0]
    print("max |decomposition - library| =", np.abs(Sigma - Sig_ref).max())
    np.savez(FIG / "fig_sigma_anatomy.npz", q=q_grid, om=om, eps=eps, S=S,
             sigma=Sigma, **{k: v for k, v in parts.items()})

    xi_k = hv * nk - EF
    i_edge = np.argmax(-Sigma.imag * (eps > xi_k + 0.02))
    eps0 = eps[i_edge]
    print(f"xi_k = {xi_k*1e3:.1f} meV, peak of -Im Sigma above xi_k at {eps0*1e3:.1f} meV")

    fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.85), gridspec_kw=dict(width_ratios=[1, 1, 1.05]))
    ax = axs[0]
    im = ax.pcolormesh(q_grid / kF, om * 1e3, np.log10(S.T + 1e-4), shading="auto", cmap="magma",
                       vmin=-2.5, vmax=1.5, rasterized=True)
    qq = q_grid
    lo = eps0 + EF - hv * (nk + qq)
    hi = np.minimum(eps0 + EF - hv * np.abs(nk - qq), eps0)
    ok = hi > np.maximum(lo, 0)
    ax.fill_between(qq[ok] / kF, np.maximum(lo[ok], 0) * 1e3, hi[ok] * 1e3, color="c", alpha=0.18, lw=0)
    ax.plot(qq[ok] / kF, hi[ok] * 1e3, "c", lw=0.8)
    ax.plot(qq[ok] / kF, np.maximum(lo[ok], 0) * 1e3, "c", lw=0.8)
    ax.set_xlabel(r"$q/k_F$"); ax.set_ylabel(r"$\omega$ (meV)")
    ax.set_title(r"(a) $\log_{10}S_{SS}(q,\omega)$", fontsize=9)
    ax.set_xlim(0, 3); ax.set_ylim(0, om.max() * 1e3)
    ax.text(1.55, 30, "allowed\nemission", color="c", fontsize=7.5, ha="left")

    ax = axs[1]
    dq = np.gradient(q_grid / kF)
    dens = -(parts["e+"] + parts["h+"] + parts["h-"]).imag / dq[:, None] * 1e3   # meV per unit q/k_F
    vmax = np.percentile(dens, 99.5)
    ax.pcolormesh(q_grid / kF, eps * 1e3, dens.T, shading="auto", cmap="viridis", vmin=0, vmax=vmax,
                  rasterized=True)
    ax.axhline(xi_k * 1e3, color="w", ls=":", lw=0.8)
    ax.axhline(eps0 * 1e3, color="c", ls="--", lw=0.8)
    ax.set_xlabel(r"$q/k_F$"); ax.set_ylabel(r"$\varepsilon$ (meV)")
    ax.set_title(r"(b) $-\partial_q\,\mathrm{Im}\,\Sigma(\mathbf{k},\varepsilon)$", fontsize=9)
    ax.set_xlim(0, 3)

    ax = axs[2]
    e = eps * 1e3
    ax.plot(e, -Sigma.imag * 1e3, "k", lw=1.2, label=r"$-\mathrm{Im}\,\Sigma$")
    ax.plot(e, -parts["e+"].sum(0).imag * 1e3, "C3", lw=0.9, label=r"electron, $s'=+$")
    ax.plot(e, -parts["h+"].sum(0).imag * 1e3, "C0", lw=0.9, label=r"hole, $s'=+$")
    ax.plot(e, -parts["h-"].sum(0).imag * 1e3, "C2", lw=0.9, label=r"hole, $s'=-$")
    ax.plot(e, Sigma.real * 1e3, color="0.5", lw=0.9, ls="--", label=r"$\mathrm{Re}\,\Sigma$")
    ax.axvline(xi_k * 1e3, color="0.6", ls=":", lw=0.8)
    ax.axvline(eps0 * 1e3, color="c", ls="--", lw=0.8)
    ax.set_xlabel(r"$\varepsilon$ (meV)"); ax.set_ylabel("meV")
    ax.set_title(r"(c) terms of $\mathcal{D}_m$", fontsize=9)
    ax.set_xlim(-350, 350)
    ax.legend(loc="upper left", frameon=False, fontsize=6.5, handlelength=1.5)
    fig.tight_layout()
    fig.savefig(FIG / "fig_sigma_anatomy.png")
    print("wrote", FIG / "fig_sigma_anatomy.png")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("quadrature", "all"):
        quadrature()
    if what in ("anatomy", "all"):
        anatomy()
