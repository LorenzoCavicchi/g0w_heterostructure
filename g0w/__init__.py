# -*- coding: utf-8 -*-
r"""
g0w -- layer-resolved RPA screening and one-shot G0W self-energy of 2D heterostructures,
with the quantum-twisting-microscope (QTM) differential conductance.

Modules
-------
constants                  physical constants, unit conventions
compat                     numpy / SciPy / pybandstructure compatibility
layer_coulomb              V(q, w) of N sheets in a stratified, gated, dispersive medium
dielectrics                dielectric presets (hBN, SiO2, Al2O3) -- re-exported from layer_helpers
layer_helpers              layer-resolved chi0 and eps from pybandstructure (layer_helpers_G)
screening                  eps = 1 - V chi0, W = eps^-1 V for a stack, loss, channel split
self_energy_ppa            plasmon-pole self-energy (poles, residues, Sigma_PPA)
self_energy_full_frequency full-frequency Sigma, band-matrix Sigma, fast resolvent kernels
spectral                   Dyson equation, spectral functions, quasiparticle analysis
qtm                        QTM kinematics and dI/dV, d2I/dV2 (SLG and TBG samples)
satellites                 plasmon-satellite energies: emission edge, energy-loss decomposition
models.dirac_chi           analytic chi0 of doped graphene
models.slg_stack           SLG|hBN|SLG junction: geometry, eps, plasmon poles
models.slg_self_energy     Dirac form factor, Sigma and A of the SLG sample
models.tbg_model           TBG continuum model (pybandstructure), densityL, chi0
models.tbg_stack           TBG|hBN|SLG three-sheet stack
models.tbg_self_energy     band-resolved Sigma and A of the TBG sample

All energies in eV, lengths in nm, momenta in nm^-1.  See docs/notes.pdf.
"""

__version__ = "1.0.0"

from . import constants, compat, layer_coulomb, screening, spectral  # noqa: F401
