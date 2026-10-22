# -*- coding: utf-8 -*-
r"""
Physical constants and unit conventions used throughout :mod:`g0w`.

Units
-----
* energies, frequencies, temperatures : eV   (hbar = 1, k_B = 1)
* lengths : nm,   momenta : nm^-1
* 2D Coulomb kernel V(q) : eV nm^2   (Fourier transform per unit area)
* polarizability chi0 : nm^-2 eV^-1
* electrostatics in Gaussian (cgs-ESU) form:  div(eps grad phi) = -4 pi rho,
  with e^2 = 1.4399764 eV nm  (the fine-structure constant times hbar c).
"""

import numpy as np

E2_EV_NM: float = 1.4399764            # e^2 [eV nm]
FOURPI_E2_EV_NM: float = 4.0 * np.pi * E2_EV_NM
TWOPI_E2_EV_NM: float = 2.0 * np.pi * E2_EV_NM

HBAR_EV_S: float = 6.582119569e-16      # hbar [eV s]
C_NM_PER_S: float = 2.99792458e17       # c [nm/s]
HBARC_EV_NM: float = HBAR_EV_S * C_NM_PER_S
M_ELECTRON_ENERGY_EV: float = 510998.950   # m_e c^2 [eV]
KB_EV_PER_K: float = 8.617333262e-5     # k_B [eV/K]

# graphene
GRAPHENE_A_NM: float = 0.246                            # lattice constant [nm]
GRAPHENE_K_NM_INV: float = 4.0 * np.pi / (3.0 * GRAPHENE_A_NM)   # |K| = 17.03 nm^-1
GRAPHENE_HV_EV_NM: float = 0.658                        # hbar v_F, v_F = 1.0e6 m/s
GRAPHENE_DEGENERACY: float = 4.0                        # spin x valley

# conductance / current prefactors
G0_SIEMENS: float = 7.748091729e-5                      # 2 e^2 / h [S]
E_OVER_HBAR_A_PER_EV: float = 0.5 * 7.747921729e-5      # e / hbar [A / eV]

# density conversions
CM2_TO_NM2: float = 1.0e-14                             # 1 cm^-2 = 1e-14 nm^-2
