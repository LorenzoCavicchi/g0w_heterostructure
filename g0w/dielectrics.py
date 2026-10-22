# -*- coding: utf-8 -*-
r"""
Dielectric functions of common substrates/spacers (frequencies in eV).

* ``hbn_eps_par``, ``hbn_eps_z`` -- uniaxial hBN, one Lorentz oscillator per axis
  (in-plane TO 168 meV, out-of-plane TO 94.2 meV; static 6.41 / 3.0);
* ``sio2_eps`` -- SiO2, factorized LO-TO model (3 pairs);
* ``al2o3_eps_perp``, ``al2o3_eps_par`` -- sapphire, factorized LO-TO model;
* ``epsilon_preset(name, omega)`` -> ``(eps_par, eps_z)`` for "hBN", "SiO2", "Al2O3";
* :class:`HBNModel`, :class:`EncapsulationEpsilon` -- an editable hBN model and a
  region provider for ``LayerCoulomb(epsilon_dispersive=...)``.

The same presets are accepted as strings in ``layer_helpers_G(epsilon_static=[...])``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from .layer_helpers import (  # noqa: F401
    _al2o3_eps_par as al2o3_eps_par,
    _al2o3_eps_perp as al2o3_eps_perp,
    _epsilon_preset as epsilon_preset,
    _hbn_eps_x as hbn_eps_par,
    _hbn_eps_z as hbn_eps_z,
    _lorentz_eps_from_eps0 as lorentz_epsilon,
    _sio2_eps as sio2_eps,
)


@dataclass
class HBNModel:
    """Uniaxial hBN, one Lorentz oscillator per axis (energies in eV)."""
    eps0_par: float = 6.41
    epsinf_par: float = 4.53
    omegaT_par: float = 168e-3
    gamma_par: float = 0.82e-3
    eps0_z: float = 3.0
    epsinf_z: float = 2.5
    omegaT_z: float = 94.2e-3
    gamma_z: float = 0.23e-3

    def eps_par(self, omega):
        return lorentz_epsilon(omega, self.eps0_par, self.epsinf_par, self.omegaT_par, self.gamma_par)

    def eps_z(self, omega):
        return lorentz_epsilon(omega, self.eps0_z, self.epsinf_z, self.omegaT_z, self.gamma_z)

    @property
    def omegaL_par(self) -> float:
        return self.omegaT_par * np.sqrt(self.eps0_par / self.epsinf_par)

    @property
    def omegaL_z(self) -> float:
        return self.omegaT_z * np.sqrt(self.eps0_z / self.epsinf_z)

    def static(self) -> Tuple[float, float]:
        return float(self.eps0_par), float(self.eps0_z)

    def reststrahlen_bands(self) -> Dict[str, Tuple[float, float]]:
        return {"out-of-plane": (self.omegaT_z, self.omegaL_z),
                "in-plane": (self.omegaT_par, self.omegaL_par)}


class EncapsulationEpsilon:
    """Region -> (eps_par(w), eps_z(w)): hBN in the outer regions, ``eps_inner`` in the spacers."""

    def __init__(self, hbn: HBNModel, eps_inner: float, n_regions: int):
        self.hbn = hbn
        self.eps_inner = complex(eps_inner)
        self.n_regions = int(n_regions)

    def __call__(self, region: int, omega) -> Tuple[complex, complex]:
        if region == 0 or region == self.n_regions - 1:
            return complex(self.hbn.eps_par(omega)), complex(self.hbn.eps_z(omega))
        return self.eps_inner, self.eps_inner
