# -*- coding: utf-8 -*-
r"""
Version-compatibility helpers.

* :func:`trapezoid` -- ``np.trapezoid`` (numpy >= 2) or ``np.trapz`` (numpy < 2).
* :func:`import_pybandstructure` -- imports pybandstructure after restoring the
  ``np.int``/``np.float`` aliases it still uses (removed in numpy 1.24) and
  after adapting ``scipy.linalg.eigh(..., eigvals=...)`` (removed in SciPy 1.14).
  If the environment variable ``PYBANDSTRUCTURE_PATH`` points to a source
  checkout (the folder that *contains* the ``pybandstructure`` package) that
  checkout is used instead of the installed copy.

Supported version: the PyPI release ``pybandstructure>=2.1`` (tested with 2.1),
which reproduces all results of the package.  Only the optional Hartree
self-consistency of doped TBG needs ``Band_Structure.from_hamiltonian(regularize_occupations=...)``,
available in the development version at commit ``3a785c6``
(https://gitlab.com/itorre/bandstructure-calculation); see ``PB_INSTALL_HARTREE``.
"""

from __future__ import annotations

import inspect
import os
import sys
import warnings

import numpy as np

PB_REPOSITORY = "https://gitlab.com/itorre/bandstructure-calculation"
PB_COMMIT = "3a785c659a0700c39febfce1000214229547f66a"
PB_INSTALL = 'pip install "pybandstructure>=2.1"'
PB_INSTALL_HARTREE = (f'pip install --force-reinstall --no-deps '
                      f'"pybandstructure @ git+{PB_REPOSITORY}.git@{PB_COMMIT}"')


def trapezoid(y, x=None, axis=-1):
    """Trapezoidal rule, independent of the numpy version."""
    f = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    if x is None:
        return f(y, axis=axis)
    return f(y, x, axis=axis)


def trapezoid_weights(x) -> np.ndarray:
    """Weights w_i with sum_i w_i f(x_i) = trapezoid(f, x) on a (non-uniform) grid."""
    x = np.asarray(x, dtype=float).ravel()
    w = np.zeros_like(x)
    if x.size > 1:
        d = np.diff(x)
        w[:-1] += 0.5 * d
        w[1:] += 0.5 * d
    else:
        w[:] = 1.0
    return w


def restore_numpy_aliases() -> None:
    """Restore the scalar aliases removed in numpy 1.24 (``np.int`` etc.) if missing."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, builtin in (("int", int), ("float", float), ("complex", complex), ("bool", bool)):
            try:
                getattr(np, name)
            except AttributeError:
                setattr(np, name, builtin)


def _eigh_compat(eigh):
    """Wrap ``scipy.linalg.eigh`` so that the legacy ``eigvals=(lo, hi)`` keyword works."""
    params = inspect.signature(eigh).parameters
    if "eigvals" in params:
        return eigh

    def eigh_with_eigvals(a, b=None, *args, eigvals=None, **kwargs):
        if eigvals is not None:
            kwargs["subset_by_index"] = [int(eigvals[0]), int(eigvals[1])]
        return eigh(a, b, *args, **kwargs)

    eigh_with_eigvals.__wrapped__ = eigh
    return eigh_with_eigvals


_PB = None


def import_pybandstructure(require_hartree: bool = False):
    """Import pybandstructure with the compatibility fixes above and return the module.

    Parameters
    ----------
    require_hartree : bool
        Raise if the imported version lacks ``regularize_occupations`` (needed
        for the Hartree self-consistency of doped TBG).
    """
    global _PB
    path = os.environ.get("PYBANDSTRUCTURE_PATH")
    if path and path not in sys.path:
        sys.path.insert(0, path)
    restore_numpy_aliases()
    if _PB is None:
        try:
            import pybandstructure as pb  # noqa: F401
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                f"pybandstructure could not be imported ({exc}).\n"
                f"Install it with\n    {PB_INSTALL}\n"
                "or set PYBANDSTRUCTURE_PATH to a source checkout.") from exc
        ops = sys.modules.get("pybandstructure.operators.operators")
        if ops is not None and hasattr(ops, "eigh"):
            ops.eigh = _eigh_compat(ops.eigh)
        _PB = pb
    if require_hartree and not has_hartree_support(_PB):
        raise ImportError(
            f"The pybandstructure at {_PB.__file__} has no 'regularize_occupations' option, "
            f"needed for the Hartree self-consistency.  Install\n    {PB_INSTALL_HARTREE}")
    return _PB


def has_hartree_support(pb) -> bool:
    """True if ``pb.Band_Structure.from_hamiltonian`` accepts ``regularize_occupations``."""
    try:
        return "regularize_occupations" in inspect.signature(
            pb.Band_Structure.from_hamiltonian).parameters
    except (AttributeError, ValueError, TypeError):
        return False
