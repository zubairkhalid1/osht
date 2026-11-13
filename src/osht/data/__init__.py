"""Data shipped with the package: stored ring placements and a test signal."""

from __future__ import annotations

from importlib import resources

import numpy as np

__all__ = ["earth_topography"]


def earth_topography(L=None):
    """Spherical harmonic coefficients of the Earth topography (real signal, band-limit 128).

    The coefficients distributed with the NSHT package (``flm_earthtopo.mat``),
    band-limited at ``L = 128`` and satisfying the conjugate symmetry of a
    real signal.

    Parameters
    ----------
    L : int, optional
        Band-limit at which to return the coefficients (``L <= 128``;
        default 128).  Lower values truncate the expansion.

    Returns
    -------
    flm : ndarray of complex, shape (L**2,)
    L : int
    """
    with resources.files("osht.data").joinpath("earth_topography_flm.npz").open("rb") as fh:
        data = np.load(fh)
        flm = data["flm"]
        L0 = int(data["L"])
    if L is None:
        L = L0
    L = int(L)
    if L < 1 or L > L0:
        raise ValueError(f"require 1 <= L <= {L0}")
    return flm[:L * L].copy(), L
