"""osht: spherical harmonic transforms on optimal-dimensionality (L**2) samples.

An O(L**3) forward spherical harmonic transform on the ``L**2`` samples of
the optimal-dimensionality sampling scheme by interpolated elimination (no
linear solves, no precomputation), with two exact back-ends, an exact
closed-form inverse, multi-pass refinement and placements of the rings
optimised for stability.

Reference: Z. Khalid, "An O(L^3) spherical harmonic transform on L^2 samples",
arXiv:2611.16441, 2026 (https://arxiv.org/abs/2611.16441).

Quick start
-----------
>>> import numpy as np, osht
>>> L = 64
>>> flm = osht.random_flm(L, rng=0)          # coefficients f_l^m at flm[l**2 + l + m]
>>> f = osht.inverse(flm, L)                 # L**2 samples, ring by ring
>>> flm_rec = osht.forward(f, L)             # interpolated elimination, two passes
>>> float(np.abs(flm_rec - flm).max()) < 1e-12
True
"""

from __future__ import annotations

from . import data, plotting
from .harmonics import elm2ind, gauss_legendre, ind2elm, legmat
from .metrics import Errors, amplification, errors, random_flm, sharmonic
from .sampling import (
    Sampling,
    cache_dir,
    cached_placements,
    equiangular_set,
    indexed_theta,
    is_equiangular,
    ordered_theta,
    ring_index,
    ring_slice,
    sampling_points,
    save_placement,
    search_kappa_placement,
    search_lambda_placement,
    stored_placements,
    theta_index,
)
from .sampling_functions import bary_weights, lambda_matrix, sampling_functions
from .transforms import backend_eq, forward, forward_lu, forward_pass, inverse, multipass
from .wigner import reflected_weights, trapani_plane

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # transforms
    "forward",
    "forward_pass",
    "inverse",
    "forward_lu",
    "multipass",
    "backend_eq",
    # sampling
    "Sampling",
    "sampling_points",
    "ordered_theta",
    "indexed_theta",
    "equiangular_set",
    "is_equiangular",
    "theta_index",
    "ring_index",
    "ring_slice",
    "stored_placements",
    "cached_placements",
    "cache_dir",
    "save_placement",
    "search_lambda_placement",
    "search_kappa_placement",
    # harmonics
    "elm2ind",
    "ind2elm",
    "legmat",
    "gauss_legendre",
    "sharmonic",
    # sampling functions
    "sampling_functions",
    "lambda_matrix",
    "bary_weights",
    # wigner
    "trapani_plane",
    "reflected_weights",
    # metrics
    "errors",
    "Errors",
    "random_flm",
    "amplification",
    # submodules
    "data",
    "plotting",
]
