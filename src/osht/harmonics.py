"""Associated Legendre functions, Gauss-Legendre quadrature and the indexing
of harmonic coefficients.

Conventions
-----------
The scalar spherical harmonics are

    Y_lm(theta, phi) = Ptilde_l^m(theta) exp(i m phi),

orthonormal on the sphere, with the Condon-Shortley phase included in
``Ptilde_l^m`` so that ``Ptilde_l^{-m} = (-1)^m Ptilde_l^m``.  Coefficients
are stored in a vector of length ``L**2`` with ``flm[l**2 + l + m] = f_l^m``.
"""

from __future__ import annotations

import numpy as np
from scipy.special import gammaln

__all__ = [
    "elm2ind",
    "ind2elm",
    "legmat",
    "gauss_legendre",
]


# ----------------------------------------------------------------------------- indexing
def elm2ind(el, m):
    """Position of the coefficient ``f_l^m`` in the coefficient vector.

    Parameters
    ----------
    el, m : int or array_like
        Degree ``0 <= el < L`` and order ``-el <= m <= el``.

    Returns
    -------
    int or ndarray
        ``el**2 + el + m``.
    """
    el = np.asarray(el)
    m = np.asarray(m)
    return el * el + el + m


def ind2elm(ind):
    """Degree and order of the coefficient stored at position ``ind``.

    Parameters
    ----------
    ind : int or array_like
        Position in the coefficient vector.

    Returns
    -------
    (el, m) : tuple of int or ndarray
    """
    ind = np.asarray(ind)
    el = np.floor(np.sqrt(ind)).astype(int)
    m = ind - el * el - el
    return el, m


# ----------------------------------------------------------------------------- scalar Legendre
def legmat(thetas, L, m):
    """Scaled associated Legendre functions ``Ptilde_l^m(theta)`` for ``l = m..L-1``.

    Parameters
    ----------
    thetas : array_like
        Co-latitudes.
    L : int
        Harmonic band-limit.
    m : int
        Order, ``0 <= m < L``.  Negative orders follow from
        ``Ptilde_l^{-m} = (-1)^m Ptilde_l^m``.

    Returns
    -------
    ndarray, shape (L - m, len(thetas))
        ``P[l - m, i] = Ptilde_l^m(thetas[i])``.

    Notes
    -----
    The three-term recurrence in ``l`` is run on a mantissa with a separate
    logarithmic scale per sample: the seed ``sin^m(theta)`` underflows deep
    inside the forbidden region while ``Ptilde_l^m`` for larger ``l`` is
    O(1).  The mantissa is rescaled whenever it exceeds 1e150, so the
    recurrence is valid for any ``L``.  The returned values are the true
    values, with harmless underflow to zero where ``Ptilde_l^m`` is below
    the smallest normal double.  At a pole every row with ``m > 0`` is zero.
    This is the recurrence of ``iesht_legmat.m`` and ``nsht_legmat.m``.
    """
    if m < 0 or m >= L:
        raise ValueError("require 0 <= m < L")
    thetas = np.asarray(thetas, dtype=float).ravel()
    n = thetas.size
    x = np.cos(thetas)
    s = np.abs(np.sin(thetas))
    P = np.zeros((L - m, n))
    with np.errstate(divide="ignore"):
        logscale = (0.5 * np.log((2 * m + 1) / (4 * np.pi))
                    + 0.5 * (gammaln(2 * m + 1) - 2 * m * np.log(2.0) - 2 * gammaln(m + 1))
                    + m * np.log(s))
    cur = np.full(n, float((-1) ** m))
    bad = ~np.isfinite(logscale)
    cur[bad] = 0.0
    logscale[bad] = 0.0
    prev = np.zeros(n)
    BIG = 1e150
    LOGBIG = np.log(BIG)
    with np.errstate(under="ignore"):
        P[0] = cur * np.exp(logscale)
        for ell in range(m, L - 1):
            a = np.sqrt((2 * ell + 1) * (2 * ell + 3) / ((ell + 1 - m) * (ell + 1 + m)))
            if ell > 0:
                b = np.sqrt((2 * ell + 3) * (ell - m) * (ell + m)
                            / ((2 * ell - 1) * (ell + 1 - m) * (ell + 1 + m)))
            else:
                b = 0.0
            nxt = a * x * cur - b * prev
            prev, cur = cur, nxt
            big = np.abs(cur) > BIG
            if big.any():
                cur[big] /= BIG
                prev[big] /= BIG
                logscale[big] += LOGBIG
            P[ell - m + 1] = cur * np.exp(logscale)
    return P


# ----------------------------------------------------------------------------- quadrature
def gauss_legendre(n):
    """Gauss-Legendre nodes (ascending in ``x = cos theta``) and weights on [-1, 1].

    Parameters
    ----------
    n : int
        Number of nodes.

    Returns
    -------
    x, w : ndarray, shape (n,)
        Nodes and weights; the rule is exact for polynomials of degree ``2n - 1``.
    """
    x, w = np.polynomial.legendre.leggauss(int(n))
    return x, w
