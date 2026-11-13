"""Wigner d-matrix planes at ``pi/2`` and the weights of the equiangular back-end."""

from __future__ import annotations

import numpy as np

__all__ = ["trapani_plane", "reflected_weights"]


def trapani_plane(dl, L, el):
    """Advance the Wigner plane ``Delta^el = d^el(pi/2)`` from ``el - 1`` to ``el``.

    The plane is stored as ``dl[m + L - 1, mm + L - 1] = d^el_{m, mm}(pi/2)``
    for ``-el <= m, mm <= el`` in a ``(2L-1, 2L-1)`` array, the layout of
    ``dl_trapani_full`` in the SSHT/EQSHT packages of J. McEwen, of which
    this is a vectorised rewrite (Trapani and Navaza, 2006).

    Parameters
    ----------
    dl : ndarray, shape (2L-1, 2L-1)
        The plane for ``el - 1`` (ignored for ``el = 0``); modified in place.
    L : int
        Harmonic band-limit.
    el : int
        Degree.

    Returns
    -------
    ndarray
        The same array, now holding the plane for ``el``.

    Examples
    --------
    >>> dl = np.zeros((2 * L - 1, 2 * L - 1))
    >>> for el in range(L):
    ...     dl = trapani_plane(dl, L, el)
    """
    o = L - 1
    if el == 0:
        dl[o, o] = 1.0
        return dl
    signs = (-1.0) ** np.arange(L + 1)
    # eighth plane: m = 0..el, mm = 0..m  (eqs. (9), (10) of Trapani and Navaza)
    dmm = np.empty(el + 1)
    dmm[0] = -np.sqrt((2 * el - 1) / (2 * el)) * dl[el - 1 + o, o]
    mm = np.arange(1, el + 1)
    dmm[1:] = (np.sqrt(el) / np.sqrt(2.0) * np.sqrt(2 * el - 1)
               / np.sqrt(el + mm) / np.sqrt(el + mm - 1) * dl[el - 1 + o, mm - 1 + o])
    dl[el + o, o:o + el + 1] = dmm
    # eq. (11): recursion downwards in m for all mm <= m at once
    mmv = np.arange(el + 1)
    m = el - 1
    if m >= 0:
        cols = mmv[:m + 1] + o
        dl[m + o, cols] = 2 * mmv[:m + 1] / np.sqrt(el - m) / np.sqrt(el + m + 1) * dl[m + 1 + o, cols]
    for m in range(el - 2, -1, -1):
        cols = mmv[:m + 1] + o
        t1 = 2 * mmv[:m + 1] / np.sqrt(el - m) / np.sqrt(el + m + 1) * dl[m + 1 + o, cols]
        t2 = (np.sqrt(el - m - 1) * np.sqrt(el + m + 2) / np.sqrt(el - m) / np.sqrt(el + m + 1)
              * dl[m + 2 + o, cols])
        dl[m + o, cols] = t1 - t2
    # diagonal symmetry (quarter): dl[m, mm] = s_m s_mm dl[mm, m] for mm > m
    sg = signs[:el + 1]
    blk = np.tril(dl[o:o + el + 1, o:o + el + 1])
    blk = blk + (sg[:, None] * sg[None, :]) * blk.T - np.diag(np.diag(blk))
    dl[o:o + el + 1, o:o + el + 1] = blk
    # symmetry in m (half): dl[-m, mm] = s_el s_mm dl[m, mm]
    dl[o - el:o, o:o + el + 1] = signs[el] * sg[None, :] * dl[o + el:o:-1, o:o + el + 1]
    # symmetry in mm (full plane): dl[m, -mm] = s_el s_|m| dl[m, mm]
    msg = signs[np.abs(np.arange(-el, el + 1))]
    dl[o - el:o + el + 1, o - el:o] = signs[el] * msg[:, None] * dl[o - el:o + el + 1, o + el:o:-1]
    return dl


def reflected_weights(mm):
    """Weights ``w(mm) = int_0^pi sin(theta) exp(i mm theta) dtheta`` of McEwen and Wiaux.

    ``w = 2 / (1 - mm**2)`` for even ``mm``, ``+-i pi / 2`` for ``mm = +-1``
    and ``0`` for the other odd ``mm``.

    Parameters
    ----------
    mm : array_like of int

    Returns
    -------
    ndarray of complex, same shape as ``mm``
    """
    mm = np.asarray(mm)
    w = np.zeros(mm.shape, dtype=complex)
    even = (mm % 2 == 0)
    w[even] = 2.0 / (1 - mm[even].astype(float) ** 2)
    w[mm == 1] = 1j * np.pi / 2
    w[mm == -1] = -1j * np.pi / 2
    return w
