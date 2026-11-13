"""Test signals, error measures and the amplification constants of a placement."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from .harmonics import elm2ind, gauss_legendre, legmat
from .sampling import ordered_theta
from .sampling_functions import bary_weights, lambda_matrix

__all__ = ["Errors", "errors", "random_flm", "amplification", "sharmonic"]


class Errors(NamedTuple):
    """Error measures between two sets of coefficients (see :func:`errors`)."""

    E_max: float
    E_mean: float
    eps: float
    SNR: float


def errors(flm_true, flm):
    """Error measures between two sets of coefficients.

    ``E_max = max |flm_true - flm|``, ``E_mean = mean |flm_true - flm|``,
    ``eps = sum |flm_true - flm|**2 / sum |flm_true|**2`` and
    ``SNR = -10 log10(eps)`` in dB.

    Parameters
    ----------
    flm_true, flm : array_like, shape (N,) or (S, N)
        Reference and computed coefficients.

    Returns
    -------
    Errors
        A named tuple ``(E_max, E_mean, eps, SNR)`` of floats, or of arrays
        of length ``S`` for a batch (one value per signal).
    """
    a = np.asarray(flm_true)
    b = np.asarray(flm)
    if a.shape != b.shape:
        raise ValueError("flm_true and flm must have the same shape")
    d = np.abs(a - b)
    E_max = d.max(axis=-1)
    E_mean = d.mean(axis=-1)
    with np.errstate(divide="ignore"):
        eps = (d ** 2).sum(axis=-1) / (np.abs(a) ** 2).sum(axis=-1)
        SNR = -10 * np.log10(eps)
    if E_max.ndim == 0:
        return Errors(float(E_max), float(E_mean), float(eps), float(SNR))
    return Errors(E_max, E_mean, eps, SNR)


def random_flm(L, rng=None, size=None, reality=False):
    """Random coefficients of a band-limited test signal.

    Real and imaginary parts uniform in ``[-1, 1]`` for ``0 <= l < L`` and
    ``|m| <= l`` (the protocol of Khalid, Kennedy and McEwen, 2014).

    Parameters
    ----------
    L : int
        Harmonic band-limit.
    rng : numpy.random.Generator or int, optional
        Generator or seed (default: a fresh generator).
    size : int, optional
        Number of signals; returns shape ``(size, L**2)`` if given, else
        ``(L**2,)``.
    reality : bool, optional
        Impose ``f_{l,-m} = (-1)^m conj(f_lm)`` and real ``f_l^0`` so that
        the signal is real.

    Returns
    -------
    ndarray of complex
    """
    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)
    L = int(L)
    S = 1 if size is None else int(size)
    N = L * L
    flm = (2 * rng.random((S, N)) - 1) + 1j * (2 * rng.random((S, N)) - 1)
    if reality:
        for el in range(L):
            flm[:, elm2ind(el, 0)] = flm[:, elm2ind(el, 0)].real
            ms = np.arange(1, el + 1)
            flm[:, elm2ind(el, -ms)] = ((-1.0) ** ms) * np.conj(flm[:, elm2ind(el, ms)])
    return flm[0] if size is None else flm


def amplification(L, thetas="lambda", kappa=False):
    """Amplification constants of a ring placement for every order ``m``.

    ``lambda_small[m] = Lambda_m = max_{k < m} sum_{s >= m} |Lambda^m_s(theta_k)|``
    (worst-case amplification of the interpolation to the small rings,
    which governs the interpolated elimination),
    ``lambda_gl[m] = Lambda_m^GL`` (amplification of the interpolation to the
    Gauss-Legendre nodes, GL back-end) and, optionally,
    ``kappa[m] = kappa(P_m)`` (condition number of the per-order system of
    the 2014 transform; one SVD per order).

    Parameters
    ----------
    L : int
        Harmonic band-limit.
    thetas : array_like or str, optional
        Ring co-latitudes or a placement name (default ``"lambda"``).
    kappa : bool, optional
        Also compute ``kappa(P_m)``.

    Returns
    -------
    dict
        Keys ``"lambda_small"``, ``"lambda_gl"`` and (if requested)
        ``"kappa"``, each an array of length ``L``.
    """
    thetas = ordered_theta(L, thetas)
    L = thetas.size
    x = np.cos(thetas)
    with np.errstate(divide="ignore"):
        logsin = np.log(np.abs(np.sin(thetas)))
    xg, _ = gauss_legendre(L)
    logsin_g = np.log(np.sqrt(1 - xg ** 2))
    lam_small = np.zeros(L)
    lam_gl = np.zeros(L)
    kap = np.zeros(L)
    for m in range(L - 1, -1, -1):
        nodes = slice(m, L)
        logw, sgnw = bary_weights(x[nodes])
        if m > 0:
            Lam = lambda_matrix(m, x[nodes], logsin[nodes], logw, sgnw, x[:m], logsin[:m])
            lam_small[m] = np.abs(Lam).sum(axis=1).max()
        LamG = lambda_matrix(m, x[nodes], logsin[nodes], logw, sgnw, xg, logsin_g)
        lam_gl[m] = np.abs(LamG).sum(axis=1).max()
        if kappa:
            kap[m] = np.linalg.cond(legmat(thetas[nodes], L, m))
    out = {"lambda_small": lam_small, "lambda_gl": lam_gl}
    if kappa:
        out["kappa"] = kap
    return out


def sharmonic(L, el, m, thetas="lambda"):
    """A single spherical harmonic sampled on the scheme.

    Parameters
    ----------
    L : int
        Harmonic band-limit.
    el, m : int
        Degree ``el < L`` and order ``|m| <= el``.
    thetas : array_like or str, optional
        Ring co-latitudes or a placement name.

    Returns
    -------
    ndarray of complex, shape (L**2,)
        ``Y_lm`` at the ``L**2`` samples in the ring-by-ring layout.
    """
    from .transforms import inverse

    L = int(L)
    if el >= L or abs(m) > el:
        raise ValueError("require el < L and |m| <= el")
    flm = np.zeros(L * L, dtype=complex)
    flm[elm2ind(el, m)] = 1.0
    return inverse(flm, L, thetas)
