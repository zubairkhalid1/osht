"""Legendre functions, quadrature and indexing."""

import numpy as np
import pytest
from scipy.special import gammaln, lpmv

import osht


def _sph_harm(el, m, theta, phi):
    """Y_lm(theta, phi) with theta the polar angle, for old and new SciPy."""
    try:
        from scipy.special import sph_harm_y

        return sph_harm_y(el, m, theta, phi)
    except ImportError:  # SciPy < 1.15
        from scipy.special import sph_harm

        return sph_harm(m, el, phi, theta)


def test_elm2ind_roundtrip():
    L = 17
    ind = np.arange(L * L)
    el, m = osht.ind2elm(ind)
    assert np.all(osht.elm2ind(el, m) == ind)
    assert np.all(np.abs(m) <= el)
    assert osht.elm2ind(3, -2) == 10 and osht.ind2elm(10) == (3, -2)


@pytest.mark.parametrize("m", [0, 1, 4, 9])
def test_legmat_against_scipy(m):
    L = 12
    thetas = np.linspace(0.05, np.pi - 0.05, 11)
    P = osht.legmat(thetas, L, m)
    for el in range(m, L):
        # scipy's lpmv includes the Condon-Shortley phase; normalise to orthonormal Y_lm
        norm = np.sqrt((2 * el + 1) / (4 * np.pi) * np.exp(gammaln(el - m + 1) - gammaln(el + m + 1)))
        ref = norm * lpmv(m, el, np.cos(thetas))
        assert np.allclose(P[el - m], ref, rtol=1e-11, atol=1e-13)


def test_legmat_against_sph_harm():
    L, m = 9, 3
    thetas = np.array([0.3, 1.1, 2.0, 2.9])
    P = osht.legmat(thetas, L, m)
    for el in range(m, L):
        ref = _sph_harm(el, m, thetas, 0.0)
        assert np.allclose(P[el - m], ref.real, rtol=1e-11, atol=1e-13)


def test_legmat_orthonormality_by_quadrature():
    L = 20
    x, w = osht.gauss_legendre(L)
    thetas = np.arccos(x)
    for m in [0, 2, 7]:
        P = osht.legmat(thetas, L, m)
        G = 2 * np.pi * (P * w) @ P.T
        assert np.allclose(G, np.eye(L - m), atol=1e-12)


def test_legmat_pole_and_large_L():
    # m > 0 rows vanish at the poles; the rescaled recurrence survives large L
    P = osht.legmat([0.0, np.pi], 10, 3)
    assert np.all(P[:, 0] == 0) and np.all(np.abs(P[:, 1]) < 1e-40)  # sin(pi) is 1e-16 in floating point
    P = osht.legmat([1e-3, np.pi / 2, np.pi - 1e-3], 2000, 1500)
    assert np.all(np.isfinite(P))
    assert np.abs(P[:, 1]).max() > 0.1  # at the equator the functions are O(1) for large l
    assert np.all(P[:, 0] == 0) and np.all(P[:, 2] == 0)  # underflow deep in the forbidden region


def test_legmat_errors():
    with pytest.raises(ValueError):
        osht.legmat([0.1], 5, 5)
    with pytest.raises(ValueError):
        osht.legmat([0.1], 5, -1)


def test_gauss_legendre():
    x, w = osht.gauss_legendre(20)
    assert abs(np.sum(w * x ** 2) - 2 / 3) < 1e-14
    assert abs(np.sum(w * x ** 38) - 2 / 39) < 1e-14
    assert np.all(np.diff(x) > 0)
