"""Error measures, test signals, shipped data, Wigner helpers and plotting."""

import numpy as np
import pytest

import osht


def test_errors():
    a = np.array([1 + 1j, 2.0, -3.0])
    b = a + np.array([1e-3, 0, 0])
    e = osht.errors(a, b)
    assert isinstance(e, osht.Errors)
    assert np.isclose(e.E_max, 1e-3) and np.isclose(e.E_mean, 1e-3 / 3)
    assert np.isclose(e.eps, 1e-6 / 15) and np.isclose(e.SNR, -10 * np.log10(1e-6 / 15))
    eb = osht.errors(np.stack([a, a]), np.stack([b, a]))
    assert eb.E_max.shape == (2,) and eb.E_max[1] == 0 and np.isinf(eb.SNR[1])
    with pytest.raises(ValueError):
        osht.errors(a, a[:2])


def test_random_flm():
    L = 7
    flm = osht.random_flm(L, rng=3)
    assert flm.shape == (49,) and np.abs(flm.real).max() <= 1 and np.abs(flm.imag).max() <= 1
    assert np.array_equal(osht.random_flm(L, rng=3), flm)  # seeded
    F = osht.random_flm(L, rng=np.random.default_rng(0), size=2)
    assert F.shape == (2, 49)
    r = osht.random_flm(L, rng=1, reality=True)
    el, m = osht.ind2elm(np.arange(L * L))
    assert np.allclose(r, (-1.0) ** m * np.conj(r[osht.elm2ind(el, -m)]))
    assert np.all(r[osht.elm2ind(np.arange(L), 0)].imag == 0)


def test_earth_topography():
    flm, L = osht.data.earth_topography()
    assert L == 128 and flm.shape == (128 * 128,)
    el, m = osht.ind2elm(np.arange(L * L))
    assert np.allclose(flm, (-1.0) ** m * np.conj(flm[osht.elm2ind(el, -m)]))  # real signal
    f = osht.inverse(flm, L, reality=True)
    assert f.dtype == float and f.max() > 1000 and f.min() < -1000  # metres, roughly
    rec = osht.forward(f, L, reality=True)
    assert osht.errors(flm, rec).SNR > 200
    flm32, L32 = osht.data.earth_topography(32)
    assert L32 == 32 and np.array_equal(flm32, flm[:32 * 32])
    with pytest.raises(ValueError):
        osht.data.earth_topography(129)


def test_reflected_weights():
    w = osht.reflected_weights(np.array([-3, -2, -1, 0, 1, 2, 3, 4]))
    assert np.allclose(w, [0, -2 / 3, -1j * np.pi / 2, 2, 1j * np.pi / 2, -2 / 3, 0, -2 / 15])
    # w(mm) is the integral of sin(theta) exp(i mm theta) over [0, pi]
    th = np.linspace(0, np.pi, 20001)
    for mm in range(-4, 5):
        trapezoid = getattr(np, "trapezoid", None) or getattr(np, "trapz")  # noqa: B009
        num = trapezoid(np.sin(th) * np.exp(1j * mm * th), th)
        assert abs(num - osht.reflected_weights(mm)) < 1e-7


def test_trapani_plane_is_orthogonal_and_matches_closed_forms():
    L = 9
    dl = np.zeros((2 * L - 1, 2 * L - 1))
    o = L - 1
    for el in range(L):
        dl = osht.trapani_plane(dl, L, el)
        D = dl[o - el:o + el + 1, o - el:o + el + 1]
        assert np.allclose(D @ D.T, np.eye(2 * el + 1), atol=1e-13)
        # d^l_{l l}(pi/2) = 2^{-l}, d^l_{0 0}(pi/2) = P_l(0)
        assert np.isclose(dl[el + o, el + o], 2.0 ** (-el))
        assert np.isclose(dl[o, o], np.polynomial.legendre.Legendre.basis(el)(0.0), atol=1e-14)


def test_plotting_smoke():
    mpl = pytest.importorskip("matplotlib")
    mpl.use("Agg")
    import matplotlib.pyplot as plt

    L = 12
    fig, ax = osht.plotting.plot_sampling(L)
    assert fig is not None
    flm, L2 = osht.data.earth_topography(16)
    f = osht.inverse(flm, L2, reality=True)
    fig, ax = osht.plotting.plot_sphere(f, L2)
    fig, ax = osht.plotting.plot_sphere(f.astype(complex), L2, resolution=20)
    with pytest.raises(ValueError):
        osht.plotting.plot_sphere(f + 1j, L2)
    err = np.abs(osht.forward(f, L2, reality=True) - flm) + 1e-17
    fig, ax = osht.plotting.plot_spectral(err, L2, log=True, clabel="log10 error")
    with pytest.raises(ValueError):
        osht.plotting.plot_spectral(err[:-1], L2)
    plt.close("all")
