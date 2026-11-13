"""Sampling scheme, placements and the placement searches."""

import numpy as np
import pytest

import osht
from osht.sampling import _describe, equiangular_set


def test_equiangular_set_and_indexed_theta():
    L = 9
    TT = equiangular_set(L)
    assert TT.size == L and np.isclose(TT[-1], np.pi) and TT[0] > 0
    ind = osht.indexed_theta(L)
    assert np.isclose(ind[0], np.pi)  # the pole is the most distant from the equator
    d = np.abs(ind - np.pi / 2)
    assert np.all(np.diff(d) <= 1e-12)  # descending distance from the equator
    # of a symmetric pair the northern member comes first
    assert ind[1] < np.pi / 2 and ind[2] > np.pi / 2
    assert osht.is_equiangular(ind, L)


@pytest.mark.parametrize("name", ["lambda", "kappa", "alternating"])
def test_ordered_theta_is_a_permutation(name):
    for L in [1, 2, 3, 10, 64, 128]:
        th = osht.ordered_theta(L, name)
        assert th.shape == (L,)
        assert osht.is_equiangular(th, L)
        if L > 1:
            assert np.isclose(th[0], np.pi)  # ring 0 (one sample) sits at the south pole
            assert np.isclose(th[-1], osht.indexed_theta(L)[-1])  # largest ring nearest the equator


def test_ordered_theta_objective_and_missing():
    th, obj = osht.ordered_theta(64, "lambda", objective=True)
    assert obj.shape == (63,) and np.isclose(obj.max(), 46.387342770304585)
    th, obj = osht.ordered_theta(64, "kappa", objective=True)
    assert obj.shape == (63,) and np.isclose(obj.max(), 10.267, atol=1e-3)
    th, obj = osht.ordered_theta(8, "alternating", objective=True)
    assert obj is None
    with pytest.raises(ValueError, match="no stored or cached lambda-placement"):
        osht.ordered_theta(129, "lambda")
    with pytest.raises(ValueError, match="no stored or cached kappa-placement"):
        osht.ordered_theta(722, "kappa")
    with pytest.raises(ValueError):
        osht.ordered_theta(8, "unknown")


def test_ordered_theta_array_passthrough():
    th = osht.ordered_theta(6, "kappa")
    out = osht.ordered_theta(6, th)
    assert np.array_equal(out, th)
    out = osht.ordered_theta(6, list(th))
    assert np.array_equal(out, th)
    with pytest.raises(ValueError):
        osht.ordered_theta(6, th[:5])
    with pytest.raises(ValueError):
        osht.ordered_theta(6, np.full(6, 0.5))  # not distinct
    with pytest.raises(ValueError):
        osht.ordered_theta(6, np.linspace(-1, 1, 6))  # outside [0, pi]
    for bad in [0, -3, 2.5, True]:
        with pytest.raises(ValueError):
            osht.ordered_theta(bad)


def test_stored_placements():
    lam = osht.stored_placements("lambda")
    kap = osht.stored_placements("kappa")
    assert lam[:128] == list(range(1, 129)) and lam[128:] == [256, 512, 1024, 1536, 2048]
    assert kap[:721] == list(range(1, 722)) and kap[721:] == [1024, 1536, 2048]
    assert _describe(lam) == "1..128, 256, 512, 1024, 1536, 2048"


def test_sampling_points_layout():
    L = 7
    theta, phi = osht.sampling_points(L)
    assert theta.shape == (L * L,) and phi.shape == (L * L,)
    th = osht.ordered_theta(L)
    for k in range(L):
        sl = osht.ring_slice(k)
        assert sl.stop - sl.start == 2 * k + 1
        assert np.all(theta[sl] == th[k])
        assert np.allclose(phi[sl], 2 * np.pi * np.arange(2 * k + 1) / (2 * k + 1))
    assert np.array_equal(osht.ring_index(L), np.repeat(np.arange(L), 2 * np.arange(L) + 1))


def test_sampling_class():
    s = osht.Sampling(10)
    assert s.L == 10 and s.nsamples == 100 and s.equiangular
    x, y, z = s.cartesian()
    assert np.allclose(x ** 2 + y ** 2 + z ** 2, 1)
    assert "L=10" in repr(s)
    s2 = osht.Sampling(64, "kappa")
    assert not np.array_equal(osht.Sampling(64).thetas, s2.thetas)
    s3 = osht.Sampling(5, np.arccos(osht.gauss_legendre(5)[0]))
    assert not s3.equiangular


def test_theta_index():
    L = 11
    th = osht.ordered_theta(L)
    t = osht.theta_index(th, L)
    assert sorted(t.tolist()) == list(range(L))
    assert osht.theta_index(th + 1e-6, L) is None
    assert not osht.is_equiangular(np.linspace(0.1, 3, L), L)


def test_search_lambda_placement_reproduces_stored():
    for L in [1, 2, 5, 16, 24]:
        th, obj = osht.search_lambda_placement(L)
        assert np.allclose(th, osht.ordered_theta(L, "lambda"))
        th_s, obj_s = osht.ordered_theta(L, "lambda", objective=True)
        if L > 1:
            assert np.allclose(obj, obj_s, rtol=1e-8)
    th_gl, obj_gl = osht.search_lambda_placement(12, include_gl=True, verbose=True)
    assert osht.is_equiangular(th_gl, 12)


def test_search_kappa_placement_reproduces_stored():
    for L in [1, 2, 4, 16, 24]:
        th, obj = osht.search_kappa_placement(L, verbose=(L == 16))
        assert np.allclose(th, osht.ordered_theta(L, "kappa"))
        th_s, obj_s = osht.ordered_theta(L, "kappa", objective=True)
        if L > 1:
            assert np.allclose(obj, obj_s, rtol=1e-5)
    th_exact, _ = osht.search_kappa_placement(12, exact=True)
    assert np.allclose(th_exact, osht.ordered_theta(12, "kappa"))


def test_placement_cache(tmp_path):
    """A placement computed once is saved in the cache and found by ordered_theta afterwards."""
    L = 130  # not shipped
    assert osht.cached_placements("lambda") == []
    with pytest.raises(ValueError, match="search=True"):
        osht.ordered_theta(L)
    th = osht.ordered_theta(L, search=True, verbose=False)
    assert osht.cached_placements("lambda") == [L]
    assert osht.cache_dir().is_dir() and osht.cache_dir() == tmp_path / "osht_cache"
    th2, obj = osht.ordered_theta(L, objective=True)  # from the cache now
    assert np.array_equal(th, th2) and obj.shape == (L - 1,)
    thk, objk = osht.search_kappa_placement(L)  # kappa-placements are shipped up to L = 721: not cached
    assert osht.cached_placements("kappa") == [] and np.array_equal(osht.ordered_theta(L, "kappa"), thk)
    osht.save_placement(722, "kappa", osht.indexed_theta(722), np.zeros(721))
    assert osht.cached_placements("kappa") == [722]
    assert np.allclose(osht.ordered_theta(722, "kappa"), osht.indexed_theta(722))
    # the transforms use the cached placement
    flm = osht.random_flm(L, rng=0)
    f = osht.inverse(flm, L)
    assert np.abs(osht.forward(f, L) - flm).max() < 1e-13
    # save=False leaves the cache alone; a shipped L is never cached
    osht.search_lambda_placement(131, save=False)
    osht.search_lambda_placement(8)
    assert osht.cached_placements("lambda") == [L]
    # manual save and validation
    path = osht.save_placement(132, "lambda", osht.indexed_theta(132), None)
    assert path.is_file() and 132 in osht.cached_placements("lambda")
    th3, obj3 = osht.ordered_theta(132, objective=True)
    assert obj3 is None and np.allclose(th3, osht.indexed_theta(132))
    with pytest.raises(ValueError):
        osht.save_placement(10, "lambda", np.linspace(0.1, 3, 10))
    with pytest.raises(ValueError):
        osht.save_placement(10, "other", osht.indexed_theta(10))
    # a different cache directory can be given explicitly
    other = tmp_path / "elsewhere"
    osht.save_placement(133, "kappa", osht.indexed_theta(133), directory=other)
    assert osht.cached_placements("kappa", directory=other) == [133]
    assert osht.cached_placements("kappa", directory=tmp_path / "missing") == []


def test_cache_dir_env(monkeypatch, tmp_path):
    monkeypatch.setenv("OSHT_CACHE_DIR", str(tmp_path / "x"))
    assert osht.cache_dir() == tmp_path / "x"
    monkeypatch.delenv("OSHT_CACHE_DIR")
    assert osht.cache_dir().name == "osht"
