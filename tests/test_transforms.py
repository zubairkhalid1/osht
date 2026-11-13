"""Exactness and stability of the transforms."""

import warnings

import numpy as np
import pytest

import osht
from osht.transforms import _as_batch, backend_eq


# ----------------------------------------------------------------------------- spectral round trips
@pytest.mark.parametrize("L", [1, 2, 3, 4, 7, 16, 33, 64])
@pytest.mark.parametrize("backend", ["eq", "gl"])
def test_single_pass_roundtrip(L, backend, rng):
    flm = osht.random_flm(L, rng)
    f = osht.inverse(flm, L)
    rec = osht.forward_pass(f, L, backend=backend)
    assert rec.shape == flm.shape
    tol = {1: 1e-14, 2: 1e-14, 3: 1e-14, 4: 1e-14, 7: 1e-13, 16: 1e-13, 33: 1e-12, 64: 2e-12}[L]
    assert np.abs(rec - flm).max() < tol


@pytest.mark.parametrize("L", [8, 32, 64])
def test_two_pass_improves(L, rng):
    flm = osht.random_flm(L, rng)
    f = osht.inverse(flm, L)
    one = osht.forward(f, L, passes=1)
    two, info = osht.forward(f, L, passes=2, return_info=True)
    e1, e2 = np.abs(one - flm).max(), np.abs(two - flm).max()
    assert e2 < 1e-13 and e2 <= e1
    assert info["passes"] == 2 and len(info["res_max"]) == 1
    assert info["lambda_small"].shape == (L,)


def test_recorded_accuracy_L64(rng):
    """Recorded for the MATLAB package (Octave): 1 pass eq 8e-14, gl 1.6e-13, 2 passes 3e-15, LU 4e-14."""
    L = 64
    worst = {"eq": 0, "gl": 0, "two": 0, "lu": 0}
    for _ in range(3):
        flm = osht.random_flm(L, rng)
        f = osht.inverse(flm, L)
        worst["eq"] = max(worst["eq"], np.abs(osht.forward_pass(f, L) - flm).max())
        worst["gl"] = max(worst["gl"], np.abs(osht.forward_pass(f, L, backend="gl") - flm).max())
        worst["two"] = max(worst["two"], np.abs(osht.forward(f, L) - flm).max())
        worst["lu"] = max(worst["lu"], np.abs(osht.forward_lu(f, L) - flm).max())
    assert worst["eq"] < 5e-13 and worst["gl"] < 2e-12 and worst["two"] < 5e-14 and worst["lu"] < 2e-13


@pytest.mark.parametrize("placement", ["kappa", "alternating"])
def test_other_placements(placement, rng):
    L = 24
    flm = osht.random_flm(L, rng)
    f = osht.inverse(flm, L, placement)
    for backend in ["eq", "gl"]:
        assert np.abs(osht.forward_pass(f, L, placement, backend=backend) - flm).max() < 1e-11
    assert np.abs(osht.forward(f, L, placement) - flm).max() < 1e-12
    assert np.abs(osht.forward_lu(f, L, placement) - flm).max() < 1e-12


def test_gl_backend_on_non_equiangular_rings(rng):
    """The GL back-end works for any distinct ring set; the eq back-end refuses it."""
    L = 12
    gl = np.arccos(osht.gauss_legendre(L)[0])
    thetas = gl[np.argsort(-np.abs(gl - np.pi / 2), kind="stable")]  # alternating order of the GL nodes
    flm = osht.random_flm(L, rng)
    f = osht.inverse(flm, L, thetas)
    assert np.abs(osht.forward_pass(f, L, thetas, backend="gl") - flm).max() < 1e-10
    assert np.abs(osht.forward_lu(f, L, thetas) - flm).max() < 1e-10
    with pytest.raises(ValueError, match="equiangular"):
        osht.forward_pass(f, L, thetas, backend="eq")


# ----------------------------------------------------------------------------- spatial round trip
def test_spatial_roundtrip(rng):
    L = 20
    ft = (2 * rng.random(L * L) - 1) + 1j * (2 * rng.random(L * L) - 1)
    fr = osht.inverse(osht.forward(ft, L), L)
    assert np.abs(fr - ft).max() < 1e-13


def test_single_harmonic(rng):
    L, el, m = 24, 10, -4
    Y = osht.sharmonic(L, el, m)
    flm = osht.forward_pass(Y, L)
    e = flm.copy()
    e[osht.elm2ind(el, m)] -= 1
    assert np.abs(e).max() < 1e-13
    # against the explicit formula
    s = osht.Sampling(L)
    P = osht.legmat(s.theta, L, abs(m))[el - abs(m)] * (-1) ** abs(m)
    assert np.allclose(Y, P * np.exp(1j * m * s.phi), atol=1e-13)
    with pytest.raises(ValueError):
        osht.sharmonic(L, L, 0)
    with pytest.raises(ValueError):
        osht.sharmonic(L, 3, 4)


# ----------------------------------------------------------------------------- real signals
@pytest.mark.parametrize("L", [1, 5, 16, 48])
def test_reality_fast_path(L, rng):
    flm = osht.random_flm(L, rng, reality=True)
    f = osht.inverse(flm, L, reality=True)
    assert f.dtype == float
    fc = osht.inverse(flm, L)
    assert np.abs(fc - f).max() < 1e-13 and np.abs(fc.imag).max() < 1e-13
    for backend in ["eq", "gl"]:
        rec = osht.forward_pass(f, L, backend=backend, reality=True)
        assert np.abs(rec - flm).max() < 1e-12
    rec2 = osht.forward(f, L, reality=True)
    assert np.abs(rec2 - flm).max() < 1e-13
    assert np.abs(osht.forward(f, L) - flm).max() < 1e-13  # the complex path on real data agrees


def test_reality_errors(rng):
    L = 6
    flm = osht.random_flm(L, rng)
    f = osht.inverse(flm, L)
    with pytest.raises(ValueError, match="imaginary"):
        osht.forward(f, L, reality=True)


# ----------------------------------------------------------------------------- batches
def test_batch_matches_single(rng):
    L = 16
    F = osht.random_flm(L, rng, size=4)
    f = osht.inverse(F, L)
    assert f.shape == (4, L * L)
    for backend in ["eq", "gl"]:
        rec = osht.forward_pass(f, L, backend=backend)
        assert rec.shape == F.shape
        for s in range(4):
            single = osht.forward_pass(f[s], L, backend=backend)
            assert np.abs(rec[s] - single).max() < 1e-13
    rec, info = osht.forward(f, L, passes=3, return_info=True)
    assert np.abs(rec - F).max() < 1e-13
    assert all(r.shape == (4,) for r in info["res_max"])
    lu = osht.forward_lu(f, L)
    assert np.abs(lu - F).max() < 1e-12
    Fr = osht.random_flm(L, rng, size=3, reality=True)
    fr = osht.inverse(Fr, L, reality=True)
    assert np.abs(osht.forward(fr, L, reality=True) - Fr).max() < 1e-13


def test_backend_eq_chunking(rng):
    L = 10
    F = osht.random_flm(L, rng, size=5)
    f = osht.inverse(F, L)
    _, info = osht.forward_pass(f, L, return_info=True)
    # recompute the back-end with tiny chunks through forward_pass' internals: compare with full chunk
    from osht.transforms import _ring_ffts  # noqa: F401  (import check)
    rec_full = osht.forward_pass(f, L)
    thetas = osht.ordered_theta(L)
    # build the profiles by the exact inverse route and feed the back-end with chunk size 1
    Gp = np.zeros((L, L, 5), dtype=complex)
    Gn = np.zeros((L, L, 5), dtype=complex)
    for m in range(L):
        ell = np.arange(m, L)
        P = osht.legmat(thetas, L, m)
        Gp[m] = 2 * np.pi * (P.T @ F[:, osht.elm2ind(ell, m)].T)
        Gn[m] = 2 * np.pi * (-1) ** m * (P.T @ F[:, osht.elm2ind(ell, -m)].T)
    small = backend_eq(Gp, Gn, L, thetas, max_bytes=1)
    big = backend_eq(Gp, Gn, L, thetas)
    assert np.abs(small - big).max() < 1e-14
    assert np.abs(big - F).max() < 1e-13
    assert np.abs(rec_full - F).max() < 1e-13


# ----------------------------------------------------------------------------- multi-pass
def test_multipass_with_lu_and_stopping(rng):
    L = 20
    flm = osht.random_flm(L, rng)
    f = osht.inverse(flm, L)
    out, info = osht.multipass(f, L, lambda g: osht.forward_lu(g, L), passes=6, verbose=True)
    assert np.abs(out - flm).max() < 1e-13
    assert 1 <= info["passes"] <= 6
    # stops when the residual stops decreasing: never more residuals than passes - 1
    assert len(info["res_max"]) <= 5
    out1, info1 = osht.multipass(f, L, lambda g: osht.forward_pass(g, L), passes=1)
    assert info1["passes"] == 1 and info1["res_max"] == []
    with pytest.raises(ValueError):
        osht.multipass(f, L, lambda g: g, passes=0)


# ----------------------------------------------------------------------------- oracle sweep
def test_oracle_sweep(rng):
    L = 32
    flm = osht.random_flm(L, rng)
    f = osht.inverse(flm, L)
    assert np.abs(osht.forward_pass(f, L, flm_oracle=flm) - flm).max() < 1e-12
    assert np.abs(osht.forward_pass(f, L, backend="gl", flm_oracle=flm) - flm).max() < 1e-12


# ----------------------------------------------------------------------------- warnings
def test_warning_for_unstable_placement(rng):
    L = 128
    thetas = osht.ordered_theta(L, "alternating")
    flm = osht.random_flm(L, rng)
    f = osht.inverse(flm, L, thetas)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        osht.forward_pass(f, L, thetas)
    assert len(w) == 1 and "amplification" in str(w[0].message)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        osht.forward(f, L, thetas)
    messages = [str(x.message) for x in w if issubclass(x.category, RuntimeWarning)]
    assert any("amplification" in msg for msg in messages)
    assert any("not converged" in msg for msg in messages)


def test_no_warning_for_scalar(rng):
    L = 64
    flm = osht.random_flm(L, rng)
    f = osht.inverse(flm, L)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        osht.forward_pass(f, L)


# ----------------------------------------------------------------------------- argument checks
def test_argument_checks(rng):
    L = 5
    flm = osht.random_flm(L, rng)
    f = osht.inverse(flm, L)
    with pytest.raises(ValueError, match="L\\*\\*2"):
        osht.forward(f[:-1], L)
    with pytest.raises(ValueError, match="shape"):
        osht.forward(f.reshape(1, 1, -1), L)
    with pytest.raises(ValueError, match="NaN"):
        osht.forward(np.full(L * L, np.nan), L)
    with pytest.raises(ValueError, match="backend"):
        osht.forward(f, L, backend="mw")
    with pytest.raises(ValueError, match="band-limit"):
        osht.forward(f, 0)
    with pytest.raises(ValueError, match="band-limit"):
        osht.inverse(flm, 2.5)
    with pytest.raises(ValueError):
        osht.forward(f, L, thetas=np.zeros(3))
    A, sq = _as_batch(f, L, "f")
    assert A.shape == (1, L * L) and sq
    A, sq = _as_batch(np.stack([f, f]), L, "f")
    assert A.shape == (2, L * L) and not sq


def test_inputs_are_not_modified(rng):
    L = 9
    flm = osht.random_flm(L, rng)
    f = osht.inverse(flm, L)
    f0, flm0 = f.copy(), flm.copy()
    osht.forward(f, L)
    osht.forward_lu(f, L)
    osht.inverse(flm, L)
    assert np.array_equal(f, f0) and np.array_equal(flm, flm0)
    fl = list(f)  # lists are accepted
    assert np.abs(osht.forward(fl, L) - flm).max() < 1e-13


def test_dtype_handling(rng):
    L = 6
    flm = osht.random_flm(L, rng, reality=True)
    f = osht.inverse(flm, L, reality=True).astype(np.float32)
    rec = osht.forward(f.astype(float), L, reality=True)
    assert rec.dtype == complex
    rec32 = osht.forward(f, L, reality=True)  # float32 input is upcast
    assert np.abs(rec32 - rec).max() < 1e-5
