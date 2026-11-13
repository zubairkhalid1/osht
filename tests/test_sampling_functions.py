"""Sampling functions: Lagrange property, exactness on the profile spaces, amplification."""

import numpy as np
import pytest

import osht
from osht.sampling_functions import bary_weights, lambda_matrix


def test_bary_weights_match_direct_product():
    x = np.array([-0.8, -0.1, 0.3, 0.9])
    logw, sgnw = bary_weights(x)
    for s in range(4):
        w = 1.0 / np.prod([x[s] - x[r] for r in range(4) if r != s])
        assert np.isclose(sgnw[s] * np.exp(logw[s]), w)
    with pytest.raises(ValueError):
        bary_weights([0.1, 0.1])


def test_lagrange_property_and_interpolation_of_profiles():
    """Lambda^m_s(theta_r) = delta_sr, and the interpolation reproduces every order-m profile."""
    L = 20
    thetas = osht.ordered_theta(L)
    rng = np.random.default_rng(1)
    for m in [1, 5, 12, 19]:
        nodes = thetas[m:]
        Lam = osht.sampling_functions(L, m, nodes)
        assert np.allclose(Lam, np.eye(L - m), atol=1e-11)
        # a random order-m profile: G_m(theta) = 2 pi sum_l f_l^m Ptilde_l^m(theta)
        c = rng.standard_normal(L - m) + 1j * rng.standard_normal(L - m)
        targets = np.concatenate([thetas[:m], np.linspace(0.07, 3.1, 25)])
        G_nodes = c @ osht.legmat(nodes, L, m)
        G_targets = c @ osht.legmat(targets, L, m)
        Lam_t = osht.sampling_functions(L, m, targets)
        assert np.allclose(Lam_t @ G_nodes, G_targets, rtol=1e-9, atol=1e-11)


def test_pole_target_rows_vanish():
    L = 10
    Lam = osht.sampling_functions(L, 3, [0.0, np.pi, 1.0])
    assert np.all(Lam[0] == 0) and np.all(np.abs(Lam[1]) < 1e-40) and np.any(np.abs(Lam[2]) > 1e-3)
    # m = 0: the pole ring is a node and the hit is exact
    Lam0 = osht.sampling_functions(L, 0, [np.pi])
    assert np.isclose(Lam0[0, 0], 1.0) and np.allclose(Lam0[0, 1:], 0)


def test_lambda_matrix_errors():
    x = np.array([0.1, 0.5])
    ls = np.log(np.sqrt(1 - x ** 2))
    lw, sw = bary_weights(x)
    hit = lambda_matrix(1, x, ls, lw, sw, np.array([0.5, 0.3]), np.array([ls[1], np.log(np.sqrt(1 - 0.09))]))
    assert np.allclose(hit[0], [0.0, 1.0])  # a target on a node: Lagrange property
    with pytest.raises(ValueError, match="pole"):
        lambda_matrix(2, np.array([1.0, 0.0]), np.array([-np.inf, 0.0]), lw, sw,
                      np.array([0.5]), np.array([-0.1]))


def test_amplification_matches_recorded_values():
    """Recorded in the project notes (MATLAB/Octave and the Python prototype)."""
    a = osht.amplification(64, kappa=True)
    assert np.isclose(a["lambda_small"].max(), 46.387, atol=2e-3)
    assert np.isclose(a["kappa"].max(), 11.247, atol=2e-3)
    assert a["lambda_small"][0] == 0 and a["lambda_gl"].shape == (64,)
    k = osht.amplification(64, "kappa", kappa=True)
    assert np.isclose(k["kappa"].max(), 10.267, atol=2e-3)
    a128 = osht.amplification(128)
    assert np.isclose(a128["lambda_small"].max(), 100.30, atol=0.01)
    assert np.isclose(a128["lambda_gl"].max(), 84.12, atol=0.01)
    k128 = osht.amplification(128, "kappa", kappa=True)
    assert np.isclose(k128["lambda_small"].max(), 170.6, atol=0.1)
    assert np.isclose(k128["kappa"].max(), 30.73, atol=0.01)
