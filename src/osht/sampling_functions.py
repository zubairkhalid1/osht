"""Order-m sampling functions on the rings and their stable evaluation.

The order-m profile ``G_m(theta) = int f(theta, phi) exp(-i m phi) dphi`` of a
signal band-limited at ``L`` lies in the space

    V_m = sin^|m|(theta) x { polynomials of degree <= L - 1 - |m| in cos theta },

whose Lagrange basis on the co-latitudes ``theta_s`` of the rings that resolve
order ``m`` (``s >= |m|``) is the set of order-m sampling functions of
Giacaglia and Lundquist (1972),

    Lambda^m_s(theta) = (sin theta / sin theta_s)^|m|
                        x prod_{r != s} (cos theta - cos theta_r) / (cos theta_s - cos theta_r).

They are evaluated here in the first barycentric form in the log domain, so
that every entry is accurate to a relative error of O(L eps_mach) even when
the individual factors overflow or underflow.
"""

from __future__ import annotations

import numpy as np

__all__ = ["bary_weights", "lambda_matrix", "sampling_functions"]


def bary_weights(x_nodes):
    """Barycentric weights of a node set in the log domain.

    Parameters
    ----------
    x_nodes : array_like
        Nodes (values of ``cos theta``), distinct.

    Returns
    -------
    logw, sgnw : ndarray
        ``log|w_s|`` and ``sign(w_s)`` of ``w_s = 1 / prod_{r != s} (x_s - x_r)``.
    """
    x_nodes = np.asarray(x_nodes, dtype=float).ravel()
    D = x_nodes[:, None] - x_nodes[None, :]
    np.fill_diagonal(D, 1.0)
    if np.any(D == 0):
        raise ValueError("nodes must be distinct")
    logw = -np.log(np.abs(D)).sum(axis=1)
    sgnw = np.prod(np.sign(D), axis=1)
    return logw, sgnw


def lambda_matrix(m, x_nodes, logsin_nodes, logw, sgnw, x_t, logsin_t):
    """Order-m sampling functions of a node set evaluated at target co-latitudes.

    Parameters
    ----------
    m : int
        Order (any sign; only ``|m|`` matters).
    x_nodes, logsin_nodes : ndarray, shape (n,)
        ``cos theta`` and ``log sin theta`` of the nodes.
    logw, sgnw : ndarray, shape (n,)
        Barycentric weights of the nodes from :func:`bary_weights`.
    x_t, logsin_t : ndarray, shape (T,)
        ``cos theta`` and ``log sin theta`` of the targets (``-inf`` at a
        pole).  A target that coincides exactly with a node gets the unit
        row of the Lagrange property.

    Returns
    -------
    ndarray, shape (T, n)
        ``Lam[t, s] = Lambda^m_s(theta_t)``; row ``t`` reconstructs
        ``G_m(theta_t) = sum_s Lam[t, s] G_m(theta_s)``.
    """
    x_nodes = np.asarray(x_nodes, dtype=float).ravel()
    logsin_nodes = np.asarray(logsin_nodes, dtype=float).ravel()
    logw = np.asarray(logw, dtype=float).ravel()
    sgnw = np.asarray(sgnw, dtype=float).ravel()
    x_t = np.asarray(x_t, dtype=float).ravel()
    logsin_t = np.asarray(logsin_t, dtype=float).ravel()
    m = abs(int(m))

    D = x_t[:, None] - x_nodes[None, :]
    hit = D == 0  # a target that coincides with a node: Lagrange property, row = unit vector
    hit_rows = hit.any(axis=1)
    if hit_rows.any():
        D = D.copy()
        D[hit_rows] = 1.0
    with np.errstate(divide="ignore"):
        logabsD = np.log(np.abs(D))
    sgnD = np.sign(D)
    logomega = logabsD.sum(axis=1)
    sgnomega = np.prod(sgnD, axis=1)
    logLam = logomega[:, None] + logw[None, :] - logabsD
    zero_rows = np.zeros(x_t.size, dtype=bool)
    if m > 0:
        if not np.all(np.isfinite(logsin_nodes)):
            raise ValueError("a node lies at a pole where the sampling functions are singular")
        with np.errstate(invalid="ignore"):
            logLam = logLam + m * (logsin_t[:, None] - logsin_nodes[None, :])
        zero_rows |= ~np.isfinite(logsin_t)
    with np.errstate(under="ignore", over="ignore", invalid="ignore"):
        Lam = np.exp(logLam) * (sgnomega[:, None] * sgnw[None, :] * sgnD)
    if zero_rows.any():
        Lam[zero_rows, :] = 0.0
    if hit_rows.any():
        Lam[hit_rows, :] = hit[hit_rows].astype(float)
    return Lam


def sampling_functions(L, m, theta_eval, thetas=None):
    """Order-m sampling functions on a grid of co-latitudes (for plotting and checks).

    Parameters
    ----------
    L : int
        Harmonic band-limit.
    m : int
        Order.
    theta_eval : array_like
        Evaluation co-latitudes.
    thetas : array_like or str, optional
        The ``L`` ring co-latitudes (ring ``k`` at ``thetas[k]``) or a
        placement name accepted by :func:`osht.ordered_theta`; default
        the Lambda-placement.

    Returns
    -------
    ndarray, shape (len(theta_eval), L - abs(m))
        ``Lam[t, j] = Lambda^m_{|m| + j}(theta_eval[t])`` for the nodes
        ``theta_|m|, ..., theta_{L-1}``.  Evaluation points that coincide
        with a node are handled exactly.
    """
    from .sampling import ordered_theta

    if thetas is None or isinstance(thetas, str):
        thetas = ordered_theta(L, thetas or "lambda")
    thetas = np.asarray(thetas, dtype=float).ravel()
    theta_eval = np.asarray(theta_eval, dtype=float).ravel()
    m = abs(int(m))
    nodes = thetas[m:]
    x = np.cos(nodes)
    with np.errstate(divide="ignore"):
        logsin = np.log(np.abs(np.sin(nodes)))
        xt = np.cos(theta_eval)
        logsin_t = np.log(np.abs(np.sin(theta_eval)))
    logw, sgnw = bary_weights(x)
    Lam = np.zeros((theta_eval.size, nodes.size))
    key_nodes = np.round(x * 1e12)
    key_t = np.round(xt * 1e12)
    hit = np.isin(key_t, key_nodes)
    off = ~hit
    if off.any():
        Lam[off] = lambda_matrix(m, x, logsin, logw, sgnw, xt[off], logsin_t[off])
    for t in np.nonzero(hit)[0]:
        Lam[t, np.nonzero(key_nodes == key_t[t])[0][0]] = 1.0
    return Lam
