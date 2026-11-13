"""Forward and inverse spherical harmonic transforms on the ``L**2`` samples.

The forward transform is the interpolated elimination: for ``m = L-1`` down
to ``0`` the order-``m`` profiles ``G_{+-m}(theta_k)`` are read exactly on the
rings ``k >= m`` by a ``(2k+1)``-point DFT, interpolated to the rings
``k < m`` by the order-m sampling functions, and the orders ``+-m`` are
subtracted from those rings.  The profiles on all ``L`` rings then feed one of
two exact back-ends (the equiangular pipeline of McEwen and Wiaux, or
Gauss-Legendre quadrature of the interpolated profiles).  No linear system is
solved anywhere and the cost is O(L**3) per pass.  The inverse transform is a
closed-form synthesis by folded inverse FFTs.
"""

from __future__ import annotations

import warnings

import numpy as np

from .harmonics import elm2ind, gauss_legendre, legmat
from .sampling import is_equiangular, ordered_theta, theta_index
from .sampling_functions import lambda_matrix
from .wigner import reflected_weights, trapani_plane

__all__ = ["forward", "forward_pass", "inverse", "forward_lu", "multipass", "backend_eq"]

_BACKENDS = ("eq", "gl")

#: Interpolation amplification above which :func:`forward_pass` emits a RuntimeWarning.
AMPLIFICATION_WARNING = 1e4
#: Relative sample residual above which :func:`multipass` (and :func:`forward`) emits a RuntimeWarning.
RESIDUAL_WARNING = 1e-9


# ----------------------------------------------------------------------------- helpers
def _check_L(L):
    if isinstance(L, (bool, np.bool_)) or int(L) != L or int(L) < 1:
        raise ValueError("the harmonic band-limit L must be a positive integer")
    return int(L)


def _as_batch(a, L, name, reality=False):
    """Return ``(A, squeeze)`` with ``A`` of shape (S, L**2), complex (real if ``reality``)."""
    a = np.asarray(a)
    N = L * L
    if a.ndim == 1:
        squeeze = True
        a = a[None, :]
    elif a.ndim == 2:
        squeeze = False
    else:
        raise ValueError(f"{name} must have shape (L**2,) or (S, L**2)")
    if a.shape[1] != N:
        raise ValueError(f"{name} must have L**2 = {N} entries along the last axis, got {a.shape[1]}")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} contains NaN or infinite values")
    if reality:
        if np.iscomplexobj(a) and np.any(a.imag != 0):
            raise ValueError(f"reality=True but {name} has a nonzero imaginary part")
        a = a.real
    return np.ascontiguousarray(a, dtype=complex if not reality else float), squeeze


def _ring_bins(L):
    ks = np.arange(L)
    return ks, 2 * ks + 1, ks * ks


def _ring_ffts(F, L):
    """DFT of every ring of every signal, stored at the ring's own positions."""
    R = np.empty(F.shape, dtype=complex)
    for k in range(L):
        sl = slice(k * k, (k + 1) * (k + 1))
        R[:, sl] = np.fft.fft(F[:, sl], axis=1)
    return R


def _ring_iffts(C, L):
    """Inverse of :func:`_ring_ffts` (unnormalised, as ``n * ifft``)."""
    F = np.empty(C.shape, dtype=complex)
    for k in range(L):
        sl = slice(k * k, (k + 1) * (k + 1))
        F[:, sl] = (2 * k + 1) * np.fft.ifft(C[:, sl], axis=1)
    return F


def _theta_logs(thetas):
    x = np.cos(thetas)
    with np.errstate(divide="ignore"):
        logsin = np.log(np.abs(np.sin(thetas)))
    return x, logsin


# ----------------------------------------------------------------------------- forward pass
def forward_pass(f, L, thetas="lambda", backend="eq", reality=False, return_info=False, flm_oracle=None):
    """One sweep of the interpolated-elimination forward transform.

    Parameters
    ----------
    f : array_like, shape (L**2,) or (S, L**2)
        Samples of one signal, or of ``S`` signals, in the ring-by-ring
        layout of :class:`osht.Sampling`.
    L : int
        Harmonic band-limit.
    thetas : array_like or str, optional
        Ring co-latitudes or a placement name (default ``"lambda"``).
    backend : {"eq", "gl"}, optional
        ``"eq"``: equiangular back-end of McEwen and Wiaux on the profiles
        of all ``L`` rings (needs the equiangular ring set).  ``"gl"``:
        Gauss-Legendre quadrature of the interpolated profiles (any ring
        set).
    reality : bool, optional
        The signal is real: orders ``-m`` follow by conjugate symmetry,
        which halves the work.
    return_info : bool, optional
        Also return a dictionary with the amplification constants
        ``lambda_small[m]`` (interpolation to the small rings) and, for the
        GL back-end, ``lambda_gl[m]``.
    flm_oracle : array_like, optional
        True coefficients, same shape as the output.  If given, the
        cleaning step subtracts the exact profiles computed from them
        instead of the interpolated ones ("oracle sweep", for error
        analysis).

    Returns
    -------
    flm : ndarray, shape (L**2,) or (S, L**2)
        Coefficients, ``flm[..., l**2 + l + m] = f_l^m``.
    info : dict
        Only if ``return_info=True``.
    """
    L = _check_L(L)
    if backend not in _BACKENDS:
        raise ValueError("backend must be 'eq' or 'gl'")
    thetas = ordered_theta(L, thetas)
    if backend == "eq" and not is_equiangular(thetas, L):
        raise ValueError("the equiangular back-end needs the ring set (2t+1)pi/(2L-1), t = 0..L-1; "
                         "use backend='gl' for other ring sets")
    F, squeeze = _as_batch(f, L, "f", reality)
    S = F.shape[0]
    N = L * L
    oracle = flm_oracle is not None
    if oracle:
        O, _ = _as_batch(flm_oracle, L, "flm_oracle")
        O = O.T  # (N, S)

    x, logsin = _theta_logs(thetas)
    ks, n_k, k2 = _ring_bins(L)
    R = _ring_ffts(F, L)
    use_gl = backend == "gl"
    if use_gl:
        xg, wg = gauss_legendre(L)
        thg = np.arccos(xg)
        logsin_g = np.log(np.sqrt(1 - xg ** 2))
        flm = np.zeros((N, S), dtype=complex)
    Gp = np.zeros((L, L, S), dtype=complex)
    Gn = None if reality else np.zeros((L, L, S), dtype=complex)
    logw = np.zeros(L)
    sgnw = np.ones(L)
    lam_small = np.zeros(L)
    lam_gl = np.zeros(L)

    for m in range(L - 1, -1, -1):
        # --- barycentric weights of the node set {x_m, ..., x_{L-1}}: node x_m joins
        if m < L - 1:
            d = x[m + 1:] - x[m]
            logw[m + 1:] -= np.log(np.abs(d))
            sgnw[m + 1:] *= np.sign(d)
            logw[m] = -np.log(np.abs(d)).sum()
            sgnw[m] = np.prod(np.sign(-d))
        else:
            logw[m] = 0.0
            sgnw[m] = 1.0
        nodes = slice(m, L)
        # --- (a) resolve on the rings k >= m
        nk = n_k[m:]
        Gp[m, m:] = ((2 * np.pi / nk) * R[:, k2[m:] + m]).T
        if not reality:
            Gn[m, m:] = ((2 * np.pi / nk) * R[:, k2[m:] + ((-m) % nk)]).T
        # --- (b) interpolate to the rings k < m and (c) clean them
        if m > 0:
            Lam = lambda_matrix(m, x[nodes], logsin[nodes], logw[nodes], sgnw[nodes], x[:m], logsin[:m])
            gp = Lam @ Gp[m, nodes]
            Gp[m, :m] = gp
            lam_small[m] = np.abs(Lam).sum(axis=1).max()
            if not reality:
                gn = Lam @ Gn[m, nodes]
                Gn[m, :m] = gn
            else:
                gn = np.conj(gp)
            if oracle:
                ell = np.arange(m, L)
                Po = legmat(thetas[:m], L, m)
                gp = 2 * np.pi * (Po.T @ O[elm2ind(ell, m)])
                gn = 2 * np.pi * (-1) ** m * (Po.T @ O[elm2ind(ell, -m)])
            nk = n_k[:m]
            R[:, k2[:m] + (m % nk)] -= ((nk / (2 * np.pi)) * gp.T)
            R[:, k2[:m] + ((-m) % nk)] -= ((nk / (2 * np.pi)) * gn.T)
        # --- GL back-end: quadrature of the profiles interpolated to the Gauss nodes
        if use_gl:
            ell = np.arange(m, L)
            LamG = lambda_matrix(m, x[nodes], logsin[nodes], logw[nodes], sgnw[nodes], xg, logsin_g)
            lam_gl[m] = np.abs(LamG).sum(axis=1).max()
            P = legmat(thg, L, m)
            flm[elm2ind(ell, m)] = P @ (wg[:, None] * (LamG @ Gp[m, nodes]))
            if m > 0 and not reality:
                flm[elm2ind(ell, -m)] = (-1) ** m * (P @ (wg[:, None] * (LamG @ Gn[m, nodes])))

    del R
    if use_gl:
        if reality:
            flm = _complete_real(flm, L)
        flm = flm.T
    else:
        flm = backend_eq(Gp, Gn, L, thetas, reality=reality)
    if squeeze:
        flm = flm[0]
    worst = max(lam_small.max(), lam_gl.max())
    if worst > AMPLIFICATION_WARNING:
        warnings.warn(
            f"the interpolation amplification reaches {worst:.2e} (L = {L}); the transform is exact "
            "in exact arithmetic but the result may carry a large rounding error. See the "
            "documentation on stability and placements.", RuntimeWarning, stacklevel=2)
    if return_info:
        return flm, {"lambda_small": lam_small, "lambda_gl": lam_gl}
    return flm


def _complete_real(flm, L):
    """Fill the orders ``m < 0`` of ``flm`` (shape (N, S)) from ``f_{l,-m} = (-1)^m conj(f_lm)``."""
    for el in range(1, L):
        ms = np.arange(1, el + 1)
        flm[elm2ind(el, -ms)] = ((-1.0) ** ms)[:, None] * np.conj(flm[elm2ind(el, ms)])
    return flm


# ----------------------------------------------------------------------------- equiangular back-end
def backend_eq(Gp, Gn, L, thetas, reality=False, max_bytes=512 * 1024 ** 2):
    """Equiangular back-end: coefficients from the order profiles on all ``L`` rings.

    Implements the forward pipeline of McEwen and Wiaux (2011) on the
    profiles ``G_m(theta_k)``: periodic extension in ``theta`` with the
    factor ``(-1)^m``, FFT in ``theta``, convolution with the reflected
    weights and the Wigner step with the Trapani-Navaza planes
    ``Delta^l = d^l(pi/2)``.

    Parameters
    ----------
    Gp : ndarray, shape (L, L, S)
        ``Gp[m, k, s] = G_m(theta_k)`` of signal ``s`` for ``m = 0..L-1``.
    Gn : ndarray, shape (L, L, S) or None
        ``Gn[m, k, s] = G_{-m}(theta_k)``; ``None`` for real signals.
    L : int
        Harmonic band-limit.
    thetas : array_like, shape (L,)
        Ring co-latitudes (a permutation of the equiangular set).
    reality : bool, optional
        Real signal: orders ``m < 0`` by conjugate symmetry.
    max_bytes : int, optional
        Memory budget per chunk of signals for the ``(2L-1)**2`` arrays.

    Returns
    -------
    ndarray, shape (S, L**2)
        Coefficients of every signal.
    """
    L = _check_L(L)
    thetas = np.asarray(thetas, dtype=float).ravel()
    n = 2 * L - 1
    t_of_k = theta_index(thetas, L)
    if t_of_k is None or len(set(t_of_k.tolist())) != L:
        raise ValueError("the equiangular back-end needs the ring set (2t+1)pi/(2L-1), t = 0..L-1")
    S = Gp.shape[2]
    N = L * L
    if reality:
        mcols = np.arange(0, L)
    else:
        mcols = np.arange(-(L - 1), L)
    mp = np.arange(-(L - 1), L)
    W = reflected_weights(mp[None, :] - mp[:, None])  # W[m', m''] = w(m'' - m')
    rows = mp % n
    phase = np.exp(-1j * mp * np.pi / n)[:, None]
    sgn = (-1.0) ** np.abs(mcols)
    o = L - 1
    pref_l = np.sqrt((2 * np.arange(L) + 1) / (4 * np.pi))
    flm = np.zeros((S, N), dtype=complex)
    chunk = max(1, int(max_bytes // (16 * n * mcols.size * 2)))
    for s0 in range(0, S, chunk):
        s1 = min(S, s0 + chunk)
        C = s1 - s0
        # profiles re-indexed by t, extended to [0, 2pi): GT[c, t, m]
        g = np.zeros((C, L, mcols.size), dtype=complex)
        if reality:
            g[:, t_of_k, :] = np.transpose(Gp[:, :, s0:s1], (2, 1, 0))
        else:
            g[:, t_of_k, L - 1:] = np.transpose(Gp[:, :, s0:s1], (2, 1, 0))
            g[:, t_of_k, :L - 1] = np.transpose(Gn[1:, :, s0:s1], (2, 1, 0))[:, :, ::-1]
        GT = np.empty((C, n, mcols.size), dtype=complex)
        GT[:, :L] = g
        GT[:, L:] = sgn[None, None, :] * g[:, L - 2::-1, :] if L > 1 else 0.0
        del g
        Fhat = np.fft.fft(GT, axis=1)
        del GT
        Fm = phase[None] * Fhat[:, rows, :] / (2 * np.pi * n)  # Fm[c, m', m]
        del Fhat
        # Gmm[c, m, m'] = 2 pi sum_{m''} W[m', m''] F[m'', m]
        Gmm = np.ascontiguousarray(2 * np.pi * np.transpose(np.matmul(W, Fm), (0, 2, 1)))
        del Fm
        dl = np.zeros((n, n))
        for el in range(L):
            dl = trapani_plane(dl, L, el)
            lo = 0 if reality else -el
            ms = np.arange(lo, el + 1)
            i0, i1 = lo - mcols[0], el + 1 - mcols[0]  # rows of Gmm for the orders lo..el
            p0, p1 = o - el, o + el + 1  # Delta^el vanishes for |m'| > el
            # D[m', m] = Delta_{m' m} Delta_{m' 0}
            D = dl[p0:p1, lo + o:el + 1 + o] * dl[p0:p1, o][:, None]
            Ssum = np.sum(Gmm[:, i0:i1, p0:p1] * D.T[None, :, :], axis=2)
            pref = (1j ** (ms % 4)) * pref_l[el]
            flm[s0:s1, elm2ind(el, ms)] = pref[None, :] * Ssum
        del Gmm
    if reality:
        flm = _complete_real(flm.T, L).T
    return flm


# ----------------------------------------------------------------------------- inverse
def inverse(flm, L, thetas="lambda", reality=False):
    """Inverse spherical harmonic transform (synthesis) on the ``L**2`` samples.

    For every order the profile ``G_m(theta_k) = 2 pi sum_l f_l^m Ptilde_l^m(theta_k)``
    is formed on the ``L`` rings and ring ``k`` is synthesised by an inverse
    FFT of length ``2k + 1`` of the orders folded modulo ``2k + 1`` (exact).

    Parameters
    ----------
    flm : array_like, shape (L**2,) or (S, L**2)
        Coefficients, ``flm[..., l**2 + l + m] = f_l^m``.
    L : int
        Harmonic band-limit.
    thetas : array_like or str, optional
        Ring co-latitudes or a placement name (default ``"lambda"``).
    reality : bool, optional
        The coefficients satisfy ``f_{l,-m} = (-1)^m conj(f_lm)``: only the
        orders ``m >= 0`` are used and a real array is returned.

    Returns
    -------
    ndarray, shape (L**2,) or (S, L**2)
        Samples in the ring-by-ring layout (real if ``reality=True``).
    """
    L = _check_L(L)
    thetas = ordered_theta(L, thetas)
    A, squeeze = _as_batch(flm, L, "flm")
    A = A.T  # (N, S)
    S = A.shape[1]
    N = L * L
    G = np.zeros((2 * L - 1, L, S), dtype=complex)  # G[m + L - 1, k, s]
    for m in range(L):
        ell = np.arange(m, L)
        P = legmat(thetas, L, m)
        G[m + L - 1] = 2 * np.pi * (P.T @ A[elm2ind(ell, m)])
        if m > 0 and not reality:
            G[-m + L - 1] = 2 * np.pi * (-1) ** m * (P.T @ A[elm2ind(ell, -m)])
    if reality:
        G[:L - 1] = np.conj(G[L:][::-1])
    Cfold = np.zeros((S, N), dtype=complex)
    mvec = np.arange(-(L - 1), L)
    for k in range(L):
        n = 2 * k + 1
        c = np.zeros((n, S), dtype=complex)
        np.add.at(c, mvec % n, G[:, k, :])
        Cfold[:, k * k:(k + 1) * (k + 1)] = c.T / (2 * np.pi)
    f = _ring_iffts(Cfold, L)
    if reality:
        f = f.real
    return f[0] if squeeze else f


# ----------------------------------------------------------------------------- multi-pass
def multipass(f, L, fwd, passes=2, thetas="lambda", reality=False, verbose=False):
    """Multi-pass refinement of a forward transform by residual iteration.

    ``flm_1 = fwd(f)``; then for ``p = 2..passes``: ``r = f - inverse(flm)``
    and ``flm <- flm + fwd(r)``.  A signal stops when the maximum of its
    residual stops decreasing (Nafees et al., 2017) and keeps its last
    estimate.

    Parameters
    ----------
    f : array_like, shape (L**2,) or (S, L**2)
        Samples.
    L : int
        Harmonic band-limit.
    fwd : callable
        One forward pass, ``g -> flm`` (same shapes as ``f``), for example
        ``lambda g: osht.forward_pass(g, L, thetas)``.
    passes : int, optional
        Maximum number of passes (default 2).
    thetas, reality
        As in :func:`inverse`, used for the residual synthesis.
    verbose : bool, optional
        Print the residual maximum before every pass.

    Returns
    -------
    flm : ndarray
        Final estimate.
    info : dict
        ``res_max`` (list, per computed residual, of the residual maxima
        per signal; NaN for signals that had stopped) and ``passes`` (number
        of passes run).
    """
    L = _check_L(L)
    if int(passes) < 1:
        raise ValueError("passes must be >= 1")
    thetas = ordered_theta(L, thetas)
    F, squeeze = _as_batch(f, L, "f", reality)
    flm = np.atleast_2d(fwd(F if not squeeze else F[0]))
    S = F.shape[0]
    res = []
    active = np.ones(S, dtype=bool)
    npass = 1
    for p in range(2, int(passes) + 1):
        r = F - inverse(flm, L, thetas, reality=reality)
        rmax = np.abs(r).max(axis=1)
        if res:
            active &= rmax < np.nan_to_num(res[-1], nan=-np.inf)
        res.append(np.where(active, rmax, np.nan))
        if verbose:
            print(f"  residual before pass {p}: max {np.nanmax(rmax):.3e}, active signals {active.sum()}/{S}")
        if not active.any():
            break
        corr = np.atleast_2d(fwd(r if not squeeze else r[0]))
        flm = flm + corr * active[:, None]
        npass = p
    if res:
        scale = max(np.abs(F).max(), np.finfo(float).tiny)
        arr = np.array(res)  # (residuals computed, S); NaN once a signal has stopped
        last = np.array([a[np.isfinite(a)][-1] if np.any(np.isfinite(a)) else 0.0 for a in arr.T])
        worst = last.max() / scale
        if worst > RESIDUAL_WARNING:
            warnings.warn(
                f"the sample residual before the last pass was {worst:.2e} relative to the signal "
                f"(L = {L}); the transform has not converged. See the documentation on stability "
                "and placements.", RuntimeWarning, stacklevel=2)
    flm = flm[0] if squeeze else flm
    return flm, {"res_max": res, "passes": npass}


# ----------------------------------------------------------------------------- forward (recommended)
def forward(f, L, thetas="lambda", backend="eq", passes=2, reality=False, return_info=False, verbose=False):
    """Forward spherical harmonic transform on the ``L**2`` samples (recommended entry point).

    Interpolated elimination with the chosen back-end, refined by the
    multi-pass correction (default two passes), on the Lambda-placement of
    the rings by default.

    Parameters
    ----------
    f : array_like, shape (L**2,) or (S, L**2)
        Samples of one signal, or of ``S`` signals, in the ring-by-ring
        layout of :class:`osht.Sampling`.
    L : int
        Harmonic band-limit.
    thetas : array_like or str, optional
        Ring co-latitudes or a placement name: ``"lambda"`` (default),
        ``"kappa"`` or ``"alternating"``.
    backend : {"eq", "gl"}, optional
        ``"eq"`` (default): equiangular back-end on the profiles of all
        rings.  ``"gl"``: Gauss-Legendre quadrature of the interpolated
        profiles.
    passes : int, optional
        Maximum number of passes of the multi-pass refinement (default 2;
        ``1`` is the single-pass transform).  The refinement stops early
        when the residual stops decreasing.
    reality : bool, optional
        The signal is real; halves the work.
    return_info : bool, optional
        Also return a dictionary with ``lambda_small``, ``lambda_gl``,
        ``res_max`` and ``passes``.
    verbose : bool, optional
        Print the residual maxima of the refinement.

    Returns
    -------
    flm : ndarray, shape (L**2,) or (S, L**2)
        Coefficients, ``flm[..., l**2 + l + m] = f_l^m``.
    info : dict
        Only if ``return_info=True``.

    Examples
    --------
    >>> import numpy as np, osht
    >>> L = 32
    >>> flm = osht.random_flm(L, rng=np.random.default_rng(0))
    >>> f = osht.inverse(flm, L)
    >>> np.abs(osht.forward(f, L) - flm).max() < 1e-12
    True
    """
    L = _check_L(L)
    thetas = ordered_theta(L, thetas)
    info = {}

    def fwd(g):
        out, inf = forward_pass(g, L, thetas, backend=backend, reality=reality, return_info=True)
        if not info:
            info.update(inf)
        return out

    flm, inf = multipass(f, L, fwd, passes=passes, thetas=thetas, reality=reality, verbose=verbose)
    info.update(inf)
    if return_info:
        return flm, info
    return flm


# ----------------------------------------------------------------------------- reference (linear solves)
def forward_lu(f, L, thetas="lambda", return_info=False):
    """Forward transform by per-order linear solves (reference of the 2014 paper).

    For ``m = L-1`` down to ``0`` the ``(L-m) x (L-m)`` system
    ``P_m f^m = g_m`` with ``P_m = 2 pi [Ptilde_l^m(theta_k)]``,
    ``k, l >= m``, is solved by LU for ``f^{+m}`` and ``f^{-m}``, and the
    contribution of the orders ``+-m`` is subtracted from the rings
    ``k < m``.  The solves cost O(L**4) in total.  Provided for the
    comparison of the interpolated elimination with the linear solves on
    the same ring placement.

    Parameters
    ----------
    f : array_like, shape (L**2,) or (S, L**2)
    L : int
    thetas : array_like or str, optional
    return_info : bool, optional
        Also return ``{"kappa": kappa}`` with ``kappa[m] = kappa(P_m)``.

    Returns
    -------
    flm : ndarray
    info : dict
        Only if ``return_info=True``.
    """
    L = _check_L(L)
    thetas = ordered_theta(L, thetas)
    F, squeeze = _as_batch(f, L, "f")
    S = F.shape[0]
    N = L * L
    ks, n_k, k2 = _ring_bins(L)
    R = _ring_ffts(F, L)
    flm = np.zeros((N, S), dtype=complex)
    kappa = np.zeros(L)
    for m in range(L - 1, -1, -1):
        ell = np.arange(m, L)
        nk = n_k[m:]
        gp = ((2 * np.pi / nk) * R[:, k2[m:] + m]).T  # (L-m, S)
        gn = ((2 * np.pi / nk) * R[:, k2[m:] + ((-m) % nk)]).T
        P = 2 * np.pi * legmat(thetas, L, m)  # (L-m, L), rows ell
        Pm = P[:, m:].T  # (L-m, L-m), rows k >= m
        if return_info:
            kappa[m] = np.linalg.cond(Pm)
        fp = np.linalg.solve(Pm, gp)
        fn = (-1) ** m * np.linalg.solve(Pm, gn)  # Ptilde_l^{-m} = (-1)^m Ptilde_l^m
        flm[elm2ind(ell, m)] = fp
        if m > 0:
            flm[elm2ind(ell, -m)] = fn
            gp_small = P[:, :m].T @ fp  # (m, S)
            gn_small = (-1) ** m * (P[:, :m].T @ fn)
            nk = n_k[:m]
            R[:, k2[:m] + (m % nk)] -= ((nk / (2 * np.pi)) * gp_small.T)
            R[:, k2[:m] + ((-m) % nk)] -= ((nk / (2 * np.pi)) * gn_small.T)
    flm = flm.T
    flm = flm[0] if squeeze else flm
    if return_info:
        return flm, {"kappa": kappa}
    return flm
