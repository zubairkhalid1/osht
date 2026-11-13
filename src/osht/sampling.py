"""The optimal-dimensionality sampling scheme and the placement of its rings.

The scheme (Khalid, Kennedy and McEwen, 2014) takes ``L**2`` samples on ``L``
iso-latitude rings: ring ``k = 0, ..., L-1`` lies at co-latitude ``theta_k``
and carries ``2k + 1`` equally spaced samples ``phi_j = 2 pi j / (2k + 1)``.
The ring co-latitudes are the equiangular set

    Theta = { (2t + 1) pi / (2L - 1), t = 0, ..., L-1 }

of McEwen and Wiaux (2011), and a *placement* is the assignment of these
``L`` co-latitudes to the ``L`` rings.  Three placements are provided:

``"lambda"``
    greedy minimisation of the interpolation amplification ``Lambda_m`` of
    the interpolated elimination (the recommended placement; stored for
    ``L = 1..128, 256, 512, 1024, 1536, 2048``);
``"kappa"``
    greedy minimisation of the condition number ``kappa(P_m)`` of the 2014
    paper, the placement of the NSHT package (stored for ``L = 1..721, 1024,
    1536, 2048``);
``"alternating"``
    the analytic ordering of the 2014 paper: the largest ring nearest the
    equator, then alternating between the hemispheres outwards.

Samples are stored ring by ring: ring ``k`` occupies the positions
``k**2, ..., (k+1)**2 - 1`` of every vector of ``L**2`` samples.
"""

from __future__ import annotations

import os
import time
from importlib import resources
from pathlib import Path

import numpy as np

__all__ = [
    "Sampling",
    "indexed_theta",
    "ordered_theta",
    "sampling_points",
    "ring_slice",
    "ring_index",
    "theta_index",
    "is_equiangular",
    "stored_placements",
    "cached_placements",
    "cache_dir",
    "save_placement",
    "search_lambda_placement",
    "search_kappa_placement",
]

_PLACEMENT_FILES = {"lambda": "lambda_placements.npz", "kappa": "kappa_placements.npz"}
_cache: dict = {}


def _load_placements(name):
    if name not in _cache:
        with resources.files("osht.data").joinpath(_PLACEMENT_FILES[name]).open("rb") as fh:
            data = np.load(fh)
            _cache[name] = {k: data[k] for k in data.files}
    return _cache[name]


def stored_placements(name="lambda"):
    """Band-limits for which a placement is shipped with the package.

    Parameters
    ----------
    name : {"lambda", "kappa"}

    Returns
    -------
    list of int
    """
    data = _load_placements(name)
    return sorted(int(k[1:]) for k in data if k.startswith("L"))


# ----------------------------------------------------------------------------- user cache
def cache_dir():
    """Directory in which placements computed by the searches are stored for reuse.

    The environment variable ``OSHT_CACHE_DIR`` overrides the default, which
    is ``%LOCALAPPDATA%\\osht`` on Windows and ``~/.cache/osht`` elsewhere
    (``$XDG_CACHE_HOME/osht`` if that variable is set).

    Returns
    -------
    pathlib.Path
    """
    env = os.environ.get("OSHT_CACHE_DIR")
    if env:
        return Path(env).expanduser()
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / "osht"
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "osht"


def _cache_file(L, name, directory=None):
    return Path(directory or cache_dir()) / f"{name}_placement_L{int(L)}.npz"


def save_placement(L, name, thetas, objective=None, directory=None):
    """Store a placement in the user cache so that :func:`ordered_theta` finds it later.

    Parameters
    ----------
    L : int
        Harmonic band-limit.
    name : {"lambda", "kappa"}
        Kind of placement.
    thetas : array_like, shape (L,)
        Ring co-latitudes, a permutation of the equiangular set.
    objective : array_like, optional
        Objective values of the search (stored alongside).
    directory : path-like, optional
        Cache directory (default :func:`cache_dir`).

    Returns
    -------
    pathlib.Path
        The file written.
    """
    L = _check_L(L)
    name = name.lower()
    if name not in _PLACEMENT_FILES:
        raise ValueError("name must be 'lambda' or 'kappa'")
    thetas = np.asarray(thetas, dtype=float).ravel()
    if not is_equiangular(thetas, L):
        raise ValueError("thetas must be a permutation of the equiangular set (2t+1)pi/(2L-1)")
    path = _cache_file(L, name, directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    obj = np.zeros(0) if objective is None else np.asarray(objective, dtype=float).ravel()
    np.savez(path, t=theta_index(thetas, L).astype(np.int32), objective=obj)
    return path


def _load_cached(L, name, directory=None):
    path = _cache_file(L, name, directory)
    if not path.is_file():
        return None
    with np.load(path) as data:
        t = data["t"].astype(int)
        obj = data["objective"]
    if sorted(t.tolist()) != list(range(L)):
        raise ValueError(f"corrupt cached placement {path}; delete it")
    return np.pi * ((2 * t + 1) / (2 * L - 1)), (obj if obj.size else None)


def cached_placements(name="lambda", directory=None):
    """Band-limits for which a placement is stored in the user cache.

    Parameters
    ----------
    name : {"lambda", "kappa"}
    directory : path-like, optional

    Returns
    -------
    list of int
    """
    d = Path(directory or cache_dir())
    if not d.is_dir():
        return []
    out = []
    for p in d.glob(f"{name}_placement_L*.npz"):
        try:
            out.append(int(p.stem.split("_L")[-1]))
        except ValueError:
            continue
    return sorted(out)


def equiangular_set(L):
    """The ``L`` equiangular co-latitudes ``(2t + 1) pi / (2L - 1)``, ``t = 0..L-1``."""
    L = _check_L(L)
    return np.pi * ((2 * np.arange(L) + 1) / (2 * L - 1))


def indexed_theta(L):
    """Candidate co-latitudes sorted by distance from the equator, descending.

    The equiangular set with the candidate most distant from the equator
    (the south pole, ``theta = pi``) first; of a symmetric pair the northern
    member comes first.  This is also the *alternating* placement of the
    2014 paper: ring ``k`` at ``indexed_theta(L)[k]``.

    Parameters
    ----------
    L : int
        Harmonic band-limit.

    Returns
    -------
    ndarray, shape (L,)
    """
    TT = equiangular_set(L)
    order = np.lexsort((np.arange(L), -np.abs(TT - np.pi / 2)))
    return TT[order]


def theta_index(thetas, L):
    """Index ``t`` of every co-latitude in the equiangular set, or ``None`` if not in it."""
    thetas = np.asarray(thetas, dtype=float).ravel()
    n = 2 * L - 1
    t = np.rint((thetas * n / np.pi - 1) / 2).astype(int)
    if np.any(t < 0) or np.any(t >= L):
        return None
    if np.max(np.abs(thetas - (2 * t + 1) * np.pi / n)) > 1e-10 * np.pi:
        return None
    return t


def is_equiangular(thetas, L):
    """True if ``thetas`` is a permutation of the equiangular set of band-limit ``L``."""
    thetas = np.asarray(thetas, dtype=float).ravel()
    if thetas.size != L:
        return False
    t = theta_index(thetas, L)
    return t is not None and len(set(t.tolist())) == L


def ordered_theta(L, placement="lambda", objective=False, search=False, verbose=True):
    """Co-latitudes of the ``L`` rings for a stored, cached, computed or analytic placement.

    A named placement is looked up, in this order, among the placements
    shipped with the package (:func:`stored_placements`), in the user cache
    (:func:`cached_placements`, written by the searches) and, if
    ``search=True``, computed by the greedy search and saved in the cache so
    that it is never computed again.

    Parameters
    ----------
    L : int
        Harmonic band-limit.
    placement : {"lambda", "kappa", "alternating"} or array_like, optional
        Placement name (default ``"lambda"``).  An array of ``L``
        co-latitudes is returned unchanged (after validation), so that
        every function of the package accepts either form.
    objective : bool, optional
        If True, also return the objective values recorded by the greedy
        search (``Lambda_m`` or ``kappa(P_m)`` at the steps
        ``m = L-2, ..., 0``), or ``None`` for the alternating placement.
    search : bool, optional
        Run the greedy search (and cache the result) when no placement is
        stored or cached for ``L``.  The search costs O(L**4): a few
        seconds at L = 128, about two minutes at L = 512 and half an hour at
        L = 1024.
    verbose : bool, optional
        Print the progress of a search.

    Returns
    -------
    thetas : ndarray, shape (L,)
        ``thetas[k]`` is the co-latitude of ring ``k`` (``2k + 1`` samples).
    objective : ndarray or None
        Only if ``objective=True``.

    Raises
    ------
    ValueError
        If no placement of that kind is stored or cached for ``L`` and
        ``search`` is False.
    """
    L = _check_L(L)
    if not isinstance(placement, str):
        thetas = np.asarray(placement, dtype=float).ravel()
        if thetas.size != L:
            raise ValueError(f"thetas must contain L = {L} co-latitudes, got {thetas.size}")
        if np.any(thetas < -1e-12) or np.any(thetas > np.pi + 1e-12):
            raise ValueError("co-latitudes must lie in [0, pi]")
        if np.unique(np.cos(thetas)).size != L:
            raise ValueError("ring co-latitudes must be distinct")
        return (thetas, None) if objective else thetas
    name = placement.lower()
    if name == "alternating":
        thetas = indexed_theta(L)
        return (thetas, None) if objective else thetas
    if name not in _PLACEMENT_FILES:
        raise ValueError("placement must be 'lambda', 'kappa', 'alternating' or an array of co-latitudes")
    data = _load_placements(name)
    key = f"L{L}"
    if key in data:
        t = data[key].astype(int)
        thetas = np.pi * ((2 * t + 1) / (2 * L - 1))
        return (thetas, data.get(f"obj{L}")) if objective else thetas
    cached = _load_cached(L, name)
    if cached is not None:
        return cached if objective else cached[0]
    if not search:
        raise ValueError(
            f"no stored or cached {name}-placement for L = {L}; stored band-limits are "
            f"{_describe(stored_placements(name))}. Call osht.ordered_theta({L}, '{name}', search=True) "
            f"or osht.search_{name}_placement({L}) once: the result is saved in {cache_dir()} and "
            f"reused afterwards. placement='alternating' needs no search but is unstable beyond small L.")
    if name == "lambda":
        thetas, obj = search_lambda_placement(L, verbose=verbose)
    else:
        thetas, obj = search_kappa_placement(L, verbose=verbose)
    return (thetas, obj) if objective else thetas


def _describe(Ls):
    if not Ls:
        return "none"
    out = []
    start = prev = Ls[0]
    for v in Ls[1:] + [None]:
        if v is not None and v == prev + 1:
            prev = v
            continue
        out.append(f"{start}..{prev}" if prev > start else f"{start}")
        if v is not None:
            start = prev = v
    return ", ".join(out)


def _check_L(L):
    if isinstance(L, (bool, np.bool_)) or int(L) != L or int(L) < 1:
        raise ValueError("the harmonic band-limit L must be a positive integer")
    return int(L)


def ring_slice(k):
    """Slice of the positions of ring ``k`` in a vector of ``L**2`` samples."""
    return slice(k * k, (k + 1) * (k + 1))


def ring_index(L):
    """Ring index ``k`` of every sample position (array of length ``L**2``)."""
    L = _check_L(L)
    return np.repeat(np.arange(L), 2 * np.arange(L) + 1)


def sampling_points(L, thetas="lambda"):
    """Co-latitude and longitude of every sample of the scheme.

    Parameters
    ----------
    L : int
        Harmonic band-limit.
    thetas : array_like or str, optional
        Ring co-latitudes or a placement name (default ``"lambda"``).

    Returns
    -------
    theta, phi : ndarray, shape (L**2,)
        Co-latitude and longitude of the samples in the ring-by-ring layout.
    """
    thetas = ordered_theta(L, thetas)
    L = thetas.size
    theta = np.repeat(thetas, 2 * np.arange(L) + 1)
    phi = np.concatenate([2 * np.pi * np.arange(2 * k + 1) / (2 * k + 1) for k in range(L)])
    return theta, phi


class Sampling:
    """The sampling scheme at band-limit ``L`` for a given placement.

    Parameters
    ----------
    L : int
        Harmonic band-limit.
    thetas : array_like or str, optional
        Ring co-latitudes or a placement name (default ``"lambda"``).

    Attributes
    ----------
    L : int
    thetas : ndarray, shape (L,)
        Co-latitude of ring ``k``.
    theta, phi : ndarray, shape (L**2,)
        Co-latitude and longitude of every sample.
    ring : ndarray, shape (L**2,)
        Ring index of every sample.
    nsamples : int
        ``L**2``.
    equiangular : bool
        Whether the ring set is the equiangular set (needed by the
        equiangular back-end).
    """

    def __init__(self, L, thetas="lambda"):
        self.thetas = ordered_theta(L, thetas)
        self.L = int(self.thetas.size)
        self.theta, self.phi = sampling_points(self.L, self.thetas)
        self.ring = ring_index(self.L)
        self.nsamples = self.L * self.L
        self.equiangular = is_equiangular(self.thetas, self.L)

    def slice(self, k):
        """Slice of the positions of ring ``k``."""
        return ring_slice(k)

    def cartesian(self, radius=1.0):
        """Cartesian coordinates ``(x, y, z)`` of the samples, each of shape ``(L**2,)``."""
        st = np.sin(self.theta)
        return (radius * st * np.cos(self.phi), radius * st * np.sin(self.phi),
                radius * np.cos(self.theta))

    def __repr__(self):
        return f"Sampling(L={self.L}, nsamples={self.nsamples}, equiangular={self.equiangular})"


# ----------------------------------------------------------------------------- searches
def search_lambda_placement(L, include_gl=False, verbose=False, save=True):
    """Greedy ring placement minimising the interpolation amplification ``Lambda_m``.

    ``theta_{L-1}`` is fixed at the candidate nearest the equator and, for
    ``m = L-2, ..., 0``, the unused candidate that minimises
    ``Lambda_m = max_{k < m} sum_{s >= m} |Lambda^m_s(theta_k)|``
    (the amplification of the interpolation to the rings that will be
    cleaned at step ``m``) is chosen as ``theta_m``.  The search costs
    O(L**4): about a second at L = 64, a few seconds at L = 128, two minutes
    at L = 512 and half an hour at L = 1024 in NumPy.  Placements for
    ``L <= 128`` and ``L = 256, 512, 1024, 1536, 2048`` are shipped with the
    package, see :func:`ordered_theta`.

    Parameters
    ----------
    L : int
        Harmonic band-limit.
    include_gl : bool, optional
        Also minimise the amplification on the Gauss-Legendre nodes
        (objective = max of both).
    verbose : bool, optional
        Print progress.
    save : bool, optional
        Store the result in the user cache (:func:`cache_dir`) so that
        :func:`ordered_theta` finds it afterwards (default True).  Nothing
        is stored for a band-limit whose placement is shipped with the
        package, nor for ``include_gl=True``, which is a different placement.

    Returns
    -------
    thetas : ndarray, shape (L,)
        Ring co-latitudes.
    objective : ndarray, shape (L - 1,)
        ``Lambda_m`` attained at the steps ``m = L-2, ..., 0``.
    """
    from .harmonics import gauss_legendre
    from .sampling_functions import bary_weights, lambda_matrix

    L = _check_L(L)
    TT = indexed_theta(L)
    if L == 1:
        return TT, np.zeros(0)
    x_all = np.cos(TT)
    with np.errstate(divide="ignore"):
        logsin_all = np.log(np.abs(np.sin(TT)))
    if include_gl:
        xg, _ = gauss_legendre(L)
        logsin_g = np.log(np.sqrt(1 - xg ** 2))
    chosen = [L - 1]
    used = np.zeros(L, dtype=bool)
    used[L - 1] = True
    obj = np.zeros(L - 1)
    t0 = time.time()
    for m in range(L - 2, -1, -1):
        unused = np.nonzero(~used)[0]
        best, bestv = int(unused[0]), np.inf
        for c in unused:
            nodes = [int(c)] + chosen
            xn, lsn = x_all[nodes], logsin_all[nodes]
            if not np.all(np.isfinite(lsn)) and m > 0:
                continue  # a pole cannot resolve an order m > 0
            logw, sgnw = bary_weights(xn)
            targets = unused[unused != c]
            if m > 0 and targets.size > 0:
                Lam = lambda_matrix(m, xn, lsn, logw, sgnw, x_all[targets], logsin_all[targets])
                v = np.abs(Lam).sum(axis=1).max()
            else:
                v = 0.0
            if include_gl:
                LamG = lambda_matrix(m, xn, lsn, logw, sgnw, xg, logsin_g)
                v = max(v, np.abs(LamG).sum(axis=1).max())
            if v < bestv:
                best, bestv = int(c), v
        chosen.insert(0, best)
        used[best] = True
        obj[L - 2 - m] = bestv
        if verbose and (m % max(1, L // 8) == 0 or m == L - 2):
            print(f"  L = {L}, m = {m:4d}, Lambda_m = {bestv:9.3f}, {time.time() - t0:6.1f} s", flush=True)
    thetas = TT[chosen]
    if save and not include_gl:
        _save_after_search(L, "lambda", thetas, obj, verbose)
    return thetas, obj


def _save_after_search(L, name, thetas, obj, verbose):
    if L in stored_placements(name):
        return
    try:
        path = save_placement(L, name, thetas, obj)
        if verbose:
            print(f"  placement saved to {path}", flush=True)
    except OSError as exc:  # pragma: no cover - read-only home directories
        import warnings

        warnings.warn(f"could not cache the placement: {exc}", stacklevel=3)


def _secular_extreme(d, Z, lo, hi, iters=100):
    Z2 = Z * Z
    lo = lo.copy()
    hi = hi.copy()
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        with np.errstate(divide="ignore", invalid="ignore"):
            f = 1.0 + np.sum(Z2 / (d[None, :] - mid[:, None]), axis=1)
        pos = f > 0
        hi = np.where(pos, mid, hi)
        lo = np.where(pos, lo, mid)
    return 0.5 * (lo + hi)


def _kappas_rank_one(F, C):
    """kappa([F; r]) for every row r of C (F: (n-1, n), C: (c, n)) by the secular equation."""
    G = F.T @ F
    d, V = np.linalg.eigh(G)
    d = np.maximum(d, 0.0)
    Z = C @ V
    znorm2 = np.sum(Z * Z, axis=1)
    c = C.shape[0]
    lam_min = _secular_extreme(d, Z, np.full(c, d[0]), np.full(c, d[1]))
    lam_max = _secular_extreme(d, Z, np.full(c, d[-1]), d[-1] + znorm2)
    with np.errstate(divide="ignore", invalid="ignore"):
        kap = np.sqrt(lam_max / lam_min)
    kap[~np.isfinite(kap)] = np.inf
    return kap


def search_kappa_placement(L, exact=False, verbose=False, save=True):
    """Greedy ring placement minimising ``kappa(P_m)`` (Khalid, Kennedy and McEwen, 2014).

    The placement of the NSHT package.  For ``m = L-2, ..., 0`` the unused
    candidate that minimises the condition number of
    ``P_m = [Ptilde_l^m(theta_k)]``, ``k, l >= m``, is chosen as ``theta_m``.
    Every trial matrix at step ``m`` is a fixed block plus one candidate row,
    so the extreme singular values are found from the secular equation of a
    rank-one update in O(L) per candidate; the whole search is O(L**4)
    (a few minutes at L = 1024) and reproduces the placements of the
    reference O(L**5) search with full SVDs (``exact=True``).  Placements for
    ``L <= 721`` and ``L = 1024, 1536, 2048`` are shipped with the package.

    Parameters
    ----------
    L : int
        Harmonic band-limit.
    exact : bool, optional
        Use a full SVD per candidate (reference; slow).
    verbose : bool, optional
        Print progress.
    save : bool, optional
        Store the result in the user cache (:func:`cache_dir`) so that
        :func:`ordered_theta` finds it afterwards (default True).  Nothing
        is stored for a band-limit whose placement is shipped with the
        package.

    Returns
    -------
    thetas : ndarray, shape (L,)
        Ring co-latitudes.
    objective : ndarray, shape (L - 1,)
        ``kappa(P_m)`` attained at the steps ``m = L-2, ..., 0``.
    """
    from .harmonics import legmat

    L = _check_L(L)
    TT = indexed_theta(L)
    if L == 1:
        return TT, np.zeros(0)
    chosen = [L - 1]
    obj = []
    t0 = time.time()
    for m in range(L - 2, -1, -1):
        n = L - m
        P_all = legmat(TT, L, m).T  # (L, n): rows candidates, columns ell
        F = P_all[chosen, :]
        unused = [i for i in range(L) if i not in set(chosen)]
        C = P_all[unused, :]
        if exact or n <= 3:
            kap = np.array([np.linalg.cond(np.vstack([C[j:j + 1], F])) for j in range(len(unused))])
        else:
            kap = _kappas_rank_one(F, C)
        j = int(np.argmin(kap))
        chosen.insert(0, unused[j])
        obj.append(float(kap[j]))
        if verbose and (m % max(1, L // 8) == 0 or m == L - 2):
            print(f"  L = {L}, m = {m:4d}, kappa(P_m) = {kap[j]:9.3f}, {time.time() - t0:6.1f} s", flush=True)
    thetas = TT[chosen]
    obj = np.array(obj)
    if save:
        _save_after_search(L, "kappa", thetas, obj, verbose)
    return thetas, obj
