"""Plotting helpers (require matplotlib, an optional dependency).

The helpers return the matplotlib ``Figure`` and ``Axes`` so that the caller
can adjust and save them; nothing is shown automatically.
"""

from __future__ import annotations

import numpy as np

from .harmonics import ind2elm
from .sampling import Sampling

__all__ = ["plot_sampling", "plot_sphere", "plot_spectral"]


def _mpl():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError("osht.plotting needs matplotlib: pip install osht[plot]") from exc
    return plt


def plot_sampling(L, thetas="lambda", ax=None, view=(30, -60), marker_size=12, color="C0"):
    """Plot the ``L**2`` sampling points on the unit sphere.

    Parameters
    ----------
    L : int
        Harmonic band-limit.
    thetas : array_like or str, optional
        Ring co-latitudes or a placement name.
    ax : matplotlib 3D axes, optional
    view : tuple of float, optional
        Elevation and azimuth of the view.
    marker_size : float, optional
    color : color, optional

    Returns
    -------
    fig, ax
    """
    plt = _mpl()
    s = Sampling(L, thetas)
    if ax is None:
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure
    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, np.pi, 30)
    ax.plot_surface(0.98 * np.outer(np.cos(u), np.sin(v)), 0.98 * np.outer(np.sin(u), np.sin(v)),
                    0.98 * np.outer(np.ones_like(u), np.cos(v)), color="whitesmoke", alpha=0.6,
                    linewidth=0, shade=True)
    x, y, z = s.cartesian()
    ax.scatter(x, y, z, s=marker_size, color=color, depthshade=True)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(*view)
    ax.set_axis_off()
    ax.set_title(f"L = {s.L}: {s.nsamples} samples on {s.L} rings")
    return fig, ax


def plot_sphere(f, L, thetas="lambda", ax=None, resolution=None, view=(30, -60), cmap="viridis"):
    """Plot a real signal sampled on the scheme, interpolated onto a regular grid.

    The samples do not form a regular grid, so the signal is interpolated
    (linearly, with periodic copies in longitude) for display;
    interpolation artifacts are possible.

    Parameters
    ----------
    f : array_like, shape (L**2,)
        Real samples in the ring-by-ring layout.
    L : int
        Harmonic band-limit.
    thetas : array_like or str, optional
    ax : matplotlib 3D axes, optional
    resolution : int, optional
        Number of grid points in co-latitude (default ``max(2L, 64)``).
    view : tuple of float, optional
    cmap : str, optional

    Returns
    -------
    fig, ax
    """
    plt = _mpl()
    import matplotlib
    from scipy.interpolate import griddata

    f = np.asarray(f)
    if np.iscomplexobj(f):
        if np.any(np.abs(f.imag) > 1e-10 * max(1.0, np.abs(f).max())):
            raise ValueError("plot_sphere expects a real signal; plot f.real or np.abs(f)")
        f = f.real
    s = Sampling(L, thetas)
    if f.size != s.nsamples:
        raise ValueError("f must have L**2 samples")
    n = resolution or max(2 * s.L, 64)
    TT, FF = np.meshgrid(np.linspace(0, np.pi, n), np.linspace(0, 2 * np.pi, 2 * n))
    tt = np.concatenate([s.theta] * 3)
    pp = np.concatenate([s.phi - 2 * np.pi, s.phi, s.phi + 2 * np.pi])
    ff = np.concatenate([f] * 3)
    V = griddata((tt, pp), ff, (TT, FF), method="linear")
    V = np.where(np.isnan(V), griddata((tt, pp), ff, (TT, FF), method="nearest"), V)
    X, Y, Z = np.sin(TT) * np.cos(FF), np.sin(TT) * np.sin(FF), np.cos(TT)
    if ax is None:
        fig = plt.figure(figsize=(5.5, 5))
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.figure
    norm = plt.Normalize(V.min(), V.max())
    ax.plot_surface(X, Y, Z, facecolors=matplotlib.colormaps[cmap](norm(V)), rstride=1, cstride=1,
                    linewidth=0, antialiased=False, shade=False)
    m = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
    fig.colorbar(m, ax=ax, shrink=0.6)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(*view)
    ax.set_axis_off()
    return fig, ax


def plot_spectral(values, L, ax=None, log=False, cmap="viridis", clabel=None):
    """Plot a real quantity over degrees and orders (for example ``log10 |error|``).

    Parameters
    ----------
    values : array_like, shape (L**2,)
        Real values indexed as the coefficients, ``values[l**2 + l + m]``.
    L : int
        Harmonic band-limit.
    ax : matplotlib axes, optional
    log : bool, optional
        Plot ``log10`` of the values.
    cmap : str, optional
    clabel : str, optional
        Colour bar label.

    Returns
    -------
    fig, ax
    """
    plt = _mpl()
    values = np.asarray(values, dtype=float).ravel()
    if values.size != L * L:
        raise ValueError("values must have L**2 entries")
    M = np.full((L, 2 * L - 1), np.nan)
    el, m = ind2elm(np.arange(L * L))
    M[el, m + L - 1] = np.log10(np.abs(values)) if log else values
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    else:
        fig = ax.figure
    im = ax.imshow(M.T, origin="lower", aspect="auto", cmap=cmap,
                   extent=(-0.5, L - 0.5, -(L - 1) - 0.5, L - 1 + 0.5))
    ax.set_xlabel("degree $\\ell$")
    ax.set_ylabel("order $m$")
    fig.colorbar(im, ax=ax, label=clabel)
    return fig, ax
