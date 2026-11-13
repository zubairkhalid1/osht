"""Demo 8: accuracy versus band-limit after J = 1, 2 (or 3) passes.

The spectral-spatial-spectral experiment behind the accuracy figure of the
paper (see paper/README.md).  For every band-limit L and every random test signal, the coefficients f_l^m
(real and imaginary parts uniform in [-1, 1]) are synthesised on the L^2
samples by the exact inverse and recovered by the forward transform with the
equiangular back-end after J = 1, ..., --passes passes (default 2; a third
pass gains nothing over the second), on the Lambda- and the
kappa-placement.  Further passes run on the sample residual and a signal
stops when the maximum of its residual stops decreasing (the rule of
osht.multipass); a stopped signal keeps its last estimate.

Three figures are written, combined in the paper as subfigures:

    accuracy_emean   E_mean versus L
    accuracy_emax    E_max versus L
    accuracy_snr     SNR versus L

each 5.8 cm wide (0.32\\textwidth of an IEEEtran paper) with 8 pt Times text,
as PDF and 300 dpi PNG.  The errors are averaged over the signals.

Every signal is saved in <out>/accuracy_vs_L.npz as soon as it is done, so
the run can be stopped and restarted; the plots are drawn from whatever has
been computed.  Run time is dominated by the largest band-limit: a few minutes
per signal at L = 1024 and about three quarters of an hour at L = 2048.

The results of the paper are stored in the folder paper/ of the repository;
"--out paper --Lmax 2048 --plot-only" draws the figures from them and
"--out paper --Lmax 2048" recomputes whatever is missing there.

Run:  python demos/demo08_accuracy_vs_L.py [options]
      --signals N     signals per band-limit (default 5)
      --Lmax L        largest band-limit, a power of two from 16 (default 1024);
                      a larger value appends to the saved results, so
                      --Lmax 2048 after a default run computes L = 2048 only
      --out DIR       output folder (default accuracy_vs_L in the current
                      directory)
      --passes J      passes computed and plotted, 1 to 3 (default 2); saved
                      results with more passes are plotted up to J, and a
                      band-limit saved with fewer is recomputed
      --plot-only     redraw the figures from the saved results
"""

import argparse
import os
import time

import numpy as np

import osht

ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("--signals", type=int, default=5)
ap.add_argument("--Lmax", type=int, default=1024)
ap.add_argument("--out", default="accuracy_vs_L")
ap.add_argument("--passes", type=int, default=2, choices=(1, 2, 3))
ap.add_argument("--plot-only", action="store_true")
args = ap.parse_args()

L_list = [L for L in (16, 32, 64, 128, 256, 512, 1024, 2048, 4096) if L <= args.Lmax]
placements = ["lambda", "kappa"]
K = 3  # passes stored in the cache (at most)
J = args.passes  # passes computed and plotted
seed = 2026
N = args.signals
os.makedirs(args.out, exist_ok=True)
cache = os.path.join(args.out, "accuracy_vs_L.npz")

# ----------------------------------------------------------------------------- results cache
# The cache holds every band-limit computed so far.  A run with a larger --Lmax
# keeps the saved band-limits and appends the new ones; a run with a smaller
# --Lmax computes and plots only up to it but keeps the rest in the file.
old = None
if os.path.isfile(cache):
    with np.load(cache) as z:
        old = {k: z[k] for k in z.files}
    if int(old["seed"]) != seed or int(old["K"]) != K or old["E_max"].shape[3] != N:
        raise SystemExit(f"{cache} was made with other settings (signals, passes, seed); "
                         "rename it or match them")
L_all = sorted(set(L_list) | (set(int(x) for x in old["L_list"]) if old is not None else set()))
shape = (len(placements), len(L_all), K, N)
D = {"L_list": np.array(L_all), "seed": np.array(seed), "K": np.array(K),
     "E_max": np.full(shape, np.nan), "E_mean": np.full(shape, np.nan), "SNR": np.full(shape, np.nan),
     "passes_run": np.zeros(shape[:2] + (N,), dtype=int), "done": np.zeros(shape[:2] + (N,), dtype=bool)}
if old is not None:  # copy the saved band-limits into their places
    idx = [L_all.index(int(L)) for L in old["L_list"]]
    for key in ("E_max", "E_mean", "SNR", "passes_run", "done"):
        D[key][:, idx] = old[key]

# a (placement, L, signal) is done for this run when it has all J passes
have = D["done"] & np.isfinite(D["E_max"][:, :, J - 1, :])

# ----------------------------------------------------------------------------- experiment
if not args.plot_only:
    for L in L_list:
        i = L_all.index(L)
        if have[:, i].all():
            continue
        thetas = {p: osht.ordered_theta(L, p) for p in placements}  # stored placements
        for s in range(N):
            if have[:, i, s].all():
                continue
            t0 = time.time()
            flm0 = osht.random_flm(L, np.random.default_rng(seed + 1000 * L + s))
            for pp, p in enumerate(placements):
                if have[pp, i, s]:
                    continue
                for key in ("E_max", "E_mean", "SNR"):  # start afresh (fewer passes saved before)
                    D[key][pp, i, :, s] = np.nan
                th = thetas[p]
                f = osht.inverse(flm0, L, th)
                flm = osht.forward_pass(f, L, th)
                e = osht.errors(flm0, flm)
                for key, val in zip(("E_max", "E_mean", "SNR"), (e.E_max, e.E_mean, e.SNR)):
                    D[key][pp, i, 0, s] = val
                npass, res_prev = 1, np.inf
                for j in range(2, J + 1):  # further passes on the residual
                    r = f - osht.inverse(flm, L, th)
                    rmax = np.abs(r).max()
                    if rmax >= res_prev:
                        break
                    res_prev = rmax
                    flm = flm + osht.forward_pass(r, L, th)
                    npass = j
                    e = osht.errors(flm0, flm)
                    for key, val in zip(("E_max", "E_mean", "SNR"), (e.E_max, e.E_mean, e.SNR)):
                        D[key][pp, i, j - 1, s] = val
                for j in range(npass, J):  # a stopped signal keeps its last estimate
                    for key in ("E_max", "E_mean", "SNR"):
                        D[key][pp, i, j, s] = D[key][pp, i, npass - 1, s]
                D["passes_run"][pp, i, s] = npass
                D["done"][pp, i, s] = have[pp, i, s] = True
                np.savez(cache, **D)
            e_lam = " ".join(f"{x:.1e}" for x in D["E_max"][0, i, :J, s])
            e_kap = " ".join(f"{x:.1e}" for x in D["E_max"][1, i, :J, s])
            print(f"L = {L:4d}, signal {s + 1}/{N}: E_max after J = 1..{J}: "
                  f"Lambda {e_lam} | kappa {e_kap}  ({time.time() - t0:.1f} s)", flush=True)


# ----------------------------------------------------------------------------- figures
def nanmean_last(x):
    """Mean over the signals (last axis), ignoring NaN; NaN where nothing is computed."""
    ok = np.isfinite(x)
    c = ok.sum(axis=-1)
    m = np.where(ok, x, 0.0).sum(axis=-1) / np.maximum(c, 1)
    return np.where(c > 0, m, np.nan)


import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, LogLocator, NullFormatter  # noqa: E402

CM = 1 / 2.54
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",  # Times-like mathematics
    "font.size": 8,
    "axes.linewidth": 0.5,
    "lines.linewidth": 1.0,
    "lines.markersize": 3.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

blue = np.array([42, 120, 214]) / 255  # Lambda-placement
orange = np.array([217, 98, 43]) / 255  # kappa-placement
pstyle = ["-", "--"]
pshort = [r"$\Lambda$", r"$\kappa$"]
markers = ["o", "s", "^"]

W8, H8 = 5.8, 5.0  # cm, as in the MATLAB figures
# axes position (figure fraction), with room for the rotated tick labels
pos8 = [1.30 / W8, 1.10 / H8, 4.30 / W8, 3.80 / H8]
sel = [i for i, L in enumerate(L_all) if L <= args.Lmax and have[:, i].any()]  # what is plotted
if not sel:
    raise SystemExit("nothing computed yet")
Ls = np.array(L_all)[sel]
Y = [nanmean_last(D[k][:, sel])[:, :, :J] for k in ("E_mean", "E_max", "SNR")]  # (2, nL, J)
names = ["accuracy_emean", "accuracy_emax", "accuracy_snr"]
ylabels = [r"E$_\mathrm{mean}$", r"E$_\mathrm{max}$", "SNR (dB)"]
legend_loc = ["upper left", "upper left", "lower left"]

for q in range(3):
    fig = plt.figure(figsize=(W8 * CM, H8 * CM))
    ax = fig.add_axes(pos8)
    for pp in range(2):
        base = blue if pp == 0 else orange
        for j in range(J):
            col = (1 - 0.3 * j) * base + 0.3 * j * np.ones(3)  # lighter for more passes
            ax.plot(Ls, Y[q][pp, :, j], pstyle[pp], marker=markers[j], color=col, mfc=col, mec=col,
                    label=f"J = {j + 1}, {pshort[pp]}")
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(FixedLocator(Ls))
    ax.set_xticklabels([str(L) for L in Ls], rotation=30)
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(Ls[0] / 1.25, Ls[-1] * 1.25)
    if q < 2:
        ax.set_yscale("log")
        lo, hi = 1e-16, 1e-6
        data = Y[q][np.isfinite(Y[q])]
        if data.size:  # widen the paper's range only if the data leave it
            lo = min(lo, 10 ** np.floor(np.log10(data.min())))
            hi = max(hi, 10 ** np.ceil(np.log10(data.max()) + 2))
        ax.set_ylim(lo, hi)
        ax.yaxis.set_major_locator(FixedLocator(10.0 ** np.arange(np.log10(lo), np.log10(hi) + 1, 2)))
        ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(1, 10), numticks=40))
        ax.yaxis.set_minor_formatter(NullFormatter())
    else:
        lo, hi = 100, 350
        data = Y[q][np.isfinite(Y[q])]
        if data.size:
            lo = min(lo, 50 * np.floor(data.min() / 50))
        ax.set_ylim(lo, hi)
        ax.set_yticks(np.arange(lo, hi + 1, 50))
    ax.grid(True, which="major", lw=0.4, color="0.85")
    ax.grid(True, which="minor", axis="y", lw=0.3, color="0.92", ls=":")
    ax.set_axisbelow(True)
    ax.set_xlabel("band-limit L", labelpad=1)
    ax.set_ylabel(ylabels[q])
    leg = ax.legend(loc=legend_loc[q], fontsize=8, frameon=True, fancybox=False, edgecolor="0.75",
                    framealpha=1, borderpad=0.3, labelspacing=0.15, handlelength=2.2, handletextpad=0.4,
                    borderaxespad=0.35)
    leg.get_frame().set_linewidth(0.4)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(args.out, f"{names[q]}.{ext}"), dpi=300)
    plt.close(fig)
    print(f"saved {os.path.join(args.out, names[q])} (.pdf, .png)")

n_done = have.all(axis=0).sum(axis=-1)
print("signals done per L: " + ", ".join(f"{L}: {n}" for L, n in zip(L_all, n_done)))
