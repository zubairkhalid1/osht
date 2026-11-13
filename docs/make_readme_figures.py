"""Draw the figures shown in README.md and in the documentation.

Reads the stored results of the accuracy and run-time experiments
(paper/accuracy_vs_L.npz and paper/runtime.csv) and writes

    docs/_static/accuracy.png   maximum coefficient error versus band-limit
    docs/_static/runtime.png    run time of one forward pass and one inverse

No transform is run.  The figures are drawn for the web (README, documentation);
the figures of the paper are not stored in the repository.

Run:  python docs/make_readme_figures.py      (needs matplotlib)
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

here = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(here)
out = os.path.join(here, "_static")
os.makedirs(out, exist_ok=True)

BLUE, ORANGE = "#2a78d6", "#eb6834"  # categorical slots 1 and 2 (checked for colour-vision deficiency)
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#ffffff"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.edgecolor": GRID,
    "axes.labelcolor": MUTED,
    "axes.linewidth": 0.8,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.titlecolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "text.color": INK,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "lines.linewidth": 2.0,
    "lines.markersize": 7,
    "legend.frameon": False,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})


def style(ax, Ls):
    """Log-2 band-limit axis, horizontal grid only, no top and right spines."""
    ax.set_xscale("log", base=2)
    ax.set_xticks(Ls)
    ax.set_xticklabels([str(int(L)) for L in Ls])
    ax.set_xlim(Ls[0] / 1.3, Ls[-1] * 2.4)  # room on the right for the end labels
    ax.set_yscale("log")
    ax.minorticks_off()
    ax.grid(True, axis="y", which="major")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
    ax.set_xlabel("band-limit L")


def marker(color):
    """Filled marker with a ring in the surface colour."""
    return {"marker": "o", "mfc": color, "mec": SURFACE, "mew": 1.5}


# ----------------------------------------------------------------------------- accuracy
with np.load(os.path.join(root, "paper", "accuracy_vs_L.npz")) as z:
    Ls = z["L_list"].astype(float)
    E = np.nanmean(z["E_max"], axis=-1)  # (placement, L, pass): mean over the signals

fig, ax = plt.subplots(figsize=(6.4, 4.2))
series = [
    (0, 0, BLUE, "-", r"$\Lambda$-placement, one pass"),
    (0, 1, BLUE, "--", r"$\Lambda$-placement, two passes"),
    (1, 0, ORANGE, "-", r"$\kappa$-placement, one pass"),
    (1, 1, ORANGE, "--", r"$\kappa$-placement, two passes"),
]
for pp, j, color, ls, label in series:
    ax.plot(Ls, E[pp, :, j], ls, color=color, label=label, **marker(color))
style(ax, Ls)
ax.set_ylim(1e-16, 1e1)
ax.set_yticks(10.0 ** np.arange(-16, 2, 4))
ax.set_ylabel("maximum coefficient error")
ax.set_title("Accuracy of the forward transform on L² samples", loc="left")
ax.legend(loc="upper left", handlelength=3.2)
# direct labels: the errors at the largest band-limit
for pp, j, _, _, _ in series:
    ax.annotate(f"{E[pp, -1, j]:.1e}", (Ls[-1], E[pp, -1, j]), xytext=(9, 0),
                textcoords="offset points", ha="left", va="center", fontsize=9, color=MUTED)
fig.tight_layout()
fig.savefig(os.path.join(out, "accuracy.png"), dpi=200)
plt.close(fig)

# ----------------------------------------------------------------------------- run time
data = np.loadtxt(os.path.join(root, "paper", "runtime.csv"), delimiter=",", skiprows=1)
Ls, t_inv, t_fwd = data[:, 0], data[:, 1], data[:, 2]
sel = Ls >= 128
guide = 3.0 * np.max(t_fwd[sel] / Ls[sel] ** 3) * Ls[sel] ** 3  # O(L^3), clear of the data

fig, ax = plt.subplots(figsize=(6.4, 4.2))
ax.plot(Ls[sel], guide, "-", color=MUTED, lw=1.0)
ax.plot(Ls, t_fwd, "-", color=BLUE, label="forward (one pass)", **marker(BLUE))
ax.plot(Ls, t_inv, "-", color=ORANGE, label="inverse", **marker(ORANGE))
style(ax, Ls)
ax.set_ylim(3e-4, 1e4)
ax.set_ylabel("run time (s)")
ax.set_title("Run time of the transforms (pure NumPy, laptop)", loc="left")
ax.legend(loc="upper left")
ax.annotate(r"$\propto L^3$", (Ls[sel][2], guide[2]), xytext=(-6, 6), textcoords="offset points",
            ha="right", va="bottom", fontsize=10, color=MUTED)
for t, dy in ((t_fwd[-1], 5), (t_inv[-1], -5)):  # direct labels: the times at the largest band-limit
    ax.annotate(f"{t:.0f} s", (Ls[-1], t), xytext=(9, dy), textcoords="offset points",
                ha="left", va="center", fontsize=9, color=MUTED)
fig.tight_layout()
fig.savefig(os.path.join(out, "runtime.png"), dpi=200)
plt.close(fig)
print(f"wrote {os.path.join(out, 'accuracy.png')} and {os.path.join(out, 'runtime.png')}")
