"""Run-time figure of the paper from the times saved by demos/demo07_runtime_vs_L.py.

Reads the CSV written by demo 7 (no transforms are run) and writes the
figure, as PDF and as a 300 dpi PNG, sized for a single journal column, with the text set by LaTeX in
Computer Modern so that it matches the body of the paper.  If no LaTeX
installation is found, or it cannot typeset (for example, MiKTeX without the
cm-super package), matplotlib's built-in Computer Modern is used instead.

The dashed line shows the O(L^3) scaling only: it is C * L^3, drawn for
L >= GUIDE_LMIN (where the cubic cost dominates the fixed overheads), with the
constant C chosen so that the line sits at least a factor GUIDE_FACTOR above
every forward-pass time in that range, clear of the data.

Run:  python paper/plot_runtime.py [csv_path] [pdf_path]
      (defaults: runtime.csv and runtime.pdf next to this script; the PNG
       is written beside the PDF)
"""

import os
import shutil
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

here = os.path.dirname(os.path.abspath(__file__))
csv_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "runtime.csv")
pdf_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(here, "runtime.pdf")

GUIDE_FACTOR = 3.0  # the O(L^3) line sits at least this factor above the forward times
GUIDE_LMIN = 128  # draw the O(L^3) line from this L on
FIG_WIDTH = 3.4  # inches, about one column of a two-column journal
FIG_HEIGHT = 2.6
FONT_SIZE = 9  # match \footnotesize to \small of a 10 pt paper; adjust to the journal


def latex_works():
    """True if matplotlib can typeset text with the local LaTeX installation."""
    if shutil.which("latex") is None:
        return False
    import io

    try:
        with plt.rc_context({"text.usetex": True}):
            fig = plt.figure(figsize=(1, 1))
            fig.text(0.5, 0.5, r"$L^3$ test")
            fig.savefig(io.BytesIO(), format="pdf")
            plt.close(fig)
        return True
    except Exception:
        plt.close("all")
        return False


use_tex = latex_works()
plt.rcParams.update({
    "text.usetex": use_tex,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"] if use_tex else ["cmr10"],
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": False,
    "font.size": FONT_SIZE,
    "axes.labelsize": FONT_SIZE,
    "legend.fontsize": FONT_SIZE - 1,
    "xtick.labelsize": FONT_SIZE - 1,
    "ytick.labelsize": FONT_SIZE - 1,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "lines.linewidth": 1.0,
    "lines.markersize": 4,
    "pdf.fonttype": 42,  # embed TrueType fonts (journals reject Type 3)
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
Ls, t_inv, t_fwd = data[:, 0], data[:, 1], data[:, 2]

sel = Ls >= GUIDE_LMIN  # the asymptotic range, where the O(L^3) cost dominates the overheads
L_guide = Ls[sel]
C = GUIDE_FACTOR * np.max(t_fwd[sel] / L_guide ** 3)  # above every forward time in that range
guide = C * L_guide ** 3

fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
ax.loglog(Ls, t_fwd, "o-", color="C0", label="forward (one pass)")
ax.loglog(Ls, t_inv, "s-", color="C1", label="inverse")
ax.loglog(L_guide, guide, "k--", lw=0.8, label=r"$\mathcal{O}(L^3)$")

ax.set_xscale("log", base=2)
ax.set_xticks(Ls)
ax.set_xticklabels([f"${int(x)}$" for x in Ls])
ax.minorticks_off()
ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10, numticks=20))
ax.yaxis.set_minor_locator(matplotlib.ticker.LogLocator(base=10, subs=np.arange(2, 10), numticks=20))
ax.set_xlim(Ls[0] / 1.25, Ls[-1] * 1.25)
lo = min(t_inv.min(), t_fwd.min(), guide.min())
hi = max(t_fwd.max(), guide.max())
ax.set_ylim(lo / 2, 10 ** np.ceil(np.log10(hi)))

ax.set_xlabel(r"band-limit $L$")
ax.set_ylabel(r"run time (s)")
ax.grid(True, which="major", lw=0.3, alpha=0.5)
ax.legend(loc="upper left", frameon=False, handlelength=2.2)

png_path = os.path.splitext(pdf_path)[0] + ".png"
fig.savefig(pdf_path)
fig.savefig(png_path, dpi=300)
print(f"saved {pdf_path} and {png_path} "
      f"({'LaTeX' if use_tex else 'mathtext'} fonts, {FIG_WIDTH} x {FIG_HEIGHT} in)")
