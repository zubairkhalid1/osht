"""Demo 7: run time of one forward pass and one inverse transform versus L.

For L = 16, 32, 64, ..., Lmax this times one forward pass (interpolated
elimination, equiangular back-end) and one inverse transform on the
Lambda-placement.  Each time is the best of a few repeats, and the ring
placement is looked up before the clock starts, so only the transform is
timed.  The times are saved to a CSV file and plotted against L on log-log
axes with an O(L^3) guide line.

A band-limit with no stored or cached Lambda-placement is skipped with a
message (placements are shipped for every power of two up to 2048).  The
whole run takes about half an hour, nearly all of it at L = 2048;
"python demos/demo07_runtime_vs_L.py 512" finishes in under a minute.

The times of the paper are stored in paper/runtime.csv and drawn in the
style of the paper by paper/plot_runtime.py (see paper/README.md).

Run:  python demos/demo07_runtime_vs_L.py [Lmax] [output_prefix]
      (defaults: Lmax = 2048, output_prefix = runtime; writes runtime.csv
       and runtime.png in the current directory)
"""

import sys
import time

import numpy as np

import osht

Lmax = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
prefix = sys.argv[2] if len(sys.argv) > 2 else "runtime"
rng = np.random.default_rng(7)


def best_time(fn, repeats):
    """Smallest wall-clock time of `repeats` calls of fn(); also returns the last result."""
    best, out = np.inf, None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - t0)
    return best, out


rows = []  # (L, t_inverse, t_forward, E_max)
print(f"{'L':>5s} {'inverse [s]':>12s} {'forward [s]':>12s} {'E_max':>9s} {'repeats':>8s}")
L = 16
while L <= Lmax:
    try:
        thetas = osht.ordered_theta(L)  # stored or cached Lambda-placement, outside the timing
    except ValueError:
        print(f"{L:5d}  skipped: no stored or cached Lambda-placement "
              f"(run osht.ordered_theta({L}, search=True) once)")
        L *= 2
        continue
    repeats = 5 if L <= 256 else (3 if L <= 512 else 1)
    flm = osht.random_flm(L, rng)
    osht.forward_pass(osht.inverse(flm, L, thetas=thetas), L, thetas=thetas)  # warm-up
    t_inv, f = best_time(lambda flm=flm, L=L, th=thetas: osht.inverse(flm, L, thetas=th), repeats)
    t_fwd, rec = best_time(lambda f=f, L=L, th=thetas: osht.forward_pass(f, L, thetas=th), repeats)
    e_max = osht.errors(flm, rec).E_max
    rows.append((L, t_inv, t_fwd, e_max))
    print(f"{L:5d} {t_inv:12.4f} {t_fwd:12.4f} {e_max:9.1e} {repeats:8d}")
    L *= 2

if not rows:
    sys.exit("no band-limit could be timed")

data = np.array(rows)
csv_path = f"{prefix}.csv"
np.savetxt(csv_path, data, delimiter=",", header="L,inverse_s,forward_pass_s,E_max",
           fmt=["%d", "%.6e", "%.6e", "%.3e"], comments="")
print(f"\ntimes saved to {csv_path}")

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit("matplotlib is not installed (pip install osht[plot]); skipping the plot")

Ls, t_inv, t_fwd = data[:, 0], data[:, 1], data[:, 2]
fig, ax = plt.subplots(figsize=(6, 4.5))
ax.loglog(Ls, t_fwd, "o-", label="forward (one pass)")
ax.loglog(Ls, t_inv, "s-", label="inverse")
ref = t_fwd[-1] * (Ls / Ls[-1]) ** 3  # O(L^3) guide through the largest forward time
ax.loglog(Ls, ref, "k--", lw=1, label=r"$\mathcal{O}(L^3)$")
ax.set_xscale("log", base=2)
ax.set_xticks(Ls)
ax.set_xticklabels([str(int(x)) for x in Ls])
ax.set_xlabel("band-limit L")
ax.set_ylabel("run time [s]")
ax.set_title("osht: run time versus band-limit")
ax.grid(True, which="both", alpha=0.3)
ax.legend()
fig.tight_layout()
png_path = f"{prefix}.png"
fig.savefig(png_path, dpi=150)
print(f"plot saved to {png_path}")
