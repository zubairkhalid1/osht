"""Demo 2: a real signal (Earth topography) with the real-signal fast path.

Synthesises the Earth topography from its stored coefficients (band-limit 128),
recovers the coefficients with reality=True (orders m < 0 by conjugate symmetry,
about half the work) and compares with the complex path.  With matplotlib
installed the signal and the error spectrum are plotted.

Run:  python demos/demo02_real_signal_earth.py [L <= 128]
"""

import sys
import time

import numpy as np

import osht

L = int(sys.argv[1]) if len(sys.argv) > 1 else 128
flm, L = osht.data.earth_topography(L)

t0 = time.time()
f = osht.inverse(flm, L, reality=True)
print(f"L = {L}: synthesis {time.time() - t0:.2f} s; samples are real ({f.dtype}), "
      f"range {f.min():.0f} .. {f.max():.0f} m")

t0 = time.time()
rec_real = osht.forward(f, L, reality=True)
t_real = time.time() - t0
t0 = time.time()
rec_cplx = osht.forward(f.astype(complex), L)
t_cplx = time.time() - t0
e_real, e_cplx = osht.errors(flm, rec_real), osht.errors(flm, rec_cplx)
print(f"  reality=True : E_max = {e_real.E_max:.2e}, SNR = {e_real.SNR:.1f} dB, {t_real:.2f} s")
print(f"  complex path : E_max = {e_cplx.E_max:.2e}, SNR = {e_cplx.SNR:.1f} dB, {t_cplx:.2f} s")

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = osht.plotting.plot_sphere(f, L, view=(20, -40))
    ax.set_title(f"Earth topography on {L * L} samples")
    fig.savefig("demo02_earth.png", dpi=120)
    err = np.abs(rec_real - flm) + 1e-18
    fig, ax = osht.plotting.plot_spectral(err, L, log=True, clabel="log10 |error|")
    fig.savefig("demo02_error_spectrum.png", dpi=120)
    plt.close("all")
    print("  wrote demo02_earth.png and demo02_error_spectrum.png")
except ImportError:
    print("  (install matplotlib to plot the signal and the error spectrum)")
