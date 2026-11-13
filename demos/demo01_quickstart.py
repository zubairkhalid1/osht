"""Demo 1: a spectral-spatial-spectral round trip on the L^2 samples.

Generates a random band-limited test signal with known coefficients, synthesises
it on the L^2 samples by the exact inverse, recovers the coefficients by the
forward transform and reports the error measures.  By Theorem 1 of the paper any
difference is numerical error.

Run:  python demos/demo01_quickstart.py [L]
"""

import sys

import numpy as np

import osht

L = int(sys.argv[1]) if len(sys.argv) > 1 else 64
rng = np.random.default_rng(0)

flm = osht.random_flm(L, rng)  # f_l^m at flm[l**2 + l + m], uniform [-1, 1] real and imaginary parts
f = osht.inverse(flm, L)  # L**2 samples, ring by ring (ring k: positions k**2 .. (k+1)**2 - 1)

print(f"L = {L}: {L * L} samples on {L} rings, Lambda-placement, equiangular back-end")
for passes in (1, 2):
    rec, info = osht.forward(f, L, passes=passes, return_info=True)
    e = osht.errors(flm, rec)
    print(f"  {passes} pass(es): E_max = {e.E_max:.2e}, E_mean = {e.E_mean:.2e}, SNR = {e.SNR:.1f} dB")

rec_gl = osht.forward_pass(f, L, backend="gl")
print(f"  GL back-end, one pass: E_max = {osht.errors(flm, rec_gl).E_max:.2e}")

rec_lu = osht.forward_lu(f, L)
print(f"  linear-solve reference (2014 transform): E_max = {osht.errors(flm, rec_lu).E_max:.2e}")

# the sampling scheme itself
s = osht.Sampling(L)
print(f"  ring 0 at theta = {s.thetas[0]:.4f} (south pole, 1 sample); "
      f"ring {L - 1} at theta = {s.thetas[-1]:.4f} ({2 * L - 1} samples)")
