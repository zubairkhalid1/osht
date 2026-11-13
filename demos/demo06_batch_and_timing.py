"""Demo 6: batches of signals and the cost of the transforms versus L.

Several signals can be transformed at once (shape (S, L**2)); the sweep and
the back-ends are shared, so a batch costs little more than a single signal
at small L.  The second part times the inverse, one forward pass with each
back-end and the linear-solve reference for a few band-limits.

Run:  python demos/demo06_batch_and_timing.py [Lmax]
"""

import sys
import time

import numpy as np

import osht

Lmax = int(sys.argv[1]) if len(sys.argv) > 1 else 256
rng = np.random.default_rng(4)

L, S = 64, 8
F = osht.random_flm(L, rng, size=S)
f = osht.inverse(F, L)
t0 = time.time()
rec = osht.forward(f, L)
t_batch = time.time() - t0
t0 = time.time()
rec_one = osht.forward(f[0], L)
t_one = time.time() - t0
e = osht.errors(F, rec)
print(f"batch of {S} signals at L = {L}: E_max per signal {np.array2string(e.E_max, precision=1)}; "
      f"{t_batch:.2f} s for the batch, {t_one:.2f} s for one signal")

print(f"\n{'L':>5s} {'inverse':>9s} {'fwd eq':>9s} {'fwd gl':>9s} {'LU':>9s} {'E_max eq':>10s}")
L = 32
while L <= Lmax:
    flm = osht.random_flm(L, rng)
    t0 = time.time()
    f = osht.inverse(flm, L)
    ti = time.time() - t0
    t0 = time.time()
    r_eq = osht.forward_pass(f, L)
    te = time.time() - t0
    t0 = time.time()
    osht.forward_pass(f, L, backend="gl")
    tg = time.time() - t0
    t0 = time.time()
    osht.forward_lu(f, L)
    tl = time.time() - t0
    print(f"{L:5d} {ti:9.2f} {te:9.2f} {tg:9.2f} {tl:9.2f} {osht.errors(flm, r_eq).E_max:10.1e}")
    L *= 2
print("times in seconds (pure NumPy); the sweep and the back-ends are O(L^3), the linear solves O(L^4)")
