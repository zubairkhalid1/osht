"""Demo 3: ring placements and their amplification constants.

Compares the three placements of the rings (Lambda-greedy, kappa-greedy of the
2014 paper, alternating) through the constants that govern the accuracy of the
transforms: Lambda_m (interpolation to the small rings), Lambda_m^GL (to the
Gauss-Legendre nodes) and kappa(P_m) (per-order condition number of the
linear-solve transform), and through the single-pass accuracy.

Run:  python demos/demo03_placements_and_amplification.py [L]
"""

import sys
import time
import warnings

import numpy as np

import osht

L = int(sys.argv[1]) if len(sys.argv) > 1 else 128
rng = np.random.default_rng(1)
flm = osht.random_flm(L, rng)

print(f"L = {L}")
print(f"{'placement':12s} {'max Lambda_m':>13s} {'max Lambda^GL':>14s} {'max kappa':>10s} "
      f"{'E_max eq':>10s} {'E_max gl':>10s} {'E_max LU':>10s}")
for name in ["lambda", "kappa", "alternating"]:
    thetas = osht.ordered_theta(L, name)
    amp = osht.amplification(L, thetas, kappa=True)
    f = osht.inverse(flm, L, thetas)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # the alternating placement is unstable at this L
        e_eq = osht.errors(flm, osht.forward_pass(f, L, thetas)).E_max
        e_gl = osht.errors(flm, osht.forward_pass(f, L, thetas, backend="gl")).E_max
        e_lu = osht.errors(flm, osht.forward_lu(f, L, thetas)).E_max
    print(f"{name:12s} {amp['lambda_small'].max():13.1f} {amp['lambda_gl'].max():14.1f} "
          f"{amp['kappa'].max():10.1f} {e_eq:10.1e} {e_gl:10.1e} {e_lu:10.1e}")

# the greedy searches reproduce the stored placements
Ls = 24
t0 = time.time()
th, obj = osht.search_lambda_placement(Ls)
print(f"\nLambda-greedy search at L = {Ls}: {time.time() - t0:.2f} s, max Lambda_m = {obj.max():.3f}, "
      f"identical to the stored placement: {np.allclose(th, osht.ordered_theta(Ls))}")
t0 = time.time()
th, obj = osht.search_kappa_placement(Ls)
print(f"kappa-greedy search at L = {Ls}: {time.time() - t0:.2f} s, max kappa(P_m) = {obj.max():.3f}, "
      f"identical to the stored placement: {np.allclose(th, osht.ordered_theta(Ls, 'kappa'))}")
print("\nStored Lambda-placements: L = 1..128, 256, 512, 1024, 1536, 2048; "
      "kappa-placements: L = 1..721, 1024, 1536, 2048.")
