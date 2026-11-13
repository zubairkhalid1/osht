"""Demo 4: multi-pass refinement and the oracle sweep.

The interpolated elimination is exact in exact arithmetic; in floating point
the interpolation error of every order propagates down the elimination chain.
This demo shows (a) how the residual iteration (multi-pass refinement) reduces
the error pass by pass, with the stopping rule on the residual, and (b) the
oracle sweep, in which the cleaning step uses the exact profiles so that no
interpolation error propagates: the per-order error then stays at rounding
level for every order, which identifies the propagation as the only loss.

Run:  python demos/demo04_multipass_and_oracle.py [L]
"""

import sys

import numpy as np

import osht

L = int(sys.argv[1]) if len(sys.argv) > 1 else 256
rng = np.random.default_rng(2)
flm = osht.random_flm(L, rng)
f = osht.inverse(flm, L)

print(f"L = {L}, Lambda-placement, equiangular back-end")
flm_pass = []
res_max = []


def fwd(g):
    return osht.forward_pass(g, L)


rec, info = osht.multipass(f, L, fwd, passes=4, verbose=True)
print(f"  passes run: {info['passes']} (stops when the residual stops decreasing)")
for p in range(1, 4):
    rec_p = osht.forward(f, L, passes=p)
    e = osht.errors(flm, rec_p)
    print(f"  after {p} pass(es): E_max = {e.E_max:.2e}, E_mean = {e.E_mean:.2e}, SNR = {e.SNR:.1f} dB")

# error per order, real sweep against oracle sweep
rec_real = osht.forward_pass(f, L)
rec_orc = osht.forward_pass(f, L, flm_oracle=flm)
el, m = osht.ind2elm(np.arange(L * L))
err_real = np.abs(rec_real - flm)
err_orc = np.abs(rec_orc - flm)
print("\n  maximum error per order band (real sweep / oracle sweep):")
bands = [(0, L // 8), (L // 8, L // 2), (L // 2, L)]
for lo, hi in bands:
    sel = (np.abs(m) >= lo) & (np.abs(m) < hi)
    print(f"    {lo:4d} <= |m| < {hi:4d}: {err_real[sel].max():.2e} / {err_orc[sel].max():.2e}")
print("  the oracle error stays at rounding level for every order; the real sweep grows towards m = 0,")
print("  so the single-pass error is set by the propagation along the elimination chain.")
