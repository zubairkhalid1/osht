"""Demo 5: the order-m sampling functions and a placement for a new band-limit.

Part A evaluates the sampling functions Lambda^m_s(theta) of the rings that
resolve order m (the Lagrange basis of the order-m profile space) and checks
that they reproduce a random order-m profile anywhere on [0, pi].

Part B transforms at a band-limit for which no placement is shipped with the
package: the greedy search runs once (search=True), the result is saved in
the user cache (osht.cache_dir()) and every later call finds it there.

Run:  python demos/demo05_sampling_functions_and_new_L.py
"""

import time

import numpy as np

import osht

# --- Part A: sampling functions
L, m = 32, 8
thetas = osht.ordered_theta(L)
theta_eval = np.linspace(0, np.pi, 2001)
Lam = osht.sampling_functions(L, m, theta_eval)  # (2001, L - m): Lambda^m_s(theta) for s = m..L-1
print(f"sampling functions at L = {L}, m = {m}: {Lam.shape[1]} functions, "
      f"max |Lambda| on [0, pi] = {np.abs(Lam).max():.2f}, "
      f"Lebesgue-type sum max = {np.abs(Lam).sum(axis=1).max():.2f}")
rng = np.random.default_rng(3)
c = rng.standard_normal(L - m)
G_nodes = c @ osht.legmat(thetas[m:], L, m)  # a random order-m profile on the resolving rings
G_eval = c @ osht.legmat(theta_eval, L, m)
print(f"  interpolation of a random order-{m} profile from the {L - m} resolving rings: "
      f"max error {np.abs(Lam @ G_nodes - G_eval).max():.1e}")

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 3.5))
    for j in [0, 3, 10]:
        ax.plot(theta_eval, Lam[:, j], label=f"$\\Lambda^{{{m}}}_{{{m + j}}}$")
    ax.plot(thetas[m:], np.zeros(L - m), "k.", label="resolving rings")
    ax.set_xlabel("$\\theta$")
    ax.legend()
    fig.tight_layout()
    fig.savefig("demo05_sampling_functions.png", dpi=120)
    plt.close(fig)
    print("  wrote demo05_sampling_functions.png")
except ImportError:
    pass

# --- Part B: a band-limit without a shipped placement
L = 150
print("\nplacements shipped with the package: L = 1..128, 256, 512, 1024, 1536, 2048")
print(f"cache directory: {osht.cache_dir()}")
try:
    osht.ordered_theta(L)
except ValueError as exc:
    print(f"  ordered_theta({L}) without search: {str(exc)[:60]}...")
t0 = time.time()
thetas = osht.ordered_theta(L, "lambda", search=True, verbose=False)  # runs the greedy search once
print(f"  search at L = {L}: {time.time() - t0:.1f} s; cached band-limits now {osht.cached_placements()}")
t0 = time.time()
thetas2 = osht.ordered_theta(L)  # found in the cache
print(f"  second call: {1e3 * (time.time() - t0):.1f} ms, identical: {np.array_equal(thetas, thetas2)}")
flm = osht.random_flm(L, rng)
f = osht.inverse(flm, L)  # the default placement "lambda" now resolves through the cache
print(f"  round trip at L = {L}: E_max = {osht.errors(flm, osht.forward(f, L)).E_max:.1e}")
