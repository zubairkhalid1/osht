# Quick start

## Conventions

* **Band-limit.** A signal band-limited at $L$ has coefficients $f_\ell^m$ for
  $0 \le \ell < L$ and $|m| \le \ell$, stored in a complex vector of length
  $L^2$ with `flm[l**2 + l + m]` $= f_\ell^m$. {func}`osht.elm2ind` and
  {func}`osht.ind2elm` convert between $(\ell, m)$ and the position.
* **Harmonics.** $Y_\ell^m(\theta, \phi) = \tilde P_\ell^m(\theta)\, e^{i m \phi}$,
  orthonormal on the sphere, with the Condon-Shortley phase inside
  $\tilde P_\ell^m$, so that a real signal has $f_\ell^{-m} = (-1)^m \overline{f_\ell^m}$.
* **Samples.** $L$ rings; ring $k = 0, \dots, L-1$ at co-latitude $\theta_k$
  with $2k+1$ samples at $\phi_j = 2\pi j/(2k+1)$. The samples are stored ring
  by ring: ring $k$ occupies the positions $k^2, \dots, (k+1)^2 - 1$ of a vector
  of length $L^2$ ({func}`osht.ring_slice`). Ring $0$ (one sample) sits at the
  south pole $\theta = \pi$ for the stored placements.
* **Batches.** Every transform accepts one signal (shape `(L**2,)`) or a batch
  (shape `(S, L**2)`), and returns the same layout.

## A round trip

```python
import numpy as np
import osht

L = 128
flm = osht.random_flm(L, rng=0)             # random coefficients, |Re|, |Im| <= 1
f = osht.inverse(flm, L)                    # synthesis on the L**2 samples
flm_rec, info = osht.forward(f, L, return_info=True)
e = osht.errors(flm, flm_rec)
print(e.E_max, e.SNR)                       # about 1e-14 and 290 dB
print(info["passes"], info["lambda_small"].max())
```

{func}`osht.forward` is the recommended entry point: the interpolated
elimination with the equiangular back-end on the $\Lambda$-placement, refined by
a second pass on the residual. Its options:

| option | default | meaning |
| --- | --- | --- |
| `thetas` | `"lambda"` | ring placement: `"lambda"`, `"kappa"`, `"alternating"` or an array of $L$ co-latitudes |
| `backend` | `"eq"` | `"eq"` equiangular back-end (needs the equiangular ring set) or `"gl"` Gauss-Legendre quadrature (any ring set) |
| `passes` | `2` | maximum number of passes of the residual refinement; `1` is the single-pass transform |
| `reality` | `False` | real signal: orders $m < 0$ by conjugate symmetry, half the work |

{func}`osht.forward_pass` is one sweep without refinement, {func}`osht.multipass`
the refinement around any one-pass transform, {func}`osht.forward_lu` the
transform of the 2014 paper by per-order linear solves (a reference, $O(L^4)$).

## The sampling scheme

```python
s = osht.Sampling(L)              # or osht.Sampling(L, "kappa"), osht.Sampling(L, thetas)
s.thetas                          # co-latitude of ring k, shape (L,)
s.theta, s.phi                    # co-latitude and longitude of every sample, shape (L**2,)
x, y, z = s.cartesian()           # Cartesian coordinates of the samples
s.slice(k)                        # positions of ring k
```

To sample your own function on the scheme, evaluate it at `(s.theta, s.phi)`
and pass the resulting vector to {func}`osht.forward`.

## Real signals

```python
flm, L = osht.data.earth_topography()          # a real signal, L = 128
f = osht.inverse(flm, L, reality=True)         # float64 samples
flm_rec = osht.forward(f, L, reality=True)
```

## Placements, amplification and sampling functions

```python
thetas, obj = osht.ordered_theta(L, "lambda", objective=True)   # shipped placement and its objective
amp = osht.amplification(L, thetas, kappa=True)                  # Lambda_m, Lambda_m^GL, kappa(P_m)
Lam = osht.sampling_functions(L, m=8, theta_eval=np.linspace(0, np.pi, 1001))
```

## A band-limit without a shipped placement

Placements are shipped for $L \le 128$ and $L = 256, 512, 1024, 1536, 2048$
(`"lambda"`) and for $L \le 721$ and $L = 1024, 1536, 2048$ (`"kappa"`).
For any other $L$ the greedy search runs once and its result is saved in the
user cache ({func}`osht.cache_dir`), where every later call finds it:

```python
thetas = osht.ordered_theta(150, "lambda", search=True)   # a few seconds; saved in the cache
flm = osht.forward(f, 150)                                 # from now on "lambda" resolves at L = 150
osht.cached_placements("lambda")                           # [150]
```

Without `search=True`, {func}`osht.ordered_theta` raises an error for such
an $L$ rather than starting a search that may take long (half an hour at
$L = 1024$). {func}`osht.save_placement` stores a placement computed elsewhere.

## Plotting

With matplotlib installed, {mod}`osht.plotting` draws the sampling points, a
real signal on the sphere and a quantity over $(\ell, m)$:

```python
osht.plotting.plot_sampling(12)
osht.plotting.plot_sphere(f, L)
osht.plotting.plot_spectral(np.abs(flm_rec - flm), L, log=True)
```

## Demos

The `demos/` folder of the repository contains runnable scripts: a quick
start, the Earth topography with the real-signal path, the comparison of the
placements, the multi-pass refinement and the oracle sweep, the sampling
functions and a new band-limit, batches and timing, and the run-time and
accuracy experiments of the paper. They are described in {doc}`demos`.
