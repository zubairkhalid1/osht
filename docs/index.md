# osht

**Spherical harmonic transforms on optimal-dimensionality ($L^2$) samples.**

`osht` computes forward and inverse spherical harmonic transforms of signals
band-limited at $L$ from exactly $L^2$ samples on the sphere, the number of
degrees of freedom of such a signal. The samples lie on $L$ iso-latitude rings,
ring $k$ carrying $2k+1$ equally spaced samples, as in the optimal-dimensionality
sampling scheme of Khalid, Kennedy and McEwen (2014). The forward transform is
the *interpolated elimination*: an $O(L^3)$ sweep over the orders that never
solves a linear system and needs no precomputation beyond the ring
co-latitudes, followed by one of two exact back-ends. The inverse transform is
a closed-form synthesis by folded inverse FFTs.

```python
import numpy as np, osht

L = 64
flm = osht.random_flm(L, rng=0)      # f_l^m at flm[l**2 + l + m]
f = osht.inverse(flm, L)             # L**2 samples, ring by ring
flm_rec = osht.forward(f, L)         # interpolated elimination, two passes
print(osht.errors(flm, flm_rec).E_max)   # about 1e-15
```

## Features

* $L^2$ samples, the smallest number any sampling theorem can use, and exact
  transforms in exact arithmetic (Theorem 1 of the paper).
* $O(L^3)$ forward transform by interpolated elimination with the order-$m$
  sampling functions of Giacaglia and Lundquist (1972), evaluated in a stable
  log-domain barycentric form.
* Two exact back-ends: the equiangular pipeline of McEwen and Wiaux (2011) and
  Gauss-Legendre quadrature of the interpolated profiles.
* Multi-pass refinement by residual iteration, with an automatic stopping rule.
* Ring placements optimised for stability (the $\Lambda$-placement, stored for
  $L \le 128$ and $L = 256, 512, 1024, 1536, 2048$; the $\kappa$-placement of the NSHT
  package for $L \le 721$ and $L = 1024, 1536, 2048$) and the greedy searches that
  produce them.
* Any band-limit: for an $L$ without a shipped placement the greedy search
  runs once, its result is saved in a user cache and reused afterwards.
* Real-signal fast path, batches of signals, the linear-solve transform of the
  2014 paper as a reference, amplification constants and error measures,
  sampling functions, plotting helpers.
* Pure NumPy and SciPy; no compiled extension.

```{toctree}
:maxdepth: 2
:caption: Contents

installation
quickstart
method
stability
placements
demos
api
changelog
references
```

## Citation

If you use `osht` in your research, please cite the paper on the interpolated
elimination, Z. Khalid, "An $O(L^3)$ spherical harmonic transform on $L^2$
samples", [arXiv:2611.16441](https://arxiv.org/abs/2611.16441), 2026, and the
2014 paper on the sampling scheme (see {doc}`references`), and the software
through its `CITATION.cff` file.

## License

MIT License, see the `LICENSE` file.
