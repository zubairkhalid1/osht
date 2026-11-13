# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-11-13

First public release.

### Added

- Forward spherical harmonic transform on the $L^2$ samples of the
  optimal-dimensionality scheme by interpolated elimination (`osht.forward`,
  `osht.forward_pass`), with the equiangular back-end of McEwen and Wiaux and
  the Gauss-Legendre back-end, and the multi-pass refinement (`osht.multipass`).
- Exact inverse transform by folded inverse FFTs (`osht.inverse`).
- Linear-solve transform of the 2014 paper as a reference (`osht.forward_lu`).
- Real-signal fast path (`reality=True`) and batches of signals.
- Stored ring placements: the Lambda-placement for `L = 1..128, 256, 512,
  1024, 1536, 2048` and the kappa-placement of the NSHT package for
  `L = 1..721, 1024, 1536, 2048`; the greedy searches
  `osht.search_lambda_placement` and `osht.search_kappa_placement` (the latter
  in `O(L^4)` by a rank-one update), whose results are saved in a user cache
  (`osht.cache_dir()`) and found by `osht.ordered_theta(L, search=True)` for
  any other band-limit.
- Sampling functions, barycentric weights, scaled Legendre functions,
  Trapani-Navaza Wigner planes, amplification constants, error measures,
  random test signals, the Earth topography test signal and plotting helpers.
- Stability warnings (`RuntimeWarning`) when the interpolation amplification
  of the sweep or the sample residual of the refinement signals an inaccurate
  transform.
- Eight demos, including the run-time experiment (`demos/demo07_runtime_vs_L.py`)
  and the accuracy experiment (`demos/demo08_accuracy_vs_L.py`, resumable) of
  the paper; their stored numerical results for `L = 16, ..., 2048` and the
  script that draws the run-time figure are in `paper/`, and the accuracy and
  run-time figures of the README are drawn from them by
  `docs/make_readme_figures.py`.
- Documentation (Sphinx), test suite and GitHub Actions workflows (tests on
  Linux, macOS and Windows, linting, demos, documentation, PyPI publishing).
