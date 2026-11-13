# Installation

`osht` is a pure Python package that depends on NumPy (1.22 or later) and
SciPy (1.8 or later) and supports Python 3.9 to 3.13.

From PyPI:

```bash
pip install osht
```

With the plotting helpers (matplotlib):

```bash
pip install "osht[plot]"
```

From the repository, for development (tests, documentation, linting):

```bash
git clone https://github.com/zubairkhalid1/osht.git
cd osht
pip install -e ".[dev]"
pytest
```

The test suite runs in a few seconds and checks the exactness of every
transform against random band-limited signals, the agreement of the stored
ring placements with a fresh search, the amplification constants against the
values recorded for the MATLAB package, the placement cache and the real-signal
path.

## Data shipped with the package

* `lambda_placements.npz`: the $\Lambda$-placements of the rings for
  $L = 1, \dots, 128$ and $L = 256, 512, 1024, 1536, 2048$ (about 175 kB).
* `kappa_placements.npz`: the $\kappa$-placements of the NSHT package for
  $L = 1, \dots, 721$ and $L = 1024, 1536, 2048$ (about 1.6 MB).
* `earth_topography_flm.npz`: coefficients of the Earth topography,
  band-limited at $L = 128$, a real test signal.

For a band-limit without a shipped placement, `osht.ordered_theta(L, search=True)`
(or {func}`osht.search_lambda_placement`) runs the greedy search once (about
two minutes at $L = 512$, half an hour at $L = 1024$ in NumPy) and saves the
result in the user cache, `~/.cache/osht` on Linux and macOS,
`%LOCALAPPDATA%\osht` on Windows, or the directory named by the environment
variable `OSHT_CACHE_DIR` ({func}`osht.cache_dir`). Every later call of the
package at that band-limit finds the placement there; {func}`osht.cached_placements`
lists them and {func}`osht.save_placement` stores a placement computed
elsewhere (for example by the MATLAB package).
