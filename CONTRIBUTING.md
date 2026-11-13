# Contributing to osht

Thank you for considering a contribution. Bug reports, questions about the
method, new placements, speed-ups and documentation fixes are all welcome.

## Reporting a problem

Open an issue on GitHub with the band-limit, the placement, the back-end,
the NumPy and SciPy versions and a short script that reproduces the
problem. For an accuracy question, please include the output of
`osht.errors` and the amplification constants from `return_info=True`.

## Development setup

```bash
git clone https://github.com/zubairkhalid1/osht.git
cd osht
pip install -e ".[dev]"
pre-commit install          # optional: runs ruff before every commit
pytest                      # the full suite runs in a few seconds
ruff check src tests demos paper docs  # linting
```

The documentation is built with Sphinx:

```bash
sphinx-build -W -b html docs docs/_build/html
```

## Guidelines

- Keep the package pure NumPy and SciPy; optional features may depend on
  matplotlib behind an import guard.
- Every transform must stay exact in exact arithmetic. A change that affects
  numerics needs a round-trip test at several band-limits and, where the
  recorded values exist (see `tests/test_sampling_functions.py` and
  `tests/test_transforms.py`), agreement with the MATLAB package.
- Public functions take `thetas` either as a placement name or as an array
  of co-latitudes, accept a single signal or a batch, and document their
  arguments in NumPy docstring style.
- New stored placements go into `src/osht/data/*.npz` as index permutations of
  the equiangular set (see `osht.theta_index`), with the objective vector of
  the search, and are checked by a test against a fresh search at a small
  band-limit.
- Please follow the writing style of the documentation: full words, no
  contractions, no dashes used as punctuation.

## Release checklist

1. Update `__version__` in `src/osht/__init__.py`, the version in
   `pyproject.toml` and `CITATION.cff`, and add a section to `CHANGELOG.md`.
2. `pytest`, `ruff check`, `sphinx-build -W docs docs/_build/html`.
3. `python -m build` and `twine check dist/*`.
4. Tag the release (`git tag v0.1.0 && git push --tags`); the `publish.yml`
   workflow uploads the distribution to PyPI through trusted publishing when
   a GitHub release is created.

`tests/test_release.py` checks that the version in `pyproject.toml`,
`CITATION.cff`, `src/osht/__init__.py` and the newest section of
`CHANGELOG.md` agree.
