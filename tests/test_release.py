"""Release consistency: one version everywhere, shipped data complete, demos present."""

import re
from pathlib import Path

import numpy as np
import pytest

import osht

ROOT = Path(__file__).resolve().parents[1]


def _read(name):
    path = ROOT / name
    if not path.is_file():
        pytest.skip(f"{name} is not available (tests run outside the repository)")
    return path.read_text(encoding="utf-8")


def test_version_is_consistent():
    version = osht.__version__
    assert re.fullmatch(r"\d+\.\d+\.\d+", version)
    assert re.search(r'^version = "(.+)"$', _read("pyproject.toml"), re.M).group(1) == version
    assert re.search(r"^version: (.+)$", _read("CITATION.cff"), re.M).group(1).strip() == version
    newest = re.search(r"^## \[(.+?)\]", _read("CHANGELOG.md"), re.M).group(1)
    assert newest == version


def test_installed_metadata_matches():
    from importlib import metadata

    try:
        installed = metadata.version("osht")
    except metadata.PackageNotFoundError:
        pytest.skip("osht is not installed (running from the source tree)")
    assert installed == osht.__version__


def test_shipped_placements_are_valid_permutations():
    """Every shipped placement is a permutation of the equiangular set with its objective vector."""
    for name in ("lambda", "kappa"):
        for L in osht.stored_placements(name):
            if 128 < L < 1024 and L % 97:  # a sample of the kappa-placements between 129 and 721
                continue
            thetas, obj = osht.ordered_theta(L, name, objective=True)
            assert osht.is_equiangular(thetas, L), (name, L)
            if L > 1:
                assert obj is not None and obj.shape == (L - 1,) and np.all(np.isfinite(obj)), (name, L)
                assert np.isclose(thetas[-1], osht.indexed_theta(L)[-1]), (name, L)


def test_lambda_placement_2048():
    thetas, obj = osht.ordered_theta(2048, "lambda", objective=True)
    assert np.isclose(thetas[0], np.pi)  # the single-sample ring sits at the south pole
    assert np.isclose(obj.max(), 2842.866, atol=1e-2)  # recorded by the search
    assert 2046 - int(obj.argmax()) == 139  # worst order


def test_demos_and_paper_material_present():
    demos = ROOT / "demos"
    if not demos.is_dir():
        pytest.skip("demos are not available (tests run outside the repository)")
    names = sorted(p.name for p in demos.glob("demo*.py"))
    assert len(names) == 8 and names[0].startswith("demo01") and names[-1].startswith("demo08")
    paper = ROOT / "paper"
    if paper.is_dir():
        with np.load(paper / "accuracy_vs_L.npz") as z:
            assert list(z["L_list"]) == [16, 32, 64, 128, 256, 512, 1024, 2048]
            assert z["E_max"].shape == (2, 8, 3, 5) and z["done"].all()
        rows = np.loadtxt(paper / "runtime.csv", delimiter=",", skiprows=1)
        assert rows.shape == (8, 4) and rows[-1, 0] == 2048
    for name in re.findall(r"docs/_static/([\w.]+)", _read("README.md")):  # the figures of the README
        assert (ROOT / "docs" / "_static" / name).is_file(), name


def test_paper_reference_is_consistent():
    """The paper is cited by the same arXiv identifier wherever it is cited."""
    arxiv = re.search(r"arXiv:(\d{4}\.\d{5})", _read("CITATION.cff")).group(1)
    for name in ("README.md", "docs/references.md", "paper/README.md"):
        assert f"arxiv.org/abs/{arxiv}" in _read(name), name
    assert f"arXiv:{arxiv}" in osht.__doc__
