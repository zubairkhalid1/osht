import numpy as np
import pytest


@pytest.fixture
def rng():
    return np.random.default_rng(20261109)


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    """Every test gets an empty placement cache so that nothing is written to the user's home."""
    monkeypatch.setenv("OSHT_CACHE_DIR", str(tmp_path / "osht_cache"))
