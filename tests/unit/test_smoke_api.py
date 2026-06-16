"""Smoke tests: the package imports cleanly and exposes its curated public API.

These are deliberately lightweight so the suite collects and passes while the
compute kernels are still stubs (the kernels themselves are covered by
partition-specific tests authored separately). They guard the scaffolding the
three parallel authors build on: import purity and the ``__all__`` surface.
"""

from __future__ import annotations

import importlib

import pytest


@pytest.mark.unit
def test_package_imports() -> None:
    """``import stockclusters`` succeeds with no import-time side effects."""
    mod = importlib.import_module("stockclusters")
    assert mod.__version__
    assert isinstance(mod.__all__, list)
    assert len(mod.__all__) == len(set(mod.__all__))


@pytest.mark.unit
def test_public_api_resolves() -> None:
    """Every name in ``__all__`` is actually bound on the package."""
    import stockclusters

    missing = [name for name in stockclusters.__all__ if not hasattr(stockclusters, name)]
    assert not missing, f"names declared in __all__ but unbound: {missing}"


@pytest.mark.unit
@pytest.mark.parametrize(
    "submodule",
    [
        "stockclusters.correlation",
        "stockclusters.clustering",
        "stockclusters.stability",
        "stockclusters.allocation",
        "stockclusters.evaluation",
        "stockclusters.metrics",
        "stockclusters.plots",
        "stockclusters.cli",
    ],
)
def test_submodules_import(submodule: str) -> None:
    """Each planned subpackage/module imports without side effects."""
    importlib.import_module(submodule)


@pytest.mark.unit
def test_fixtures_are_dataframes(
    one_block_correlation: object,
    k_blocks: object,
    pure_noise: object,
) -> None:
    """The seeded conftest fixtures return non-empty return panels."""
    import pandas as pd

    for panel in (one_block_correlation, k_blocks, pure_noise):
        assert isinstance(panel, pd.DataFrame)
        assert not panel.empty
        assert panel.shape[1] >= 2
