"""Shared, seeded test fixtures for the clustering suite.

Every fixture is deterministic (driven by :func:`stockclusters._rng.make_rng`) and
returns a wide pandas ``DataFrame`` of synthetic log-returns with known correlation
structure, so tests across the suite share identical data:

- ``one_block_correlation`` — a single common factor: every asset positively
  correlated (one true cluster). The "structure is real but trivial" case.
- ``k_blocks`` — ``K`` blocks of assets with high within-block and low cross-block
  correlation: the canonical case the clustering must recover (``K`` clusters).
- ``pure_noise`` — independent assets, population correlation = identity: the null,
  where honest clustering and the gap statistic should find no structure and the
  diversification horse race must come back insignificant.

Importing this module has no side effects beyond fixture registration.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stockclusters._rng import make_rng

_SEED = 20260616

#: Default panel shape used by the synthetic return fixtures.
_N_OBS = 750


def _asset_labels(n: int) -> list[str]:
    """Return ``n`` deterministic asset labels ``A00, A01, ...``."""
    return [f"A{i:02d}" for i in range(n)]


def _returns_from_corr(
    corr: np.ndarray,
    *,
    n_obs: int,
    seed: int,
    daily_vol: float = 0.01,
) -> pd.DataFrame:
    """Draw a seeded Gaussian return panel with population correlation ``corr``.

    Cholesky-factorizes ``corr`` and applies it to i.i.d. standard-normal draws,
    scaling to a realistic daily volatility. The resulting panel has the requested
    population correlation structure up to sampling error.
    """
    n_assets = corr.shape[0]
    gen = make_rng(seed)
    chol = np.linalg.cholesky(corr)
    z = gen.standard_normal((n_obs, n_assets))
    data = (z @ chol.T) * daily_vol
    index = pd.date_range("2020-01-01", periods=n_obs, freq="B")
    return pd.DataFrame(data, index=index, columns=_asset_labels(n_assets))


@pytest.fixture
def rng() -> np.random.Generator:
    """A seeded PCG64 generator shared by tests that need raw randomness."""
    return make_rng(_SEED)


@pytest.fixture
def one_block_correlation() -> pd.DataFrame:
    """One-block returns panel: a single common factor (one true cluster).

    Shape ``(750, 9)``. Every pair has correlation ``0.6``, so there is exactly one
    genuine cluster — the gap statistic should not split it, and clustering should
    largely re-discover the single block.
    """
    n_assets = 9
    rho = 0.6
    corr = np.full((n_assets, n_assets), rho)
    np.fill_diagonal(corr, 1.0)
    return _returns_from_corr(corr, n_obs=_N_OBS, seed=_SEED)


@pytest.fixture
def k_blocks() -> pd.DataFrame:
    """K-block returns panel: high within-block, low cross-block correlation.

    Shape ``(750, 12)``, four blocks of three assets. Within a block the
    correlation is ``0.75``; across blocks it is ``0.10``. This is the canonical
    structure the clustering is designed to recover (``K = 4`` clusters); the
    recovery guard pins ARI-vs-truth above a threshold on this fixture.
    """
    n_assets = 12
    block_size = 3
    within, across = 0.75, 0.10
    corr = np.full((n_assets, n_assets), across)
    for b in range(0, n_assets, block_size):
        corr[b : b + block_size, b : b + block_size] = within
    np.fill_diagonal(corr, 1.0)
    return _returns_from_corr(corr, n_obs=_N_OBS, seed=_SEED + 1)


@pytest.fixture
def k_blocks_truth() -> pd.Series:
    """Ground-truth cluster labels for the :func:`k_blocks` fixture.

    Four blocks of three assets -> labels ``[0,0,0,1,1,1,2,2,2,3,3,3]`` indexed by
    asset ticker. Used by the recovery guard (ARI vs truth).
    """
    n_assets = 12
    block_size = 3
    labels = [i // block_size for i in range(n_assets)]
    return pd.Series(labels, index=_asset_labels(n_assets), dtype=int)


@pytest.fixture
def pure_noise() -> pd.DataFrame:
    """Independent-asset returns panel: the no-structure null.

    Shape ``(750, 9)``. Every column is i.i.d. Gaussian noise (population
    correlation = identity), so clustering should find no meaningful structure, the
    gap statistic should not reject ``k = 1``, and the diversification horse race
    must come back insignificant (Memmel-JK p not significant, DSR <= 0).
    """
    n_assets = 9
    corr = np.eye(n_assets)
    return _returns_from_corr(corr, n_obs=_N_OBS, seed=_SEED + 2)
