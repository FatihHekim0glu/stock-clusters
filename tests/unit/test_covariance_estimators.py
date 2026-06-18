"""Unit tests for the covariance estimators in ``estimators.covariance``.

Pins the three estimators (sample, Ledoit-Wolf, OAS) against ``sklearn`` and
against basic structural invariants: labelled output, exact symmetry, a positive
diagonal, and positive-definiteness for the shrinkage estimators (which must stay
well conditioned even when the window is short).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stockclusters._exceptions import InsufficientDataError
from stockclusters.estimators.covariance import ledoit_wolf_cov, oas_cov, sample_cov


def _panel(n_obs: int, n_assets: int, seed: int = 0) -> pd.DataFrame:
    gen = np.random.default_rng(seed)
    data = gen.standard_normal((n_obs, n_assets)) * 0.01
    idx = pd.date_range("2020-01-01", periods=n_obs, freq="B")
    cols = [f"A{i:02d}" for i in range(n_assets)]
    return pd.DataFrame(data, index=idx, columns=cols)


@pytest.mark.unit
def test_sample_cov_matches_numpy() -> None:
    """The sample covariance matches ``numpy.cov`` (ddof=1) and keeps asset labels."""
    panel = _panel(120, 5, seed=1)
    cov = sample_cov(panel)
    expected = np.cov(panel.to_numpy(), rowvar=False, ddof=1)
    np.testing.assert_allclose(cov.to_numpy(), expected, rtol=1e-12, atol=1e-15)
    assert list(cov.index) == list(panel.columns)
    assert list(cov.columns) == list(panel.columns)


@pytest.mark.unit
def test_sample_cov_is_symmetric() -> None:
    """The returned matrix is exactly symmetric after the explicit symmetrize."""
    cov = sample_cov(_panel(80, 6, seed=2)).to_numpy()
    np.testing.assert_array_equal(cov, cov.T)


@pytest.mark.unit
def test_ledoit_wolf_matches_sklearn() -> None:
    """Ledoit-Wolf parity against ``sklearn.covariance.ledoit_wolf`` to 1e-10."""
    sklearn_cov = pytest.importorskip("sklearn.covariance")
    panel = _panel(90, 7, seed=3)
    cov = ledoit_wolf_cov(panel).to_numpy()
    expected, _ = sklearn_cov.ledoit_wolf(panel.to_numpy())
    np.testing.assert_allclose(cov, expected, rtol=1e-10, atol=1e-12)


@pytest.mark.unit
def test_oas_matches_sklearn() -> None:
    """OAS parity against ``sklearn.covariance.oas`` to 1e-10."""
    sklearn_cov = pytest.importorskip("sklearn.covariance")
    panel = _panel(90, 7, seed=4)
    cov = oas_cov(panel).to_numpy()
    expected, _ = sklearn_cov.oas(panel.to_numpy())
    np.testing.assert_allclose(cov, expected, rtol=1e-10, atol=1e-12)


@pytest.mark.unit
@pytest.mark.parametrize("estimator", [ledoit_wolf_cov, oas_cov])
def test_shrinkage_is_positive_definite_when_t_le_n(estimator) -> None:  # type: ignore[no-untyped-def]
    """Shrinkage estimators stay positive-definite even with fewer obs than assets."""
    # T < N: the raw sample covariance would be singular here.
    panel = _panel(8, 12, seed=5)
    cov = estimator(panel).to_numpy()
    eigvals = np.linalg.eigvalsh(cov)
    assert eigvals.min() > 0.0
    np.testing.assert_array_equal(cov, cov.T)


@pytest.mark.unit
@pytest.mark.parametrize("estimator", [sample_cov, ledoit_wolf_cov, oas_cov])
def test_too_few_observations_rejected(estimator) -> None:  # type: ignore[no-untyped-def]
    """A single-observation panel cannot define a covariance and is rejected."""
    one_row = _panel(1, 4, seed=6)
    with pytest.raises(InsufficientDataError):
        estimator(one_row)
