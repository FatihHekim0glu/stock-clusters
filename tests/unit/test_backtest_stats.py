"""Unit tests for the performance statistics in ``backtest.stats``.

These pin the numerical definitions (Sharpe, annualized volatility, turnover,
maximum drawdown) against hand-computed values and basic invariants, and cover the
degenerate paths (flat series, non-declining series, disjoint weight labels).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from stockclusters.backtest.stats import (
    annualized_vol,
    max_drawdown,
    sharpe_ratio,
    turnover,
)


def _series(values: list[float]) -> pd.Series:
    idx = pd.date_range("2020-01-01", periods=len(values), freq="B")
    return pd.Series(values, index=idx, dtype="float64")


@pytest.mark.unit
def test_sharpe_matches_manual_formula() -> None:
    """Sharpe equals ``mean/std * sqrt(ppy)`` on a known series."""
    arr = np.array([0.01, -0.005, 0.02, 0.0, 0.012, -0.003], dtype="float64")
    series = _series(list(arr))
    expected = (arr.mean() / arr.std(ddof=1)) * math.sqrt(252)
    assert sharpe_ratio(series) == pytest.approx(expected, rel=1e-12)


@pytest.mark.unit
def test_sharpe_flat_series_is_nan() -> None:
    """A constant return series has zero volatility, so the Sharpe is NaN."""
    flat = _series([0.001] * 20)
    assert math.isnan(sharpe_ratio(flat))


@pytest.mark.unit
def test_sharpe_risk_free_shifts_mean() -> None:
    """Subtracting a per-period risk-free rate lowers the Sharpe of a positive series."""
    series = _series([0.01, 0.012, 0.009, 0.011, 0.013])
    sr0 = sharpe_ratio(series)
    sr_rf = sharpe_ratio(series, risk_free=0.005)
    assert sr_rf < sr0


@pytest.mark.unit
def test_annualized_vol_matches_manual_formula() -> None:
    """Annualized volatility equals ``std(ddof=1) * sqrt(ppy)``."""
    arr = np.array([0.01, -0.02, 0.015, 0.0, -0.005], dtype="float64")
    expected = arr.std(ddof=1) * math.sqrt(252)
    assert annualized_vol(_series(list(arr))) == pytest.approx(expected, rel=1e-12)


@pytest.mark.unit
def test_annualized_vol_scales_with_periods() -> None:
    """A larger annualization factor scales the volatility by ``sqrt`` of the ratio."""
    series = _series([0.01, -0.02, 0.015, 0.0, -0.005])
    v252 = annualized_vol(series, periods_per_year=252)
    v63 = annualized_vol(series, periods_per_year=63)
    assert v252 == pytest.approx(v63 * math.sqrt(252 / 63), rel=1e-12)


@pytest.mark.unit
def test_turnover_full_rotation_is_one() -> None:
    """Rotating fully out of one asset and into another is turnover 1.0."""
    prev = pd.Series({"A": 1.0, "B": 0.0})
    new = pd.Series({"A": 0.0, "B": 1.0})
    assert turnover(prev, new) == pytest.approx(1.0)


@pytest.mark.unit
def test_turnover_no_change_is_zero() -> None:
    """Identical weights produce zero turnover."""
    w = pd.Series({"A": 0.5, "B": 0.5})
    assert turnover(w, w) == pytest.approx(0.0)


@pytest.mark.unit
def test_turnover_aligns_disjoint_labels() -> None:
    """Assets absent on one side are treated as zero weight on that side."""
    prev = pd.Series({"A": 1.0})
    new = pd.Series({"B": 1.0})
    # Out of A (1.0) and into B (1.0): half the absolute change is 1.0.
    assert turnover(prev, new) == pytest.approx(1.0)


@pytest.mark.unit
def test_max_drawdown_non_positive() -> None:
    """Maximum drawdown is non-positive and matches a hand-computed trough."""
    # +10%, then two -10% steps: wealth 1.1, 0.99, 0.891; peak 1.1.
    series = _series([0.10, -0.10, -0.10])
    expected = 0.891 / 1.1 - 1.0
    assert max_drawdown(series) == pytest.approx(expected, rel=1e-12)
    assert max_drawdown(series) <= 0.0


@pytest.mark.unit
def test_max_drawdown_monotone_up_is_zero() -> None:
    """A series that never declines has zero drawdown."""
    series = _series([0.01, 0.02, 0.005, 0.03])
    assert max_drawdown(series) == pytest.approx(0.0)
