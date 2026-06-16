"""Parity oracles for the inference scalars used by the Group-B horse race.

- ARI vs ``sklearn.metrics.adjusted_rand_score`` to 1e-12 on fixed vectors.
- PSR / DSR vs an independent ``scipy.stats``-based reference to 1e-8.

The library's PSR/DSR avoid a SciPy dependency (Acklam ``_norm_ppf`` + ``erf``);
this module re-derives the same quantities with ``scipy.stats.norm`` so the two
independent implementations must agree to 1e-8.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm
from sklearn.metrics import adjusted_rand_score

from stockclusters.evaluation.dsr import (
    deflated_sharpe_ratio,
    probabilistic_sharpe_ratio,
)
from stockclusters.stability.ari import adjusted_rand_index

_DSR_TOL = 1e-8
_ARI_TOL = 1e-12
_EULER = 0.5772156649015329


def _ari_series(values: list[int]) -> pd.Series:
    return pd.Series(values, index=[f"A{i:02d}" for i in range(len(values))], dtype=int)


# --------------------------------------------------------------------------- #
# ARI 1e-12 on fixed vectors                                                  #
# --------------------------------------------------------------------------- #
@pytest.mark.parity
@pytest.mark.parametrize(
    ("a", "b"),
    [
        ([0, 0, 1, 1, 2, 2], [0, 0, 1, 1, 2, 2]),  # identical
        ([0, 0, 1, 1, 2, 2], [1, 1, 2, 2, 0, 0]),  # permuted ids
        ([0, 0, 0, 1, 1, 1], [0, 1, 0, 1, 0, 1]),  # adversarial split
        ([0, 1, 2, 3, 4, 5], [0, 0, 0, 0, 0, 0]),  # singletons vs one cluster
        ([0, 0, 1, 1, 2, 2, 3, 3], [0, 0, 1, 2, 2, 2, 3, 1]),  # partial overlap
    ],
)
def test_ari_matches_sklearn_fixed(a: list[int], b: list[int]) -> None:
    """Hand-rolled ARI matches scikit-learn to 1e-12 on fixed labelings."""
    mine = adjusted_rand_index(_ari_series(a), _ari_series(b))
    ref = float(adjusted_rand_score(np.asarray(a), np.asarray(b)))
    assert abs(mine - ref) <= _ARI_TOL


# --------------------------------------------------------------------------- #
# PSR vs scipy reference (1e-8)                                                #
# --------------------------------------------------------------------------- #
def _psr_ref(sr: float, *, n_obs: int, skew: float, kurt: float, benchmark: float) -> float:
    var = 1.0 - skew * sr + 0.25 * (kurt - 1.0) * sr * sr
    z = (sr - benchmark) * math.sqrt(n_obs - 1) / math.sqrt(var)
    return float(norm.cdf(z))


@pytest.mark.parity
@pytest.mark.parametrize(
    ("sr", "n_obs", "skew", "kurt", "benchmark"),
    [
        (0.05, 250, 0.0, 3.0, 0.0),
        (0.10, 500, -0.5, 5.0, 0.0),
        (0.08, 1000, 0.3, 4.0, 0.02),
        (0.02, 120, 0.0, 3.0, 0.0),
    ],
)
def test_psr_matches_scipy(
    sr: float, n_obs: int, skew: float, kurt: float, benchmark: float
) -> None:
    """PSR matches an independent scipy.stats.norm reference to 1e-8."""
    mine = probabilistic_sharpe_ratio(
        sr, n_obs=n_obs, skew=skew, kurtosis=kurt, benchmark_sharpe=benchmark
    )
    ref = _psr_ref(sr, n_obs=n_obs, skew=skew, kurt=kurt, benchmark=benchmark)
    assert abs(mine - ref) <= _DSR_TOL


# --------------------------------------------------------------------------- #
# DSR vs scipy reference (1e-8)                                                #
# --------------------------------------------------------------------------- #
def _dsr_ref(
    sr: float, *, n_obs: int, n_trials: int, var_trials: float, skew: float, kurt: float
) -> float:
    sqrt_v = math.sqrt(var_trials)
    if n_trials == 1 or sqrt_v == 0.0:
        benchmark = 0.0
    else:
        n = float(n_trials)
        z1 = float(norm.ppf(1.0 - 1.0 / n))
        z2 = float(norm.ppf(1.0 - 1.0 / (n * math.e)))
        benchmark = sqrt_v * ((1.0 - _EULER) * z1 + _EULER * z2)
    return _psr_ref(sr, n_obs=n_obs, skew=skew, kurt=kurt, benchmark=benchmark)


@pytest.mark.parity
@pytest.mark.parametrize(
    ("sr", "n_obs", "n_trials", "var_trials"),
    [
        (0.10, 500, 1, 0.001),
        (0.10, 500, 10, 0.001),
        (0.08, 1000, 100, 0.0005),
        (0.12, 750, 250, 0.002),
        (0.06, 300, 24, 0.0008),
    ],
)
def test_dsr_matches_scipy(sr: float, n_obs: int, n_trials: int, var_trials: float) -> None:
    """DSR matches an independent scipy.stats.norm reference to 1e-8."""
    mine = deflated_sharpe_ratio(
        sr,
        n_obs=n_obs,
        n_trials=n_trials,
        variance_of_trial_sharpes=var_trials,
        skew=0.0,
        kurtosis=3.0,
    )
    ref = _dsr_ref(sr, n_obs=n_obs, n_trials=n_trials, var_trials=var_trials, skew=0.0, kurt=3.0)
    assert abs(mine - ref) <= _DSR_TOL


@pytest.mark.parity
def test_dsr_non_increasing_in_n_trials() -> None:
    """The DSR is non-increasing in the trial count (multiplicity penalty)."""
    prev = math.inf
    for n_trials in (1, 2, 5, 10, 50, 200, 1000):
        dsr = deflated_sharpe_ratio(
            0.09,
            n_obs=400,
            n_trials=n_trials,
            variance_of_trial_sharpes=0.0012,
        )
        assert dsr <= prev + 1e-15
        prev = dsr
