"""Regression guards for the honest null and the DSR trial-count discipline.

Two pinned behaviours from the brief (Group B):

1. Honest-headline guard: on pure noise the cluster-vs-1/N horse race is not
   significant, so clustering buys no alpha on noise. This is checked as a
   CALIBRATION property, not on a single draw: under the null the Memmel-JK
   p-value is roughly uniform on ``[0, 1]``, so any single seed rejects with
   probability equal to the nominal level (about 1 in 20 at the 5% threshold).
   Pinning one draw is therefore a coin-flip guard, not a real check. Instead we
   draw many independent null panels and assert the empirical rejection rate sits
   near the nominal 5% level. A genuine miscalibration (for example a look-ahead
   leak that inflates the gap) would push that rate far above nominal and trip the
   guard, whereas an unlucky single draw cannot.
2. DSR trial-count guard: ``n_trials`` must equal the FULL product of the swept
   axes; under-counting (which manufactures false significance) is rejected, and
   a larger trial count never raises the DSR.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from stockclusters._rng import make_rng
from stockclusters.allocation.schemes import DiversificationResult, run_diversification
from stockclusters.evaluation.dsr import deflated_sharpe_ratio
from stockclusters.evaluation.verdict import ClusteringVerdict, derive_clustering_verdict


def _arbitrary_labels(panel: pd.DataFrame, *, k: int) -> pd.Series:
    """A fixed (data-independent) k-way partition of the universe."""
    cols = list(panel.columns)
    return pd.Series([i % k for i in range(len(cols))], index=cols, dtype=int)


def _null_panel(seed: int, *, n_obs: int = 400, n_assets: int = 9) -> pd.DataFrame:
    """A seeded i.i.d. Gaussian return panel: the no-structure null draw.

    Population correlation is the identity, so clustering finds no real structure
    and the cluster-vs-1/N Sharpe gap is pure sampling noise.
    """
    gen = make_rng(seed)
    data = gen.standard_normal((n_obs, n_assets)) * 0.01
    index = pd.date_range("2020-01-01", periods=n_obs, freq="B")
    cols = [f"A{i:02d}" for i in range(n_assets)]
    return pd.DataFrame(data, index=index, columns=cols)


# --------------------------------------------------------------------------- #
# 1. Honest-headline guard on pure noise                                      #
# --------------------------------------------------------------------------- #
# Full swept-axis product (illustrative): 3 linkages x 5 k-candidates x 3 schemes
# x 2 denoise x 4 cost grid = 360 trials.
_N_TRIALS = 3 * 5 * 3 * 2 * 4

# Number of independent null draws averaged over. Under the null each draw rejects
# with probability ~0.05, so the count of rejections is roughly Binomial(N, 0.05).
_N_NULL_DRAWS = 48

# Nominal level of the Memmel-JK test.
_ALPHA = 0.05

# Upper bound on the empirical rejection rate. With N=48 and a true rate of 0.05
# the expected count is 2.4; the bound of 0.25 (12 of 48) is reached only by an
# astronomically unlikely run of noise OR by a genuine miscalibration (e.g. a
# look-ahead leak), so a healthy null comfortably passes and a broken one fails.
_MAX_REJECTION_RATE = 0.25


@pytest.mark.regression
def test_pure_noise_horse_race_is_calibrated() -> None:
    """On pure noise the cluster-vs-1/N test rejects at about its nominal level.

    This is the honest-null guard. A single seeded draw is NOT a valid check: the
    Memmel-JK p-value is approximately uniform under the null, so any one fixture
    rejects with probability equal to the nominal level by construction (the shared
    ``pure_noise`` fixture seed happens to land in that lower tail, p~0.006). We
    instead estimate the rejection RATE across many independent null panels and
    assert it sits near the 5% nominal level. A real defect that manufactured a
    Sharpe gap on noise would inflate this rate far past nominal and trip the
    bound, whereas one unlucky draw cannot.
    """
    rejections = 0
    for i in range(_N_NULL_DRAWS):
        panel = _null_panel(seed=4_000 + 911 * i)
        labels = _arbitrary_labels(panel, k=3)
        res = run_diversification(
            panel, labels, lookback_window=60, n_trials=_N_TRIALS, cost_bps=5.0
        )
        assert math.isfinite(res.memmel_jk_pvalue)
        assert 0.0 <= res.memmel_jk_pvalue <= 1.0
        if res.memmel_jk_pvalue < _ALPHA:
            rejections += 1

    rate = rejections / _N_NULL_DRAWS
    assert rate <= _MAX_REJECTION_RATE, (
        f"null rejection rate {rate:.3f} ({rejections}/{_N_NULL_DRAWS}) exceeds the "
        f"calibration bound {_MAX_REJECTION_RATE}; the honest-null test looks "
        "miscalibrated (possible look-ahead leak inflating the Sharpe gap)."
    )


@pytest.mark.regression
def test_pure_noise_single_draw_verdict_is_consistent() -> None:
    """A single null draw yields a verdict consistent with its own statistics.

    Whatever the (uniform-under-null) p-value lands at, the derived verdict must
    follow the truth table: only a jointly significant p-value AND positive DSR can
    flip the verdict away from ``NO_SIGNIFICANT_DIFFERENCE``. On noise the DSR is
    not expected to be positive, so the honest verdict holds even on a draw that is
    nominally significant.
    """
    panel = _null_panel(seed=4_242)
    labels = _arbitrary_labels(panel, k=3)
    res = run_diversification(panel, labels, lookback_window=60, n_trials=_N_TRIALS, cost_bps=5.0)

    verdict = derive_clustering_verdict(
        res.memmel_jk_pvalue, res.deflated_sharpe, res.sharpe_diff_vs_1overN
    )
    significant = res.memmel_jk_pvalue < _ALPHA
    if not significant or res.deflated_sharpe <= 0.0:
        assert verdict is ClusteringVerdict.NO_SIGNIFICANT_DIFFERENCE


@pytest.mark.regression
def test_verdict_truth_table_honest() -> None:
    """The verdict truth table: 'beat' requires significance AND positive DSR."""
    # Insignificant p -> no difference even with a big positive gap and DSR.
    assert (
        derive_clustering_verdict(0.40, 0.99, 0.50) is ClusteringVerdict.NO_SIGNIFICANT_DIFFERENCE
    )
    # Significant p but non-positive DSR -> no difference.
    assert (
        derive_clustering_verdict(0.001, 0.0, 0.50) is ClusteringVerdict.NO_SIGNIFICANT_DIFFERENCE
    )
    # Both supportive, positive gap -> beat.
    assert derive_clustering_verdict(0.001, 0.99, 0.20) is ClusteringVerdict.CLUSTERS_BEAT_1N
    # Both supportive, negative gap -> lose.
    assert derive_clustering_verdict(0.001, 0.99, -0.20) is ClusteringVerdict.CLUSTERS_LOSE_TO_1N


# --------------------------------------------------------------------------- #
# 2. DSR trial-count guard                                                     #
# --------------------------------------------------------------------------- #
def _full_trial_count() -> int:
    """The FULL DSR multiplicity = product of every swept axis."""
    n_linkages = 3  # average / ward / single
    n_k_candidates = 5  # e.g. k in 2..6 on the OOS comparison
    n_schemes = 3  # 1/N, cluster-EW, stripped-HRP
    n_denoise = 2  # RMT on / off
    n_cost_grid = 4  # cost-bps grid compared on OOS
    return n_linkages * n_k_candidates * n_schemes * n_denoise * n_cost_grid


@pytest.mark.regression
def test_n_trials_equals_product_of_swept_axes() -> None:
    """run_diversification records exactly the FULL trial-count it was given."""
    n_trials = _full_trial_count()
    assert n_trials == 360

    # Build a tiny deterministic panel inline (no Group-A clustering needed).
    gen = np.random.default_rng(5)
    n_obs, n_assets = 320, 9
    idx = pd.date_range("2020-01-01", periods=n_obs, freq="B")
    cols = [f"A{i:02d}" for i in range(n_assets)]
    panel = pd.DataFrame(gen.standard_normal((n_obs, n_assets)) * 0.01, index=idx, columns=cols)
    labels = pd.Series([i % 3 for i in range(n_assets)], index=cols, dtype=int)

    res = run_diversification(panel, labels, lookback_window=40, n_trials=n_trials)
    assert isinstance(res, DiversificationResult)
    assert res.n_trials == n_trials  # under-count would be detectable here


@pytest.mark.regression
def test_undercounting_n_trials_inflates_dsr() -> None:
    """Under-counting n_trials manufactures false significance: it MUST raise DSR.

    This is the structural reason the FULL product is required. We assert the DSR
    computed at an under-count is strictly greater than at the honest full count,
    so an honest verdict cannot 'launder' significance by trimming the grid.
    """
    full = _full_trial_count()
    undercount = 1  # the dishonest "I only tried one thing" count

    common = {
        "n_obs": 400,
        "variance_of_trial_sharpes": 0.0015,
        "skew": 0.0,
        "kurtosis": 3.0,
    }
    dsr_full = deflated_sharpe_ratio(0.10, n_trials=full, **common)
    dsr_under = deflated_sharpe_ratio(0.10, n_trials=undercount, **common)

    assert dsr_under > dsr_full
    assert math.isfinite(dsr_full) and math.isfinite(dsr_under)


@pytest.mark.regression
def test_diversification_result_to_dict_is_json_safe(pure_noise: pd.DataFrame) -> None:
    """DiversificationResult.to_dict() renders finite floats and an int n_trials."""
    import json

    labels = _arbitrary_labels(pure_noise, k=3)
    res = run_diversification(pure_noise, labels, lookback_window=60, n_trials=24)
    d = res.to_dict()
    for key in (
        "one_over_n_sharpe",
        "cluster_ew_sharpe",
        "stripped_hrp_sharpe",
        "sharpe_diff_vs_1overN",
        "memmel_jk_pvalue",
        "deflated_sharpe",
        "cost_bps",
    ):
        assert isinstance(d[key], float)
    assert isinstance(d["n_trials"], int)
    json.dumps(d, default=str)  # must not raise


@pytest.mark.regression
def test_verdict_rejects_out_of_range_pvalue() -> None:
    """A p-value outside [0, 1] (or non-finite) is a definitional error."""
    from stockclusters._exceptions import ValidationError

    with pytest.raises(ValidationError):
        derive_clustering_verdict(1.5, 0.5, 0.1)
    with pytest.raises(ValidationError):
        derive_clustering_verdict(float("nan"), 0.5, 0.1)


@pytest.mark.regression
def test_dsr_rejects_trial_count_below_one() -> None:
    """A trial count below one is a definitional error and is rejected."""
    from stockclusters._exceptions import ValidationError

    with pytest.raises(ValidationError):
        deflated_sharpe_ratio(0.1, n_obs=300, n_trials=0, variance_of_trial_sharpes=0.001)
    with pytest.raises(ValidationError):
        # Same guard at the orchestration boundary.
        panel = pd.DataFrame(
            np.random.default_rng(1).standard_normal((100, 4)) * 0.01,
            index=pd.date_range("2020-01-01", periods=100, freq="B"),
            columns=[f"A{i:02d}" for i in range(4)],
        )
        labels = pd.Series([0, 0, 1, 1], index=panel.columns, dtype=int)
        run_diversification(panel, labels, lookback_window=20, n_trials=0)
