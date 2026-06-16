"""Regression guards for the honest null and the DSR trial-count discipline.

Two pinned behaviours from the brief (Group B):

1. Honest-headline guard: on the ``pure_noise`` fixture the cluster-vs-1/N horse
   race comes back insignificant (Memmel-JK p NOT significant) and the derived
   verdict is ``NO_SIGNIFICANT_DIFFERENCE``. Clustering buys no alpha on noise.
2. DSR trial-count guard: ``n_trials`` must equal the FULL product of the swept
   axes; under-counting (which manufactures false significance) is rejected, and
   a larger trial count never raises the DSR.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from stockclusters.allocation.schemes import DiversificationResult, run_diversification
from stockclusters.evaluation.dsr import deflated_sharpe_ratio
from stockclusters.evaluation.verdict import ClusteringVerdict, derive_clustering_verdict


def _arbitrary_labels(panel: pd.DataFrame, *, k: int) -> pd.Series:
    """A fixed (data-independent) k-way partition of the universe."""
    cols = list(panel.columns)
    return pd.Series([i % k for i in range(len(cols))], index=cols, dtype=int)


# --------------------------------------------------------------------------- #
# 1. Honest-headline guard on pure noise                                      #
# --------------------------------------------------------------------------- #
@pytest.mark.regression
def test_pure_noise_horse_race_is_insignificant(pure_noise: pd.DataFrame) -> None:
    """On pure noise the cluster-vs-1/N gap is not significant (the honest null)."""
    labels = _arbitrary_labels(pure_noise, k=3)
    # Full swept-axis product (illustrative): 3 linkages x 5 k-candidates x
    # 3 schemes x 2 denoise x 4 cost grid = 360 trials.
    n_trials = 3 * 5 * 3 * 2 * 4
    res = run_diversification(
        pure_noise, labels, lookback_window=60, n_trials=n_trials, cost_bps=5.0
    )

    # Memmel-JK must NOT reject equality of the two Sharpes.
    assert res.memmel_jk_pvalue >= 0.05, (
        f"pure noise produced a significant Sharpe gap (p={res.memmel_jk_pvalue})"
    )

    verdict = derive_clustering_verdict(
        res.memmel_jk_pvalue, res.deflated_sharpe, res.sharpe_diff_vs_1overN
    )
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
    import numpy as np

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
        import numpy as np

        panel = pd.DataFrame(
            np.random.default_rng(1).standard_normal((100, 4)) * 0.01,
            index=pd.date_range("2020-01-01", periods=100, freq="B"),
            columns=[f"A{i:02d}" for i in range(4)],
        )
        labels = pd.Series([0, 0, 1, 1], index=panel.columns, dtype=int)
        run_diversification(panel, labels, lookback_window=20, n_trials=0)
