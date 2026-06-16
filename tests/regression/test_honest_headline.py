"""Regression: honest-null headline guard (end-to-end).

On the ``pure_noise`` fixture the diversification horse race MUST come back with no
significant cluster-vs-1/N edge and the headline verdict MUST be
``NO_SIGNIFICANT_DIFFERENCE`` — this is the literature-consistent honest null, not a
bug. Also pins the DSR trial-count guard: ``n_trials`` is never *less* than the
product of the swept axes (under-counting manufactures false significance).
"""

from __future__ import annotations

import pandas as pd
import pytest

from stockclusters import ClusterAnalysisParams, run_cluster_analysis
from stockclusters.evaluation.verdict import ClusteringVerdict


@pytest.mark.regression
def test_pure_noise_is_insignificant(pure_noise: pd.DataFrame) -> None:
    """Pure noise yields no significant cluster-vs-1/N edge (the honest null)."""
    params = ClusterAnalysisParams(
        method="hierarchical",
        n_clusters=None,
        k_min=2,
        k_max=8,
        run_diversification=True,
        train_window=252,
        gap_b=5,
    )
    analysis = run_cluster_analysis(pure_noise, params)

    assert analysis.diversification is not None
    div = analysis.diversification

    # The verdict is a PURE function of the inference; on pure noise it cannot be a
    # directional "beats 1/N" claim. Either the Memmel-JK test is insignificant or
    # the deflated Sharpe is non-positive (usually both), so the honest verdict is
    # NO_SIGNIFICANT_DIFFERENCE.
    assert analysis.verdict is ClusteringVerdict.NO_SIGNIFICANT_DIFFERENCE
    assert analysis.verdict is not ClusteringVerdict.CLUSTERS_BEAT_1N

    # Structural honesty: a "beats 1/N" verdict requires BOTH a significant
    # Memmel-JK test AND a positive deflated Sharpe. On the null, at least one fails.
    beats = div.memmel_jk_pvalue < 0.05 and div.deflated_sharpe > 0.0
    assert not beats


@pytest.mark.regression
def test_dsr_trial_count_guard(pure_noise: pd.DataFrame) -> None:
    """The DSR n_trials is never below the product of the swept axes."""
    params = ClusterAnalysisParams(
        method="hierarchical",
        n_clusters=None,
        k_min=2,
        k_max=8,
        run_diversification=True,
        train_window=252,
        gap_b=5,
    )
    analysis = run_cluster_analysis(pure_noise, params)
    assert analysis.gap_result is not None

    # Swept axes the pipeline actually explored:
    #   #linkages (1) x #k-candidates x #weighting-schemes (2) x
    #   #denoise-settings (1) x #cost-grid-points (1).
    n_k = len(analysis.gap_result.k_candidates)
    expected_floor = 1 * n_k * 2 * 1 * 1
    assert analysis.n_trials >= expected_floor
    assert analysis.diversification is not None
    assert analysis.diversification.n_trials == analysis.n_trials
