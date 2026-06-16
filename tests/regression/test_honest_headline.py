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

    # The verdict is a PURE function of the inference; on pure noise it can NEVER be
    # the directional "clusters beat 1/N" over-claim. With the leakage fixed the
    # clusters are RE-FIT per walk-forward train window on noise, so they either
    # match 1/N (NO_SIGNIFICANT_DIFFERENCE) or honestly LOSE to it
    # (CLUSTERS_LOSE_TO_1N) — both are honest null outcomes; only CLUSTERS_BEAT_1N
    # is forbidden.
    assert analysis.verdict is not ClusteringVerdict.CLUSTERS_BEAT_1N
    assert analysis.verdict in (
        ClusteringVerdict.NO_SIGNIFICANT_DIFFERENCE,
        ClusteringVerdict.CLUSTERS_LOSE_TO_1N,
    )

    # Structural honesty: a "beats 1/N" verdict requires a significant Memmel-JK
    # test AND a positive deflated Sharpe AND a positive cluster-minus-1/N gap.
    # On the null that full conjunction never holds.
    beats = (
        div.memmel_jk_pvalue < 0.05
        and div.deflated_sharpe > 0.0
        and div.sharpe_diff_vs_1overN > 0.0
    )
    assert not beats


def _independent_trial_floor(params: ClusterAnalysisParams, n_assets: int) -> int:
    """Independent DSR-trial floor derived from the REQUEST/config only.

    Deliberately re-derives the swept-axis product from the public request
    parameters (NOT from the pipeline's internal constants), so an accidental
    under-count inside the pipeline fails this guard:

        floor = #clustering-families x #k-candidates x #weighting-schemes
                x #denoise-settings x #cost-grid-points

    - families: 2 for ``method="both"`` (hierarchical AND kmeans), else 1;
    - k-candidates: ``k_max - k_min + 1`` for auto-k (the gap grid), clamped to the
      assets, else 1 for a fixed ``n_clusters``;
    - schemes: 2 (cluster-EW + stripped-HRP are both raced against 1/N);
    - denoise settings: 1 (the pipeline reports a single setting);
    - cost-grid points: 1 (a single OOS cost is reported).
    """
    families = 2 if params.method == "both" else 1
    if params.n_clusters is not None and int(params.n_clusters) > 0:
        n_k = 1
    else:
        k_max = min(int(params.k_max), n_assets - 1)
        k_min = max(2, min(int(params.k_min), k_max))
        n_k = k_max - k_min + 1
    schemes, denoise_settings, cost_points = 2, 1, 1
    return families * n_k * schemes * denoise_settings * cost_points


@pytest.mark.regression
@pytest.mark.parametrize("method", ["hierarchical", "both"])
def test_dsr_trial_count_guard(pure_noise: pd.DataFrame, method: str) -> None:
    """The DSR n_trials is never below the INDEPENDENTLY-computed swept-axis floor.

    The floor is derived from the request/config parameters by
    :func:`_independent_trial_floor` (NOT re-derived from the same hardcoded
    constants the pipeline used), so an accidental under-count fails CI. The
    clustering-family axis (2 for ``method="both"``) must be included.
    """
    params = ClusterAnalysisParams(
        method=method,
        n_clusters=None,
        k_min=2,
        k_max=8,
        run_diversification=True,
        train_window=252,
        gap_b=5,
    )
    analysis = run_cluster_analysis(pure_noise, params)
    assert analysis.gap_result is not None

    expected_floor = _independent_trial_floor(params, n_assets=pure_noise.shape[1])
    assert analysis.n_trials >= expected_floor
    # "both" must actually raise the count via the family axis (catches a silent
    # family-axis under-count even if the gap grid alone happens to clear the floor).
    if method == "both":
        assert analysis.n_trials >= 2 * len(analysis.gap_result.k_candidates)
    assert analysis.diversification is not None
    assert analysis.diversification.n_trials == analysis.n_trials
