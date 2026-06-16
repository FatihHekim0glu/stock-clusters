"""Unit tests for the pipeline orchestration helpers and branches.

Covers the DSR trial-count helper, the K-means method branch (no linkage -> no
dendrogram figure), the GICS post-hoc ARI hook, and the explicit-``None`` figure
contract of :func:`stockclusters.assemble_figures`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stockclusters import (
    ClusterAnalysisParams,
    assemble_figures,
    run_cluster_analysis,
)
from stockclusters.pipeline import _dsr_trial_count


def _k_blocks_panel(seed: int = 11) -> pd.DataFrame:
    n_assets, block_size, within, across, n_obs = 12, 3, 0.75, 0.10, 600
    corr = np.full((n_assets, n_assets), across)
    for b in range(0, n_assets, block_size):
        corr[b : b + block_size, b : b + block_size] = within
    np.fill_diagonal(corr, 1.0)
    gen = np.random.default_rng(seed)
    chol = np.linalg.cholesky(corr)
    data = (gen.standard_normal((n_obs, n_assets)) @ chol.T) * 0.01
    idx = pd.date_range("2020-01-01", periods=n_obs, freq="B")
    return pd.DataFrame(data, index=idx, columns=[f"A{i:02d}" for i in range(n_assets)])


@pytest.mark.unit
def test_dsr_trial_count_is_full_product() -> None:
    # 3 linkages x 5 k x 2 schemes x 1 denoise x 4 cost points = 120.
    assert (
        _dsr_trial_count(
            n_linkages=3,
            n_k_candidates=5,
            n_weighting_schemes=2,
            n_denoise_settings=1,
            n_cost_points=4,
        )
        == 120
    )
    # Every factor is floored at 1.
    assert _dsr_trial_count(n_linkages=0, n_k_candidates=0) == 2  # 1 * 1 * 2 * 1 * 1


@pytest.mark.unit
def test_pipeline_kmeans_branch_has_no_dendrogram() -> None:
    panel = _k_blocks_panel()
    params = ClusterAnalysisParams(method="kmeans", n_clusters=4)
    analysis = run_cluster_analysis(panel, params)

    # K-means is a flat method: no linkage, so no dendrogram figure.
    assert analysis.cluster_result.linkage is None
    figures = assemble_figures(analysis)
    assert figures["dendrogram_figure"] is None
    # Heatmap / MST / embedding still produced.
    assert figures["heatmap_figure"] is not None
    assert figures["mst_figure"] is not None


@pytest.mark.unit
def test_pipeline_gics_posthoc_ari() -> None:
    panel = _k_blocks_panel()
    gics = {f"A{i:02d}": f"sector{i // 3}" for i in range(12)}
    params = ClusterAnalysisParams(method="hierarchical", n_clusters=4)
    analysis = run_cluster_analysis(panel, params, gics=gics)

    assert analysis.ari_vs_gics is not None
    # The planted blocks align with the GICS map, so ARI should be high.
    assert analysis.ari_vs_gics > 0.5
    assert analysis.to_dict()["ari_vs_gics"] is not None


@pytest.mark.unit
def test_pipeline_no_gics_leaves_ari_none() -> None:
    panel = _k_blocks_panel()
    params = ClusterAnalysisParams(method="hierarchical", n_clusters=3)
    analysis = run_cluster_analysis(panel, params)
    assert analysis.ari_vs_gics is None
    assert analysis.to_dict()["ari_vs_gics"] is None


@pytest.mark.unit
def test_pipeline_default_params_run() -> None:
    """run_cluster_analysis with no params uses defaults (method="both", auto-k)."""
    panel = _k_blocks_panel()
    analysis = run_cluster_analysis(panel)
    assert analysis.diversification is None
    assert analysis.stability is None
    assert analysis.gap_result is not None
    # The default method is "both": the gap rule is the base selector and the
    # "+both(won=...)" suffix records that both families ran and which one won.
    assert analysis.selection_method.startswith("tibshirani_1se")
    assert "+both(won=" in analysis.selection_method


@pytest.mark.unit
def test_pipeline_both_runs_both_families_and_counts_family_axis() -> None:
    """method="both" fits BOTH families, keeps the higher-silhouette display map,
    records the winner in selection_method, and counts the family as a DSR axis."""
    panel = _k_blocks_panel()
    n_clusters = 4

    hier = run_cluster_analysis(
        panel, ClusterAnalysisParams(method="hierarchical", n_clusters=n_clusters)
    )
    kmn = run_cluster_analysis(panel, ClusterAnalysisParams(method="kmeans", n_clusters=n_clusters))
    both = run_cluster_analysis(panel, ClusterAnalysisParams(method="both", n_clusters=n_clusters))

    # The display map kept by "both" is the higher-silhouette of the two families.
    assert "+both(won=" in both.selection_method
    won = "kmeans" if "won=kmeans" in both.selection_method else "hierarchical"
    expected = kmn if won == "kmeans" else hier
    assert both.cluster_result.silhouette == pytest.approx(expected.cluster_result.silhouette)
    assert both.cluster_result.silhouette >= min(
        hier.cluster_result.silhouette, kmn.cluster_result.silhouette
    )

    # Fixed k -> no gap grid -> family is the only swept axis beyond the schemes.
    # "both" counts the family axis (2) where a single family counts 1.
    assert both.n_trials == 2 * hier.n_trials
    assert hier.n_trials == kmn.n_trials
