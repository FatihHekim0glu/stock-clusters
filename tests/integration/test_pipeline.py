"""End-to-end pipeline integration tests.

Exercises :func:`stockclusters.run_cluster_analysis` on the seeded ``k_blocks``
fixture (with and without the diversification horse race) and asserts the response
shape the backend router relies on: a frozen :class:`ClusterAnalysis`, a clean
JSON-serializable summary, and a full set of Plotly figures (with explicit ``None``
for figures that did not run).
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from stockclusters import (
    ClusterAnalysis,
    ClusterAnalysisParams,
    assemble_figures,
    run_cluster_analysis,
)
from stockclusters.allocation.schemes import DiversificationResult
from stockclusters.evaluation.verdict import ClusteringVerdict
from stockclusters.stability.resample import StabilityResult

_FIGURE_KEYS = (
    "heatmap_figure",
    "dendrogram_figure",
    "mst_figure",
    "embedding_figure",
    "equity_curve_figure",
    "stability_figure",
)


def _assert_valid_figure(fig: dict | None, *, required: bool) -> None:
    """A figure is either a valid ``{data, layout}`` mapping or explicit ``None``."""
    if fig is None:
        assert not required
        return
    assert isinstance(fig, dict)
    assert "data" in fig and "layout" in fig
    # Must be JSON-serializable (no numpy/pandas/NaN leaking through).
    json.dumps(fig)


@pytest.mark.integration
def test_pipeline_cluster_only_k_blocks(k_blocks: pd.DataFrame, k_blocks_truth: pd.Series) -> None:
    """Cluster-only run recovers the four planted blocks and shapes correctly."""
    params = ClusterAnalysisParams(
        method="hierarchical",
        linkage="average",
        n_clusters=4,
        run_diversification=False,
        run_stability=False,
    )
    analysis = run_cluster_analysis(k_blocks, params, data_source="synthetic")

    assert isinstance(analysis, ClusterAnalysis)
    assert analysis.n_clusters == 4
    assert analysis.selection_method == "fixed"
    # No diversification / stability requested -> explicit None.
    assert analysis.diversification is None
    assert analysis.stability is None
    assert analysis.verdict is None

    # Recovery: the recovered labels match the planted truth above threshold.
    from sklearn.metrics import adjusted_rand_score

    aligned_truth = k_blocks_truth.reindex(analysis.cluster_result.labels.index)
    ari = adjusted_rand_score(aligned_truth.to_numpy(), analysis.cluster_result.labels.to_numpy())
    assert ari >= 0.5

    # Summary is a clean, JSON-serializable scalar+membership bundle.
    summary = analysis.to_dict()
    json.dumps(summary)
    assert summary["n_clusters"] == 4
    assert summary["n_assets"] == 12
    assert summary["verdict"] is None
    assert isinstance(summary["clusters"], dict)

    # Figures: heatmap/dendrogram/mst/embedding present; equity/stability None.
    figures = assemble_figures(analysis)
    assert set(figures) == set(_FIGURE_KEYS)
    for key in ("heatmap_figure", "dendrogram_figure", "mst_figure", "embedding_figure"):
        _assert_valid_figure(figures[key], required=True)
    assert figures["equity_curve_figure"] is None
    assert figures["stability_figure"] is None


@pytest.mark.integration
def test_pipeline_with_diversification_k_blocks(k_blocks: pd.DataFrame) -> None:
    """Full run (diversification + stability) produces the complete response shape."""
    params = ClusterAnalysisParams(
        method="hierarchical",
        linkage="average",
        n_clusters=4,
        run_diversification=True,
        run_stability=True,
        train_window=252,
        cost_bps=5.0,
        gap_b=5,
    )
    analysis = run_cluster_analysis(k_blocks, params, data_source="synthetic")

    assert isinstance(analysis.diversification, DiversificationResult)
    assert isinstance(analysis.stability, StabilityResult)
    assert isinstance(analysis.verdict, ClusteringVerdict)

    # The DSR trial count must be the FULL product of swept axes: with a fixed k
    # there is no gap candidate grid, so n_trials == #weighting-schemes (2).
    assert analysis.n_trials >= 2

    div = analysis.diversification
    # All horse-race scalars are finite floats.
    for value in (
        div.one_over_n_sharpe,
        div.cluster_ew_sharpe,
        div.stripped_hrp_sharpe,
        div.sharpe_diff_vs_1overN,
        div.memmel_jk_pvalue,
        div.deflated_sharpe,
    ):
        assert isinstance(value, float)
    assert 0.0 <= div.memmel_jk_pvalue <= 1.0

    summary = analysis.to_dict()
    json.dumps(summary)
    assert "diversification" in summary
    assert summary["verdict"] in {v.value for v in ClusteringVerdict}
    assert "stability_ari_mean" in summary

    figures = assemble_figures(analysis)
    for key in _FIGURE_KEYS:
        _assert_valid_figure(figures[key], required=True)


@pytest.mark.integration
def test_pipeline_auto_k_folds_gap_into_trial_count(k_blocks: pd.DataFrame) -> None:
    """Auto-k selection folds the gap candidate grid into the DSR trial count."""
    params = ClusterAnalysisParams(
        method="hierarchical",
        n_clusters=None,
        k_min=2,
        k_max=8,
        run_diversification=True,
        train_window=252,
        gap_b=5,
    )
    analysis = run_cluster_analysis(k_blocks, params)

    assert analysis.gap_result is not None
    assert analysis.selection_method == "tibshirani_1se"
    n_k = len(analysis.gap_result.k_candidates)
    # FULL trial count: #k-candidates x #weighting-schemes (>= product of axes).
    assert analysis.n_trials >= n_k * 2
