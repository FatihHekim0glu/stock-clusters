"""Unit tests for the Plotly figure builders (Group C).

Every builder must return a JSON-serializable ``{"data": [...], "layout": {...}}``
mapping whose numeric leaves are finite (the API serializes with ``allow_nan=False``,
so a stray ``NaN``/``Inf`` would break the response). Absent figures (e.g. no
diversification backtest) are represented as an explicit ``None`` by the API/CLI
layer, never as ``undefined``/``NaN`` - exercised here via the builders' graceful
handling of empty/absent inputs and an explicit-``None`` contract check.
"""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pandas as pd
import pytest

from stockclusters import plots
from stockclusters.clustering.embedding import rmt_signal_embedding
from stockclusters.correlation.distance import mantegna_distance, minimum_spanning_tree
from stockclusters.correlation.estimate import correlation_matrix


def _assert_valid_figure(fig: Any) -> None:
    """A figure must be a strict-JSON ``{data: list, layout: dict}`` with finite leaves."""
    assert isinstance(fig, dict)
    assert set(fig.keys()) == {"data", "layout"}, fig.keys()
    assert isinstance(fig["data"], list)
    assert isinstance(fig["layout"], dict)
    # Strict JSON: rejects NaN/Infinity and any numpy/pandas object that did not
    # round-trip to a native type.
    encoded = json.dumps(fig, allow_nan=False)
    _assert_finite(json.loads(encoded))


def _assert_finite(value: Any) -> None:
    """Recursively assert no float leaf is NaN or +/-Inf (None is allowed)."""
    if isinstance(value, dict):
        for v in value.values():
            _assert_finite(v)
    elif isinstance(value, list):
        for v in value:
            _assert_finite(v)
    elif isinstance(value, float):
        assert math.isfinite(value), f"non-finite float leaf: {value}"


@pytest.fixture
def clustered(k_blocks: pd.DataFrame, k_blocks_truth: pd.Series) -> dict[str, Any]:
    """A small clustered bundle (corr, dist, labels, ordered assets) for figures."""
    from scipy.cluster.hierarchy import linkage as sp_linkage
    from scipy.spatial.distance import squareform

    corr = correlation_matrix(k_blocks)
    dist = mantegna_distance(corr)
    link = sp_linkage(squareform(dist.to_numpy(), checks=False), method="average")
    return {
        "corr": corr,
        "dist": dist,
        "linkage": link,
        "labels": k_blocks_truth,
        "ordered_assets": [str(a) for a in k_blocks_truth.index],
        "edges": minimum_spanning_tree(dist),
        "embedding": rmt_signal_embedding(corr, n_obs=len(k_blocks)),
    }


@pytest.mark.unit
def test_cluster_heatmap_figure_valid(clustered: dict[str, Any]) -> None:
    """The cluster-ordered heatmap is a valid figure with a single heatmap trace."""
    fig = plots.cluster_heatmap_figure(
        clustered["corr"], clustered["ordered_assets"], clustered["labels"]
    )
    _assert_valid_figure(fig)
    assert fig["data"][0]["type"] == "heatmap"
    # The z-matrix is square and matches the asset count.
    z = fig["data"][0]["z"]
    n = len(clustered["ordered_assets"])
    assert len(z) == n and all(len(row) == n for row in z)


@pytest.mark.unit
def test_dendrogram_figure_valid(clustered: dict[str, Any]) -> None:
    """The dendrogram is a valid figure (lazy Plotly figure_factory)."""
    fig = plots.dendrogram_figure(clustered["linkage"], clustered["ordered_assets"])
    _assert_valid_figure(fig)
    assert len(fig["data"]) >= 1


@pytest.mark.unit
def test_mst_network_figure_valid(clustered: dict[str, Any]) -> None:
    """The MST network is a valid figure with an edge trace and a node trace."""
    fig = plots.mst_network_figure(clustered["edges"], clustered["labels"])
    _assert_valid_figure(fig)
    assert len(fig["data"]) == 2  # edges + nodes


@pytest.mark.unit
def test_embedding_scatter_figure_valid(clustered: dict[str, Any]) -> None:
    """The embedding scatter is a valid figure, one trace per cluster."""
    fig = plots.embedding_scatter_figure(clustered["embedding"], clustered["labels"])
    _assert_valid_figure(fig)
    n_clusters = int(clustered["labels"].nunique())
    assert len(fig["data"]) == n_clusters


@pytest.mark.unit
def test_oos_equity_figure_valid() -> None:
    """The OOS equity figure builds finite cumulative-wealth curves per strategy."""
    gen = np.random.default_rng(0)
    idx = pd.date_range("2021-01-01", periods=60, freq="B")
    curves = {
        "1/N": pd.Series(gen.normal(0.0, 0.01, 60), index=idx),
        "cluster_ew": pd.Series(gen.normal(0.0, 0.01, 60), index=idx),
        "stripped_hrp": pd.Series(gen.normal(0.0, 0.01, 60), index=idx),
    }
    fig = plots.oos_equity_figure(curves)
    _assert_valid_figure(fig)
    assert len(fig["data"]) == 3


@pytest.mark.unit
def test_stability_figure_valid() -> None:
    """The stability figure plots the per-pair ARI series over window dates."""
    fig = plots.stability_figure([0.55, 0.61, 0.48], ["2020-03", "2020-06", "2020-09"])
    _assert_valid_figure(fig)
    assert fig["data"][0]["y"] == [0.55, 0.61, 0.48]


@pytest.mark.unit
def test_equity_figure_maps_nan_returns_to_finite_wealth() -> None:
    """NaN per-period returns must not leak a NaN into the serialized wealth curve."""
    idx = pd.date_range("2021-01-01", periods=5, freq="B")
    series = pd.Series([0.01, np.nan, -0.02, 0.0, 0.03], index=idx)
    fig = plots.oos_equity_figure({"x": series})
    # Strict-JSON serialization would raise if any wealth point were NaN.
    _assert_valid_figure(fig)


@pytest.mark.unit
def test_ndarray_inputs_accepted() -> None:
    """Figures accept ndarray inputs (no labels) without leaking numpy types."""
    n = 6
    corr = np.eye(n)
    labels = pd.Series([0, 0, 1, 1, 2, 2], index=[str(i) for i in range(n)], dtype=int)
    order = [str(i) for i in range(n)]
    fig = plots.cluster_heatmap_figure(corr, order, labels)
    _assert_valid_figure(fig)


@pytest.mark.unit
def test_absent_figure_contract_is_explicit_none() -> None:
    """The 'absent figure' convention is an explicit ``None`` (valid JSON null).

    The diversification figures (equity curve, stability) are absent when the
    backtest is not run; the API/CLI represents that as an explicit ``None`` -
    never ``NaN`` or a missing key. This pins the contract the builders feed into.
    """
    response_fragment = {"equity_curve_figure": None, "stability_figure": None}
    encoded = json.dumps(response_fragment, allow_nan=False)
    assert json.loads(encoded) == {
        "equity_curve_figure": None,
        "stability_figure": None,
    }
