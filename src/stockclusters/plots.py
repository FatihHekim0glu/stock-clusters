"""Plotly figure builders for the clustering tool.

Each builder returns a plain ``dict`` shaped ``{"data": [...], "layout": {...}}``
— the same JSON shape the FastAPI layer serializes and the Next.js ``PlotlyChart``
component renders — so the figures cross the API boundary with no Plotly object
leaking through. Plotly is an OPTIONAL dependency (the ``viz`` extra) and is
imported LAZILY inside each builder; importing this module has no side effects and
does not require Plotly.

Importing this module has no side effects.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from stockclusters._typing import MatrixLike, ReturnsLike

#: A Plotly figure serialized as a plain mapping with ``data`` and ``layout`` keys.
FigureDict = dict[str, Any]

__all__ = [
    "FigureDict",
    "cluster_heatmap_figure",
    "dendrogram_figure",
    "embedding_scatter_figure",
    "mst_network_figure",
    "oos_equity_figure",
    "stability_figure",
]


def cluster_heatmap_figure(
    corr: MatrixLike,
    ordered_assets: Sequence[str],
    labels: pd.Series,
) -> FigureDict:
    """Cluster-ordered correlation heatmap.

    Reorders the correlation matrix by dendrogram leaf order so blocks of
    high-correlation assets appear on the diagonal, annotated by cluster.

    Parameters
    ----------
    corr:
        The ``N x N`` correlation matrix, labelled by asset.
    ordered_assets:
        Asset tickers in dendrogram-leaf order.
    labels:
        Integer cluster labels indexed by asset.

    Returns
    -------
    FigureDict
        A ``{"data", "layout"}`` mapping (lazy Plotly import).
    """
    raise NotImplementedError


def dendrogram_figure(linkage: np.ndarray, labels: Sequence[str]) -> FigureDict:
    """Dendrogram of the agglomerative clustering.

    Parameters
    ----------
    linkage:
        The ``(N - 1) x 4`` SciPy linkage matrix.
    labels:
        Asset tickers in the original (pre-linkage) order.

    Returns
    -------
    FigureDict
        A ``{"data", "layout"}`` mapping (lazy Plotly import).
    """
    raise NotImplementedError


def mst_network_figure(
    edges: pd.DataFrame,
    labels: pd.Series,
) -> FigureDict:
    """Minimum-spanning-tree network of the correlation graph.

    Parameters
    ----------
    edges:
        The MST edge list (columns ``["source", "target", "weight"]``).
    labels:
        Integer cluster labels indexed by asset (colours the nodes).

    Returns
    -------
    FigureDict
        A ``{"data", "layout"}`` mapping (lazy Plotly import).
    """
    raise NotImplementedError


def embedding_scatter_figure(
    embedding: MatrixLike,
    labels: pd.Series,
) -> FigureDict:
    """2-D scatter of the RMT-signal embedding, coloured by cluster.

    Parameters
    ----------
    embedding:
        An ``N x d`` embedding (the first two components are plotted).
    labels:
        Integer cluster labels indexed by asset.

    Returns
    -------
    FigureDict
        A ``{"data", "layout"}`` mapping (lazy Plotly import).
    """
    raise NotImplementedError


def oos_equity_figure(equity_curves: Mapping[str, ReturnsLike]) -> FigureDict:
    """Out-of-sample equity curves for each allocation strategy.

    Parameters
    ----------
    equity_curves:
        A mapping ``{strategy_name: oos_return_series}`` (1/N, cluster-EW,
        stripped-HRP).

    Returns
    -------
    FigureDict
        A ``{"data", "layout"}`` mapping (lazy Plotly import).
    """
    raise NotImplementedError


def stability_figure(
    ari_series: Sequence[float],
    window_dates: Sequence[str],
) -> FigureDict:
    """Adjacent-window ARI over time (cluster-stability chart).

    Parameters
    ----------
    ari_series:
        The per-adjacent-pair ARI values.
    window_dates:
        The (end) date of each window pair, parallel to ``ari_series``.

    Returns
    -------
    FigureDict
        A ``{"data", "layout"}`` mapping (lazy Plotly import).
    """
    raise NotImplementedError
