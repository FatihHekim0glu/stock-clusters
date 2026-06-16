"""Agglomerative hierarchical clustering of the correlation network.

Runs ``scipy.cluster.hierarchy`` linkage (average / ward / single) on a Mantegna
distance matrix, then cuts the dendrogram at ``k`` clusters. The frozen
:class:`ClusterResult` is the canonical output bundle carried through the rest of
the pipeline (allocation, stability, evaluation).

Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from stockclusters._typing import MatrixLike

__all__ = ["ClusterResult", "cut_tree", "hierarchical_clusters"]

#: Linkage methods accepted by :func:`hierarchical_clusters`. ``average`` is the
#: reported default for correlation networks; ``ward``/``single`` are ablations.
VALID_METHODS: frozenset[str] = frozenset({"average", "ward", "single"})


@dataclass(frozen=True, slots=True)
class ClusterResult:
    """Immutable result of a clustering of the asset universe.

    Attributes
    ----------
    labels:
        Integer cluster label per asset, indexed by asset ticker.
    n_clusters:
        The number of distinct clusters (``labels.nunique()``).
    method:
        The clustering method used (e.g. ``"hierarchical:average"`` or
        ``"kmeans"``).
    linkage:
        The ``(N - 1) x 4`` SciPy linkage matrix when a hierarchical method was
        used; ``None`` for flat methods (e.g. K-means).
    ordered_assets:
        Asset tickers in dendrogram-leaf order (for the cluster-ordered heatmap);
        falls back to input order for flat methods.
    silhouette:
        Mean silhouette score of ``labels`` on the distance matrix.
    """

    labels: pd.Series
    n_clusters: int
    method: str
    ordered_assets: list[str]
    silhouette: float
    linkage: np.ndarray | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this result.

        ``labels`` is rendered as an ordered ``{ticker: cluster_id}`` mapping and
        ``linkage`` as a nested list (or ``None``) so the result crosses the API
        boundary without numpy/pandas types leaking through.
        """
        out = asdict(self)
        out["labels"] = {str(k): int(v) for k, v in self.labels.items()}
        out["linkage"] = (
            None if self.linkage is None else np.asarray(self.linkage).tolist()
        )
        out["silhouette"] = float(self.silhouette)
        return out


def hierarchical_clusters(
    dist: MatrixLike,
    *,
    n_clusters: int,
    method: str = "average",
) -> ClusterResult:
    r"""Agglomerative clustering of a distance matrix cut at ``n_clusters``.

    Condenses ``dist`` to vector form, runs ``scipy.cluster.hierarchy.linkage``
    with ``method``, and cuts the resulting tree into exactly ``n_clusters`` flat
    clusters.

    HONESTY REQUIREMENT: clustering fits on the supplied (train-window) distance
    matrix only; the resulting labels are frozen before being applied to any
    out-of-sample window.

    Parameters
    ----------
    dist:
        An ``N x N`` symmetric Mantegna distance matrix.
    n_clusters:
        The number of flat clusters to cut the dendrogram into (``1 <= k <= N``).
    method:
        One of :data:`VALID_METHODS`. Defaults to ``"average"``.

    Returns
    -------
    ClusterResult
        The frozen clustering bundle.

    Raises
    ------
    ValidationError
        If ``dist`` is not square/symmetric, ``method`` is unknown, or
        ``n_clusters`` is out of range.
    """
    raise NotImplementedError


def cut_tree(linkage: np.ndarray, *, n_clusters: int, labels: list[str]) -> pd.Series:
    r"""Cut a SciPy linkage matrix into ``n_clusters`` flat clusters.

    Thin wrapper over ``scipy.cluster.hierarchy.fcluster`` with
    ``criterion="maxclust"`` that returns a label Series indexed by asset.

    MONOTONICITY REQUIREMENT: ``n_clusters = N`` yields singletons and
    ``n_clusters = 1`` yields a single cluster; a higher cut never increases the
    cluster count (property-tested).

    Parameters
    ----------
    linkage:
        The ``(N - 1) x 4`` SciPy linkage matrix.
    n_clusters:
        The number of flat clusters to produce.
    labels:
        Asset tickers, in the original (pre-linkage) order, used to index the
        returned Series.

    Returns
    -------
    pandas.Series
        Integer cluster labels indexed by asset ticker.

    Raises
    ------
    ValidationError
        If ``n_clusters`` is out of range or ``labels`` length mismatches.
    """
    raise NotImplementedError
