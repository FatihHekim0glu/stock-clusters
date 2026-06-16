"""Cluster-quality and post-hoc diagnostic metrics.

Internal-validity metrics (silhouette, cophenetic correlation, modularity) and the
post-hoc ARI-vs-GICS comparison.

HONESTY REQUIREMENT: GICS sector membership is used ONLY post-hoc, to report how
much the correlation clusters re-discover sectors (the expected ARI ~0.4-0.7). GICS
NEVER enters the distance computation or the ``k`` selection.

PARITY: ``silhouette_score`` matches ``sklearn.metrics`` to ``1e-10``.

Importing this module has no side effects.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from stockclusters._typing import MatrixLike

__all__ = [
    "ari_vs_gics",
    "cophenetic_correlation",
    "modularity",
    "silhouette_score",
]


def silhouette_score(dist: MatrixLike, labels: pd.Series) -> float:
    r"""Mean silhouette score of a labeling on a precomputed distance matrix.

    Uses the silhouette definition with ``metric="precomputed"``: for each asset,
    ``(b - a) / max(a, b)`` where ``a`` is mean intra-cluster distance and ``b`` the
    mean nearest-other-cluster distance, averaged over assets.

    PARITY REQUIREMENT: matches ``sklearn.metrics.silhouette_score`` (precomputed)
    to ``1e-10``.

    Parameters
    ----------
    dist:
        An ``N x N`` symmetric distance matrix.
    labels:
        Integer cluster labels indexed by asset (aligned to ``dist``).

    Returns
    -------
    float
        The mean silhouette score in ``[-1, 1]``.

    Raises
    ------
    ValidationError
        If there are fewer than two clusters or labels/dist disagree.
    """
    raise NotImplementedError


def cophenetic_correlation(linkage: np.ndarray, dist: MatrixLike) -> float:
    r"""Cophenetic correlation between a linkage tree and the original distances.

    Pearson correlation between the condensed original distances and the
    cophenetic distances implied by ``linkage`` — a measure of how faithfully the
    dendrogram preserves pairwise distances.

    PARITY REQUIREMENT: matches ``scipy.cluster.hierarchy.cophenet`` to ``1e-10``.

    Parameters
    ----------
    linkage:
        The ``(N - 1) x 4`` SciPy linkage matrix.
    dist:
        The ``N x N`` distance matrix the linkage was built from.

    Returns
    -------
    float
        The cophenetic correlation coefficient.

    Raises
    ------
    ValidationError
        If shapes are inconsistent.
    """
    raise NotImplementedError


def modularity(labels: pd.Series, corr: MatrixLike) -> float:
    r"""Newman modularity of a labeling over a correlation-weighted graph.

    Treats the (thresholded, non-negative) correlation matrix as a weighted graph
    and computes the modularity ``Q`` of the cluster partition — the fraction of
    within-cluster edge weight minus its expectation under a degree-preserving
    null. Reported as an MST/network cross-check on the gap selection.

    Parameters
    ----------
    labels:
        Integer cluster labels indexed by asset.
    corr:
        The ``N x N`` correlation matrix, labelled by asset.

    Returns
    -------
    float
        The modularity ``Q`` (higher = stronger community structure).

    Raises
    ------
    ValidationError
        If ``labels`` and ``corr`` labels disagree.
    """
    raise NotImplementedError


def ari_vs_gics(labels: pd.Series, gics: Mapping[str, str]) -> float:
    r"""Post-hoc Adjusted Rand Index between clusters and GICS sectors.

    HONESTY REQUIREMENT: this is a POST-HOC diagnostic only. GICS labels never
    enter the distance matrix or the ``k`` selection; this just measures how much
    the correlation clusters re-discover the sector taxonomy (expected
    ARI ~0.4-0.7).

    Parameters
    ----------
    labels:
        Integer cluster labels indexed by asset ticker.
    gics:
        A mapping ``{ticker: gics_sector}`` for (a superset of) the clustered
        assets.

    Returns
    -------
    float
        The Adjusted Rand Index between the clusters and the GICS sectors on their
        shared assets.

    Raises
    ------
    ValidationError
        If fewer than two assets have a GICS sector.
    """
    raise NotImplementedError
